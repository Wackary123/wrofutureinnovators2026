"""
ATLAS — Museum Helmet main script (streaming + interruptible version).

What changed vs the previous version
------------------------------------
1. Gemini calls STREAM token-by-token (generate_content_stream) and are
   broken into sentences on the fly. Piper starts speaking sentence 1 while
   Gemini is still writing sentence 3. Perceived latency drops a lot.

2. Continuous background STT worker. The mic is ALWAYS transcribing into
   a queue. The main loop decides what to do with each utterance based on
   the current mode.

3. Two modes:
     - IDLE: only wake words matter. Camera auto-triggers still work.
       Random bystander speech is ignored -> safe in a loud demo.
     - ACTIVE: short window (default 10s) after the last exchange during
       which follow-ups don't need a wake word. Interrupt keywords cancel
       the current response.

4. Interrupts wait for the current SENTENCE to finish, not the whole
   response. Feels responsive without a jarring hard-cut.

5. Noise defenses:
     - min word count (>= 3) and min duration (>= 1s) to accept an
       utterance outside of wake-word matches
     - optional Vosk confidence gate (falls back gracefully if the
       wrapper doesn't expose per-word conf scores)

6. Pluggable STT. Today it still uses Fusion HAT's Vosk wrapper. When
   external USB mic + speakers arrive, swap only the `_listen_forever`
   method (marked with TODO blocks).

Hardware notes
--------------
Current: Fusion HAT+ mic + speaker.
Incoming: MillSO MQ5 USB lavalier (USB PnP -> appears as an ALSA capture
card, no driver needed) + YARCHONN speaker (USB is power only, audio is
3.5mm analog from the Pi).

Audio device selection: set AUDIO_OUT_DEVICE below. `None` lets ALSA pick
the system default, which is the right thing to try first after plugging
in new hardware. If aplay routes to the wrong device, set this to
"plughw:X,0" where X is the card index from `aplay -l`.
"""

import os
import re
import time
import threading
import queue
import subprocess
import tempfile

import cv2
from dotenv import load_dotenv
from google import genai

# TODO(external-mic): when USB mic arrives, we can keep this import or
# swap to a vosk+sounddevice implementation. See _listen_forever below.
from fusion_hat.stt import Vosk as STT  # type: ignore

from picamera2 import Picamera2  # type: ignore
from ultralytics import YOLOE  # type: ignore


# --------------------------------------------------------------------------
# Tunable constants — most of what you'll want to change lives here.
# --------------------------------------------------------------------------

# Audio output device for aplay. None = system default. Override like
# "plughw:2,0" after checking `aplay -l` on the Pi.
AUDIO_OUT_DEVICE: str | None = None

# Interaction mode timing.
ACTIVE_WINDOW_SECONDS = 10.0   # how long "active" mode lingers after last turn
IDLE_STT_MIN_WORDS = 2         # in IDLE, any utterance shorter than this is ignored unless it contains a wake word
ACTIVE_STT_MIN_WORDS = 3       # in ACTIVE, anything shorter than this is ignored (bystander guard)
ACTIVE_STT_MIN_SECONDS = 1.0   # minimum speaking duration to consider valid
VOSK_MIN_CONF = 0.5            # per-word avg confidence threshold; ignored if not available

# Keywords.
WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
INTERRUPT_WORDS = ("stop", "wait", "actually", "never mind", "nevermind", "cancel", "hold on")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")
QUIET_WORDS = ("stop", "be quiet", "silence")  # "stop" is overloaded; resolved by context


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        # --- AI setup ---
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_model = "gemini-2.5-flash"

        # --- STT ---
        self.stt = STT(language="en-us")

        # --- Piper TTS ---
        self.piper_model = "en_US-ryan-medium"
        self.piper_data_dir = os.path.expanduser("~/piper_voices")

        # --- Camera / YOLOE ---
        self.camera_size = (800, 800)
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "dog", "phone", "clock", "hoodie", "computer", "box", "plant",
            "tape", "mona lisa", "vase", "hair", "person", "table",
            "light", "fruit", "chair", "couch",
        ]
        self.confidence_threshold = 0.20
        self.object_hold_seconds = 2
        self.object_cooldown_seconds = 8.0
        self.model_imgsz = 160  # lower for more FPS; raise for more detail

        # Detection tracking
        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        # --- Inter-thread state ---
        # Raw transcriptions coming in from the mic.
        self.utterance_queue: queue.Queue = queue.Queue()

        # Global stop flag for shutdown.
        self.stop_event = threading.Event()

        # Signals from main loop to the TTS/Gemini worker that the
        # current response should be canceled at the next sentence
        # boundary.
        self.cancel_response_event = threading.Event()

        # Set while the helmet is actively speaking or streaming. Lets
        # the camera logic avoid stepping on the current interaction.
        self.is_busy_event = threading.Event()

        # Timestamp of last finalized turn. Used to decide IDLE vs ACTIVE mode.
        self.last_turn_time = 0.0

        # Protects piper subprocess handle (used for interrupt hard-stop on shutdown).
        self._piper_lock = threading.Lock()
        self._current_aplay_proc: subprocess.Popen | None = None

        # --- System prompt ---
        self.system_prompt = """
You are an AI museum guide embedded in a wearable helmet, speaking directly to a visitor in front of an exhibit. You also have a secondary, non-intrusive safety role.

Personality & Style
Speak like a real human guide: warm, natural, and conversational
Avoid sounding robotic, scripted, or like a textbook
Keep responses concise, but prioritize clarity and engagement over strict brevity
Usually respond in 1–3 sentences (up to 4–5 if needed for clarity)
Prefer short back-and-forth interaction over long explanations
Occasionally ask one short follow-up question, especially after explaining an exhibit
These rules guide behavior, but natural conversation should always come first
When first addressing a visitor, briefly greet them naturally
Adjust energy depending on the exhibit

Core Behavior
Give clear, simple, and meaningful explanations
Focus on culture, history, artifacts, symbolism, and context
When explaining an object:
say what it is
explain why it matters
add one interesting or surprising detail
When appropriate, include a short story, human element, or engaging fact
Adapt to the visitor's level:
simplify for beginners or children
add depth for advanced questions
If unsure, acknowledge uncertainty calmly while still giving helpful context
Avoid vague or generic explanations—always give a specific reason

Interaction Rules
Do not overwhelm the visitor with too much information
Avoid long monologues unless explicitly requested
Break information into small, digestible pieces
Match the visitor's tone (curious, excited, confused)
If the visitor is incorrect, correct them politely and briefly
If the question is off-topic, gently guide the conversation back to the exhibit

Vision & Context Awareness
If an object may be misidentified, acknowledge uncertainty and still provide helpful context
If input is unclear, politely ask the visitor to repeat
Vary phrasing to avoid sounding repetitive
When the interaction ends, close the conversation naturally

Privacy & Safety
Keep the safety role subtle and secondary
When relevant, highlight awareness and protection in a calm, reassuring way
Never mention storing, tracking, or saving personal data

Overall Goal
Act like a knowledgeable, friendly guide beside the visitor—making the experience interactive, human, engaging, and memorable.
"""

    # --------------------------------------------------------------------
    # TTS — speak a single chunk of text. Generates WAV with Piper, plays
    # with aplay. Tracks the aplay subprocess so shutdown can kill it.
    # --------------------------------------------------------------------
    def _speak_chunk(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name
        try:
            subprocess.run(
                [
                    "python3", "-m", "piper",
                    "--model", self.piper_model,
                    "--data-dir", self.piper_data_dir,
                    "--output-file", wav_path,
                ],
                input=text,
                text=True,
                check=True,
            )
            aplay_cmd = ["aplay"]
            if AUDIO_OUT_DEVICE:
                aplay_cmd += ["-D", AUDIO_OUT_DEVICE]
            aplay_cmd.append(wav_path)

            with self._piper_lock:
                self._current_aplay_proc = subprocess.Popen(aplay_cmd)
            self._current_aplay_proc.wait()
            with self._piper_lock:
                self._current_aplay_proc = None
        except subprocess.CalledProcessError as e:
            print(f"[TTS] piper/aplay failed: {e}")
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _hard_stop_aplay(self) -> None:
        """Kill any currently playing audio (used on full shutdown)."""
        with self._piper_lock:
            proc = self._current_aplay_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def say_blocking(self, text: str) -> None:
        """Speak a short fixed string and block until done. Used for system
        utterances like 'Museum helmet ready.' — never used for Gemini output,
        which goes through the streaming path."""
        print(f"🤖 {text}")
        self.is_busy_event.set()
        try:
            self._speak_chunk(text)
        finally:
            self.is_busy_event.clear()

    # --------------------------------------------------------------------
    # Sentence splitter. Consumes streaming text deltas and yields full
    # sentences as soon as they complete. Leftover tail returned via the
    # flush method.
    # --------------------------------------------------------------------
    class _SentenceSplitter:
        # Split on . ! ? and keep reasonable handling of abbreviations by
        # requiring the punctuation be followed by whitespace or end of text.
        _split_re = re.compile(r"([.!?]+)(\s+|$)")

        def __init__(self) -> None:
            self._buf = ""

        def feed(self, text: str):
            """Yield complete sentences from the buffer."""
            self._buf += text
            while True:
                m = self._split_re.search(self._buf)
                if not m:
                    break
                end = m.end()
                sentence = self._buf[:end].strip()
                self._buf = self._buf[end:]
                if sentence:
                    yield sentence

        def flush(self) -> str:
            tail = self._buf.strip()
            self._buf = ""
            return tail

    # --------------------------------------------------------------------
    # Gemini — streaming. Yields sentences as they're produced, stopping
    # early if cancel_response_event is set.
    # --------------------------------------------------------------------
    def _stream_sentences(self, prompt: str):
        splitter = self._SentenceSplitter()
        try:
            stream = self.client.models.generate_content_stream(
                model=self.gemini_model,
                contents=prompt,
            )
        except Exception as e:
            print(f"[Gemini] stream start failed: {e}")
            yield "Sorry, I could not reach the knowledge service right now."
            return

        try:
            for chunk in stream:
                if self.cancel_response_event.is_set():
                    print("[Gemini] canceled mid-stream.")
                    return
                delta = getattr(chunk, "text", None)
                if not delta:
                    continue
                for sentence in splitter.feed(delta):
                    yield sentence
                    if self.cancel_response_event.is_set():
                        print("[Gemini] canceled after sentence.")
                        return
            tail = splitter.flush()
            if tail and not self.cancel_response_event.is_set():
                yield tail
        except Exception as e:
            print(f"[Gemini] stream error: {e}")

    def _build_user_prompt(self, user_text: str) -> str:
        return f"""
{self.system_prompt}

Visitor says: {user_text}

Reply as a museum guide speaking naturally to a visitor beside you.
Sound warm, human, and conversational—not like a robot or textbook.
Keep your response short (1–3 sentences) unless more detail is clearly requested.
When explaining something:
- briefly say what it is
- explain why it matters
- add one interesting or engaging detail when possible
Adapt to the visitor's level and tone.
If appropriate, ask one short follow-up question to keep the interaction engaging.
"""

    def _build_object_prompt(self, object_name: str) -> str:
        return f"""
{self.system_prompt}

Camera event:
The visitor has been steadily looking at an object detected as "{object_name}" for at least {self.object_hold_seconds} seconds.

Task:
Give a short, natural museum-guide explanation related to "{object_name}".
- If it is a known or meaningful artwork, briefly explain:
  • what it is
  • why it matters
  • one interesting detail
- If the object is generic (e.g., chair, table, phone, person):
  • only respond if it makes sense in a museum context
  • otherwise give a very short, neutral response or gently shift attention

Guidelines:
- Do NOT mention detection or observation (avoid phrases like "I see you looking at…")
- If unsure, use soft uncertainty (e.g., "This appears to be…")
- Keep it natural, warm, and conversational
- Keep it short (1–3 sentences)
- Vary phrasing to avoid repetition
- Adjust tone slightly based on how interesting the object is
- When appropriate, include one engaging detail or a light follow-up question
"""

    def speak_stream(self, prompt: str) -> bool:
        """Stream a Gemini response and speak it sentence-by-sentence.

        Returns True if the response ran to completion, False if it was
        interrupted mid-response.
        """
        self.cancel_response_event.clear()
        self.is_busy_event.set()
        completed = True
        try:
            for sentence in self._stream_sentences(prompt):
                if self.cancel_response_event.is_set():
                    completed = False
                    break
                print(f"🤖 {sentence}")
                # _speak_chunk itself is blocking. Interrupts land between
                # sentences, giving the polished feel we want.
                self._speak_chunk(sentence)
                if self.cancel_response_event.is_set():
                    completed = False
                    break
        finally:
            self.is_busy_event.clear()
            self.last_turn_time = time.time()
        return completed

    # --------------------------------------------------------------------
    # Continuous STT worker. Runs in the background, pushing utterances
    # into utterance_queue. Each item is a dict:
    #    {"text": str, "conf": float|None, "duration": float}
    # --------------------------------------------------------------------
    # TODO(external-mic): when swapping to USB mic, replace the body of
    # this method with a vosk KaldiRecognizer driven by a sounddevice
    # InputStream bound to the USB device. The queue item format should
    # stay the same so nothing else needs to change.
    def _listen_forever(self) -> None:
        while not self.stop_event.is_set():
            try:
                # Each call to .listen(stream=True) is one "utterance
                # session" for the Fusion HAT wrapper. When it finalizes,
                # we push the result and start another.
                start_time = time.time()
                last_partial = ""
                final_text = ""
                final_conf: float | None = None

                for result in self.stt.listen(stream=True):
                    if self.stop_event.is_set():
                        break
                    partial = result.get("partial", "")
                    if partial:
                        last_partial = partial
                    if result.get("done"):
                        final_text = result.get("final", "") or last_partial
                        # Try to read per-word confidence if the wrapper
                        # exposes it. Fall back silently if not.
                        words = result.get("result") or result.get("words")
                        if isinstance(words, list) and words:
                            confs = [w.get("conf") for w in words if isinstance(w, dict) and "conf" in w]
                            if confs:
                                final_conf = sum(confs) / len(confs)
                        break

                duration = time.time() - start_time
                text = (final_text or "").strip().lower()
                if text:
                    self.utterance_queue.put({
                        "text": text,
                        "conf": final_conf,
                        "duration": duration,
                    })
            except Exception as e:
                print(f"[STT] listener error: {e}. Restarting in 0.5s.")
                time.sleep(0.5)

    # --------------------------------------------------------------------
    # Camera worker — unchanged detection logic. Kept here so the file
    # stays self-contained.
    # --------------------------------------------------------------------
    def camera_worker(self) -> None:
        try:
            picam2 = Picamera2()
            picam2.preview_configuration.main.size = self.camera_size
            picam2.preview_configuration.main.format = "RGB888"
            picam2.preview_configuration.align()
            picam2.configure("preview")
            picam2.start()
            time.sleep(0.2)

            # Show raw camera preview while YOLOE loads.
            preview_start = time.time()
            while time.time() - preview_start < 1.0 and not self.stop_event.is_set():
                frame = picam2.capture_array()
                cv2.imshow("YOLOE Museum Helmet", frame)
                if cv2.waitKey(1) == ord("q"):
                    self.stop_event.set()
                    return

            print("[Camera] Preview opened, loading YOLOE model...")
            model = YOLOE(self.model_path)
            model.set_classes(self.prompt_names)
            print("[Camera] YOLOE model loaded.")

            while not self.stop_event.is_set():
                frame = picam2.capture_array()
                results = model.predict(frame, imgsz=self.model_imgsz)
                result = results[0]
                annotated_frame = result.plot(boxes=True, masks=False)

                detections: list[dict] = []
                boxes = result.boxes
                if boxes is not None and boxes.cls is not None and boxes.conf is not None:
                    class_ids = boxes.cls.tolist()
                    confidences = boxes.conf.tolist()
                    for cls_id, conf in zip(class_ids, confidences):
                        if conf < self.confidence_threshold:
                            continue
                        cls_index = int(cls_id)
                        name = result.names.get(cls_index, str(cls_index))
                        detections.append({
                            "name": str(name).lower(),
                            "confidence": float(conf),
                        })
                self.maybe_trigger_object_explanation(detections)

                inference_time = result.speed["inference"]
                fps = 1000 / inference_time if inference_time > 0 else 0.0
                text = f"FPS: {fps:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = annotated_frame.shape[1] - text_size[0] - 10
                text_y = text_size[1] + 10
                cv2.putText(
                    annotated_frame, text, (text_x, text_y), font, 1,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

                cv2.imshow("YOLOE Museum Helmet", annotated_frame)
                if cv2.waitKey(1) == ord("q"):
                    self.stop_event.set()
                    break
        except Exception as e:
            print("Camera worker error:", e)
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def maybe_trigger_object_explanation(self, detections: list[dict]) -> None:
        current_time = time.time()
        detected_names = [d["name"] for d in detections]
        if detected_names:
            unique_names = sorted(set(detected_names))
            if unique_names != self.last_terminal_objects:
                print(f"[Camera detected objects]: {', '.join(unique_names)}")
                self.last_terminal_objects = unique_names
        else:
            if self.last_terminal_objects is not None:
                print("[Camera detected objects]: none")
                self.last_terminal_objects = None

        if not detections:
            self.last_seen_object = None
            self.object_first_seen_time = None
            return

        dominant = max(detections, key=lambda d: d["confidence"])
        dominant_name = dominant["name"]

        if dominant_name != self.last_seen_object:
            self.last_seen_object = dominant_name
            self.object_first_seen_time = current_time
            return

        if self.object_first_seen_time is None:
            self.object_first_seen_time = current_time
            return

        held_long_enough = (current_time - self.object_first_seen_time) >= self.object_hold_seconds
        off_cooldown = (current_time - self.last_object_trigger_time.get(dominant_name, 0.0)) >= self.object_cooldown_seconds

        # Only trigger when nothing else is happening. The main loop can
        # still interrupt us after we start by setting cancel_response_event.
        if held_long_enough and off_cooldown and not self.is_busy_event.is_set():
            print(f"[Camera trigger]: {dominant_name} held for {self.object_hold_seconds} seconds")
            self.last_object_trigger_time[dominant_name] = current_time
            prompt = self._build_object_prompt(dominant_name)
            self.speak_stream(prompt)
            self.object_first_seen_time = current_time

    # --------------------------------------------------------------------
    # Utterance classification.
    # --------------------------------------------------------------------
    def _contains_wake_word(self, text: str) -> str | None:
        for w in WAKE_WORDS:
            if w in text:
                return w
        return None

    def _is_interrupt(self, text: str) -> bool:
        # Wake word during active response = new query, treated as interrupt.
        if self._contains_wake_word(text):
            return True
        # Explicit interrupt phrases.
        for kw in INTERRUPT_WORDS:
            if kw in text:
                return True
        return False

    def _utterance_passes_noise_gate(self, utt: dict, in_active_mode: bool) -> bool:
        """Return True if the utterance is plausibly from the user, not noise."""
        text = utt["text"]
        conf = utt.get("conf")
        duration = utt.get("duration", 0.0)

        # Wake word always passes — it's what we listen for in the first place.
        if self._contains_wake_word(text):
            return True

        word_count = len(text.split())
        if in_active_mode:
            if word_count < ACTIVE_STT_MIN_WORDS:
                return False
            if duration < ACTIVE_STT_MIN_SECONDS:
                return False
        else:
            if word_count < IDLE_STT_MIN_WORDS:
                return False

        # Confidence gate — only rejects if we actually got a number back.
        if conf is not None and conf < VOSK_MIN_CONF:
            return False

        return True

    # --------------------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------------------
    def _currently_active_mode(self) -> bool:
        return (time.time() - self.last_turn_time) <= ACTIVE_WINDOW_SECONDS

    def _strip_wake_word(self, text: str, wake: str) -> str:
        return text.replace(wake, "", 1).strip()

    def _process_user_turn(self, user_text: str) -> None:
        """Run one full user question -> streamed response cycle, honoring
        interrupts that arrive mid-response."""
        prompt = self._build_user_prompt(user_text)
        # Start streaming in this thread; we drain interrupt checks
        # between sentences via speak_stream's internal cancel polling.
        # But we also need to watch the utterance queue while speaking,
        # so we spin a tiny watcher that flips cancel_response_event if
        # a new valid utterance arrives.
        watcher_stop = threading.Event()

        pending_followup: list[str] = []  # queued non-interrupt speech

        def watcher():
            while not watcher_stop.is_set() and not self.stop_event.is_set():
                try:
                    utt = self.utterance_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                text = utt["text"]
                if not self._utterance_passes_noise_gate(utt, in_active_mode=True):
                    continue
                if self._is_interrupt(text):
                    print(f"[Interrupt heard]: {text}")
                    # If it was a wake-word-style interrupt, queue the
                    # cleaned-up content as the next turn.
                    wake = self._contains_wake_word(text)
                    if wake:
                        cleaned = self._strip_wake_word(text, wake)
                        if cleaned:
                            pending_followup.append(cleaned)
                    self.cancel_response_event.set()
                    # Keep watching in case more speech comes in, but
                    # the cancel flag will stop speak_stream soon.
                else:
                    # Queue as a follow-up after the current answer finishes.
                    print(f"[Follow-up queued]: {text}")
                    pending_followup.append(text)

        w = threading.Thread(target=watcher, daemon=True)
        w.start()
        try:
            self.speak_stream(prompt)
        finally:
            watcher_stop.set()
            w.join(timeout=0.5)

        # Handle any queued follow-ups in order. Each one is a fresh turn
        # so the user can keep chaining naturally.
        for text in pending_followup:
            if self.stop_event.is_set():
                return
            # Check exit first.
            if any(kw in text for kw in EXIT_WORDS):
                self.say_blocking("Goodbye.")
                self.stop_event.set()
                return
            # "be quiet" style requests just silence us briefly.
            if text in ("stop", "be quiet", "silence"):
                # No response needed — the prior cancel already stopped us.
                continue
            self._process_user_turn(text)

    def start(self) -> None:
        # Background threads: camera + continuous STT.
        camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        camera_thread.start()

        stt_thread = threading.Thread(target=self._listen_forever, daemon=True)
        stt_thread.start()

        self.say_blocking("Museum helmet ready.")

        try:
            while not self.stop_event.is_set():
                try:
                    utt = self.utterance_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                text = utt["text"]
                active = self._currently_active_mode()

                if not self._utterance_passes_noise_gate(utt, in_active_mode=active):
                    continue

                print(f"\n[Heard]: {text}  (active={active}, conf={utt.get('conf')})")

                # Exit phrases always win.
                if any(kw in text for kw in EXIT_WORDS):
                    self.say_blocking("Goodbye.")
                    break

                wake = self._contains_wake_word(text)

                if active:
                    # In active mode, follow-ups don't need a wake word.
                    query = self._strip_wake_word(text, wake) if wake else text
                    if not query:
                        self.say_blocking("Yes? What would you like to know?")
                        continue
                    self._process_user_turn(query)
                else:
                    # In idle mode, we need a wake word to do anything.
                    if not wake:
                        # Bystander noise or off-topic speech — ignore.
                        continue
                    query = self._strip_wake_word(text, wake)
                    if not query:
                        self.say_blocking("Yes? What would you like to know?")
                        continue
                    self._process_user_turn(query)
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self.cancel_response_event.set()
            self._hard_stop_aplay()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()
