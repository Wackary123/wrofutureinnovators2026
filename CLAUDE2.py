"""
ATLAS — Museum Helmet main script.

This version drops the Fusion HAT STT wrapper entirely and drives the
MillSO MQ5 USB lavalier mic directly via sounddevice + vosk.

Architecture
------------
Three background threads:
  1. Camera thread   — YOLOE every N frames, enqueues trigger events.
  2. Gemini worker   — drains request_queue, streams Gemini responses.
  3. STT thread      — sounddevice (48 kHz) -> decimate -> vosk (16 kHz).

Main loop reads utterances, decides IDLE vs ACTIVE mode, and enqueues
user-turn requests.

Hardware
--------
Mic:     MillSO MQ5 USB lavalier on sounddevice index 1 (USB ENC Audio).
         Records at 48 kHz natively, decimated to 16 kHz for Vosk.
Speaker: USB speaker on ALSA card 4 (UACDemoV1.0), routed via plughw:4,0.
"""

import json
import os
import re
import time
import threading
import queue
import subprocess
import tempfile

import cv2
import numpy as np
import sounddevice as sd
import vosk  # type: ignore

from dotenv import load_dotenv
from google import genai

from picamera2 import Picamera2  # type: ignore
from ultralytics import YOLOE  # type: ignore

try:
    import torch  # type: ignore
    torch.set_num_threads(3)
except Exception:
    pass

# Quiet Vosk's stderr chatter (model load spam).
vosk.SetLogLevel(-1)


# --------------------------------------------------------------------------
# Tunable constants.
# --------------------------------------------------------------------------

# --- Audio input (mic) ---
# Run `python3 -c "import sounddevice as sd; print(sd.query_devices())"` to
# find the right index. Currently 1 = USB ENC Audio Device (MillSO mic).
MIC_DEVICE = 1
MIC_NATIVE_RATE = 48000         # rate the USB mic actually records at
MIC_SAMPLE_RATE = 16000         # what Vosk wants; we downsample to this
MIC_BLOCKSIZE = 12000           # frames per callback at NATIVE rate (~0.25 s)

# --- Audio output (speaker) ---
# Run `aplay -l` to find the right card. Currently card 4 = UACDemoV1.0 USB.
# None lets ALSA pick the default (useful for debugging only).
AUDIO_OUT_DEVICE: str | None = "plughw:4,0"

# --- Vosk ---
VOSK_MODEL_PATH = "/opt/vosk_models/vosk-model-small-en-us-0.15"

# --- Interaction-mode timing ---
ACTIVE_WINDOW_SECONDS = 10.0
IDLE_STT_MIN_WORDS = 2
ACTIVE_STT_MIN_WORDS = 3
ACTIVE_STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.5

# --- Keywords ---
WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
INTERRUPT_WORDS = ("stop", "wait", "actually", "never mind", "nevermind", "cancel", "hold on")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")

# --- Vision / detection ---
DETECT_EVERY_N_FRAMES = 2
TRIGGER_OBJECTS = {"mona lisa", "vase"}


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        # --- AI setup ---
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_model = "gemini-2.5-flash"

        # --- Vosk STT model (loaded once, shared) ---
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(
                f"Vosk model not found at {VOSK_MODEL_PATH}. "
                f"Check the path or download a model."
            )
        print(f"[STT] Loading Vosk model from {VOSK_MODEL_PATH} ...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        print("[STT] Vosk model loaded.")

        # --- Piper TTS ---
        self.piper_model = "en_US-ryan-medium"
        self.piper_data_dir = os.path.expanduser("~/piper_voices")

        # --- Camera / YOLOE ---
        self.camera_size = (800, 800)
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "mona lisa", "computer", "person", "vase", "iphone", "head",
        ]
        self.confidence_threshold = 0.20
        self.object_hold_seconds = 1.5
        self.object_cooldown_seconds = 8.0
        self.model_imgsz = 224

        # Detection tracking
        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        # --- Inter-thread state ---
        self.utterance_queue: queue.Queue = queue.Queue()
        self.request_queue: queue.Queue = queue.Queue()

        self.stop_event = threading.Event()
        self.cancel_response_event = threading.Event()
        self.is_busy_event = threading.Event()

        self.last_turn_time = 0.0

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
    # TTS primitives.
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
                input=text, text=True, check=True,
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
        with self._piper_lock:
            proc = self._current_aplay_proc
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def say_blocking(self, text: str) -> None:
        print(f"🤖 {text}")
        self.is_busy_event.set()
        try:
            self._speak_chunk(text)
        finally:
            self.is_busy_event.clear()
            self.last_turn_time = time.time()

    # --------------------------------------------------------------------
    # Sentence splitter.
    # --------------------------------------------------------------------
    class _SentenceSplitter:
        _split_re = re.compile(r"([.!?]+)(\s+|$)")

        def __init__(self) -> None:
            self._buf = ""

        def feed(self, text: str):
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
    # Gemini streaming.
    # --------------------------------------------------------------------
    def _stream_sentences(self, prompt: str):
        splitter = self._SentenceSplitter()
        try:
            stream = self.client.models.generate_content_stream(
                model=self.gemini_model, contents=prompt,
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

Guidelines:
- Do NOT mention detection or observation (avoid phrases like "I see you looking at…")
- If unsure, use soft uncertainty (e.g., "This appears to be…")
- Keep it natural, warm, and conversational
- Keep it short (1–3 sentences)
- Vary phrasing to avoid repetition
- When appropriate, include one engaging detail or a light follow-up question
"""

    def speak_stream(self, prompt: str) -> bool:
        self.cancel_response_event.clear()
        self.is_busy_event.set()
        completed = True
        try:
            for sentence in self._stream_sentences(prompt):
                if self.cancel_response_event.is_set():
                    completed = False
                    break
                print(f"🤖 {sentence}")
                self._speak_chunk(sentence)
                if self.cancel_response_event.is_set():
                    completed = False
                    break
        finally:
            self.is_busy_event.clear()
            self.last_turn_time = time.time()
        return completed

    # --------------------------------------------------------------------
    # Gemini/TTS worker thread.
    # --------------------------------------------------------------------
    def _gemini_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            kind = req.get("kind")
            text = req.get("text", "")

            if kind == "object":
                if self.is_busy_event.is_set():
                    continue
                prompt = self._build_object_prompt(text)
                self._with_voice_watcher(lambda: self.speak_stream(prompt))
            elif kind == "user":
                prompt = self._build_user_prompt(text)
                self._with_voice_watcher(lambda: self.speak_stream(prompt))
            else:
                print(f"[Worker] unknown request kind: {kind!r}")

    def _with_voice_watcher(self, action):
        watcher_stop = threading.Event()

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
                    wake = self._contains_wake_word(text)
                    if wake:
                        cleaned = self._strip_wake_word(text, wake)
                        if cleaned:
                            self.request_queue.put({"kind": "user", "text": cleaned})
                    self.cancel_response_event.set()
                else:
                    print(f"[Follow-up queued]: {text}")
                    self.request_queue.put({"kind": "user", "text": text})

        w = threading.Thread(target=watcher, daemon=True)
        w.start()
        try:
            action()
        finally:
            watcher_stop.set()
            w.join(timeout=0.5)

    # --------------------------------------------------------------------
    # STT: sounddevice (native rate) -> decimate -> Vosk (16 kHz).
    # --------------------------------------------------------------------
    def _listen_forever(self) -> None:
        """Continuous capture from MIC_DEVICE at MIC_NATIVE_RATE, downsample
        to MIC_SAMPLE_RATE, feed into Vosk. Emits utterances into
        self.utterance_queue as:
            {"text": str, "conf": float|None, "duration": float}
        """
        audio_q: queue.Queue = queue.Queue()

        # Integer decimation factor from native rate -> Vosk rate.
        if MIC_NATIVE_RATE % MIC_SAMPLE_RATE != 0:
            raise RuntimeError(
                f"MIC_NATIVE_RATE ({MIC_NATIVE_RATE}) must be an integer "
                f"multiple of MIC_SAMPLE_RATE ({MIC_SAMPLE_RATE})."
            )
        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            # indata is a CFFI buffer of int16 mono. Convert to numpy,
            # decimate, then back to bytes for Vosk.
            samples = np.frombuffer(bytes(indata), dtype=np.int16)
            if decim > 1:
                samples = samples[::decim]
            audio_q.put(samples.tobytes())

        while not self.stop_event.is_set():
            try:
                recognizer = vosk.KaldiRecognizer(self.vosk_model, MIC_SAMPLE_RATE)
                recognizer.SetWords(True)

                with sd.RawInputStream(
                    samplerate=MIC_NATIVE_RATE,
                    blocksize=MIC_BLOCKSIZE,
                    dtype="int16",
                    channels=1,
                    device=MIC_DEVICE,
                    callback=audio_callback,
                ):
                    print(f"[STT] Listening on device {MIC_DEVICE} "
                          f"@ {MIC_NATIVE_RATE} Hz -> {MIC_SAMPLE_RATE} Hz for Vosk")
                    utt_start: float | None = None

                    while not self.stop_event.is_set():
                        try:
                            data = audio_q.get(timeout=0.2)
                        except queue.Empty:
                            continue

                        if utt_start is None:
                            utt_start = time.time()

                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            text = (result.get("text") or "").strip().lower()
                            conf = self._avg_word_conf(result.get("result"))
                            duration = time.time() - (utt_start or time.time())
                            utt_start = None

                            if text:
                                self.utterance_queue.put({
                                    "text": text,
                                    "conf": conf,
                                    "duration": duration,
                                })
            except Exception as e:
                print(f"[STT] listener error: {e}. Restarting in 0.5s.")
                time.sleep(0.5)

    @staticmethod
    def _avg_word_conf(words) -> float | None:
        if not isinstance(words, list) or not words:
            return None
        confs = [w.get("conf") for w in words if isinstance(w, dict) and "conf" in w]
        if not confs:
            return None
        return sum(confs) / len(confs)

    # --------------------------------------------------------------------
    # Camera worker.
    # --------------------------------------------------------------------
    def camera_worker(self) -> None:
        picam2 = None
        try:
            picam2 = Picamera2()
            picam2.preview_configuration.main.size = self.camera_size
            picam2.preview_configuration.main.format = "RGB888"
            picam2.preview_configuration.align()
            picam2.configure("preview")
            picam2.start()
            time.sleep(0.2)

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

            frame_idx = 0
            last_annotated = None
            last_fps = 0.0
            detections: list[dict] = []

            while not self.stop_event.is_set():
                frame = picam2.capture_array()
                frame_idx += 1

                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    results = model.predict(frame, imgsz=self.model_imgsz, verbose=False)
                    result = results[0]
                    last_annotated = result.plot(boxes=True, masks=False)

                    detections = []
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

                    inference_time = result.speed.get("inference", 0.0)
                    last_fps = 1000.0 / inference_time if inference_time > 0 else 0.0
                    self.maybe_trigger_object_explanation(detections)

                display = last_annotated if last_annotated is not None else frame

                text = f"FPS (infer): {last_fps:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = display.shape[1] - text_size[0] - 10
                text_y = text_size[1] + 10
                cv2.putText(display, text, (text_x, text_y), font, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

                cv2.imshow("YOLOE Museum Helmet", display)
                if cv2.waitKey(1) == ord("q"):
                    self.stop_event.set()
                    break
        except Exception as e:
            print("Camera worker error:", e)
        finally:
            try:
                if picam2 is not None:
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

        triggerable = [d for d in detections if d["name"] in TRIGGER_OBJECTS]
        if not triggerable:
            self.last_seen_object = None
            self.object_first_seen_time = None
            return

        dominant = max(triggerable, key=lambda d: d["confidence"])
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

        if held_long_enough and off_cooldown and not self.is_busy_event.is_set():
            print(f"[Camera trigger]: {dominant_name} held for {self.object_hold_seconds}s — enqueuing")
            self.last_object_trigger_time[dominant_name] = current_time
            self.request_queue.put({"kind": "object", "text": dominant_name})
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
        if self._contains_wake_word(text):
            return True
        for kw in INTERRUPT_WORDS:
            if kw in text:
                return True
        return False

    def _utterance_passes_noise_gate(self, utt: dict, in_active_mode: bool) -> bool:
        text = utt["text"]
        conf = utt.get("conf")
        duration = utt.get("duration", 0.0)

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

        if conf is not None and conf < VOSK_MIN_CONF:
            return False

        return True

    def _strip_wake_word(self, text: str, wake: str) -> str:
        return text.replace(wake, "", 1).strip()

    # --------------------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------------------
    def _currently_active_mode(self) -> bool:
        return (time.time() - self.last_turn_time) <= ACTIVE_WINDOW_SECONDS

    def start(self) -> None:
        camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        camera_thread.start()

        stt_thread = threading.Thread(target=self._listen_forever, daemon=True)
        stt_thread.start()

        worker_thread = threading.Thread(target=self._gemini_worker, daemon=True)
        worker_thread.start()

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

                if any(kw in text for kw in EXIT_WORDS):
                    self.say_blocking("Goodbye.")
                    break

                if self.is_busy_event.is_set():
                    continue

                wake = self._contains_wake_word(text)
                if active:
                    query = self._strip_wake_word(text, wake) if wake else text
                    if not query:
                        self.say_blocking("Yes? What would you like to know?")
                        continue
                    self.request_queue.put({"kind": "user", "text": query})
                else:
                    if not wake:
                        continue
                    query = self._strip_wake_word(text, wake)
                    if not query:
                        self.say_blocking("Yes? What would you like to know?")
                        continue
                    self.request_queue.put({"kind": "user", "text": query})
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self.cancel_response_event.set()
            self._hard_stop_aplay()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()
