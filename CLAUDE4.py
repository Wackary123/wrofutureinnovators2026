"""
ATLAS — Museum Helmet main script.

Major changes in this version:
- Camera hold = 2s, with hysteresis for YOLOE detection flicker.
- Persistent Piper + aplay subprocess pair: no per-sentence startup cost,
  Gemini streams sentences directly to Piper via stdin.
- Mid-speech interrupts allowed, with stricter gate (>=6 words, conf>=0.6,
  dur>=1.5s) to avoid false interrupts from noise / self-hearing.
- Echo dampener: recent spoken text is compared to incoming utterances and
  high-overlap matches are dropped as probable self-echo.
- Conversation memory (10 turns), bystander SKIP filter, startup greeting,
  no-wake-word-required — all kept from previous version.

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
from collections import deque

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

vosk.SetLogLevel(-1)


# --------------------------------------------------------------------------
# Tunable constants.
# --------------------------------------------------------------------------

# --- Audio input (mic) ---
MIC_DEVICE = 1
MIC_NATIVE_RATE = 48000
MIC_SAMPLE_RATE = 16000
MIC_BLOCKSIZE = 12000

# --- Audio output (speaker) ---
AUDIO_OUT_DEVICE: str | None = "plughw:4,0"

# --- Piper ---
PIPER_SAMPLE_RATE = 22050  # Piper's native output rate for most voices

# --- Vosk ---
VOSK_MODEL_PATH = "/opt/vosk_models/vosk-model-small-en-us-0.15"

# --- Noise gate (idle) ---
STT_MIN_WORDS = 3
STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.55

# --- Interrupt gate (stricter — used while helmet is speaking) ---
INTERRUPT_MIN_WORDS = 6
INTERRUPT_MIN_SECONDS = 1.5
INTERRUPT_MIN_CONF = 0.60

# --- Echo dampener ---
# After we speak a sentence, remember its words for a short window. If an
# incoming utterance overlaps heavily with recently-spoken text, drop it.
ECHO_MEMORY_SECONDS = 2.5
ECHO_OVERLAP_FRACTION = 0.55

# --- Keywords ---
WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
INTERRUPT_WORDS = ("stop", "wait", "actually", "never mind", "nevermind", "cancel", "hold on")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")

# --- Memory ---
MEMORY_TURNS = 10

# --- Vision ---
DETECT_EVERY_N_FRAMES = 4
OBJECT_HOLD_SECONDS = 2.0
OBJECT_COOLDOWN_SECONDS = 8.0
# Hysteresis: allow this many seconds of "not seen" without resetting hold.
OBJECT_FLICKER_TOLERANCE = 0.5
TRIGGER_OBJECTS = {"mona lisa", "vase"}

# --- Greeting ---
GREETING = (
    "Hi, I'm your museum guide. You can ask me anything, or just stop in "
    "front of an exhibit and I'll tell you about it."
)


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        # --- AI setup ---
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)
        self.gemini_model = "gemini-2.5-flash"

        # --- Vosk STT ---
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}.")
        print(f"[STT] Loading Vosk model from {VOSK_MODEL_PATH} ...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        print("[STT] Vosk model loaded.")

        # --- Piper TTS ---
        self.piper_voice = "en_US-ryan-medium"
        self.piper_data_dir = os.path.expanduser("~/piper_voices")
        self._piper_proc: subprocess.Popen | None = None
        self._aplay_proc: subprocess.Popen | None = None
        self._piper_lock = threading.Lock()

        # --- Camera / YOLOE ---
        self.camera_size = (800, 800)
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "mona lisa", "computer", "person", "vase", "iphone", "head",
        ]
        self.confidence_threshold = 0.20
        self.model_imgsz = 192

        # Detection tracking for TRIGGER_OBJECTS.
        self.last_seen_object: str | None = None
        self.object_first_seen_time: float | None = None
        self.object_last_seen_time: float | None = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        # --- Inter-thread state ---
        self.utterance_queue: queue.Queue = queue.Queue()
        self.request_queue: queue.Queue = queue.Queue()

        self.stop_event = threading.Event()
        self.cancel_response_event = threading.Event()
        self.is_busy_event = threading.Event()

        # --- Conversation memory ---
        self.memory: deque = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

        # --- Echo dampener state ---
        # List of (timestamp, set-of-words) for recently spoken sentences.
        self._echo_log: list[tuple[float, set[str]]] = []
        self._echo_lock = threading.Lock()

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
Adjust energy depending on the exhibit

Core Behavior
Give clear, simple, and meaningful explanations
Focus on culture, history, artifacts, symbolism, and context
When explaining an object: say what it is, why it matters, and add one interesting detail
Adapt to the visitor's level: simplify for beginners, add depth for advanced questions
If unsure, acknowledge uncertainty calmly while still giving helpful context
Avoid vague or generic explanations—always give a specific reason
Do not use bold as it will be said in the tts (asterisk ---- asterisk)

Interaction Rules
Do not overwhelm the visitor with too much information
Avoid long monologues unless explicitly requested
Match the visitor's tone (curious, excited, confused)
If the visitor is incorrect, correct them politely and briefly
If the question is off-topic, gently guide the conversation back to the exhibit

Vision & Context Awareness
Treat [Camera] notes as context about what the visitor is looking at right now
If something may be misidentified, acknowledge uncertainty and still provide helpful context
Vary phrasing to avoid sounding repetitive

Privacy & Safety
Keep the safety role subtle and secondary
Never mention storing, tracking, or saving personal data

Overall Goal
Act like a knowledgeable, friendly guide beside the visitor.
"""

    # --------------------------------------------------------------------
    # Memory helpers.
    # --------------------------------------------------------------------
    def _memory_append(self, role: str, text: str) -> None:
        with self.memory_lock:
            self.memory.append((role, text))

    def _memory_pop_if_last(self, role: str, text: str) -> None:
        with self.memory_lock:
            if self.memory and self.memory[-1] == (role, text):
                self.memory.pop()

    def _memory_as_transcript(self) -> str:
        with self.memory_lock:
            items = list(self.memory)
        lines = []
        for role, text in items:
            if role == "user":
                lines.append(f"Visitor: {text}")
            elif role == "assistant":
                lines.append(f"Guide: {text}")
            elif role == "camera":
                lines.append(f"[Camera] Visitor is now looking at: {text}")
        return "\n".join(lines) if lines else "(no prior turns)"

    # --------------------------------------------------------------------
    # Echo dampener.
    # --------------------------------------------------------------------
    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z']+", text.lower()))

    def _note_spoken(self, text: str) -> None:
        words = self._tokenize(text)
        if not words:
            return
        with self._echo_lock:
            self._echo_log.append((time.time(), words))

    def _looks_like_echo(self, text: str) -> bool:
        now = time.time()
        heard = self._tokenize(text)
        if not heard:
            return False
        with self._echo_lock:
            # Drop stale entries.
            self._echo_log = [
                (t, w) for (t, w) in self._echo_log
                if now - t <= ECHO_MEMORY_SECONDS
            ]
            for _, spoken in self._echo_log:
                if not spoken:
                    continue
                overlap = len(heard & spoken) / max(1, len(heard))
                if overlap >= ECHO_OVERLAP_FRACTION:
                    return True
        return False

    # --------------------------------------------------------------------
    # Persistent Piper -> aplay pipeline.
    # --------------------------------------------------------------------
    def _start_audio_pipeline(self) -> None:
        """Launch persistent Piper and aplay processes, wire Piper stdout
        into aplay stdin so raw PCM streams straight through."""
        with self._piper_lock:
            if self._piper_proc is not None or self._aplay_proc is not None:
                return

            aplay_cmd = [
                "aplay", "-q",
                "-r", str(PIPER_SAMPLE_RATE),
                "-f", "S16_LE",
                "-c", "1",
                "-t", "raw",
            ]
            if AUDIO_OUT_DEVICE:
                aplay_cmd += ["-D", AUDIO_OUT_DEVICE]

            # aplay reads raw PCM from stdin (pipe from Piper).
            self._aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            piper_cmd = [
                "python3", "-m", "piper",
                "--model", self.piper_voice,
                "--data-dir", self.piper_data_dir,
                "--output-raw",
            ]
            self._piper_proc = subprocess.Popen(
                piper_cmd,
                stdin=subprocess.PIPE,
                stdout=self._aplay_proc.stdin,
                stderr=subprocess.DEVNULL,
            )
            # Close our copy of aplay's stdin — Piper holds the write end.
            try:
                self._aplay_proc.stdin.close()
            except Exception:
                pass

    def _restart_audio_pipeline(self) -> None:
        """Kill and relaunch Piper+aplay. Used on hard-cut interrupts: once
        audio has been queued into aplay's buffer, the only way to silence
        it immediately is to kill the process."""
        self._stop_audio_pipeline()
        self._start_audio_pipeline()

    def _stop_audio_pipeline(self) -> None:
        with self._piper_lock:
            for attr in ("_piper_proc", "_aplay_proc"):
                proc = getattr(self, attr, None)
                if proc is not None:
                    try:
                        if proc.stdin and not proc.stdin.closed:
                            try:
                                proc.stdin.close()
                            except Exception:
                                pass
                        proc.terminate()
                        try:
                            proc.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    except Exception:
                        pass
                    setattr(self, attr, None)

    def _speak_chunk(self, text: str) -> None:
        """Send one sentence to the persistent Piper process."""
        text = text.strip()
        if not text:
            return
        with self._piper_lock:
            proc = self._piper_proc
        if proc is None or proc.poll() is not None:
            # Pipeline died — rebuild it.
            self._start_audio_pipeline()
            with self._piper_lock:
                proc = self._piper_proc
            if proc is None:
                return
        try:
            # Piper reads one line per utterance. Make sure there's a newline.
            payload = text.rstrip() + "\n"
            proc.stdin.write(payload.encode("utf-8"))
            proc.stdin.flush()
        except BrokenPipeError:
            # Pipeline died mid-write; try once more after restart.
            self._start_audio_pipeline()
            with self._piper_lock:
                proc = self._piper_proc
            if proc is not None:
                try:
                    proc.stdin.write((text + "\n").encode("utf-8"))
                    proc.stdin.flush()
                except Exception:
                    pass
        except Exception as e:
            print(f"[TTS] write failed: {e}")

    def say_blocking(self, text: str) -> None:
        """Speak a short fixed string. Uses the same pipeline; returns
        quickly since aplay buffers independently. Marks busy for the
        approximate duration of the utterance."""
        print(f"🤖 {text}")
        self.is_busy_event.set()
        self._speak_chunk(text)
        self._note_spoken(text)
        # Estimate playback time from word count (~2.5 wps).
        approx = max(1.0, len(text.split()) / 2.5)
        time.sleep(approx)
        self.is_busy_event.clear()

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
                    return
                delta = getattr(chunk, "text", None)
                if not delta:
                    continue
                for sentence in splitter.feed(delta):
                    yield sentence
                    if self.cancel_response_event.is_set():
                        return
            tail = splitter.flush()
            if tail and not self.cancel_response_event.is_set():
                yield tail
        except Exception as e:
            print(f"[Gemini] stream error: {e}")

    # --------------------------------------------------------------------
    # Prompt building.
    # --------------------------------------------------------------------
    _skip_instructions = """
Bystander filter:
If the visitor's latest line looks like random background chatter, off-topic
noise, or clearly not directed at you, reply with exactly:
SKIP
and nothing else.

Otherwise, answer normally as the museum guide.
"""

    def _build_user_prompt(self, user_text: str) -> str:
        return f"""{self.system_prompt}

Conversation so far (most recent last):
{self._memory_as_transcript()}

Visitor's latest line: {user_text}

{self._skip_instructions}

If you answer, keep it short (1–3 sentences), warm and conversational.
Use prior turns when relevant so follow-ups feel natural.
"""

    def _build_object_prompt(self, object_name: str) -> str:
        return f"""{self.system_prompt}

Conversation so far (most recent last):
{self._memory_as_transcript()}

Camera event:
The visitor has been steadily looking at an object detected as "{object_name}"
for at least {OBJECT_HOLD_SECONDS} seconds.

Task:
Give a short, natural museum-guide explanation of "{object_name}".
Do NOT mention detection or observation; speak as if you noticed it yourself.
Keep it to 1–3 sentences. If unsure, use soft uncertainty.
Never reply SKIP for a camera event. Just speak.
"""

    # --------------------------------------------------------------------
    # Response pipeline.
    # --------------------------------------------------------------------
    def _handle_request(self, kind: str, text: str) -> None:
        if kind == "user":
            prompt = self._build_user_prompt(text)
            self._memory_append("user", text)
        elif kind == "object":
            prompt = self._build_object_prompt(text)
            self._memory_append("camera", text)
        else:
            return

        self.cancel_response_event.clear()
        self.is_busy_event.set()

        collected: list[str] = []
        first_sentence_seen = False
        skipped = False

        try:
            for sentence in self._stream_sentences(prompt):
                if self.cancel_response_event.is_set():
                    break

                if not first_sentence_seen and kind == "user":
                    stripped = sentence.strip().rstrip(".").upper()
                    if stripped == "SKIP":
                        print("[Gemini] SKIP — probable bystander, staying silent.")
                        skipped = True
                        break
                first_sentence_seen = True

                print(f"🤖 {sentence}")
                self._speak_chunk(sentence)
                self._note_spoken(sentence)
                collected.append(sentence)
        finally:
            # Keep is_busy_event set for a short tail so aplay's buffer drains.
            # Estimate tail time from last collected text.
            tail_text = collected[-1] if collected else ""
            tail_wait = max(0.2, len(tail_text.split()) / 2.5) if tail_text else 0.2
            time.sleep(tail_wait)
            self.is_busy_event.clear()

        if skipped:
            self._memory_pop_if_last("user", text)
            return

        if collected:
            self._memory_append("assistant", " ".join(collected))

    def _gemini_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_request(req.get("kind"), req.get("text", ""))

    # --------------------------------------------------------------------
    # STT: sounddevice -> decimate -> Vosk.
    # --------------------------------------------------------------------
    def _listen_forever(self) -> None:
        audio_q: queue.Queue = queue.Queue()

        if MIC_NATIVE_RATE % MIC_SAMPLE_RATE != 0:
            raise RuntimeError("MIC_NATIVE_RATE must be integer multiple of MIC_SAMPLE_RATE.")
        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE

        def audio_callback(indata, frames, time_info, status):
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
                                    "was_speaking": self.is_busy_event.is_set(),
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

            while not self.stop_event.is_set():
                frame = picam2.capture_array()
                frame_idx += 1

                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    results = model.predict(frame, imgsz=self.model_imgsz, verbose=False)
                    result = results[0]
                    last_annotated = result.plot(boxes=True, masks=False)

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

                    inference_time = result.speed.get("inference", 0.0)
                    last_fps = 1000.0 / inference_time if inference_time > 0 else 0.0
                    self._maybe_trigger_object_explanation(detections)

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

    def _maybe_trigger_object_explanation(self, detections: list[dict]) -> None:
        current_time = time.time()
        detected_names = [d["name"] for d in detections]
        if detected_names:
            unique_names = sorted(set(detected_names))
            if unique_names != self.last_terminal_objects:
                print(f"[Camera detected]: {', '.join(unique_names)}")
                self.last_terminal_objects = unique_names
        else:
            if self.last_terminal_objects is not None:
                print("[Camera detected]: none")
                self.last_terminal_objects = None

        triggerable = [d for d in detections if d["name"] in TRIGGER_OBJECTS]

        # Hysteresis: if a triggerable object is seen now, update last-seen.
        # If not, only reset tracking once we've exceeded the flicker tolerance.
        if triggerable:
            dominant = max(triggerable, key=lambda d: d["confidence"])
            dominant_name = dominant["name"]

            if dominant_name != self.last_seen_object:
                self.last_seen_object = dominant_name
                self.object_first_seen_time = current_time
            self.object_last_seen_time = current_time
        else:
            if (self.last_seen_object is not None
                    and self.object_last_seen_time is not None
                    and (current_time - self.object_last_seen_time) > OBJECT_FLICKER_TOLERANCE):
                self.last_seen_object = None
                self.object_first_seen_time = None
                self.object_last_seen_time = None
            return

        if self.object_first_seen_time is None:
            self.object_first_seen_time = current_time
            return

        held_long_enough = (current_time - self.object_first_seen_time) >= OBJECT_HOLD_SECONDS
        off_cooldown = (current_time - self.last_object_trigger_time.get(self.last_seen_object, 0.0)) >= OBJECT_COOLDOWN_SECONDS

        if (held_long_enough and off_cooldown
                and not self.is_busy_event.is_set()
                and self.request_queue.empty()):
            name = self.last_seen_object
            print(f"[Camera trigger]: {name} held {OBJECT_HOLD_SECONDS}s — enqueuing")
            self.last_object_trigger_time[name] = current_time
            self.request_queue.put({"kind": "object", "text": name})
            self.object_first_seen_time = current_time

    # --------------------------------------------------------------------
    # Utterance classification + gates.
    # --------------------------------------------------------------------
    def _contains_wake_word(self, text: str) -> str | None:
        for w in WAKE_WORDS:
            if w in text:
                return w
        return None

    def _strip_wake_word(self, text: str) -> str:
        out = text
        for w in WAKE_WORDS:
            out = out.replace(w, "", 1)
        return out.strip()

    def _passes_idle_gate(self, utt: dict) -> bool:
        text = utt["text"]
        conf = utt.get("conf")
        duration = utt.get("duration", 0.0)

        if self._contains_wake_word(text):
            return True

        word_count = len(text.split())
        if word_count < STT_MIN_WORDS:
            return False
        if duration < STT_MIN_SECONDS:
            return False
        if conf is not None and conf < VOSK_MIN_CONF:
            return False
        return True

    def _passes_interrupt_gate(self, utt: dict) -> bool:
        text = utt["text"]
        conf = utt.get("conf")
        duration = utt.get("duration", 0.0)

        # Wake word + explicit interrupt word always pass (override).
        if self._contains_wake_word(text):
            return True

        word_count = len(text.split())
        if word_count < INTERRUPT_MIN_WORDS:
            return False
        if duration < INTERRUPT_MIN_SECONDS:
            return False
        if conf is not None and conf < INTERRUPT_MIN_CONF:
            return False
        return True

    # --------------------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------------------
    def start(self) -> None:
        # Start the persistent audio pipeline first.
        self._start_audio_pipeline()

        camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        camera_thread.start()

        stt_thread = threading.Thread(target=self._listen_forever, daemon=True)
        stt_thread.start()

        worker_thread = threading.Thread(target=self._gemini_worker, daemon=True)
        worker_thread.start()

        self.say_blocking(GREETING)

        try:
            while not self.stop_event.is_set():
                try:
                    utt = self.utterance_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                text = utt["text"]
                was_speaking = utt.get("was_speaking", False)

                # Echo dampener — drop anything that looks like our own recent speech.
                if self._looks_like_echo(text):
                    print(f"[Echo dropped]: {text}")
                    continue

                # Pick the right gate.
                if was_speaking:
                    if not self._passes_interrupt_gate(utt):
                        continue
                else:
                    if not self._passes_idle_gate(utt):
                        continue

                print(f"\n[Heard{' during speech' if was_speaking else ''}]: "
                      f"{text}  (conf={utt.get('conf')}, "
                      f"dur={utt.get('duration', 0):.2f}s)")

                if any(kw in text for kw in EXIT_WORDS):
                    self.say_blocking("Goodbye.")
                    break

                query = self._strip_wake_word(text) if self._contains_wake_word(text) else text
                if not query:
                    self.say_blocking("Yes?")
                    continue

                if self.is_busy_event.is_set() or was_speaking:
                    # Hard-cut: kill the audio pipeline, cancel Gemini,
                    # drain any queued camera triggers.
                    print("[Interrupt] cutting current response")
                    self.cancel_response_event.set()
                    self._restart_audio_pipeline()
                    # Drop any camera triggers still queued.
                    try:
                        while True:
                            self.request_queue.get_nowait()
                    except queue.Empty:
                        pass
                    time.sleep(0.05)

                self.request_queue.put({"kind": "user", "text": query})
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self.cancel_response_event.set()
            self._stop_audio_pipeline()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()