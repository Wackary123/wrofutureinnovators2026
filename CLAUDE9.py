"""
ATLAS — Museum Helmet main script.

This version:
- No interrupts. User utterances during playback are queued and handled
  after the current response finishes.
- Prompts explicitly forbid markdown/asterisks/bullets so Piper never
  reads formatting characters aloud.
- Fast voice (en_US-ryan-low) + faster length scale for lower latency.
- Bystander SKIP filter still in place.
- Camera triggers, memory, self-hearing guard all unchanged.

Hardware
--------
Mic:     MillSO MQ5 USB lavalier on sounddevice index 1.
Speaker: USB speaker on ALSA card 4, routed via plughw:4,0.
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

# --- Vosk ---
VOSK_MODEL_PATH = "/opt/vosk_models/vosk-model-small-en-us-0.15"

# --- Piper voice / speed ---
# Download with:
#   python3 -m piper.download_voices en_US-ryan-low --data-dir ~/piper_voices
# If en_US-ryan-low isn't available, try en_US-amy-low or en_US-kathleen-low
# and change this constant to match.
PIPER_VOICE = "en_US-ryan-low"
PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 0.90   # lower = faster; 0.90 is ~10% faster than default

# --- Noise gate ---
STT_MIN_WORDS = 3
STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.55

# --- Self-hearing settle ---
POST_SPEAK_SETTLE_SECONDS = 0.60

# --- Keywords ---
WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")

# --- Memory ---
MEMORY_TURNS = 10

# --- Vision ---
DETECT_EVERY_N_FRAMES = 4
OBJECT_HOLD_SECONDS = 2.0
OBJECT_COOLDOWN_SECONDS = 8.0
TRIGGER_OBJECTS = {"mona lisa painting", "vase", "sword"}

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

        # --- Camera / YOLOE ---
        self.camera_size = (1536, 864)
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "mona lisa painting","vase", "sword"
        ]
        self.confidence_threshold = 0.20
        self.model_imgsz = 192

        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        # --- Inter-thread state ---
        self.utterance_queue: queue.Queue = queue.Queue()
        self.request_queue: queue.Queue = queue.Queue()

        self.stop_event = threading.Event()
        self.is_busy_event = threading.Event()

        self.speak_start_time = 0.0
        self.last_speak_end_time = 0.0

        # Subprocess handles (kept so shutdown can terminate them cleanly).
        self._proc_lock = threading.Lock()
        self._piper_proc: subprocess.Popen | None = None
        self._aplay_proc: subprocess.Popen | None = None

        # --- Memory ---
        self.memory: deque = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

        # --- System prompt ---
        self.system_prompt = """
You are an AI museum guide embedded in a wearable helmet, speaking directly to a visitor in front of an exhibit. You also have a secondary, non-intrusive safety role.

Personality & Style
Speak like a real human guide: warm, natural, and conversational
Avoid sounding robotic, scripted, or like a textbook
Keep responses concise
Usually respond in 1–2 short sentences; 3 only if clarity really needs it
Prefer short back-and-forth interaction over long explanations
Adjust energy depending on the exhibit

Core Behavior
Give clear, simple, and meaningful explanations
Focus on culture, history, artifacts, symbolism, and context
When explaining an object: say what it is, why it matters, and one interesting detail
Adapt to the visitor's level: simplify for beginners, add depth for advanced questions
If unsure, acknowledge uncertainty calmly while still giving helpful context
Avoid vague or generic explanations—always give a specific reason

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

        # --- Formatting rule (appended to every prompt) ---
        self.formatting_rules = """
CRITICAL OUTPUT FORMAT RULES — these are read aloud by a text-to-speech engine:
- Do NOT use asterisks (*), underscores (_), backticks (`), or any markdown.
- Do NOT use bold, italics, or any emphasis markers.
- Do NOT use bullet points, numbered lists, or dashes for lists.
- Do NOT use headers, titles, or section labels.
- Do NOT use emoji.
- Write ONLY plain spoken prose — continuous sentences, like a person talking.
- Every character you write will be spoken out loud, so anything that isn't
  a natural spoken word will sound wrong.
"""

    # --------------------------------------------------------------------
    # Memory.
    # --------------------------------------------------------------------
    def _memory_append(self, role: str, text: str) -> None:
        with self.memory_lock:
            self.memory.append((role, text))

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
    # TTS — one shot per response.
    # --------------------------------------------------------------------
    def _sanitize_for_tts(self, text: str) -> str:
        """Safety net: even if Gemini ignores the format rule and emits
        formatting characters, strip them before Piper sees them."""
        if not text:
            return text
        # Remove markdown emphasis and code markers.
        text = re.sub(r"[*_`~]", "", text)
        # Remove leading list/numbering artifacts at line starts.
        text = re.sub(r"^\s*[-•]+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
        # Collapse repeated whitespace and newlines.
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _speak_full(self, text: str) -> None:
        text = self._sanitize_for_tts(text)
        if not text:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        try:
            piper_cmd = [
                "python3", "-m", "piper",
                "--model", PIPER_VOICE,
                "--data-dir", PIPER_DATA_DIR,
                "--length-scale", str(PIPER_LENGTH_SCALE),
                "--output-file", wav_path,
            ]
            with self._proc_lock:
                self._piper_proc = subprocess.Popen(
                    piper_cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            try:
                if self._piper_proc.stdin:
                    self._piper_proc.stdin.write(text.encode("utf-8"))
                    self._piper_proc.stdin.close()
            except Exception:
                pass
            self._piper_proc.wait()
            with self._proc_lock:
                self._piper_proc = None

            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                return

            aplay_cmd = ["aplay", "-q"]
            if AUDIO_OUT_DEVICE:
                aplay_cmd += ["-D", AUDIO_OUT_DEVICE]
            aplay_cmd.append(wav_path)

            with self._proc_lock:
                self._aplay_proc = subprocess.Popen(
                    aplay_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._aplay_proc.wait()
            with self._proc_lock:
                self._aplay_proc = None
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

    def _hard_stop_all_audio(self) -> None:
        """Only used on shutdown."""
        with self._proc_lock:
            for p in (self._piper_proc, self._aplay_proc):
                if p and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    def say_blocking(self, text: str) -> None:
        print(f"🤖 {text}")
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            self._speak_full(text)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

    # --------------------------------------------------------------------
    # Gemini — collect full response.
    # --------------------------------------------------------------------
    def _collect_full_response(self, prompt: str) -> str:
        chunks: list[str] = []
        try:
            stream = self.client.models.generate_content_stream(
                model=self.gemini_model, contents=prompt,
            )
            for chunk in stream:
                delta = getattr(chunk, "text", None)
                if delta:
                    chunks.append(delta)
        except Exception as e:
            print(f"[Gemini] stream error: {e}")
            return "Sorry, I could not reach the knowledge service right now."
        return "".join(chunks).strip()

    # --------------------------------------------------------------------
    # Prompt building.
    # --------------------------------------------------------------------
    _skip_instructions = """
Bystander filter:
If the visitor's latest line looks like random background chatter, off-topic
noise, an unrelated side conversation, or clearly not directed at you, reply
with exactly the single token:
SKIP
and nothing else. Do not explain.

Otherwise, answer normally as the museum guide.
"""

    def _build_user_prompt(self, user_text: str) -> str:
        transcript = self._memory_as_transcript()
        return f"""{self.system_prompt}

{self.formatting_rules}

Conversation so far (most recent last):
{transcript}

Visitor's latest line: {user_text}

{self._skip_instructions}

If you answer, keep it to 1–2 short sentences. Warm and conversational, plain prose.
Use prior turns when relevant so follow-ups feel natural.
"""

    def _build_object_prompt(self, object_name: str) -> str:
        transcript = self._memory_as_transcript()
        return f"""{self.system_prompt}

{self.formatting_rules}

Conversation so far (most recent last):
{transcript}

Camera event:
The visitor has been steadily looking at an object detected as "{object_name}"
for at least {OBJECT_HOLD_SECONDS} seconds.

Task:
Give a short, natural museum-guide explanation of "{object_name}".
Do NOT mention detection or observation; just speak as if you noticed it yourself.
Keep it to 1–2 short sentences. If unsure, use soft uncertainty.
This is NOT a bystander event — never reply SKIP for a camera event.
"""

    # --------------------------------------------------------------------
    # Handle one request end-to-end.
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

        print(f"[Gemini] thinking ...")
        response = self._collect_full_response(prompt)

        if not response:
            return

        if kind == "user":
            first_token = response.split(None, 1)[0].strip().rstrip(".").upper() if response else ""
            if first_token == "SKIP":
                print("[Gemini] SKIP — bystander noise, staying silent.")
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
                return

        sanitized = self._sanitize_for_tts(response)
        print(f"🤖 {sanitized}")
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            self._speak_full(sanitized)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

        self._memory_append("assistant", sanitized)

    def _gemini_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_request(req.get("kind"), req.get("text", ""))

    # --------------------------------------------------------------------
    # STT — drops audio during speech (no interrupts needed).
    # --------------------------------------------------------------------
    def _listen_forever(self) -> None:
        audio_q: queue.Queue = queue.Queue()

        if MIC_NATIVE_RATE % MIC_SAMPLE_RATE != 0:
            raise RuntimeError("MIC_NATIVE_RATE must be integer multiple of MIC_SAMPLE_RATE.")
        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE

        def audio_callback(indata, frames, time_info, status):
            if self.is_busy_event.is_set():
                return
            if (time.time() - self.last_speak_end_time) < POST_SPEAK_SETTLE_SECONDS:
                return
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
                    was_busy = False

                    while not self.stop_event.is_set():
                        busy_now = self.is_busy_event.is_set()
                        if busy_now and not was_busy:
                            recognizer.Reset()
                            utt_start = None
                            while not audio_q.empty():
                                try:
                                    audio_q.get_nowait()
                                except queue.Empty:
                                    break
                        was_busy = busy_now

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
                            my_start = utt_start or time.time()
                            utt_start = None

                            if not text:
                                continue

                            # Safety: if this utterance's start overlapped with
                            # a speaking window, drop it as possible self-echo.
                            speak_window_start = self.speak_start_time
                            speak_window_end = self.last_speak_end_time + POST_SPEAK_SETTLE_SECONDS
                            if (speak_window_start > 0 and
                                    my_start <= speak_window_end and
                                    my_start + duration >= speak_window_start):
                                print(f"[STT] Discarding self-hearing echo: {text!r}")
                                continue

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
            try:
                picam2.set_controls({"ScalerCrop": (0,0,4608,2592)})
            except Exception as e:
                print(f"[Camera] Could not set ScalerCrop: {e}")

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

        held_long_enough = (current_time - self.object_first_seen_time) >= OBJECT_HOLD_SECONDS
        off_cooldown = (current_time - self.last_object_trigger_time.get(dominant_name, 0.0)) >= OBJECT_COOLDOWN_SECONDS

        if held_long_enough and off_cooldown and not self.is_busy_event.is_set() \
                and self.request_queue.empty():
            print(f"[Camera trigger]: {dominant_name} held {OBJECT_HOLD_SECONDS}s — enqueuing")
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

    def _utterance_passes_noise_gate(self, utt: dict) -> bool:
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

    def _strip_wake_word(self, text: str) -> str:
        out = text
        for w in WAKE_WORDS:
            out = out.replace(w, "", 1)
        return out.strip()

    # --------------------------------------------------------------------
    # Main loop.
    # --------------------------------------------------------------------
    def start(self) -> None:
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
                if not self._utterance_passes_noise_gate(utt):
                    continue

                print(f"\n[Heard]: {text}  (conf={utt.get('conf')}, "
                      f"dur={utt.get('duration', 0):.2f}s)")

                if any(kw in text for kw in EXIT_WORDS):
                    self.say_blocking("Goodbye.")
                    break

                query = self._strip_wake_word(text) if self._contains_wake_word(text) else text
                if not query:
                    self.say_blocking("Yes?")
                    continue

                # No interrupt logic — just enqueue. If the helmet is busy,
                # this turn will be handled after the current one finishes.
                self.request_queue.put({"kind": "user", "text": query})
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self._hard_stop_all_audio()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()
