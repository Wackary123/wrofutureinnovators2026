"""
ATLAS — Museum Helmet main script.

This version:
- Mid-speech interrupts work. STT stays active during playback.
- When user speech is detected while the helmet is speaking, playback is
  hard-cut and the new utterance becomes the next turn.
- Partial old response (whatever was said before the cut) is still saved
  to memory so follow-ups like "finish what you were saying" or "what did
  you mean about X" can reference it.
- Self-hearing guard is mostly disabled (physical speaker/mic separation
  per user). Only a small POST_SPEAK_SETTLE window remains as a light
  safety for the moment right after the speaker stops.
- Mid-speech utterances must pass a STRICTER noise gate
  (MID_SPEECH_MIN_CONF = 0.80) to reduce false interrupts from any stray
  audio the mic might pick up. Tune down if needed.

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
PIPER_VOICE = "en_US-ryan-medium"
PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 0.95

# --- Noise gate (idle) ---
STT_MIN_WORDS = 3
STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.55

# --- Noise gate (mid-speech interrupts) ---
# Raised to 0.80 to reduce false self-interrupts from mic picking up
# helmet's own voice. Raise further (0.85-0.90) if it still self-interrupts,
# lower (0.65-0.70) if real interrupts get ignored too often.
MID_SPEECH_MIN_CONF = 0.80
MID_SPEECH_MIN_WORDS = 3

# --- Self-hearing guard (tiny settle only) ---
POST_SPEAK_SETTLE_SECONDS = 0.20

# --- Keywords ---
WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")

# --- Memory ---
MEMORY_TURNS = 10

# --- Vision ---
DETECT_EVERY_N_FRAMES = 4
OBJECT_HOLD_SECONDS = 2.0
OBJECT_COOLDOWN_SECONDS = 8.0
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

        # --- Camera / YOLOE ---
        self.camera_size = (1280, 960)
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "mona lisa", "computer", "person", "vase", "iphone", "head",
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
        self.cancel_response_event = threading.Event()
        self.is_busy_event = threading.Event()

        self.speak_start_time = 0.0
        self.last_speak_end_time = 0.0

        # Subprocess handles for hard-cut.
        self._proc_lock = threading.Lock()
        self._piper_proc: subprocess.Popen | None = None
        self._aplay_proc: subprocess.Popen | None = None

        # Tracks what's currently being spoken + how much the user likely
        # heard before any interrupt. Used to preserve partial replies in
        # memory.
        self._speaking_text: str = ""
        self._speaking_duration_estimate: float = 0.0

        # --- Memory ---
        self.memory: deque = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

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
            elif role == "assistant_interrupted":
                lines.append(f"Guide (interrupted): {text}")
            elif role == "camera":
                lines.append(f"[Camera] Visitor is now looking at: {text}")
        return "\n".join(lines) if lines else "(no prior turns)"

    # --------------------------------------------------------------------
    # TTS — one shot per response, hard-cut capable.
    # --------------------------------------------------------------------
    def _speak_full(self, text: str) -> bool:
        """Synthesize + play. Returns True if played to completion, False if
        canceled mid-playback."""
        text = text.strip()
        if not text:
            return True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        completed = True
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

            while True:
                if self._piper_proc.poll() is not None:
                    break
                if self.cancel_response_event.is_set():
                    try:
                        self._piper_proc.terminate()
                    except Exception:
                        pass
                    completed = False
                    break
                time.sleep(0.02)
            with self._proc_lock:
                self._piper_proc = None

            if not completed or self.cancel_response_event.is_set():
                return False
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                return False

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

            while True:
                if self._aplay_proc.poll() is not None:
                    break
                if self.cancel_response_event.is_set():
                    try:
                        self._aplay_proc.terminate()
                    except Exception:
                        pass
                    completed = False
                    break
                time.sleep(0.02)
            with self._proc_lock:
                self._aplay_proc = None
        finally:
            if os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        return completed

    def _hard_stop_all_audio(self) -> None:
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
        self._speaking_text = text
        try:
            self._speak_full(text)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            self._speaking_text = ""

    # --------------------------------------------------------------------
    # Gemini — collect full response, respecting cancel.
    # --------------------------------------------------------------------
    def _collect_full_response(self, prompt: str) -> str:
        chunks: list[str] = []
        try:
            stream = self.client.models.generate_content_stream(
                model=self.gemini_model, contents=prompt,
            )
            for chunk in stream:
                if self.cancel_response_event.is_set():
                    return ""
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

Conversation so far (most recent last):
{transcript}

Visitor's latest line: {user_text}

{self._skip_instructions}

If you do answer, keep it short (1–3 sentences), warm and conversational.
Write your reply as ONE continuous answer — no bullet points, no list items,
no odd formatting. Just plain spoken sentences.
Use prior turns when relevant so follow-ups feel natural. If a prior turn is
labeled "Guide (interrupted)", it means you were cut off; if the visitor
refers back to it (e.g. "finish that" or "what was that about"), you can
continue from where you were interrupted.
"""

    def _build_object_prompt(self, object_name: str) -> str:
        transcript = self._memory_as_transcript()
        return f"""{self.system_prompt}

Conversation so far (most recent last):
{transcript}

Camera event:
The visitor has been steadily looking at an object detected as "{object_name}"
for at least {OBJECT_HOLD_SECONDS} seconds.

Task:
Give a short, natural museum-guide explanation of "{object_name}".
Do NOT mention detection or observation; just speak as if you noticed it yourself.
Keep it to 1–3 sentences. If unsure, use soft uncertainty.
Write your reply as ONE continuous answer — plain spoken sentences, no lists.
This is NOT a bystander event — never reply SKIP for a camera event. Just speak.
"""

    # --------------------------------------------------------------------
    # Partial-reply estimator for interrupted responses.
    # --------------------------------------------------------------------
    def _estimate_spoken_portion(self, full_text: str,
                                 speak_start: float,
                                 cut_time: float) -> str:
        if not full_text:
            return ""
        elapsed = max(0.0, cut_time - speak_start)
        chars_per_sec = 13.0 / PIPER_LENGTH_SCALE
        audible_chars = int(elapsed * chars_per_sec)
        if audible_chars <= 0:
            return ""
        if audible_chars >= len(full_text):
            return full_text
        slice_ = full_text[:audible_chars]
        last_sentence_end = max(
            slice_.rfind(". "),
            slice_.rfind("! "),
            slice_.rfind("? "),
        )
        if last_sentence_end > 0:
            return full_text[:last_sentence_end + 1].strip()
        last_space = slice_.rfind(" ")
        if last_space > 0:
            return full_text[:last_space].strip() + "…"
        return slice_.strip() + "…"

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

        self.cancel_response_event.clear()
        self._speaking_text = ""

        print(f"[Gemini] thinking ...")
        response = self._collect_full_response(prompt)

        if self.cancel_response_event.is_set():
            if kind == "user":
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
            return

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

        print(f"🤖 {response}")
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        self._speaking_text = response
        completed = True
        try:
            completed = self._speak_full(response)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

        if completed:
            self._memory_append("assistant", response)
        else:
            spoken_portion = self._estimate_spoken_portion(
                response, self.speak_start_time, self.last_speak_end_time
            )
            if spoken_portion:
                print(f"[Memory] saving interrupted portion: {spoken_portion!r}")
                self._memory_append("assistant_interrupted", spoken_portion)
        self._speaking_text = ""

    def _gemini_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_request(req.get("kind"), req.get("text", ""))

    # --------------------------------------------------------------------
    # STT — stays active during playback so interrupts work.
    # --------------------------------------------------------------------
    def _listen_forever(self) -> None:
        audio_q: queue.Queue = queue.Queue()

        if MIC_NATIVE_RATE % MIC_SAMPLE_RATE != 0:
            raise RuntimeError("MIC_NATIVE_RATE must be integer multiple of MIC_SAMPLE_RATE.")
        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE

        def audio_callback(indata, frames, time_info, status):
            if (time.time() - self.last_speak_end_time) < POST_SPEAK_SETTLE_SECONDS \
                    and not self.is_busy_event.is_set():
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
                    utt_mid_speech = False

                    while not self.stop_event.is_set():
                        try:
                            data = audio_q.get(timeout=0.2)
                        except queue.Empty:
                            continue

                        if utt_start is None:
                            utt_start = time.time()
                            utt_mid_speech = self.is_busy_event.is_set()

                        if recognizer.AcceptWaveform(data):
                            result = json.loads(recognizer.Result())
                            text = (result.get("text") or "").strip().lower()
                            conf = self._avg_word_conf(result.get("result"))
                            duration = time.time() - (utt_start or time.time())
                            utt_start = None

                            if not text:
                                continue

                            self.utterance_queue.put({
                                "text": text,
                                "conf": conf,
                                "duration": duration,
                                "mid_speech": utt_mid_speech,
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
        mid_speech = utt.get("mid_speech", False)

        if self._contains_wake_word(text):
            return True

        word_count = len(text.split())
        min_words = MID_SPEECH_MIN_WORDS if mid_speech else STT_MIN_WORDS
        min_conf = MID_SPEECH_MIN_CONF if mid_speech else VOSK_MIN_CONF

        if word_count < min_words:
            return False
        if duration < STT_MIN_SECONDS:
            return False
        if conf is not None and conf < min_conf:
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
                      f"dur={utt.get('duration', 0):.2f}s, "
                      f"mid_speech={utt.get('mid_speech', False)})")

                if any(kw in text for kw in EXIT_WORDS):
                    if self.is_busy_event.is_set():
                        self.cancel_response_event.set()
                        self._hard_stop_all_audio()
                        time.sleep(0.05)
                    self.say_blocking("Goodbye.")
                    break

                query = self._strip_wake_word(text) if self._contains_wake_word(text) else text
                if not query:
                    self.say_blocking("Yes?")
                    continue

                if self.is_busy_event.is_set():
                    print("[Interrupt] cutting current response")
                    self.cancel_response_event.set()
                    self._hard_stop_all_audio()
                    time.sleep(0.10)

                self.request_queue.put({"kind": "user", "text": query})
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self.cancel_response_event.set()
            self._hard_stop_all_audio()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()
