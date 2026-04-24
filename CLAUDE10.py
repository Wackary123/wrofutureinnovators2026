"""
ATLAS — Museum Helmet main script.

Hardware
--------
Mic:     MillSO MQ5 USB lavalier on sounddevice index 1.
Speaker: USB speaker on ALSA card 4, routed via plughw:4,0.
Camera:  Raspberry Pi Camera Module 3, mounted upside down (180° flipped).
"""

import json
import os
import re
import time
import random
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
from libcamera import Transform  # type: ignore
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
PIPER_VOICE = "en_US-ryan-low"
PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 0.90

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
TRIGGER_OBJECTS = {"vase", "sword", "pharaoh mask", "mona lisa"}
OBJECT_PRIORITY = {
    "pharaoh mask": 1,
    "mona lisa": 2,
    "vase": 3,
    "sword": 4,
}
DETECT_CONFIDENCE_THRESHOLD = 0.10
TRIGGER_CONFIDENCE_THRESHOLD = 0.10

# --- Camera ---
CAMERA_FULL_SENSOR = (0, 0, 4608, 2592)
CAMERA_PREVIEW_SIZE = (1200, 1200)
CAMERA_FLIP_180 = True

# --- Gemini retry / acknowledgment ---
GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"
# Time to wait after hearing question before playing "let me think".
ACK_DELAY_SECONDS = 1.5
# Acknowledgment phrases (pre-rendered). Random first-try choice.
ACK_FIRST_TRY_PHRASES = [
    "Let me think.",
    "One moment.",
    "Good question.",
]
ACK_SECOND_TRY_PHRASE = "Sorry, one second, let me think."
FAILURE_PHRASE = "There's a problem with the connection right now. Please try again in a moment."

# --- Greeting ---
GREETING = (
    "Hi, I'm your museum guide. You can ask me anything, or just stop in "
    "front of an exhibit and I'll tell you about it."
)


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------
def _piper_synthesize(text: str, out_path: str) -> bool:
    """Blocking: run piper once and write a WAV. Returns True on success."""
    try:
        proc = subprocess.run(
            [
                "python3", "-m", "piper",
                "--model", PIPER_VOICE,
                "--data-dir", PIPER_DATA_DIR,
                "--length-scale", str(PIPER_LENGTH_SCALE),
                "--output-file", out_path,
            ],
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except subprocess.CalledProcessError:
        return False


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        # --- AI setup ---
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)

        # --- Vosk STT ---
        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}.")
        print(f"[STT] Loading Vosk model from {VOSK_MODEL_PATH} ...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        print("[STT] Vosk model loaded.")

        # --- Camera / YOLOE ---
        self.camera_size = CAMERA_PREVIEW_SIZE
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "vase","mona lisa ", "cement","head",
            "pharaoh mask", "royal crown", " bone", "statue", "diamonds", "shiny"
        ]
        self.detect_confidence_threshold = DETECT_CONFIDENCE_THRESHOLD
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

        self._proc_lock = threading.Lock()
        self._piper_proc: subprocess.Popen | None = None
        self._aplay_proc: subprocess.Popen | None = None

        # --- Memory ---
        self.memory: deque = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

        # --- Pre-rendered acknowledgment WAV paths ---
        # Populated by _prepare_ack_wavs() at startup.
        self._ack_first_try_wavs: list[str] = []
        self._ack_second_try_wav: str | None = None
        self._failure_wav: str | None = None
        self._ack_tempdir = tempfile.mkdtemp(prefix="atlas_ack_")

        # --- System prompt ---
        self.system_prompt = """
You are an AI museum guide called Atlas embedded in a wearable helmet, speaking directly to a visitor in front of an exhibit. You also have a secondary, non-intrusive safety role as a security guard.
Personality & Style
Your name is Atlas , an AI cultural and educational guide.
Speak like a real human guide: warm, natural, and conversational.
Avoid sounding robotic, scripted, or like a textbook.
Keep responses concise: usually 1–2 short sentences, 3 only if clarity really needs it.
Prefer short back-and-forth interaction over long explanations.
Adjust energy depending on the subject.

What you will answer
You are an educational and cultural guide first called Atlas. ANSWER any reasonable question about:
art, history, culture, artifacts, artworks, artists, architecture, literature, mythology,
religion, science, nature, geography, historical events, historical figures, museums, and
general knowledge that an educated museum guide would know. This is true whether or not
the subject is physically in front of the visitor. If the visitor asks "what is the Mona Lisa"
or "who was Napoleon" or "how old is the Colosseum", answer directly and helpfully even if
that topic is not the current exhibit.

Only gently redirect if the question is clearly unrelated to education or culture — things
like: personal advice, medical advice, financial advice, live sports scores, current news,
directions to specific addresses, or explicit political debate. For those, briefly decline
and steer back.

Style rules when answering
Give clear, simple, meaningful explanations.
When explaining an object: say what it is, why it matters, and one interesting detail.
Adapt to the visitor's level: simplify for beginners, add depth for advanced questions.
If unsure, acknowledge uncertainty calmly while still giving helpful context.
Avoid vague or generic explanations — always give a specific reason or fact.

Interaction Rules
Do not overwhelm the visitor with too much information.
Avoid long monologues unless explicitly requested.
Match the visitor's tone (curious, excited, confused).
If the visitor is incorrect, correct them politely and briefly.

Vision & Context Awareness
Treat [Camera] notes as context about what the visitor is looking at right now.
If something may be misidentified, acknowledge uncertainty and still provide helpful context.
Vary phrasing to avoid sounding repetitive.

Privacy & Safety
Keep the safety role subtle and secondary.
Never mention storing, tracking, or saving personal data.

Overall Goal
Act like a knowledgeable, friendly guide beside the visitor.
"""

        self.formatting_rules = """
CRITICAL OUTPUT FORMAT RULES — these are read aloud by a text-to-speech engine:
- Do NOT use asterisks (*), underscores (_), backticks (`), or any markdown.
- Do NOT use bold, italics, or any emphasis markers.
- Do NOT use bullet points, numbered lists, or dashes for lists.
- Do NOT use headers, titles, or section labels.
- Do NOT use emojis
- Write ONLY plain spoken prose — continuous sentences, like a person talking.
- Every character you write will be spoken out loud, so anything that isn't
  a natural spoken word will sound wrong.
"""

    # --------------------------------------------------------------------
    # Pre-render acknowledgment WAVs.
    # --------------------------------------------------------------------
    def _prepare_ack_wavs(self) -> None:
        print("[Piper] Pre-rendering acknowledgment audio...")
        for i, phrase in enumerate(ACK_FIRST_TRY_PHRASES):
            path = os.path.join(self._ack_tempdir, f"ack_first_{i}.wav")
            if _piper_synthesize(phrase, path):
                self._ack_first_try_wavs.append(path)

        path = os.path.join(self._ack_tempdir, "ack_second.wav")
        if _piper_synthesize(ACK_SECOND_TRY_PHRASE, path):
            self._ack_second_try_wav = path

        path = os.path.join(self._ack_tempdir, "failure.wav")
        if _piper_synthesize(FAILURE_PHRASE, path):
            self._failure_wav = path
        print("[Piper] Acknowledgment audio ready.")

    def _play_cached_wav(self, wav_path: str) -> None:
        if not wav_path or not os.path.exists(wav_path):
            return
        aplay_cmd = ["aplay", "-q"]
        if AUDIO_OUT_DEVICE:
            aplay_cmd += ["-D", AUDIO_OUT_DEVICE]
        aplay_cmd.append(wav_path)
        try:
            with self._proc_lock:
                self._aplay_proc = subprocess.Popen(
                    aplay_cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._aplay_proc.wait()
        finally:
            with self._proc_lock:
                self._aplay_proc = None

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
        if not text:
            return text
        text = re.sub(r"[*_`~]", "", text)
        text = re.sub(r"^\s*[-•]+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
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
    # Gemini with retry + fallback model.
    # --------------------------------------------------------------------
    def _gemini_try_once(self, model: str, prompt: str) -> str:
        """One attempt. Returns response text, or raises on error."""
        chunks: list[str] = []
        stream = self.client.models.generate_content_stream(
            model=model, contents=prompt,
        )
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if delta:
                chunks.append(delta)
        return "".join(chunks).strip()

    def _gemini_request_with_retries(self, prompt: str,
                                     ack_enabled: bool) -> tuple[str, str]:
        """Run the request with up to 3 attempts and acknowledgments.

        Returns (response_text, status) where status is one of:
          "ok"            — got a response
          "empty"         — got back empty string
          "failed"        — all attempts failed

        Acknowledgment timing (only if ack_enabled):
          - t=0      : request sent to primary model
          - t=1.5s   : if still pending, play random "let me think"
          - if that attempt fails: play "sorry one second let me think" and retry primary
          - if second also fails: silently switch to fallback model and retry
          - if all three fail: play failure phrase
        """
        # Phase 1: attempt primary. Launch in a thread so we can play ack
        # after 1.5s if it hasn't returned.
        result_holder: dict = {}

        def attempt(model, key):
            try:
                result_holder[key] = ("ok", self._gemini_try_once(model, prompt))
            except Exception as e:
                result_holder[key] = ("err", str(e))

        t1 = threading.Thread(target=attempt, args=(GEMINI_MODEL_PRIMARY, "t1"), daemon=True)
        t1.start()

        t1_start = time.time()
        ack_played = False
        while t1.is_alive():
            if ack_enabled and not ack_played and (time.time() - t1_start) >= ACK_DELAY_SECONDS:
                # Play first-try acknowledgment.
                if self._ack_first_try_wavs:
                    choice = random.choice(self._ack_first_try_wavs)
                    print("🤖 [ack] let me think")
                    self._play_cached_wav(choice)
                ack_played = True
            time.sleep(0.05)
        t1.join(timeout=0.1)

        status, payload = result_holder.get("t1", ("err", "unknown"))
        if status == "ok" and payload:
            return (payload, "ok")
        if status == "ok" and not payload:
            # Got empty response — retry like an error.
            print("[Gemini] primary returned empty, retrying.")
        else:
            print(f"[Gemini] primary attempt 1 failed: {payload}")

        # Phase 2: retry primary, with second-try acknowledgment.
        if ack_enabled and self._ack_second_try_wav:
            print("🤖 [ack] sorry, one second, let me think")
            self._play_cached_wav(self._ack_second_try_wav)

        try:
            text = self._gemini_try_once(GEMINI_MODEL_PRIMARY, prompt)
            if text:
                return (text, "ok")
            print("[Gemini] primary attempt 2 returned empty, falling back.")
        except Exception as e:
            print(f"[Gemini] primary attempt 2 failed: {e}")

        # Phase 3: fall back to lite, silently.
        try:
            print(f"[Gemini] switching to fallback model {GEMINI_MODEL_FALLBACK}")
            text = self._gemini_try_once(GEMINI_MODEL_FALLBACK, prompt)
            if text:
                return (text, "ok")
            print("[Gemini] fallback returned empty.")
        except Exception as e:
            print(f"[Gemini] fallback failed: {e}")

        return ("", "failed")

    # --------------------------------------------------------------------
    # Prompts.
    # --------------------------------------------------------------------
    _skip_instructions = """
Bystander filter:
If the visitor's latest line looks like random background chatter, off-topic
noise (not a real question), an unrelated side conversation, or clearly not
directed at you, reply with exactly the single token:
SKIP
and nothing else. Do not explain.

Otherwise, answer normally as the museum guide. Remember: educational and
cultural questions about anything in the world are ON TOPIC — do not reply
SKIP just because the topic isn't the current exhibit.
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
    # Request handling.
    # --------------------------------------------------------------------
    def _handle_request(self, kind: str, text: str) -> None:
        if kind == "user":
            prompt = self._build_user_prompt(text)
            self._memory_append("user", text)
            ack_enabled = True
        elif kind == "object":
            prompt = self._build_object_prompt(text)
            self._memory_append("camera", text)
            ack_enabled = False  # camera triggers don't get "let me think"
        else:
            return

        print("[Gemini] thinking ...")
        # The request function itself may play acknowledgments during retries.
        # Set is_busy_event so STT drops frames during those ack plays.
        self.is_busy_event.set()
        try:
            response, status = self._gemini_request_with_retries(prompt, ack_enabled)
        finally:
            # Don't clear yet — we still need to speak the actual answer below.
            pass

        if status == "failed":
            if self._failure_wav:
                print("🤖 [failure] connection problem")
                self._play_cached_wav(self._failure_wav)
            # Roll back the user memory entry since we never answered.
            if kind == "user":
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return

        if not response:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return

        if kind == "user":
            first_token = response.split(None, 1)[0].strip().rstrip(".").upper() if response else ""
            if first_token == "SKIP":
                print("[Gemini] SKIP — bystander noise, staying silent.")
                with self.memory_lock:
                    if self.memory and self.memory[-1] == ("user", text):
                        self.memory.pop()
                self.is_busy_event.clear()
                self.last_speak_end_time = time.time()
                return

        sanitized = self._sanitize_for_tts(response)
        print(f"🤖 {sanitized}")
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
    # STT.
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
            transform = Transform(hflip=True, vflip=True) if CAMERA_FLIP_180 else Transform()
            config = picam2.create_preview_configuration(
                main={"size": self.camera_size, "format": "RGB888"},
                transform=transform,
            )
            picam2.configure(config)
            picam2.start()
            time.sleep(0.2)

            try:
                picam2.set_controls({"ScalerCrop": CAMERA_FULL_SENSOR})
                print(f"[Camera] ScalerCrop set to full sensor {CAMERA_FULL_SENSOR}")
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
                            if conf < self.detect_confidence_threshold:
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
        if detections:
            summary = ", ".join(f"{d['name']} ({d['confidence']:.2f})" for d in detections)
            if summary != self.last_terminal_objects:
                print(f"[Camera detected]: {summary}")
                self.last_terminal_objects = summary
        else:
            if self.last_terminal_objects is not None:
                print("[Camera detected]: none")
                self.last_terminal_objects = None

        triggerable = [
            d for d in detections
            if d["name"] in TRIGGER_OBJECTS
            and d["confidence"] >= TRIGGER_CONFIDENCE_THRESHOLD
        ]
        if not triggerable:
            self.last_seen_object = None
            self.object_first_seen_time = None
            return

        dominant = sorted(
            triggerable,
            key=lambda d: (
                OBJECT_PRIORITY.get(d["name"], 999),
            -d["confidence"]
            )
        )[0]
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
            print(f"[Camera trigger]: {dominant_name} "
                  f"(conf={dominant['confidence']:.2f}) held {OBJECT_HOLD_SECONDS}s — enqueuing")
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
        # Pre-render acknowledgments before anything else.
        self._prepare_ack_wavs()

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

                self.request_queue.put({"kind": "user", "text": query})
        except KeyboardInterrupt:
            print("\n[Ctrl-C] shutting down.")
        finally:
            self.stop_event.set()
            self._hard_stop_all_audio()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()
