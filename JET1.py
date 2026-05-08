"""
ATLAS — Museum Helmet main script for Jetson Orin Nano.
Multilanguage edition: English / French / Spanish.

Hardware:
  - Jetson Orin Nano (JetPack 6.x, CUDA 12.6)
  - USB webcam (HD Camera, MJPG @ 1280x720, /dev/video0)
  - USB mic on sounddevice index 0 (hw:1,0), stereo input, downmixed to mono
  - USB speaker on plughw:0,0 (UACDemoV1.0)

Notes after the reflash:
  - PulseAudio is masked. Audio uses raw ALSA via aplay.
  - Camera is plain V4L2 + OpenCV. No GStreamer / nvargus / IMX708 pipeline.
  - Mic is stereo on this flash; we capture stereo and take left channel.
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
import unicodedata
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import vosk  # type: ignore

from dotenv import load_dotenv
from google import genai
from ultralytics import YOLOE  # type: ignore

vosk.SetLogLevel(-1)


# --------------------------------------------------------------------------
# Tunable constants.
# --------------------------------------------------------------------------

# --- Audio input (mic) ---
MIC_DEVICE = 0           # USB ENC Audio Device on this flash (sounddevice index)
MIC_CHANNELS = 2         # mic is stereo; we downmix to mono in code
MIC_NATIVE_RATE = 48000
MIC_SAMPLE_RATE = 16000  # what Vosk expects
MIC_BLOCKSIZE = 12000

# --- Audio output (speaker) ---
AUDIO_OUT_DEVICE: str | None = "plughw:0,0"

# --- Piper global ---
PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 1.00

# --- Multilanguage definitions ---
LANGUAGES: dict[str, dict] = {
    "english": {
        "vosk_model": "/opt/vosk_models/vosk-model-small-en-us-0.15",
        "piper_voice": "en_US-ryan-low",
        "switch_phrases": [
            # English commands
            "switch to english", "speak english", "english please", "in english",
            # French phrases for switching TO English (if French is active)
            "passez a l anglais", "passe a l anglais", "passe en anglais",
            "passez en anglais", "parle anglais", "en anglais",
            # Spanish phrases for switching TO English (if Spanish is active)
            "cambia al ingles", "cambiar al ingles", "habla ingles",
            "en ingles", "ingles por favor",
        ],
        "exit_phrases": [
            "goodbye", "good bye", "exit", "quit", "stop program", "see you",
        ],
        "ack_first": [
            "Let me think.",
            "One moment please.",
            "Good question, give me a second.",
        ],
        "ack_second": "Sorry, one second, let me think.",
        "failure": "There's a problem with the connection right now. Please try again in a moment.",
        "switch_confirmation": "Switching to English.",
        "exit_phrase": "Goodbye.",
        "greeting": (
            "Hi, I'm Atlas, your museum guide. You can ask me anything about art, "
            "history, or culture, or just stop in front of an exhibit and I'll tell you about it."
        ),
        "gemini_directive": "Respond ONLY in English.",
    },
    "french": {
        "vosk_model": "/opt/vosk_models/vosk-model-small-fr-0.22",
        "piper_voice": "fr_FR-siwis-medium",
        "switch_phrases": [
            "passez au francais", "passe au francais", "passe en francais",
            "passez en francais", "parle francais", "en francais",
            "francais s il vous plait",
            "switch to french", "speak french", "french please", "in french",
            "cambia al frances", "cambiar al frances", "habla frances",
            "en frances", "frances por favor",
        ],
        "exit_phrases": [
            "au revoir", "salut", "a bientot", "termine", "quitte",
            "arrete le programme",
        ],
        "ack_first": [
            "Laisse-moi réfléchir.",
            "Un instant, s'il vous plaît.",
            "Bonne question, un moment.",
        ],
        "ack_second": "Désolé, un moment, je réfléchis.",
        "failure": "Il y a un problème de connexion en ce moment. Veuillez réessayer dans un instant.",
        "switch_confirmation": "Je passe au français.",
        "exit_phrase": "Au revoir.",
        "greeting": (
            "Bonjour, je suis Atlas, votre guide de musée. Vous pouvez me poser "
            "n'importe quelle question sur l'art, l'histoire ou la culture."
        ),
        "gemini_directive": "Respond ONLY in French.",
    },
    "spanish": {
        "vosk_model": "/opt/vosk_models/vosk-model-small-es-0.42",
        "piper_voice": "es_MX-claude-high",
        "switch_phrases": [
            "cambia al espanol", "cambiar al espanol", "habla espanol",
            "en espanol", "espanol por favor",
            "pasa al espanol", "pasar al espanol",
            "switch to spanish", "speak spanish", "spanish please", "in spanish",
            "passez a l espagnol", "passe a l espagnol", "passe en espagnol",
            "parle espagnol", "en espagnol",
        ],
        "exit_phrases": [
            "adios", "hasta luego", "hasta la vista", "salir",
            "termina el programa",
        ],
        "ack_first": [
            "Déjame pensar.",
            "Un momento, por favor.",
            "Buena pregunta, un momento.",
        ],
        "ack_second": "Disculpe, un momento, estoy pensando.",
        "failure": "Hay un problema de conexión ahora mismo. Por favor, inténtelo de nuevo en un momento.",
        "switch_confirmation": "Cambiando al español.",
        "exit_phrase": "Adiós.",
        "greeting": (
            "Hola, soy Atlas, su guía del museo. Puede preguntarme cualquier cosa "
            "sobre arte, historia o cultura."
        ),
        "gemini_directive": "Respond ONLY in Spanish.",
    },
}

DEFAULT_LANGUAGE = "english"

# --- STT shared ---
STT_MIN_WORDS = 3
STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.55
POST_SPEAK_SETTLE_SECONDS = 0.60

# --- Wake words ---
WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")

# --- Memory ---
MEMORY_TURNS = 10

# --- Vision ---
DETECT_EVERY_N_FRAMES = 2
OBJECT_HOLD_SECONDS = 2.0
OBJECT_COOLDOWN_SECONDS = 8.0
TRIGGER_OBJECTS = {"mona lisa painting", "vase", "sword", "pharaoh mask"}
DETECT_CONFIDENCE_THRESHOLD = 0.40
TRIGGER_CONFIDENCE_THRESHOLD = 0.50

# --- USB Camera ---
CAMERA_INDEX = 0                      # /dev/video0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
# Set to one of:
#   None                          -> no rotation
#   cv2.ROTATE_90_CLOCKWISE       -> 90° clockwise
#   cv2.ROTATE_180                -> upside down -> rotate 180°
#   cv2.ROTATE_90_COUNTERCLOCKWISE
CAMERA_ROTATION = None
CAMERA_PROCESS_SIZE = (1024, 1024)    # final size YOLOE sees
YOLOE_IMGSZ = 320

# --- Gemini ---
GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"
ACK_DELAY_SECONDS = 1.5


# --------------------------------------------------------------------------
# Module-level helpers.
# --------------------------------------------------------------------------
def _strip_accents(text: str) -> str:
    """Normalize text for phrase matching: lowercase + remove accents.
    Vosk often transcribes 'français' as 'francais', so we normalize both
    sides when matching switch/exit phrases.
    """
    text = (text or "").lower().strip()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _piper_synthesize(voice: str, text: str, out_path: str) -> bool:
    """Run Piper to produce a WAV. `voice` is the model name like 'en_US-ryan-low'."""
    try:
        subprocess.run(
            [
                "python3", "-m", "piper",
                "--model", voice,
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
        if not self.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Put it in .env.")
        self.client = genai.Client(api_key=self.gemini_api_key)

        # --- Vosk: load ALL language models at startup ---
        self.vosk_models: dict[str, vosk.Model] = {}
        for lang_name, cfg in LANGUAGES.items():
            path = cfg["vosk_model"]
            if not os.path.isdir(path):
                raise RuntimeError(f"Vosk model for {lang_name} not found at {path}.")
            print(f"[STT] Loading Vosk model for {lang_name} from {path} ...")
            self.vosk_models[lang_name] = vosk.Model(path)
        print("[STT] All Vosk models loaded.")

        # --- Active language state ---
        self.current_language = DEFAULT_LANGUAGE
        self.language_lock = threading.Lock()
        self.language_change_event = threading.Event()

        # --- YOLOE ---
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            # Triggerable artifacts
            "mona lisa painting", "vase", "sword", "pharaoh mask",
            # Distractors
            "person", "face", "hand", "background wall",
        ]
        self.yoloe_imgsz = YOLOE_IMGSZ

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

        # --- Pre-rendered acknowledgment WAVs per language ---
        self._ack_wavs: dict[str, dict] = {}
        self._ack_tempdir = tempfile.mkdtemp(prefix="atlas_ack_")

        # --- System prompt ---
        self.system_prompt = """
You are Atlas, an AI museum guide embedded in a wearable helmet, speaking directly to a visitor in front of an exhibit.

Personality & Style
Speak warmly, naturally, and conversationally — like a real human guide.
Avoid sounding robotic, scripted, or like a textbook.
Keep responses concise: usually 1–2 short sentences, 3 only if clarity really needs it.
Prefer short back-and-forth interaction over long explanations.
Adjust energy depending on the subject.

What you will answer
You are an educational and cultural guide first. ANSWER any reasonable question about
art, history, culture, artifacts, artworks, artists, architecture, literature, mythology,
religion, science, nature, geography, historical events, historical figures, museums, and
general knowledge an educated museum guide would know — whether or not the subject is
physically in front of the visitor.

Only gently redirect for things clearly unrelated to education or culture: personal advice,
medical advice, financial advice, live sports scores, current news, directions to specific
addresses, or explicit political debate.

Style rules when answering
Give clear, simple, meaningful explanations.
When explaining an object: say what it is, why it matters, and one interesting detail.
Adapt to the visitor's level: simplify for beginners, add depth for advanced questions.
If unsure, acknowledge uncertainty calmly while still giving helpful context.

Vision & Context Awareness
Treat [Camera] notes as context about what the visitor is looking at right now.
If something may be misidentified, acknowledge uncertainty and still provide helpful context.
Vary phrasing to avoid sounding repetitive.

Privacy & Safety
Never mention storing, tracking, or saving personal data.
"""

        self.formatting_rules = """
CRITICAL OUTPUT FORMAT RULES — these are read aloud by a text-to-speech engine:
- Do NOT use asterisks (*), underscores (_), backticks (`), or any markdown.
- Do NOT use bold, italics, or any emphasis markers.
- Do NOT use bullet points, numbered lists, or dashes for lists.
- Do NOT use headers, titles, or section labels.
- Do NOT use emoji.
- Write ONLY plain spoken prose — continuous sentences, like a person talking.
"""

    # --------------------------------------------------------------------
    # Pre-render acknowledgment WAVs for all languages.
    # --------------------------------------------------------------------
    def _prepare_ack_wavs(self) -> None:
        print("[Piper] Pre-rendering acknowledgment audio for all languages...")
        for lang_name, cfg in LANGUAGES.items():
            voice = cfg["piper_voice"]
            self._ack_wavs[lang_name] = {"first_try": []}
            for i, phrase in enumerate(cfg["ack_first"]):
                path = os.path.join(self._ack_tempdir, f"ack_first_{lang_name}_{i}.wav")
                if _piper_synthesize(voice, phrase, path):
                    self._ack_wavs[lang_name]["first_try"].append(path)

            path = os.path.join(self._ack_tempdir, f"ack_second_{lang_name}.wav")
            if _piper_synthesize(voice, cfg["ack_second"], path):
                self._ack_wavs[lang_name]["second_try"] = path

            path = os.path.join(self._ack_tempdir, f"failure_{lang_name}.wav")
            if _piper_synthesize(voice, cfg["failure"], path):
                self._ack_wavs[lang_name]["failure"] = path

            path = os.path.join(self._ack_tempdir, f"switch_{lang_name}.wav")
            if _piper_synthesize(voice, cfg["switch_confirmation"], path):
                self._ack_wavs[lang_name]["switch_confirmation"] = path

            path = os.path.join(self._ack_tempdir, f"greeting_{lang_name}.wav")
            if _piper_synthesize(voice, cfg["greeting"], path):
                self._ack_wavs[lang_name]["greeting"] = path

            path = os.path.join(self._ack_tempdir, f"exit_{lang_name}.wav")
            if _piper_synthesize(voice, cfg["exit_phrase"], path):
                self._ack_wavs[lang_name]["exit_phrase"] = path

            print(f"[Piper] Acknowledgments ready for {lang_name}.")
        print("[Piper] All language acknowledgments ready.")

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
    # Active language helpers.
    # --------------------------------------------------------------------
    def _get_active_language(self) -> str:
        with self.language_lock:
            return self.current_language

    def _get_active_voice(self) -> str:
        return LANGUAGES[self._get_active_language()]["piper_voice"]

    def _get_active_gemini_directive(self) -> str:
        return LANGUAGES[self._get_active_language()]["gemini_directive"]

    def _switch_language(self, new_lang: str) -> None:
        if new_lang not in LANGUAGES:
            print(f"[Lang] unknown language: {new_lang}")
            return
        with self.language_lock:
            if new_lang == self.current_language:
                print(f"[Lang] already in {new_lang}")
                return
            print(f"[Lang] switching {self.current_language} -> {new_lang}")
            self.current_language = new_lang
        self.language_change_event.set()
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            wav = self._ack_wavs.get(new_lang, {}).get("switch_confirmation")
            if wav:
                self._play_cached_wav(wav)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

    def _detect_switch_command(self, text: str) -> str | None:
        normalized = _strip_accents(text)
        for lang_name, cfg in LANGUAGES.items():
            for phrase in cfg["switch_phrases"]:
                if _strip_accents(phrase) in normalized:
                    return lang_name
        return None

    def _is_exit_phrase(self, text: str) -> bool:
        normalized = _strip_accents(text)
        cfg = LANGUAGES[self._get_active_language()]
        for phrase in cfg.get("exit_phrases", []):
            if _strip_accents(phrase) in normalized:
                return True
        return False

    # --------------------------------------------------------------------
    # TTS.
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

        voice = self._get_active_voice()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            wav_path = tmp_wav.name

        try:
            piper_cmd = [
                "python3", "-m", "piper",
                "--model", voice,
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
    # Gemini.
    # --------------------------------------------------------------------
    def _gemini_try_once(self, model: str, prompt: str) -> str:
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
                first_try_wavs = self._ack_wavs.get(self._get_active_language(), {}).get("first_try", [])
                if first_try_wavs:
                    print("🤖 [ack] (let me think)")
                    self._play_cached_wav(random.choice(first_try_wavs))
                ack_played = True
            time.sleep(0.05)
        t1.join(timeout=0.1)

        status, payload = result_holder.get("t1", ("err", "unknown"))
        if status == "ok" and payload:
            return (payload, "ok")
        print(f"[Gemini] primary attempt 1 failed/empty: {payload[:120] if payload else '(empty)'}")

        if ack_enabled:
            second_wav = self._ack_wavs.get(self._get_active_language(), {}).get("second_try")
            if second_wav:
                print("🤖 [ack] (sorry, one second)")
                self._play_cached_wav(second_wav)

        try:
            text = self._gemini_try_once(GEMINI_MODEL_PRIMARY, prompt)
            if text:
                return (text, "ok")
            print("[Gemini] primary attempt 2 returned empty.")
        except Exception as e:
            print(f"[Gemini] primary attempt 2 failed: {e}")

        try:
            print(f"[Gemini] falling back to {GEMINI_MODEL_FALLBACK}")
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
noise (not a real question), or clearly not directed at you, reply with exactly:
SKIP
and nothing else.

Otherwise, answer normally as the museum guide.
"""

    def _build_user_prompt(self, user_text: str) -> str:
        directive = self._get_active_gemini_directive()
        return f"""{self.system_prompt}

LANGUAGE INSTRUCTION: {directive}

{self.formatting_rules}

Conversation so far (most recent last):
{self._memory_as_transcript()}

Visitor's latest line: {user_text}

{self._skip_instructions}

If you answer, keep it to 1–2 short sentences. Warm and conversational, plain prose.
Use prior turns when relevant so follow-ups feel natural.
{directive}
"""

    def _build_object_prompt(self, object_name: str) -> str:
        directive = self._get_active_gemini_directive()
        return f"""{self.system_prompt}

LANGUAGE INSTRUCTION: {directive}

{self.formatting_rules}

Conversation so far (most recent last):
{self._memory_as_transcript()}

Camera event:
The visitor has been steadily looking at an object detected as "{object_name}"
for at least {OBJECT_HOLD_SECONDS} seconds.

Task:
Give a short, natural museum-guide explanation of "{object_name}".
Do NOT mention detection or observation; just speak as if you noticed it yourself.
Keep it to 1–2 short sentences. If unsure, use soft uncertainty.
This is NOT a bystander event — never reply SKIP for a camera event.
{directive}
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
            ack_enabled = False
        else:
            return

        print("[Gemini] thinking ...")
        self.is_busy_event.set()
        try:
            response, status = self._gemini_request_with_retries(prompt, ack_enabled)
        except Exception as e:
            print(f"[Gemini] unexpected error: {e}")
            response, status = "", "failed"

        if status == "failed":
            failure_wav = self._ack_wavs.get(self._get_active_language(), {}).get("failure")
            if failure_wav:
                print("🤖 [failure] connection problem")
                self._play_cached_wav(failure_wav)
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
    # STT — reloads recognizer when language changes. Stereo -> mono.
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
            # Downmix stereo -> mono (Vosk wants mono).
            if MIC_CHANNELS > 1:
                samples = samples.reshape(-1, MIC_CHANNELS)
                samples = samples[:, 0].copy()
            if decim > 1:
                samples = samples[::decim]
            audio_q.put(samples.tobytes())

        while not self.stop_event.is_set():
            try:
                active_lang = self._get_active_language()
                model = self.vosk_models[active_lang]
                recognizer = vosk.KaldiRecognizer(model, MIC_SAMPLE_RATE)
                recognizer.SetWords(True)
                self.language_change_event.clear()

                with sd.RawInputStream(
                    samplerate=MIC_NATIVE_RATE,
                    blocksize=MIC_BLOCKSIZE,
                    dtype="int16",
                    channels=MIC_CHANNELS,
                    device=MIC_DEVICE,
                    callback=audio_callback,
                ):
                    print(f"[STT] Listening in {active_lang} on device {MIC_DEVICE} "
                          f"({MIC_CHANNELS}ch) @ {MIC_NATIVE_RATE} Hz -> {MIC_SAMPLE_RATE} Hz")

                    utt_start: float | None = None
                    was_busy = False

                    while not self.stop_event.is_set():
                        if self.language_change_event.is_set():
                            print("[STT] Language change detected, reloading recognizer.")
                            break

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
    # Camera worker — plain V4L2 USB camera, no GStreamer.
    # --------------------------------------------------------------------
    def camera_worker(self) -> None:
        cap = None
        try:
            print(f"[Camera] Opening /dev/video{CAMERA_INDEX} via V4L2 ...")
            cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open /dev/video{CAMERA_INDEX}")

            # Request MJPG @ 1280x720 @ 30fps. MJPG is required at this
            # resolution/framerate; raw YUYV is too much USB bandwidth.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, test_frame = cap.read()
            if not ret:
                raise RuntimeError("Camera opened but could not read first frame.")
            print(f"[Camera] Open OK, frame shape: {test_frame.shape}")

            print("[Camera] Loading YOLOE model on CUDA ...")
            model = YOLOE(self.model_path)
            model.set_classes(self.prompt_names)
            try:
                model.to("cuda")
                print("[Camera] YOLOE loaded on CUDA.")
            except Exception as e:
                print(f"[Camera] WARNING: could not move YOLOE to CUDA, will run on CPU: {e}")

            frame_idx = 0
            last_annotated = None
            last_fps = 0.0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if CAMERA_ROTATION is not None:
                    frame = cv2.rotate(frame, CAMERA_ROTATION)

                frame = cv2.resize(frame, CAMERA_PROCESS_SIZE)

                frame_idx += 1
                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    results = model.predict(
                        frame,
                        imgsz=self.yoloe_imgsz,
                        verbose=False,
                        device="cuda",
                    )
                    result = results[0]
                    last_annotated = result.plot(boxes=True, masks=False)

                    detections: list[dict] = []
                    boxes = result.boxes
                    if boxes is not None and boxes.cls is not None and boxes.conf is not None:
                        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                            if conf < DETECT_CONFIDENCE_THRESHOLD:
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

                lang_label = f"Lang: {self._get_active_language()}"
                fps_label = f"FPS: {last_fps:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(display, lang_label, (10, 30), font, 0.8,
                            (255, 255, 255), 2, cv2.LINE_AA)
                ts = cv2.getTextSize(fps_label, font, 0.8, 2)[0]
                cv2.putText(display, fps_label, (display.shape[1] - ts[0] - 10, 30),
                            font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                try:
                    cv2.imshow("ATLAS Museum Helmet", display)
                    if cv2.waitKey(1) == ord("q"):
                        self.stop_event.set()
                        break
                except cv2.error:
                    cv2.imwrite("/tmp/atlas_latest_frame.jpg", display)
        except Exception as e:
            print("Camera worker error:", e)
        finally:
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

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

        # Debug print so we know what's blocking trigger if anything.
        if held_long_enough:
            busy = self.is_busy_event.is_set()
            queue_empty = self.request_queue.empty()
            if not (off_cooldown and not busy and queue_empty):
                cooldown_remaining = OBJECT_COOLDOWN_SECONDS - (current_time - self.last_object_trigger_time.get(dominant_name, 0.0))
                print(f"[Trigger BLOCKED] {dominant_name} held "
                      f"{current_time - self.object_first_seen_time:.1f}s but: "
                      f"cooldown_remaining={cooldown_remaining:.1f}s, "
                      f"busy={busy}, queue_empty={queue_empty}")
                return

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
        if self._detect_switch_command(text):
            return True
        if self._is_exit_phrase(text):
            return True

        if len(text.split()) < STT_MIN_WORDS:
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
        self._prepare_ack_wavs()

        camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        camera_thread.start()

        stt_thread = threading.Thread(target=self._listen_forever, daemon=True)
        stt_thread.start()

        worker_thread = threading.Thread(target=self._gemini_worker, daemon=True)
        worker_thread.start()

        # Greeting in default language (use pre-rendered WAV).
        greeting_wav = self._ack_wavs.get(DEFAULT_LANGUAGE, {}).get("greeting")
        if greeting_wav:
            print(f"🤖 [greeting in {DEFAULT_LANGUAGE}]")
            self.is_busy_event.set()
            self.speak_start_time = time.time()
            try:
                self._play_cached_wav(greeting_wav)
            finally:
                self.is_busy_event.clear()
                self.last_speak_end_time = time.time()
        else:
            self.say_blocking(LANGUAGES[DEFAULT_LANGUAGE]["greeting"])

        try:
            while not self.stop_event.is_set():
                try:
                    utt = self.utterance_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                text = utt["text"]
                if not self._utterance_passes_noise_gate(utt):
                    continue

                print(f"\n[Heard]: {text}  (lang={self._get_active_language()}, "
                      f"conf={utt.get('conf')}, dur={utt.get('duration', 0):.2f}s)")

                target_lang = self._detect_switch_command(text)
                if target_lang is not None:
                    self._switch_language(target_lang)
                    continue

                if self._is_exit_phrase(text):
                    exit_wav = self._ack_wavs.get(self._get_active_language(), {}).get("exit_phrase")
                    if exit_wav:
                        print(f"🤖 [exit in {self._get_active_language()}]")
                        self.is_busy_event.set()
                        self.speak_start_time = time.time()
                        try:
                            self._play_cached_wav(exit_wav)
                        finally:
                            self.is_busy_event.clear()
                            self.last_speak_end_time = time.time()
                    else:
                        self.say_blocking(
                            LANGUAGES[self._get_active_language()].get("exit_phrase", "Goodbye.")
                        )
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
