import json, os, re, time, random, threading, queue, subprocess, tempfile
from collections import deque

import cv2
import numpy as np
import sounddevice as sd
import vosk
from dotenv import load_dotenv
from google import genai
from ultralytics import YOLOE

try:
    import torch
    torch.set_num_threads(3)
except Exception:
    pass

vosk.SetLogLevel(-1)

MIC_DEVICE = None
MIC_NATIVE_RATE = 48000
MIC_SAMPLE_RATE = 16000
MIC_BLOCKSIZE = 12000

AUDIO_OUT_DEVICE = "default"

VOSK_MODEL_PATH = "/opt/vosk_models/vosk-model-small-en-us-0.15"
PIPER_VOICE = "en_US-ryan-low"
PIPER_DATA_DIR = os.path.expanduser("~/piper_voices")
PIPER_LENGTH_SCALE = 1.10

CAMERA_DEVICE = "/dev/video0"
CAMERA_FLIP_180 = True
CAMERA_PROCESS_SIZE = (1200, 1200)
SHOW_PREVIEW = True

DETECT_EVERY_N_FRAMES = 4
OBJECT_HOLD_SECONDS = 1.5
OBJECT_COOLDOWN_SECONDS = 3.0
DETECT_CONFIDENCE_THRESHOLD = 0.10
TRIGGER_CONFIDENCE_THRESHOLD = 0.10

TRIGGER_OBJECTS = {"vase", "sword", "pharaoh mask", "mona lisa"}
OBJECT_PRIORITY = {"pharaoh mask": 1, "mona lisa": 2, "crown": 3, "vase": 4}

WAKE_WORDS = ("atlas", "helmet", "guide", "assistant")
EXIT_WORDS = ("goodbye", "good bye", "exit", "quit", "stop program")

STT_MIN_WORDS = 3
STT_MIN_SECONDS = 1.0
VOSK_MIN_CONF = 0.55
POST_SPEAK_SETTLE_SECONDS = 0.60

MEMORY_TURNS = 10

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash-lite"
GEMINI_MODEL_FALLBACK = "gemini-2.5-flash-lite"

ACK_DELAY_SECONDS = 1
ACK_FIRST_TRY_PHRASES = [
    "Hmmmm... let me think for a second please",
    "One moment please, let me think",
    "Let me think for a moment, please wait."
]
ACK_SECOND_TRY_PHRASE = "Sorry, one second, let me think."
FAILURE_PHRASE = "There's a problem with the connection right now. Please try again in a moment."

GREETING = "Hi, I'm your museum guide. You can ask me anything, or just stop in front of something and I'll tell you about it."


def _aplay_command(wav_path: str) -> list[str]:
    cmd = ["aplay", "-q"]
    if AUDIO_OUT_DEVICE:
        cmd += ["-D", AUDIO_OUT_DEVICE]
    cmd.append(wav_path)
    return cmd


def _piper_synthesize(text: str, out_path: str) -> bool:
    try:
        subprocess.run(
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
    except Exception as e:
        print("[Piper] synth failed:", e)
        return False


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)

        if not os.path.isdir(VOSK_MODEL_PATH):
            raise RuntimeError(f"Vosk model not found at {VOSK_MODEL_PATH}")

        print("[STT] Loading Vosk model...")
        self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
        print("[STT] Vosk model loaded.")

        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "vase", "mona lisa", "cement", "head",
            "pharaoh mask", "royal crown", "bone",
            "statue", "diamonds", "shiny", "sword"
        ]

        self.detect_confidence_threshold = DETECT_CONFIDENCE_THRESHOLD
        self.model_imgsz = 192

        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        self.utterance_queue = queue.Queue()
        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_busy_event = threading.Event()

        self.speak_start_time = 0.0
        self.last_speak_end_time = 0.0

        self._proc_lock = threading.Lock()
        self._piper_proc = None
        self._aplay_proc = None

        self.memory = deque(maxlen=MEMORY_TURNS * 2 + 5)
        self.memory_lock = threading.Lock()

        self._ack_first_try_wavs = []
        self._ack_second_try_wav = None
        self._failure_wav = None
        self._ack_tempdir = tempfile.mkdtemp(prefix="atlas_ack_")

        self.system_prompt = """
You are Atlas, a warm AI museum guide inside a wearable helmet.
Speak naturally and briefly, usually 1 to 2 short sentences.
Answer educational questions about art, history, culture, artifacts, museums, science, geography, mythology, religion, and general knowledge.
If the visitor is looking at an object, explain what it is, why it matters, and one interesting detail.
Never use markdown, bullets, emojis, asterisks, underscores, or backticks.
Everything you write will be spoken aloud, so write plain spoken prose only.
"""

    def _prepare_ack_wavs(self):
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

    def _play_cached_wav(self, wav_path):
        if not wav_path or not os.path.exists(wav_path):
            return
        try:
            with self._proc_lock:
                self._aplay_proc = subprocess.Popen(
                    _aplay_command(wav_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            self._aplay_proc.wait()
        finally:
            with self._proc_lock:
                self._aplay_proc = None

    def _sanitize_for_tts(self, text):
        text = re.sub(r"[*_`~]", "", text or "")
        text = re.sub(r"^\s*[-•]+\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+[.)]\s*", "", text, flags=re.MULTILINE)
        return re.sub(r"\s+", " ", text).strip()

    def _speak_full(self, text):
        text = self._sanitize_for_tts(text)
        if not text:
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            with self._proc_lock:
                self._piper_proc = subprocess.Popen(
                    [
                        "python3", "-m", "piper",
                        "--model", PIPER_VOICE,
                        "--data-dir", PIPER_DATA_DIR,
                        "--length-scale", str(PIPER_LENGTH_SCALE),
                        "--output-file", wav_path,
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            if self._piper_proc.stdin:
                self._piper_proc.stdin.write(text.encode("utf-8"))
                self._piper_proc.stdin.close()

            self._piper_proc.wait()

            with self._proc_lock:
                self._piper_proc = None

            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                with self._proc_lock:
                    self._aplay_proc = subprocess.Popen(
                        _aplay_command(wav_path),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                self._aplay_proc.wait()

        finally:
            with self._proc_lock:
                self._aplay_proc = None
            if os.path.exists(wav_path):
                os.remove(wav_path)

    def say_blocking(self, text):
        print(text)
        self.is_busy_event.set()
        self.speak_start_time = time.time()
        try:
            self._speak_full(text)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

    def _hard_stop_all_audio(self):
        with self._proc_lock:
            for p in (self._piper_proc, self._aplay_proc):
                if p and p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass

    def _memory_append(self, role, text):
        with self.memory_lock:
            self.memory.append((role, text))

    def _memory_as_transcript(self):
        with self.memory_lock:
            items = list(self.memory)
        lines = []
        for role, text in items:
            if role == "user":
                lines.append(f"Visitor: {text}")
            elif role == "assistant":
                lines.append(f"Guide: {text}")
            elif role == "camera":
                lines.append(f"[Camera] Visitor is looking at: {text}")
        return "\n".join(lines) if lines else "(no prior turns)"

    def _gemini_try_once(self, model, prompt):
        chunks = []
        stream = self.client.models.generate_content_stream(model=model, contents=prompt)
        for chunk in stream:
            delta = getattr(chunk, "text", None)
            if delta:
                chunks.append(delta)
        return "".join(chunks).strip()

    def _gemini_request_with_retries(self, prompt, ack_enabled):
        result_holder = {}

        def attempt(model, key):
            try:
                result_holder[key] = ("ok", self._gemini_try_once(model, prompt))
            except Exception as e:
                result_holder[key] = ("err", str(e))

        t1 = threading.Thread(target=attempt, args=(GEMINI_MODEL_PRIMARY, "t1"), daemon=True)
        t1.start()

        start = time.time()
        ack_played = False

        while t1.is_alive():
            if ack_enabled and not ack_played and time.time() - start >= ACK_DELAY_SECONDS:
                if self._ack_first_try_wavs:
                    self._play_cached_wav(random.choice(self._ack_first_try_wavs))
                ack_played = True
            time.sleep(0.05)

        status, payload = result_holder.get("t1", ("err", "unknown"))

        if status == "ok" and payload:
            return payload, "ok"

        if ack_enabled and self._ack_second_try_wav:
            self._play_cached_wav(self._ack_second_try_wav)

        for model in (GEMINI_MODEL_PRIMARY, GEMINI_MODEL_FALLBACK):
            try:
                text = self._gemini_try_once(model, prompt)
                if text:
                    return text, "ok"
            except Exception as e:
                print("[Gemini] failed:", e)

        return "", "failed"

    def _build_user_prompt(self, user_text):
        return f"""
{self.system_prompt}

Conversation so far:
{self._memory_as_transcript()}

Visitor's latest line: {user_text}

If the latest line is random background chatter or not directed at you, reply exactly:
SKIP

Otherwise answer as Atlas in 1 to 2 short spoken sentences.
"""

    def _build_object_prompt(self, object_name):
        return f"""
{self.system_prompt}

Conversation so far:
{self._memory_as_transcript()}

The visitor has been steadily looking at an object detected as "{object_name}".

Give a short natural museum-guide explanation. Do not mention detection. Keep it to 1 to 2 short sentences.
"""

    def _handle_request(self, kind, text):
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

        print("[Gemini] thinking...")
        self.is_busy_event.set()

        response, status = self._gemini_request_with_retries(prompt, ack_enabled)

        if status == "failed":
            if self._failure_wav:
                self._play_cached_wav(self._failure_wav)
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return

        response = self._sanitize_for_tts(response)

        if not response:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return

        if kind == "user" and response.split(None, 1)[0].strip().rstrip(".").upper() == "SKIP":
            print("[Gemini] SKIP")
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()
            return

        print(response)
        self.speak_start_time = time.time()

        try:
            self._speak_full(response)
        finally:
            self.is_busy_event.clear()
            self.last_speak_end_time = time.time()

        self._memory_append("assistant", response)

    def _gemini_worker(self):
        while not self.stop_event.is_set():
            try:
                req = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            self._handle_request(req.get("kind"), req.get("text", ""))

    @staticmethod
    def _avg_word_conf(words):
        if not isinstance(words, list) or not words:
            return None
        confs = [w.get("conf") for w in words if isinstance(w, dict) and "conf" in w]
        return sum(confs) / len(confs) if confs else None

    def _listen_forever(self):
        audio_q = queue.Queue()

        decim = MIC_NATIVE_RATE // MIC_SAMPLE_RATE

        def audio_callback(indata, frames, time_info, status):
            if self.is_busy_event.is_set():
                return
            if time.time() - self.last_speak_end_time < POST_SPEAK_SETTLE_SECONDS:
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
                    print(f"[STT] Listening on device {MIC_DEVICE}")
                    utt_start = None

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
                print("[STT] listener error:", e)
                time.sleep(0.5)

    def camera_worker(self):
        cap = None

        try:
            cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

            if not cap.isOpened():
                raise RuntimeError("Camera failed to open.")

            print("[Camera] OpenCV camera opened.")

            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Camera opened but could not read frames.")

            print(f"[Camera] First frame shape: {frame.shape}")
            print("[Camera] Loading YOLOE model...")

            model = YOLOE(self.model_path)
            model.set_classes(self.prompt_names)

            print("[Camera] YOLOE model loaded.")

            frame_idx = 0
            last_annotated = None
            last_fps = 0.0

            while not self.stop_event.is_set():
                ret, frame = cap.read()

                if not ret:
                    continue

                if CAMERA_FLIP_180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)

                if len(frame.shape) == 2:
                    try:
                        raw = frame[:, :, 0] if len(frame.shape) == 3 else frame
                        frame = cv2.cvtColor(raw, cv2.COLOR_BAYER_BGGR2BGR)
                    except Exception:
                        print("[Camera] Bayer convert failed:", e)

                frame = cv2.resize(frame, CAMERA_PROCESS_SIZE)

                frame_idx += 1

                if frame_idx % DETECT_EVERY_N_FRAMES == 0:
                    results = model.predict(frame, imgsz=self.model_imgsz, verbose=False)
                    result = results[0]

                    last_annotated = result.plot(boxes=True, masks=False)

                    detections = []
                    boxes = result.boxes

                    if boxes is not None and boxes.cls is not None and boxes.conf is not None:
                        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
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

                cv2.putText(
                    display,
                    f"FPS infer: {last_fps:.1f}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                if SHOW_PREVIEW:
                    cv2.imshow("ATLAS Live Camera", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self.stop_event.set()
                        break

        except Exception as e:
            print("Camera worker error:", e)

        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()

    def _maybe_trigger_object_explanation(self, detections):
        current_time = time.time()

        if detections:
            summary = ", ".join(f"{d['name']} ({d['confidence']:.2f})" for d in detections)
            if summary != self.last_terminal_objects:
                print("[Camera detected]:", summary)
                self.last_terminal_objects = summary
        else:
            if self.last_terminal_objects is not None:
                print("[Camera detected]: none")
                self.last_terminal_objects = None

        triggerable = [
            d for d in detections
            if d["name"] in TRIGGER_OBJECTS and d["confidence"] >= TRIGGER_CONFIDENCE_THRESHOLD
        ]

        if not triggerable:
            self.last_seen_object = None
            self.object_first_seen_time = None
            return

        dominant = sorted(
            triggerable,
            key=lambda d: (OBJECT_PRIORITY.get(d["name"], 999), -d["confidence"])
        )[0]

        dominant_name = dominant["name"]

        if dominant_name != self.last_seen_object:
            self.last_seen_object = dominant_name
            self.object_first_seen_time = current_time
            return

        if self.object_first_seen_time is None:
            self.object_first_seen_time = current_time
            return

        held_long_enough = current_time - self.object_first_seen_time >= OBJECT_HOLD_SECONDS
        off_cooldown = current_time - self.last_object_trigger_time.get(dominant_name, 0.0) >= OBJECT_COOLDOWN_SECONDS

        if held_long_enough and off_cooldown and not self.is_busy_event.is_set() and self.request_queue.empty():
            print("[Camera trigger]:", dominant_name)
            self.last_object_trigger_time[dominant_name] = current_time
            self.request_queue.put({"kind": "object", "text": dominant_name})
            self.object_first_seen_time = current_time

    def _contains_wake_word(self, text):
        for w in WAKE_WORDS:
            if w in text:
                return w
        return None

    def _utterance_passes_noise_gate(self, utt):
        text = utt["text"]
        conf = utt.get("conf")
        duration = utt.get("duration", 0.0)

        if self._contains_wake_word(text):
            return True
        if len(text.split()) < STT_MIN_WORDS:
            return False
        if duration < STT_MIN_SECONDS:
            return False
        if conf is not None and conf < VOSK_MIN_CONF:
            return False

        return True

    def _strip_wake_word(self, text):
        out = text
        for w in WAKE_WORDS:
            out = out.replace(w, "", 1)
        return out.strip()

    def start(self):
        self._prepare_ack_wavs()

        threading.Thread(target=self.camera_worker, daemon=True).start()
        threading.Thread(target=self._listen_forever, daemon=True).start()
        threading.Thread(target=self._gemini_worker, daemon=True).start()

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

                print("\n[Heard]:", text)

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
