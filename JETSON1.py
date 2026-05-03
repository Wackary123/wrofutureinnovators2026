import cv2
import numpy as np
import sounddevice as sd
import vosk
import queue
import threading
import time
import json
import os
import subprocess
import tempfile
import re
from collections import deque
from dotenv import load_dotenv
from google import genai
from ultralytics import YOLO

# ------------------- CONFIG -------------------

MIC_DEVICE = 1
MIC_RATE = 16000

AUDIO_OUT_DEVICE = None  # set later with aplay -l

VOSK_MODEL_PATH = "/opt/vosk_models/vosk-model-small-en-us-0.15"

MODEL_PATH = "yolov8n.pt"

# ------------------- TTS -------------------

def speak(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name

    subprocess.run(
        ["python3", "-m", "piper", "--output-file", path],
        input=text.encode(),
    )

    subprocess.run(["aplay", path])
    os.remove(path)

# ------------------- MAIN CLASS -------------------

class JetsonHelmet:
    def __init__(self):
        load_dotenv()

        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # STT
        self.model = vosk.Model(VOSK_MODEL_PATH)

        # YOLO
        self.yolo = YOLO(MODEL_PATH)

        # queues
        self.audio_q = queue.Queue()
        self.request_q = queue.Queue()

        self.stop_event = threading.Event()
        self.is_busy = threading.Event()

        self.memory = deque(maxlen=10)

    # ------------------- CAMERA -------------------

    def camera_loop(self):
        cap = cv2.VideoCapture(0)

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.yolo(frame, imgsz=320)

            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0])
                        name = self.yolo.names[cls]

                        if name in ["vase", "person", "statue"]:
                            print("[DETECTED]", name)
                            self.request_q.put({"kind": "object", "text": name})

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    # ------------------- STT -------------------

    def audio_callback(self, indata, frames, time_info, status):
        if self.is_busy.is_set():
            return
        self.audio_q.put(bytes(indata))

    def stt_loop(self):
        rec = vosk.KaldiRecognizer(self.model, MIC_RATE)

        with sd.RawInputStream(
            samplerate=MIC_RATE,
            blocksize=8000,
            dtype="int16",
            channels=1,
            callback=self.audio_callback,
            device=MIC_DEVICE,
        ):
            print("Listening...")

            while not self.stop_event.is_set():
                data = self.audio_q.get()

                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get("text", "")

                    if text:
                        print("[HEARD]", text)
                        self.request_q.put({"kind": "user", "text": text})

    # ------------------- GEMINI -------------------

    def ask_gemini(self, prompt):
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        return response.text.strip()

    # ------------------- WORKER -------------------

    def worker(self):
        while not self.stop_event.is_set():
            try:
                req = self.request_q.get(timeout=0.2)
            except queue.Empty:
                continue

            text = req["text"]

            self.is_busy.set()

            try:
                print("[THINKING]")
                response = self.ask_gemini(text)

                print("[AI]", response)
                speak(response)

                self.memory.append((req["kind"], text))
            finally:
                self.is_busy.clear()

    # ------------------- RUN -------------------

    def run(self):
        threads = [
            threading.Thread(target=self.camera_loop),
            threading.Thread(target=self.stt_loop),
            threading.Thread(target=self.worker),
        ]

        for t in threads:
            t.daemon = True
            t.start()

        speak("Hello, I am your museum guide.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_event.set()

# ------------------- MAIN -------------------

if __name__ == "__main__":
    helmet = JetsonHelmet()
    helmet.run()