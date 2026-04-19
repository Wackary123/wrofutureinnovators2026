import os
import time
import threading
import queue
import subprocess
import tempfile

import cv2
from dotenv import load_dotenv
from google import genai
from fusion_hat.stt import Vosk as STT  # type: ignore
from picamera2 import Picamera2  # type: ignore
from ultralytics import YOLOE  # type: ignore


class MuseumHelmet:
    def __init__(self):
        load_dotenv()

        # AI setup
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.gemini_api_key)

        # Speech-to-text / text-to-speech
        self.stt = STT(language="en-us")

        # Piper settings
        self.piper_model = "en_US-ryan-medium"
        self.piper_data_dir = os.path.expanduser("~/piper_voices")

        # Wake words
        self.wake_words = ["atlas", "helmet", "guide", "assistant"]

        # Silence timing
        self.silence_timeout = 3
        self.followup_timeout = 3

        # Camera / YOLOE settings
        self.camera_size = (800, 800)
        self.model_path = "yoloe-11s-seg.pt"
        self.prompt_names = [
            "dog", "phone", "clock", "hoodie", "computer", "box", "plant",
            "tape", "mona lisa", "vase", "hair", "person", "table",
            "light", "fruit", "chair", "couch"
        ]
        self.confidence_threshold = 0.20
        self.object_hold_seconds = 1.5
        self.object_cooldown_seconds = 8.0
        self.model_imgsz = 320  # lower for more FPS; raise for more detail

        # Detection tracking
        self.last_seen_object = None
        self.object_first_seen_time = None
        self.last_object_trigger_time = {name: 0.0 for name in self.prompt_names}
        self.last_terminal_objects = None

        # Thread / state control
        self.speaking_lock = threading.Lock()
        self.is_speaking = threading.Event()
        self.stop_camera_event = threading.Event()

        # AI system prompt
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
Adapt to the visitor’s level:
simplify for beginners or children
add depth for advanced questions
If unsure, acknowledge uncertainty calmly while still giving helpful context
Avoid vague or generic explanations—always give a specific reason
Interaction Rules
Do not overwhelm the visitor with too much information
Avoid long monologues unless explicitly requested
Break information into small, digestible pieces
Match the visitor’s tone (curious, excited, confused)
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

    def say(self, text: str) -> None:
        with self.speaking_lock:
            self.is_speaking.set()
            print(f"🤖 {text}")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                wav_path = tmp_wav.name

            try:
                subprocess.run(
                    [
                        "python3",
                        "-m",
                        "piper",
                        "--model",
                        self.piper_model,
                        "--data-dir",
                        self.piper_data_dir,
                        "--output-file",
                        wav_path,
                    ],
                    input=text,
                    text=True,
                    check=True,
                )
                subprocess.run(["aplay", wav_path], check=True)

            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                self.is_speaking.clear()

    def _stt_worker(self, result_queue: queue.Queue, stop_event: threading.Event) -> None:
        try:
            for result in self.stt.listen(stream=True):
                if stop_event.is_set():
                    break
                result_queue.put(result)
                if result.get("done"):
                    break
        except Exception as e:
            result_queue.put({"error": str(e)})

    def listen_once(self, silence_timeout: int = 3) -> str:
        print("Listening...")

        result_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._stt_worker,
            args=(result_queue, stop_event),
            daemon=True
        )
        worker.start()

        last_partial = ""
        heard_anything = False
        last_speech_time = None

        while True:
            try:
                result = result_queue.get(timeout=0.1)
            except queue.Empty:
                if heard_anything and last_speech_time is not None:
                    if time.time() - last_speech_time >= silence_timeout:
                        stop_event.set()
                        print(f"\r\x1b[Kfinal (timeout): {last_partial}")
                        return last_partial.strip().lower()
                continue

            if "error" in result:
                raise RuntimeError(result["error"])

            if result.get("done"):
                final_text = result.get("final", "").strip().lower()
                stop_event.set()

                if final_text:
                    print(f"\r\x1b[Kfinal: {final_text}")
                    return final_text

                if last_partial:
                    print(f"\r\x1b[Kfinal (fallback): {last_partial}")
                    return last_partial.strip().lower()

                return ""

            partial = result.get("partial", "").strip().lower()
            if partial:
                heard_anything = True
                last_speech_time = time.time()

                if partial != last_partial:
                    last_partial = partial
                    print(f"\r\x1b[Kpartial: {partial}", end="", flush=True)

    def ask_ai(self, user_text: str) -> str:
        prompt = f"""
{self.system_prompt}

Visitor says: {user_text}

Reply as a museum guide speaking naturally to a visitor beside you.
Sound warm, human, and conversational—not like a robot or textbook.

Keep your response short (1–3 sentences) unless more detail is clearly requested.

When explaining something:
- briefly say what it is
- explain why it matters
- add one interesting or engaging detail when possible

Adapt to the visitor’s level and tone.
If appropriate, ask one short follow-up question to keep the interaction engaging.
"""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = getattr(response, "text", None)
        print("\n[Gemini text]:", text)

        if text:
            return text.strip()

        return "Sorry, I could not answer that."

    def ask_ai_about_detected_object(self, object_name: str) -> str | None:
        prompt = f"""
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
- Do NOT mention detection or observation (avoid phrases like “I see you looking at…”)
- If unsure, use soft uncertainty (e.g., “This appears to be…”)
- Keep it natural, warm, and conversational
- Keep it short (1–3 sentences)
- Vary phrasing to avoid repetition
- Adjust tone slightly based on how interesting the object is
- When appropriate, include one engaging detail or a light follow-up question
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = getattr(response, "text", None)
        print(f"\n[Gemini camera text - {object_name}]:", text)

        if text:
            return text.strip()
        return None

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

        # Choose the strongest detection
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

        if held_long_enough and off_cooldown and not self.is_speaking.is_set():
            print(f"[Camera trigger]: {dominant_name} held for {self.object_hold_seconds} seconds")
            self.last_object_trigger_time[dominant_name] = current_time

            answer = self.ask_ai_about_detected_object(dominant_name)
            if answer:
                self.say(answer)

            self.object_first_seen_time = current_time

    def camera_worker(self) -> None:
        try:
            picam2 = Picamera2()
            picam2.preview_configuration.main.size = self.camera_size
            picam2.preview_configuration.main.format = "RGB888"
            picam2.preview_configuration.align()
            picam2.configure("preview")
            picam2.start()
            time.sleep(0.2)

            # Show raw camera preview immediately
            preview_start = time.time()
            while time.time() - preview_start < 1.0 and not self.stop_camera_event.is_set():
                frame = picam2.capture_array()
                cv2.imshow("YOLOE Museum Helmet", frame)
                if cv2.waitKey(1) == ord("q"):
                    self.stop_camera_event.set()
                    return

            print("[Camera] Preview opened, loading YOLOE model...")

            model = YOLOE(self.model_path)
            model.set_classes(self.prompt_names)

            print("[Camera] YOLOE model loaded.")

            while not self.stop_camera_event.is_set():
                frame = picam2.capture_array()

                results = model.predict(frame, imgsz=self.model_imgsz)
                result = results[0]

                annotated_frame = result.plot(boxes=True, masks=False)

                # Build a clean detection list above threshold
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

                # FPS overlay
                inference_time = result.speed["inference"]
                fps = 1000 / inference_time if inference_time > 0 else 0.0
                text = f"FPS: {fps:.1f}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, 1, 2)[0]
                text_x = annotated_frame.shape[1] - text_size[0] - 10
                text_y = text_size[1] + 10
                cv2.putText(
                    annotated_frame,
                    text,
                    (text_x, text_y),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow("YOLOE Museum Helmet", annotated_frame)

                if cv2.waitKey(1) == ord("q"):
                    self.stop_camera_event.set()
                    break

        except Exception as e:
            print("Camera worker error:", e)
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def handle_command(self, text: str) -> bool:
        print(f"\n[Heard]: {text}")

        if text in ["goodbye", "good bye", "exit", "quit", "stop program"]:
            self.say("Goodbye.")
            return False

        if text in ["stop", "be quiet", "silence"]:
            self.say("Okay.")
            return True

        matched_wake_word = None
        for wake_word in self.wake_words:
            if wake_word in text:
                matched_wake_word = wake_word
                break

        if matched_wake_word:
            cleaned = text.replace(matched_wake_word, "").strip()

            if cleaned == "":
                self.say("Yes? What would you like to know?")
                question = self.listen_once(silence_timeout=self.silence_timeout)

                if question:
                    answer = self.ask_ai(question)
                    self.say(answer)

                    followup = self.listen_once(silence_timeout=self.followup_timeout)
                    while followup:
                        answer = self.ask_ai(followup)
                        self.say(answer)
                        followup = self.listen_once(silence_timeout=self.followup_timeout)
                else:
                    self.say("I did not hear a question.")
            else:
                answer = self.ask_ai(cleaned)
                self.say(answer)

                followup = self.listen_once(silence_timeout=self.followup_timeout)
                while followup:
                    answer = self.ask_ai(followup)
                    self.say(answer)
                    followup = self.listen_once(silence_timeout=self.followup_timeout)

            return True

        return True

    def start(self) -> None:
        camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        camera_thread.start()

        self.say("Museum helmet ready.")

        running = True
        while running:
            try:
                text = self.listen_once(silence_timeout=self.silence_timeout)

                if not text:
                    continue

                running = self.handle_command(text)

            except KeyboardInterrupt:
                self.say("Shutting down.")
                break
            except Exception as e:
                print("\nError:", e)
                self.say("Sorry, something went wrong.")

        self.stop_camera_event.set()


if __name__ == "__main__":
    helmet = MuseumHelmet()
    helmet.start()