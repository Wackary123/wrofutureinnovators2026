import os
import time
import threading
import queue
import subprocess
import tempfile
import cv2
import importlib
from dotenv import load_dotenv
from google import genai
from fusion_hat.stt import Vosk as STT  # type: ignore


HISTORY_WINDOW = 12
MIN_HISTORY_MATCHES = 4
POSITION_STABILITY_TOLERANCE = 37
SIZE_STABILITY_TOLERANCE = 12
SIZE_CHANGE_DELTA_TOLERANCE = 14
SIZE_CHANGE_FREQUENCY_THRESHOLD = 0.75
KEEP_UNCONFIRMED_DETECTIONS = False  # no touchies
MATCH_SCORE_TOLERANCE = 70
NESTED_BOX_MARGIN = 3  # no touchies
NESTED_REQUIRE_SAME_COLOR = True  # no touchies
MIN_CONTOUR_AREA = 450  # no touchies
CAMERA_SIZE = (640, 480)
CAMERA_FORMAT = "BGR888"
CAMERA_WARMUP_SECONDS = 0.2


RED_LOWER_1 = (0, 120, 80)
RED_UPPER_1 = (10, 255, 255)
RED_LOWER_2 = (170, 120, 80)
RED_UPPER_2 = (179, 255, 255)
BLUE_LOWER = (95, 100, 70)
BLUE_UPPER = (130, 255, 255)
GREEN_LOWER = (40, 40, 40)
GREEN_UPPER = (70, 255, 255)


def create_picamera2_instance():
    """Loads Picamera2 only at runtime so this file can still be edited off-device."""
    picamera2_module = importlib.import_module("picamera2")
    return picamera2_module.Picamera2()


def detect_color(frame):
    """
    Detects color patches in the image.
    Returns list of tuples: (color, position, size)
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = {
        "red": [(RED_LOWER_1, RED_UPPER_1), (RED_LOWER_2, RED_UPPER_2)],
        "blue": [(BLUE_LOWER, BLUE_UPPER)],
        "green": [(GREEN_LOWER, GREEN_UPPER)]
    }

    detected_colors = []
    for color_name, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            range_mask = cv2.inRange(hsv, lower, upper)
            mask = range_mask if mask is None else cv2.bitwise_or(mask, range_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > MIN_CONTOUR_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                detected_colors.append((color_name, (x + w // 2, y + h // 2), (w, h)))

    return detected_colors


def filter_detections(past_detections):
    """
    Filters detections to keep only stable/static ones.
    """
    if not past_detections:
        return []

    history = past_detections[-HISTORY_WINDOW:]
    latest_detections = history[-1]

    filtered = []
    for color_name, (x, y), (w, h) in latest_detections:
        matched_history = []

        for frame_detections in history:
            best_match = None
            best_score = None

            for c_name, (fx, fy), (fw, fh) in frame_detections:
                if c_name != color_name:
                    continue

                score = abs(fx - x) + abs(fy - y) + abs(fw - w) + abs(fh - h)
                if best_score is None or score < best_score:
                    best_score = score
                    best_match = (fx, fy, fw, fh)

            if best_match is not None and best_score <= MATCH_SCORE_TOLERANCE:
                matched_history.append(best_match)

        if len(matched_history) < MIN_HISTORY_MATCHES:
            if KEEP_UNCONFIRMED_DETECTIONS:
                filtered.append((color_name, (x, y), (w, h)))
            continue

        xs = [item[0] for item in matched_history]
        ys = [item[1] for item in matched_history]
        ws = [item[2] for item in matched_history]
        hs = [item[3] for item in matched_history]

        is_position_static = (
            (max(xs) - min(xs) <= POSITION_STABILITY_TOLERANCE) and
            (max(ys) - min(ys) <= POSITION_STABILITY_TOLERANCE)
        )
        is_size_static = (
            (max(ws) - min(ws) <= SIZE_STABILITY_TOLERANCE) and
            (max(hs) - min(hs) <= SIZE_STABILITY_TOLERANCE)
        )

        size_change_count = 0
        for i in range(1, len(matched_history)):
            prev_w, prev_h = matched_history[i - 1][2], matched_history[i - 1][3]
            curr_w, curr_h = matched_history[i][2], matched_history[i][3]
            if abs(curr_w - prev_w) > SIZE_CHANGE_DELTA_TOLERANCE or abs(curr_h - prev_h) > SIZE_CHANGE_DELTA_TOLERANCE:
                size_change_count += 1

        change_frequency = size_change_count / max(1, len(matched_history) - 1)
        if change_frequency >= SIZE_CHANGE_FREQUENCY_THRESHOLD:
            continue

        if not (is_position_static and is_size_static):
            continue

        filtered.append((color_name, (x, y), (w, h)))

    return remove_nested_detections(filtered)


def _to_corners(position, size):
    x, y = position
    w, h = size
    left = x - w // 2
    top = y - h // 2
    right = x + w // 2
    bottom = y + h // 2
    return left, top, right, bottom


def _is_inside(inner_detection, outer_detection, margin):
    _, inner_pos, inner_size = inner_detection
    _, outer_pos, outer_size = outer_detection
    il, it, ir, ib = _to_corners(inner_pos, inner_size)
    ol, ot, or_, ob = _to_corners(outer_pos, outer_size)
    return il >= ol + margin and it >= ot + margin and ir <= or_ - margin and ib <= ob - margin


def remove_nested_detections(detections):
    keep = [True] * len(detections)

    for i, det_i in enumerate(detections):
        color_i, _, size_i = det_i
        area_i = size_i[0] * size_i[1]

        for j, det_j in enumerate(detections):
            if i == j:
                continue

            color_j, _, size_j = det_j
            if NESTED_REQUIRE_SAME_COLOR and color_i != color_j:
                continue

            area_j = size_j[0] * size_j[1]
            if _is_inside(det_i, det_j, NESTED_BOX_MARGIN):
                if area_i < area_j or (area_i == area_j and i > j):
                    keep[i] = False
                    break

    return [det for idx, det in enumerate(detections) if keep[idx]]


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

        # Silence timeout in seconds
        self.silence_timeout = 3
        self.followup_timeout = 3

        # Camera-to-museum behavior
        self.color_hold_seconds = 1.5
        self.color_cooldown_seconds = 8.0
        self.last_seen_color = None
        self.color_first_seen_time = None
        self.last_color_trigger_time = {
            "red": 0.0,
            "blue": 0.0,
            "green": 0.0,
        }

        self.color_object_prompts = {
            "red": "The visitor seems to be looking at the Mona Lisa. Give a short, natural museum-guide style explanation about the Mona Lisa: what it is, why it matters, and one interesting detail.",
            "blue": "The visitor seems to be looking at diamonds or a precious blue gem display. Give a short, natural museum-guide style explanation about diamonds or blue precious gems: what they are, why they matter, and one interesting detail.",
            "green": "The visitor seems to be looking at an emerald or green artifact display. Give a short, natural museum-guide style explanation about emeralds or green historical artifacts: what they are, why they matter, and one interesting detail.",
        }

        self.speaking_lock = threading.Lock()
        self.is_speaking = threading.Event()
        self.stop_camera_event = threading.Event()

        # AI system prompt
        self.system_prompt = """
You are a museum cultural guide helmet with a secondary safety role.

Your personality and speaking style:
- Speak like a real human guide, not like a robot, textbook, or narrator
- Sound warm, natural, conversational, and socially aware
- Do not give long monologues unless the user clearly asks for a detailed explanation
- Prefer short back-and-forth conversation over long speeches
- Usually answer in 1 to 3 short sentences
- After answering, sometimes invite the visitor forward with one small follow-up, not every time
- Vary your wording so you do not sound repetitive or scripted

Core behavior:
- Give clear, short, natural answers
- Keep answers easy to understand
- Focus on culture, history, museums, traditions, artifacts, exhibits, symbolism, and historical context
- Adapt to the user's tone and level of knowledge
- If the user sounds confused, simplify immediately
- If the user asks a broad question, answer briefly first, then offer more detail
- If the user asks about an object, explain what it is, why it matters, and one interesting detail when possible
- If the user asks about safety or security, answer briefly, respectfully, and calmly

Conversation rules:
- Never give more than 3 short sentences unless the visitor explicitly asks for more detail
- Do not dominate the conversation
- Do not dump too much information at once
- Do not give a lecture unless asked
- Break information into small, digestible pieces
- Prefer interaction over performance
- If a topic is uncertain, say so simply instead of guessing
- Ask at most one short follow-up question unless the user wants a deeper discussion

Privacy and safety:
- Respect privacy
- Do not mention storing, saving, tracking, or remembering personal data
- Do not sound surveillance-like
- Keep the safety role secondary unless the user brings it up

Always sound like a helpful museum guide beside the visitor, not an audio encyclopedia.
"""

    def say(self, text):
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

    def _stt_worker(self, result_queue, stop_event):
        try:
            for result in self.stt.listen(stream=True):
                if stop_event.is_set():
                    break
                result_queue.put(result)
                if result.get("done"):
                    break
        except Exception as e:
            result_queue.put({"error": str(e)})

    def listen_once(self, silence_timeout=3):
        print("🎤 Listening...")

        result_queue = queue.Queue()
        stop_event = threading.Event()
        worker = threading.Thread(
            target=self._stt_worker,
            args=(result_queue, stop_event),
            daemon=True
        )
        worker.start()

        final_text = ""
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

    def ask_ai(self, user_text):
        prompt = f"""
{self.system_prompt}

Visitor says: {user_text}

Reply as the museum guide helmet in a natural, human, conversational way.
Keep it short unless the visitor asks for more.
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

    def ask_ai_from_camera_color(self, color_name):
        color_prompt = self.color_object_prompts.get(
            color_name,
            f"The visitor seems to be looking at something {color_name}. Give a short, natural museum-guide explanation about an object associated with that color."
        )

        prompt = f"""
{self.system_prompt}

Camera event:
The visitor has been looking steadily at a {color_name} object for at least {self.color_hold_seconds} seconds.

Task:
{color_prompt}

Reply as the museum guide helmet in a natural, human, conversational way.
Keep it short, around 1 to 3 short sentences.
"""
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        text = getattr(response, "text", None)
        print(f"\n[Gemini camera text - {color_name}]:", text)

        if text:
            return text.strip()

        return None

    def maybe_trigger_color_explanation(self, relevant_detections):
        current_time = time.time()

        detected_colors = [det[0] for det in relevant_detections]
        dominant_color = detected_colors[0] if detected_colors else None

        if dominant_color is None:
            self.last_seen_color = None
            self.color_first_seen_time = None
            return

        if dominant_color != self.last_seen_color:
            self.last_seen_color = dominant_color
            self.color_first_seen_time = current_time
            return

        if self.color_first_seen_time is None:
            self.color_first_seen_time = current_time
            return

        held_long_enough = (current_time - self.color_first_seen_time) >= self.color_hold_seconds
        off_cooldown = (current_time - self.last_color_trigger_time[dominant_color]) >= self.color_cooldown_seconds

        if held_long_enough and off_cooldown and not self.is_speaking.is_set():
            self.last_color_trigger_time[dominant_color] = current_time

            answer = self.ask_ai_from_camera_color(dominant_color)
            if answer:
                self.say(answer)

            # reset so it must be held again before retriggering
            self.color_first_seen_time = current_time

    def camera_worker(self):
        try:
            picam2 = create_picamera2_instance()
        except ModuleNotFoundError:
            print("Picamera2 is required. Install python3-picamera2 on Raspberry Pi.")
            return
        except Exception as e:
            print("Camera init error:", e)
            return

        try:
            camera_config = picam2.create_preview_configuration(
                main={"size": CAMERA_SIZE, "format": CAMERA_FORMAT}
            )
            picam2.configure(camera_config)
            picam2.start()
            time.sleep(CAMERA_WARMUP_SECONDS)

            detection_results = []

            while not self.stop_camera_event.is_set():
                frame = picam2.capture_array()

                detected_colors = detect_color(frame)
                detection_results.append(detected_colors)
                if len(detection_results) > HISTORY_WINDOW:
                    detection_results.pop(0)

                relevant_detections = filter_detections(detection_results)

                self.maybe_trigger_color_explanation(relevant_detections)

                # Optional preview window:
                # uncomment these lines if you want to see the camera feed
                # draw_colors = {
                #     "red": (0, 0, 255),
                #     "blue": (255, 0, 0),
                #     "green": (0, 255, 0),
                # }
                # preview = frame.copy()
                # for color_name, position, size in relevant_detections:
                #     x, y = position
                #     w, h = size
                #     draw_color = draw_colors.get(color_name, (255, 255, 255))
                #     cv2.rectangle(preview, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), draw_color, 2)
                #     cv2.putText(preview, color_name, (x - w // 2, y - h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
                # cv2.imshow("Color Detection", preview)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     self.stop_camera_event.set()
                #     break

                time.sleep(0.05)

        except Exception as e:
            print("Camera worker error:", e)
        finally:
            try:
                picam2.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()

    def handle_command(self, text):
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

    def start(self):
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