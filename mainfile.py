import cv2
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from openai import OpenAI
from google import genai
from langdetect import detect
import tempfile
import os
import json
import time
from pathlib import Path


def load_env_file(env_path=".env"):
    """Load KEY=VALUE lines from .env into os.environ if not already set."""
    path = Path(env_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file()

# -----------------------------
# CONFIG
# -----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GOOGLE_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

SAMPLE_RATE = 16000
RECORD_SECONDS = 4
MEMORY_FILE = "memory.json"

ENABLE_CAMERA_DETECTION = False
ENABLE_PAINTING_INTRO = False
ENABLE_VOICE_INPUT = True
ENABLE_VOICE_OUTPUT = True
ENABLE_MEMORY = False

# -----------------------------
# LOAD / SAVE MEMORY
# -----------------------------
def load_memory():
    if not ENABLE_MEMORY:
        return []
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(history):
    if not ENABLE_MEMORY:
        return
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

history_data = load_memory()

# -----------------------------
# GEMINI SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are an enthusiastic museum tour guide in a robotics and science museum.
RULES:
- Only talk about museum exhibits and paintings
- Be engaging and tell short stories
- Ask follow-up questions sometimes
- Adapt to visitor's language
- Use simple explanations for kids if needed
"""

# -----------------------------
# RECORD AUDIO (NO WEBRTCVAD)
# -----------------------------
def record_audio_fixed():
    print("\n🎤 Listening...")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    write(temp.name, SAMPLE_RATE, audio)
    return temp.name

# -----------------------------
# SPEECH TO TEXT
# -----------------------------
def speech_to_text(audio_file):
    with open(audio_file, "rb") as f:
        result = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f
        )
    return result.text.strip()

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"

# -----------------------------
# PAINTING RECOGNITION
# -----------------------------
painting_database = {
    "red_circle": "The Red Circle",
    "blue_square": "Blue Square Painting",
    "yellow_triangle": "Triangle of Light",
    "green_rectangle": "Emerald Landscape"
}

def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return frame
    return None

def detect_dominant_color(image):
    img = cv2.resize(image, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img.reshape((-1, 3))
    avg_color = np.mean(pixels, axis=0)
    r, g, b = avg_color

    if r > g and r > b:
        return "red"
    elif g > r and g > b:
        return "green"
    elif b > r and b > g:
        return "blue"
    elif r > 200 and g > 200 and b < 100:
        return "yellow"
    return "unknown"

def detect_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return "unknown"

    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
    vertices = len(approx)

    if vertices == 3:
        return "triangle"
    elif vertices == 4:
        return "square"
    elif vertices > 4:
        return "circle"
    return "unknown"

def map_to_painting(color, shape):
    key = f"{color}_{shape}"
    return painting_database.get(key, "Unknown Painting")

# -----------------------------
# GEMINI RESPONSE WITH MEMORY
# -----------------------------
def build_history_text():
    parts = [SYSTEM_PROMPT]
    if not ENABLE_MEMORY:
        return "\n".join(parts)
    for item in history_data[-10:]:
        role = item.get("role", "user")
        text = item.get("text", "")
        parts.append(f"{role.capitalize()}: {text}")
    return "\n".join(parts)

def get_ai_response(user_text, lang):
    prompt = f"""
{build_history_text()}

Visitor language: {lang}
Respond in the SAME language.

Visitor: {user_text}
Guide:
"""

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    answer = response.text if response.text else "Sorry, I could not answer that."
    if ENABLE_MEMORY:
        history_data.append({"role": "user", "text": user_text})
        history_data.append({"role": "model", "text": answer})
        save_memory(history_data)
    return answer

# -----------------------------
# OPENAI TTS
# -----------------------------
def speak(text):
    if not ENABLE_VOICE_OUTPUT:
        return

    temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text
    ) as response:
        response.stream_to_file(temp.name)

    os.system(f'aplay "{temp.name}"')

# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    print("🤖 Museum AI Guide Ready!\n")

    while True:
        try:
            if ENABLE_CAMERA_DETECTION:
                # Step 1: Capture painting
                image = capture_image()
                if image is None:
                    print("Failed to capture image, retrying...")
                    time.sleep(1)
                    continue

                color = detect_dominant_color(image)
                shape = detect_shape(image)
                painting_title = map_to_painting(color, shape)
                print(f"Detected Painting: {painting_title}")
            else:
                painting_title = "the current exhibit"

            if ENABLE_PAINTING_INTRO:
                # Step 2: Explain painting
                explanation = get_ai_response(
                    f"Visitor is looking at {painting_title}. Explain briefly and ask if they want to know more.",
                    "en"
                )
                print("🤖 Guide:", explanation)
                speak(explanation)

            # Step 3: Listen to visitor
            if ENABLE_VOICE_INPUT:
                audio_file = record_audio_fixed()
                visitor_text = speech_to_text(audio_file)
            else:
                visitor_text = input("🧑 Visitor (type your message): ").strip()

            print("🧑 Visitor:", visitor_text)

            if visitor_text.lower() in ["exit", "quit", "stop"]:
                speak("Goodbye! Enjoy the museum.")
                break

            lang = detect_language(visitor_text)

            # Step 4: Continue conversation
            reply = get_ai_response(visitor_text, lang)
            print("🤖 Guide:", reply)
            speak(reply)

        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print("Error:", e)
            try:
                speak("Sorry, something went wrong.")
            except Exception:
                pass

if __name__ == "__main__":
    main()
