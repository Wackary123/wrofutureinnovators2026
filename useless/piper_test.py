import os
import subprocess
from pathlib import Path

# Folder where your piper voice models are stored
VOICE_FOLDER = Path.home() / ".local" / "share" / "piper-voices"


def find_voices():
    """
    Find all .onnx voice models in the Piper voices folder.
    """
    if not VOICE_FOLDER.exists():
        return []

    return sorted(VOICE_FOLDER.rglob("*.onnx"))


def speak_with_piper(voice_path, text):
    """
    Speak text using Piper CLI.
    """
    try:
        command = [
            "python3",
            "-m",
            "piper",
            "-m",
            str(voice_path),
            "--output-raw",
        ]

        p1 = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        p2 = subprocess.Popen(
            ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
            stdin=p1.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        p1.stdout.close()
        p1.communicate(input=text)
        p2.wait()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Make sure Piper and aplay are installed.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def main():
    print("=== Piper TTS Voice Tester ===")

    voices = find_voices()

    if not voices:
        print(f"\nNo voice models found in: {VOICE_FOLDER}")
        print("Make sure your Piper voices are downloaded there.")
        return

    while True:
        print("\nAvailable voices:")
        for i, voice in enumerate(voices, start=1):
            print(f"{i}. {voice.name}")

        choice = input("\nChoose a voice number (or type 'q' to quit): ").strip()

        if choice.lower() == "q":
            print("Goodbye.")
            break

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        choice_num = int(choice)
        if choice_num < 1 or choice_num > len(voices):
            print("That number is out of range.")
            continue

        selected_voice = voices[choice_num - 1]
        print(f"\nSelected voice: {selected_voice.name}")

        text = input("What should it say? (or type 'q' to quit): ").strip()

        if text.lower() == "q":
            print("Goodbye.")
            break

        if not text:
            print("You entered empty text.")
            continue

        print(f"\nSpeaking with {selected_voice.name}...")
        speak_with_piper(selected_voice, text)


if __name__ == "__main__":
    main()