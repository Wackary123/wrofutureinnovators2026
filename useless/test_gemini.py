from google import genai
from dotenv import load_dotenv
import os

class HelmetTester:
    def __init__(self):
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file")

        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"

        self.system_prompt = """
You are a helpful museum cultural guide helmet.
Rules:
- Give short, clear answers
- Focus on museums, culture, history, artifacts, and traditions
- Be natural and easy to understand
- If asked general questions, still answer helpfully
"""

    def ask_ai(self, user_text):
        prompt = f"""
{self.system_prompt}

User: {user_text}
Assistant:
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip() if response.text else "No response."

    def start(self):
        print("Museum Helmet Gemini Test")
        print("Type 'exit' to quit.\n")

        while True:
            user_text = input("You: ").strip()

            if user_text.lower() in ["exit", "quit", "stop"]:
                print("Helmet: Goodbye.")
                break

            try:
                answer = self.ask_ai(user_text)
                print(f"Helmet: {answer}\n")
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    app = HelmetTester()
    app.start()