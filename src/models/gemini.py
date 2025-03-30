from google import genai

class Gemini:
    def __init__(self, model:str, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def run(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text
