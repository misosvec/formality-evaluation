import google.generativeai as genai

class Gemini:
    def __init__(self, model:str, api_key: str) -> None:
        self.client = genai.GenerativeModel(model_name=model, api_key=api_key)

    def run(self, prompt: str, system_prompt: str = "You are a helpful AI assistant.") -> str:
        response = self.client.generate_content(
            contents=[
                {"role": "system", "parts": [{"text": system_prompt}]},  # System message
                {"role": "user", "parts": [{"text": prompt}]},  # User message
            ]
        )
        return response.text  # Extract response text
