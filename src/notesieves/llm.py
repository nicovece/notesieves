import anthropic
from dotenv import load_dotenv

load_dotenv()


class LLMService:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 1024):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response from Claude."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
        )
        return response.content[0].text
