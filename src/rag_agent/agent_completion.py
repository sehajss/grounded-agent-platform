"""Groq chat completion adapter for the developer agent."""

from __future__ import annotations

from groq import AsyncGroq


class GroqAgentCompletion:
    """Return plain-text completions using Groq's async client."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A Groq API key is required.")
        if not model.strip():
            raise ValueError("A Groq model name is required.")

        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def __call__(self, prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty completion.")
        return content

    async def close(self) -> None:
        await self.client.close()