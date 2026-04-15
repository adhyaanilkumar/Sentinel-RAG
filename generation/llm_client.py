"""LLM API wrapper supporting OpenAI and Gemini."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    model: str = ""


class LLMClient:
    """Unified LLM client for OpenAI and Gemini."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = provider
        self.model = model
        self._client = None

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024, temperature: float = 0.1) -> LLMResponse:
        start = time.time()

        if self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, max_tokens, temperature, start)
        elif self.provider == "gemini":
            return self._generate_gemini(prompt, system_prompt, max_tokens, temperature, start)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_openai(self, prompt, system_prompt, max_tokens, temperature, start) -> LLMResponse:
        client = self._get_openai_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.time() - start
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            latency_seconds=latency,
            model=self.model,
        )

    def _generate_gemini(self, prompt, system_prompt, max_tokens, temperature, start) -> LLMResponse:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(self.model)
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        resp = model.generate_content(full_prompt)
        latency = time.time() - start
        return LLMResponse(
            text=resp.text or "",
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_seconds=latency,
            model=self.model,
        )
