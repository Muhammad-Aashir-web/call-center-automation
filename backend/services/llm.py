"""Provider-agnostic LLM client with Groq, Gemini, and Ollama fallback support."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from groq import Groq
from google import genai
from google.genai import types

from config import settings


logger = logging.getLogger(__name__)


class AllProvidersFailedError(RuntimeError):
    """Raised when every configured LLM provider fails to produce a response."""


class LLMClient:
    """Provider-agnostic LLM client with ordered fallback across Groq, Gemini, and Ollama.

    The client reads its provider order and credentials from `config.settings` and
    attempts each configured provider in sequence. If a provider raises any exception,
    the failure is logged and the next provider is tried until one succeeds or all fail.
    """

    def __init__(self) -> None:
        provider_order = getattr(settings, "LLM_PROVIDER_ORDER", ["groq", "gemini", "ollama"])
        if isinstance(provider_order, str):
            provider_order = [provider_order]

        self.provider_order = [provider.strip().lower() for provider in provider_order if provider]
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.gemini_api_key = settings.GOOGLE_API_KEY
        self.ollama_base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.gemini_client = genai.Client(api_key=self.gemini_api_key) if self.gemini_api_key else None

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a completion using the configured provider fallback chain.

        The client tries providers in order and falls through on any exception.
        Groq is used first, Gemini second, and Ollama third by default. If every
        provider fails, an `AllProvidersFailedError` is raised with the providers tried.
        """

        attempts: list[str] = []

        for provider in self.provider_order:
            attempts.append(provider)
            try:
                if provider == "groq":
                    return await self._complete_with_groq(prompt, system_prompt, temperature, max_tokens)
                if provider == "gemini":
                    return await self._complete_with_gemini(prompt, system_prompt, temperature, max_tokens)
                if provider == "ollama":
                    return await self._complete_with_ollama(prompt, system_prompt, temperature, max_tokens)

                logger.error("Unknown LLM provider configured: %s", provider)
            except Exception:
                logger.exception("LLM provider %s failed", provider)

        raise AllProvidersFailedError(
            f"All LLM providers failed. Tried: {', '.join(attempts) if attempts else 'none'}"
        )

    async def _complete_with_groq(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        def _call() -> str:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content or ""

        return await asyncio.to_thread(_call)

    async def _complete_with_gemini(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        if self.gemini_client is None:
            raise RuntimeError("Gemini API key is not configured")

        response = await self.gemini_client.aio.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text or ""

    async def _complete_with_ollama(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload: dict[str, Any] = {
            "model": "llama3.1",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(f"{self.ollama_base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "") or ""


llm_client = LLMClient()
