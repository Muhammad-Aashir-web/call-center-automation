"""Response Suggestion Agent for live call center assistance.

This implements Agent 4 (Response Suggestion) from the project's technical spec.
It generates prompt-grounded recommendations using Agent 3's retrieved knowledge-
base content rather than general knowledge, so the live agent gets a concise,
relevant next-best response tied to the current call context.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from services.llm import llm_client


logger = logging.getLogger(__name__)


class ResponseSuggestionAgent:
    """Generate a grounded next-best response for a live call center agent."""

    async def suggest(
        self,
        transcript: str,
        intent: str | None,
        kb_suggestions: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Return a safe, prompt-grounded response suggestion and rationale."""

        fallback = _fallback_suggestion()

        try:
            system_prompt = (
                "You are assisting a live call center agent. Base your answer ONLY on the "
                "provided knowledge-base content and the call transcript. Do not use general "
                "knowledge or invent policy details. Keep the tone professional and concise. "
                "Respond in strict JSON with exactly two keys: suggested_response and rationale. "
                "The rationale should briefly explain why this response was chosen and cite "
                "which KB article(s) informed it."
            )

            kb_entries: list[str] = []
            for index, suggestion in enumerate(kb_suggestions, start=1):
                title = suggestion.get("title")
                document = suggestion.get("document")
                if not title or not document:
                    continue
                kb_entries.append(f"{index}. Title: {title}\n{document}")

            if kb_entries:
                kb_block = "\n\n".join(kb_entries)
                user_prompt = (
                    f"Detected intent: {intent or 'unknown'}\n\n"
                    f"Call transcript so far:\n{transcript}\n\n"
                    f"Retrieved KB content:\n{kb_block}\n\n"
                    "Write a concise next-best response that the live agent can say to the customer. "
                    "Base the response only on the KB content above and the transcript."
                )
            else:
                user_prompt = (
                    f"Detected intent: {intent or 'unknown'}\n\n"
                    f"Call transcript so far:\n{transcript}\n\n"
                    "No KB content was found for this turn. Give a general professional acknowledgment "
                    "response rather than inventing specific policy details."
                )

            raw_output = await llm_client.complete(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=512,
            )

            try:
                parsed = json.loads(_strip_json_fences(raw_output))
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response was not a JSON object")

                suggested_response = parsed.get("suggested_response")
                rationale = parsed.get("rationale")
                if not isinstance(suggested_response, str) or not suggested_response.strip():
                    raise ValueError("Missing or empty suggested_response")
                if not isinstance(rationale, str) or not rationale.strip():
                    raise ValueError("Missing or empty rationale")

                return {
                    "suggested_response": suggested_response.strip(),
                    "rationale": rationale.strip(),
                }
            except Exception:
                logger.warning("Malformed response suggestion output: %s", raw_output)
                return fallback
        except Exception:
            logger.exception("Unexpected failure in response suggestion agent")
            return fallback


def _strip_json_fences(text: str) -> str:
    """Remove optional markdown JSON code fences from an LLM response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()
    return cleaned


def _fallback_suggestion() -> dict[str, str]:
    """Return a safe default suggestion when the LLM output is unusable."""

    return {
        "suggested_response": "I'm looking into this for you now.",
        "rationale": "Automated suggestion unavailable; showing a safe default.",
    }