"""Quality Assurance Agent for live call center assistance.

This implements the call-scoring agent from the project's technical spec. It
scores a transcript against a fixed QA rubric using the local LLM client, so the
result can be used for coaching and post-call review without introducing another
external service dependency.
"""

from __future__ import annotations

import json
import logging

from services.llm import llm_client


logger = logging.getLogger(__name__)

RUBRIC_DIMENSIONS = ["greeting", "active_listening", "compliance", "resolution", "tone", "closing"]


class QualityAssuranceAgent:
    """Score a call transcript against the project QA rubric."""

    async def score_call(self, transcript: str, agent_id: str) -> dict:
        """Return rubric scores, an overall score, coaching flags, and rationale."""

        fallback = _fallback_quality_assessment(agent_id)

        try:
            prompt = self._build_rubric_prompt(transcript)
            raw_output = await llm_client.complete(
                prompt=prompt,
                system_prompt=(
                    "You are a quality assurance analyst for a live call center. "
                    "Score the transcript exactly as requested and return strict JSON only."
                ),
                temperature=0.2,
                max_tokens=512,
            )

            try:
                parsed = json.loads(_strip_json_fences(raw_output))
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response was not a JSON object")

                dimensions = parsed.get("dimensions")
                overall = parsed.get("overall")
                coaching_flags = parsed.get("coaching_flags")
                rationale = parsed.get("rationale")

                if not isinstance(dimensions, dict):
                    raise ValueError("Missing or invalid dimensions object")
                if not isinstance(overall, int):
                    raise ValueError("Missing or invalid overall score")
                if not isinstance(coaching_flags, list) or not all(isinstance(flag, str) for flag in coaching_flags):
                    raise ValueError("Missing or invalid coaching_flags list")
                if not isinstance(rationale, str) or not rationale.strip():
                    raise ValueError("Missing or invalid rationale")

                normalized_dimensions: dict[str, int] = {}
                for dimension in RUBRIC_DIMENSIONS:
                    value = dimensions.get(dimension)
                    if not isinstance(value, int):
                        raise ValueError(f"Missing or invalid rubric score for {dimension}")
                    normalized_dimensions[dimension] = value

                return {
                    "agent_id": agent_id,
                    "dimension_scores": normalized_dimensions,
                    "overall_score": overall,
                    "coaching_flags": coaching_flags,
                    "rationale": rationale.strip(),
                }
            except Exception:
                logger.exception("Malformed QA scoring output for agent_id=%s: %s", agent_id, raw_output)
                return fallback
        except Exception:
            logger.exception("Unexpected failure in quality assurance agent for agent_id=%s", agent_id)
            return fallback

    def _build_rubric_prompt(self, transcript: str) -> str:
        """Build the QA rubric prompt for the provided transcript."""

        dimensions_text = ", ".join(RUBRIC_DIMENSIONS)
        return (
            "Score the following call transcript against each QA rubric dimension on a 0-100 integer scale. "
            f"The rubric dimensions are: {dimensions_text}. Return STRICT JSON with exactly this shape:\n"
            "{\n"
            '  "dimensions": {"greeting": <int 0-100>, "active_listening": <int 0-100>, "compliance": <int 0-100>, '
            '"resolution": <int 0-100>, "tone": <int 0-100>, "closing": <int 0-100>},\n'
            '  "overall": <int 0-100>,\n'
            '  "coaching_flags": [<list of short string flags for any dimension scoring below 60>],\n'
            '  "rationale": "<2-3 sentence explanation of the overall score, referencing specific transcript content>"\n'
            "}\n\n"
            "Scoring rules:\n"
            "- Use whole numbers only for every score.\n"
            "- For each dimension scoring below 60, add one coaching_flag in this exact format: "
            "'{dimension}: {specific action or missing action observed in the transcript}'. "
            "Reference an actual moment, quote, or absence from THIS transcript — never a generic "
            "description like 'needs improvement' or 'low score' that could apply to any call.\n"
            "- overall should reflect the transcript quality across the full rubric, not just one dimension.\n"
            "- The rationale must reference at least one specific moment, phrase, or omission from the "
            "transcript by name — not a general impression of the call's quality.\n\n"
            f"Transcript:\n{transcript}"
        )


def _strip_json_fences(text: str) -> str:
    """Remove optional markdown JSON code fences from an LLM response."""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```")
        cleaned = cleaned.strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[: -3].strip()
    return cleaned


def _fallback_quality_assessment(agent_id: str) -> dict:
    """Return a safe default QA assessment when scoring is unavailable."""

    return {
        "agent_id": agent_id,
        "dimension_scores": {dimension: 0 for dimension in RUBRIC_DIMENSIONS},
        "overall_score": 0,
        "coaching_flags": ["QA scoring unavailable"],
        "rationale": "Automated QA scoring failed for this call; manual review recommended.",
    }


quality_agent = QualityAssuranceAgent()
