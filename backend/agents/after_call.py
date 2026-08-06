"""After-Call Work Agent for live call center assistance.

This implements the post-call summarization agent from the project's technical
spec. It produces a structured call summary, follow-up tasks, and review flags
using the local LLM client so the result stays fast and self-contained.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, Field

from services.llm_utils import strip_json_fences
from services.llm import llm_client


logger = logging.getLogger(__name__)


class FollowUpTask(BaseModel):
    description: str
    priority: str
    due_by: str


class AfterCallSummary(BaseModel):
    call_summary: str = Field(default="")
    resolution_status: str = Field(default="unknown")
    commitments: list[str] = Field(default_factory=list)
    compliance_notes: list[str] = Field(default_factory=list)
    follow_up_tasks: list[FollowUpTask] = Field(default_factory=list)
    requires_human_review: bool = True
    review_reason: str | None = "Automated summary generation failed"


class AfterCallWorkAgent:
    """Generate a structured after-call summary for a finished call."""

    async def generate_summary(
        self,
        transcript: str,
        qa_score: dict | None,
        agent_id: str,
    ) -> AfterCallSummary:
        """Return a structured after-call summary with safe fallback behavior."""

        fallback = _fallback_after_call_summary()

        try:
            prompt = self._build_summary_prompt(transcript, qa_score, agent_id)
            raw_output = await llm_client.complete(
                prompt=prompt,
                system_prompt=(
                    "You are an after-call work analyst for a live call center. "
                    "Summarize the transcript accurately and return strict JSON only. "
                    "Do not invent facts. Be specific, factual, and tied to the provided transcript."
                ),
                temperature=0.2,
                max_tokens=768,
            )

            try:
                parsed = json.loads(strip_json_fences(raw_output))
                if not isinstance(parsed, dict):
                    raise ValueError("LLM response was not a JSON object")

                summary = parsed.get("call_summary")
                resolution_status = parsed.get("resolution_status")
                commitments = parsed.get("commitments")
                compliance_notes = parsed.get("compliance_notes")
                follow_up_tasks = parsed.get("follow_up_tasks")
                requires_human_review = parsed.get("requires_human_review")
                review_reason = parsed.get("review_reason")

                if not isinstance(summary, str):
                    raise ValueError("Missing or invalid call_summary")
                if resolution_status not in {"resolved", "unresolved", "escalated", "unknown"}:
                    raise ValueError("Missing or invalid resolution_status")
                if not isinstance(commitments, list) or not all(isinstance(item, str) for item in commitments):
                    raise ValueError("Missing or invalid commitments list")
                if not isinstance(compliance_notes, list) or not all(
                    isinstance(item, str) for item in compliance_notes
                ):
                    raise ValueError("Missing or invalid compliance_notes list")
                if not isinstance(follow_up_tasks, list):
                    raise ValueError("Missing or invalid follow_up_tasks list")
                if not isinstance(requires_human_review, bool):
                    raise ValueError("Missing or invalid requires_human_review flag")
                if review_reason is not None and not isinstance(review_reason, str):
                    raise ValueError("Missing or invalid review_reason")

                validated_follow_up_tasks = [self._validate_follow_up_task(task) for task in follow_up_tasks]
                if any(task is None for task in validated_follow_up_tasks):
                    raise ValueError("Invalid follow_up_tasks entry")

                return AfterCallSummary(
                    call_summary=summary.strip(),
                    resolution_status=resolution_status,
                    commitments=[item.strip() for item in commitments if item.strip()],
                    compliance_notes=[item.strip() for item in compliance_notes if item.strip()],
                    follow_up_tasks=[task for task in validated_follow_up_tasks if task is not None],
                    requires_human_review=requires_human_review,
                    review_reason=(
                        review_reason.strip()
                        if requires_human_review and isinstance(review_reason, str) and review_reason.strip()
                        else None
                    ),
                )
            except Exception:
                logger.exception("Malformed after-call summary output for agent_id=%s: %s", agent_id, raw_output)
                return fallback
        except Exception:
            logger.exception("Unexpected failure in after-call work agent for agent_id=%s", agent_id)
            return fallback

    def _build_summary_prompt(self, transcript: str, qa_score: dict | None, agent_id: str) -> str:
        """Build the after-call summary prompt for the provided transcript."""

        qa_context = self._build_qa_context(qa_score)
        return (
            "Summarize the following finished call transcript for after-call work. "
            "Return STRICT JSON with exactly this shape:\n"
            "{\n"
            '  "call_summary": "<2-4 factual sentences, no editorializing>",\n'
            '  "resolution_status": "resolved|unresolved|escalated|unknown",\n'
            '  "commitments": ["<promises or commitments made by the agent>"],\n'
            '  "compliance_notes": ["<compliance-relevant statements, disclosures, or verification steps>"],\n'
            '  "follow_up_tasks": [{"description": "<action>", "priority": "low|medium|high", "due_by": "<human-readable relative deadline>"}],\n'
            '  "requires_human_review": <true|false>,\n'
            '  "review_reason": "<one-sentence reason or null>"\n'
            "}\n\n"
            "Scoring and summarization rules:\n"
            "- call_summary must be 2-4 sentences, factual, and grounded only in the transcript.\n"
            "- resolution_status must be one of resolved, unresolved, escalated, or unknown.\n"
            "- commitments should include any promises or follow-up commitments the agent made, or [] if none.\n"
            "- compliance_notes should list only real compliance-relevant items from the transcript, or [] if none.\n"
            "- follow_up_tasks should contain concrete actionable items, or [] if no follow-up is needed.\n"
            "- requires_human_review should be true when the transcript is too short, garbled, or ambiguous to summarize confidently.\n"
            "- review_reason must be null when requires_human_review is false, and a specific one-sentence reason when true.\n"
            "- Do not invent facts, policy details, or hidden context.\n\n"
            f"Agent ID: {agent_id}\n"
            f"QA context:\n{qa_context}\n\n"
            f"Transcript:\n{transcript}"
        )

    def _build_qa_context(self, qa_score: dict | None) -> str:
        if not qa_score:
            return "No QA score was provided."

        overall_score = qa_score.get("overall_score")
        dimension_scores = qa_score.get("dimension_scores")
        low_dimensions: list[str] = []

        if isinstance(dimension_scores, dict):
            for dimension, score in dimension_scores.items():
                try:
                    if int(score) < 50:
                        low_dimensions.append(f"{dimension}={score}")
                except Exception:
                    continue

        context_lines = [f"Overall QA score: {overall_score}"]
        if low_dimensions:
            context_lines.append("Low-scoring dimensions below 50: " + ", ".join(low_dimensions))
        else:
            context_lines.append("No QA dimension scored below 50.")
        return "\n".join(context_lines)

    def _validate_follow_up_task(self, task: Any) -> FollowUpTask | None:
        if not isinstance(task, dict):
            return None

        description = task.get("description")
        priority = task.get("priority")
        due_by = task.get("due_by")

        if not isinstance(description, str) or not description.strip():
            return None
        if priority not in {"low", "medium", "high"}:
            return None
        if not isinstance(due_by, str) or not due_by.strip():
            return None

        return FollowUpTask(
            description=description.strip(),
            priority=priority,
            due_by=due_by.strip(),
        )
def _fallback_after_call_summary() -> AfterCallSummary:
    """Return a safe default summary when the LLM output is unusable."""

    return AfterCallSummary(
        call_summary="",
        resolution_status="unknown",
        commitments=[],
        compliance_notes=[],
        follow_up_tasks=[],
        requires_human_review=True,
        review_reason="Automated summary generation failed",
    )


after_call_agent = AfterCallWorkAgent()
