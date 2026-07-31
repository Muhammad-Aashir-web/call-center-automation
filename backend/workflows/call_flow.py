"""Reduced LangGraph call-flow orchestration for live call center assistance.

This is the three-node foundation of the full seven-agent orchestration graph
described in the project's technical spec. Additional nodes such as
suggest_response, predict_escalation, score_quality, and after_call_work will be
added here in later phases once those agents exist. The graph is intended to be
extended, not rebuilt.
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.intent import intent_classifier
from agents.knowledge import KnowledgeRetrievalAgent


logger = logging.getLogger(__name__)


class CallState(TypedDict):
    """State container for the live call-flow graph."""

    call_id: str
    transcript_segment: str
    transcript: str
    intent: str | None
    intent_confidence: float | None
    low_confidence: bool | None
    kb_suggestions: list[dict]


knowledge_agent = KnowledgeRetrievalAgent()


def transcribe_node(state: CallState) -> dict[str, str]:
    """Fold the latest transcript segment into the running full transcript."""

    try:
        transcript_segment = state["transcript_segment"].strip()
        current_transcript = state.get("transcript", "").strip()

        if not transcript_segment:
            return {"transcript": current_transcript}

        updated_transcript = (
            transcript_segment
            if not current_transcript
            else f"{current_transcript} {transcript_segment}"
        )
        return {"transcript": updated_transcript}
    except Exception:
        logger.exception("Failed to fold transcript segment into running transcript")
        return {"transcript": state.get("transcript", "")}


def classify_intent_node(state: CallState) -> dict[str, str | float | bool | None]:
    """Classify the current transcript segment into one of the configured intents."""

    try:
        result = intent_classifier.classify(state["transcript_segment"])
        return {
            "intent": result["intent"],
            "intent_confidence": result["confidence"],
            "low_confidence": result["low_confidence"],
        }
    except Exception:
        logger.exception("Failed to classify intent for transcript segment")
        return {
            "intent": None,
            "intent_confidence": None,
            "low_confidence": None,
        }


def retrieve_knowledge_node(state: CallState) -> dict[str, list[dict]]:
    """Retrieve knowledge-base suggestions for the current transcript segment."""

    try:
        suggestions = knowledge_agent.retrieve(state["transcript_segment"], top_k=5)
        return {"kb_suggestions": suggestions}
    except Exception:
        logger.exception("Failed to retrieve knowledge suggestions for transcript segment")
        return {"kb_suggestions": []}


workflow = StateGraph(CallState)
workflow.add_node("transcribe", transcribe_node)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.set_entry_point("transcribe")
workflow.add_edge("transcribe", "classify_intent")
workflow.add_edge("classify_intent", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", END)
call_graph = workflow.compile()