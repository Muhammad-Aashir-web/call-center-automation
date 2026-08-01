"""Reduced LangGraph call-flow orchestration for live call center assistance.

This is the foundation of the full seven-agent orchestration graph described in
the project's technical spec. Additional nodes such as score_quality and
after_call_work will be added here in later phases once those agents exist. The
graph is intended to be extended, not rebuilt.
"""

from __future__ import annotations


import time
import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.escalation import escalation_predictor
from agents.intent import intent_classifier
from agents.knowledge import KnowledgeRetrievalAgent
from services.sentiment import SentimentTracker
from agents.suggestion import ResponseSuggestionAgent


logger = logging.getLogger(__name__)


class CallState(TypedDict):
    """State container for the live call-flow graph."""

    call_id: str
    transcript_segment: str
    transcript: str
    sentiment_history: list[float]
    turn_count: int
    last_segment_timestamp: float | None
    hold_time_proxy: float
    intent: str | None
    intent_confidence: float | None
    low_confidence: bool | None
    kb_suggestions: list[dict]
    suggested_response: dict[str, str] | None
    escalation_risk: float | None
    escalation_alert: bool | None


knowledge_agent = KnowledgeRetrievalAgent()
suggestion_agent = ResponseSuggestionAgent()


def transcribe_node(state: CallState) -> dict[str, str | int | float | None]:
    """Fold the latest transcript segment into the running full transcript."""

    try:
        transcript_segment = state["transcript_segment"].strip()
        current_transcript = state.get("transcript", "").strip()
        turn_count = int(state.get("turn_count", 0)) + 1
        previous_timestamp = state.get("last_segment_timestamp")
        current_timestamp = time.time()
        hold_time_proxy = (
            float(current_timestamp - previous_timestamp)
            if previous_timestamp is not None
            else 0.0
        )

        if not transcript_segment:
            return {
                "transcript": current_transcript,
                "turn_count": turn_count,
                "last_segment_timestamp": current_timestamp,
                "hold_time_proxy": hold_time_proxy,
            }

        updated_transcript = (
            transcript_segment
            if not current_transcript
            else f"{current_transcript} {transcript_segment}"
        )
        return {
            "transcript": updated_transcript,
            "turn_count": turn_count,
            "last_segment_timestamp": current_timestamp,
            "hold_time_proxy": hold_time_proxy,
        }
    except Exception:
        logger.exception("Failed to fold transcript segment into running transcript")
        return {
            "transcript": state.get("transcript", ""),
            "turn_count": state.get("turn_count", 0),
            "last_segment_timestamp": state.get("last_segment_timestamp"),
            "hold_time_proxy": state.get("hold_time_proxy", 0.0),
        }


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


async def suggest_response_node(state: CallState) -> dict[str, dict[str, str]]:
    """Generate a grounded response suggestion for the current transcript segment.

    This node runs after retrieve_knowledge (not parallel to it), because
    ResponseSuggestionAgent.suggest() requires kb_suggestions as a mandatory
    argument -- it needs retrieval to have already completed for this segment.
    """

    try:
        result = await suggestion_agent.suggest(
            transcript=state.get("transcript", ""),
            intent=state.get("intent"),
            kb_suggestions=state.get("kb_suggestions", []),
        )
        return {"suggested_response": result}
    except Exception:
        logger.exception("Failed to generate response suggestion for transcript segment")
        return {
            "suggested_response": {
                "suggested_response": "I'm looking into this for you now.",
                "rationale": "Automated suggestion unavailable; showing a safe default.",
            }
        }


def predict_escalation_node(state: CallState) -> dict[str, float | bool]:
    """Score escalation risk from the current intent and sentiment history."""

    try:
        # Use SentimentTracker's shared analyzer directly so we can compute the
        # next history state from the plain list stored in CallState without a
        # live tracker instance; compute_features_from_history is designed for
        # exactly this JSON-serializable workflow state shape.
        current_score = float(
            SentimentTracker._analyzer.polarity_scores(state["transcript_segment"])["compound"]
        )
        updated_history = list(state.get("sentiment_history", [])) + [current_score]
        sentiment_features = SentimentTracker.compute_features_from_history(updated_history)
        # Use a trained in-vocabulary category rather than an empty string so the
        # model receives a value it could plausibly have seen during training.
        escalation_result = escalation_predictor.predict(
            intent=state.get("intent") or "general_inquiry",
            intent_confidence=float(state.get("intent_confidence") or 0.0),
            low_confidence=bool(state.get("low_confidence") or False),
            sentiment_features=sentiment_features,
            turn_count=int(state.get("turn_count", 0)),
            transcript_length=len(state.get("transcript", "").split()),
            hold_time_proxy=float(state.get("hold_time_proxy", 0.0)),
        )
        return {
            "escalation_risk": float(escalation_result["escalation_risk"]),
            "escalation_alert": bool(escalation_result["alert"]),
            "sentiment_history": updated_history,
        }
    except Exception:
        logger.exception("Failed to predict escalation risk for transcript segment")
        return {
            "escalation_risk": 0.0,
            "escalation_alert": False,
            "sentiment_history": list(state.get("sentiment_history", [])),
        }


workflow = StateGraph(CallState)
workflow.add_node("transcribe", transcribe_node)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.add_node("suggest_response", suggest_response_node)
workflow.add_node("predict_escalation", predict_escalation_node)
workflow.set_entry_point("transcribe")
workflow.add_edge("transcribe", "classify_intent")
workflow.add_edge("classify_intent", "retrieve_knowledge")
workflow.add_edge("classify_intent", "predict_escalation")
workflow.add_edge("retrieve_knowledge", "suggest_response")
workflow.add_edge("suggest_response", END)
workflow.add_edge("predict_escalation", END)
call_graph = workflow.compile()