"""Agent 6 (Escalation Prediction Agent) for the call center automation system.

This agent predicts escalation risk in real time using a locally trained XGBoost
classifier over live transcript-derived features: sentiment trend from
SentimentTracker, intent classification results, and a hold-time proxy. A local
model is preferred over an LLM approach because it keeps latency low and avoids
rate-limit dependence in the call pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from config import settings


logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "escalation_classifier" / "model.json"
FEATURE_COLUMNS_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "escalation_classifier" / "feature_columns.json"
)


class EscalationModelLoadError(Exception):
    """Raised when the escalation model or its feature metadata cannot be loaded."""


try:
    escalation_model = XGBClassifier()
    escalation_model.load_model(str(MODEL_PATH))

    with FEATURE_COLUMNS_PATH.open("r", encoding="utf-8") as file_handle:
        escalation_feature_columns = json.load(file_handle)

    if not isinstance(escalation_feature_columns, list) or not all(
        isinstance(column, str) for column in escalation_feature_columns
    ):
        raise ValueError("feature_columns.json must contain a list of string column names")
except Exception as exc:
    logger.exception(
        "Failed to load escalation model artifacts from %s and %s",
        MODEL_PATH,
        FEATURE_COLUMNS_PATH,
    )
    raise EscalationModelLoadError(
        f"Failed to load escalation model artifacts from {MODEL_PATH} and {FEATURE_COLUMNS_PATH}"
    ) from exc


ESCALATION_ALERT_THRESHOLD = float(settings.ESCALATION_ALERT_THRESHOLD)


class EscalationPredictor:
    """Predict escalation risk from live call segment features."""

    def __init__(self) -> None:
        self.model = escalation_model
        self.feature_columns = escalation_feature_columns
        self.alert_threshold = ESCALATION_ALERT_THRESHOLD

    def _extract_sentiment_feature(self, sentiment_features: dict[str, float], key: str) -> float:
        try:
            return float(sentiment_features[key])
        except KeyError:
            logger.exception("Missing sentiment feature '%s'; defaulting to 0.0", key)
            return 0.0
        except Exception:
            logger.exception("Invalid sentiment feature '%s'; defaulting to 0.0", key)
            return 0.0

    def predict(
        self,
        intent: str,
        intent_confidence: float,
        low_confidence: bool,
        sentiment_features: dict[str, float],
        turn_count: int,
        transcript_length: int,
        hold_time_proxy: float,
    ) -> dict[str, Any]:
        """Predict escalation risk for a single call segment."""

        try:
            sentiment_current = self._extract_sentiment_feature(sentiment_features, "sentiment_current")
            sentiment_trend = self._extract_sentiment_feature(sentiment_features, "sentiment_trend")
            sentiment_min_so_far = self._extract_sentiment_feature(
                sentiment_features,
                "sentiment_min_so_far",
            )

            row = {
                "intent": intent,
                "intent_confidence": float(intent_confidence),
                "low_confidence": int(bool(low_confidence)),
                "sentiment_current": sentiment_current,
                "sentiment_trend": sentiment_trend,
                "sentiment_min_so_far": sentiment_min_so_far,
                "turn_count": int(turn_count),
                "transcript_length": int(transcript_length),
                "hold_time_proxy": float(hold_time_proxy),
            }

            frame = pd.DataFrame([row])
            frame = pd.get_dummies(frame, columns=["intent"], drop_first=False)

            # A single live prediction only contains one intent value, so get_dummies
            # will create just one intent column here. Reindexing restores the full
            # trained feature layout by inserting the missing all-zero intent columns
            # in the exact order saved during training.
            frame = frame.reindex(columns=self.feature_columns, fill_value=0)

            probabilities = self.model.predict_proba(frame)
            escalation_risk = float(probabilities[0][1])
            alert = escalation_risk >= self.alert_threshold

            return {
                "escalation_risk": escalation_risk,
                "alert": alert,
            }
        except Exception:
            logger.exception(
                "Escalation prediction failed for intent=%s, turn_count=%s, transcript_length=%s",
                intent,
                turn_count,
                transcript_length,
            )
            logger.warning(
                "Automated escalation prediction was unavailable for this call segment; returning a safe fallback"
            )
            return {
                "escalation_risk": 0.0,
                "alert": False,
            }


# Usage pattern:
# intent_result = intent_classifier.classify(...)
# sentiment_result = tracker.update(...)
# escalation_result = escalation_predictor.predict(
#     intent=intent_result["intent"],
#     intent_confidence=intent_result["confidence"],
#     low_confidence=intent_result["low_confidence"],
#     sentiment_features=sentiment_result,
#     turn_count=..., transcript_length=..., hold_time_proxy=...
# )
escalation_predictor = EscalationPredictor()
