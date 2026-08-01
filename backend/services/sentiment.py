"""Track sentiment as a trend over an in-progress call using VADER.

This service is intentionally a per-call object, not a shared singleton. One
instance represents one active call's sentiment state, which keeps the lifecycle
aligned with the streaming call flow. VADER is used here because it is zero-cost,
fully local, and low-latency compared with an LLM-based sentiment approach, which
fits the same practical reasoning used for the project's local XGBoost escalation
classifier.

Usage example:
    tracker = SentimentTracker()
    features = tracker.update(segment_text)
    # feed features into escalation prediction after each transcribed segment
"""

from __future__ import annotations

import logging

import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


logger = logging.getLogger(__name__)


class SentimentTracker:
    """Track sentiment over time for one in-progress call instance.

    This is not a singleton. Create one tracker at call start, call `update()` for
    each new transcript segment, and discard or reset it when the call ends.
    """

    TREND_WINDOW = 5
    _analyzer = SentimentIntensityAnalyzer()

    def __init__(self) -> None:
        """Initialize an empty sentiment history for a single call."""

        self._history: list[float] = []

    @property
    def segment_count(self) -> int:
        """Return the number of non-empty transcript segments processed so far."""

        return len(self._history)

    def update(self, text: str) -> dict[str, float]:
        """Update sentiment history with one transcript segment and return features."""

        if not text.strip():
            logger.warning(
                "Ignoring empty sentiment segment to avoid scoring blank transcription as falsely neutral/positive"
            )
            if self._history:
                return {
                    "sentiment_current": self._history[-1],
                    "sentiment_trend": self._compute_trend(),
                    "sentiment_min_so_far": min(self._history),
                }
            return {
                "sentiment_current": 0.0,
                "sentiment_trend": 0.0,
                "sentiment_min_so_far": 0.0,
            }

        try:
            sentiment_scores = self._analyzer.polarity_scores(text)
            sentiment_current = float(sentiment_scores["compound"])
            self._history.append(sentiment_current)
        except Exception:
            logger.exception("Failed to score sentiment for transcript segment")
            if self._history:
                return {
                    "sentiment_current": self._history[-1],
                    "sentiment_trend": self._compute_trend(),
                    "sentiment_min_so_far": min(self._history),
                }
            return {
                "sentiment_current": 0.0,
                "sentiment_trend": 0.0,
                "sentiment_min_so_far": 0.0,
            }

        sentiment_trend = self._compute_trend()
        sentiment_min_so_far = min(self._history)

        return {
            "sentiment_current": sentiment_current,
            "sentiment_trend": sentiment_trend,
            "sentiment_min_so_far": sentiment_min_so_far,
        }

    def reset(self) -> None:
        """Clear the tracked history so the instance can be reused for another call."""

        self._history.clear()

    @staticmethod
    def compute_features_from_history(history: list[float]) -> dict[str, float]:
        """Compute sentiment features from a plain VADER history list.

        LangGraph's CallState must stay a plain, JSON-serializable dict, so this
        staticmethod lets graph node code compute the same trend features from a
        plain list stored in state without needing a live SentimentTracker instance
        to persist across node invocations.
        """

        try:
            sentiment_current = float(history[-1]) if history else 0.0
            sentiment_trend = SentimentTracker._compute_trend_from_history(history)
            sentiment_min_so_far = float(min(history)) if history else 0.0

            return {
                "sentiment_current": sentiment_current,
                "sentiment_trend": sentiment_trend,
                "sentiment_min_so_far": sentiment_min_so_far,
            }
        except Exception:
            logger.exception("Failed to compute sentiment features from history")
            return {
                "sentiment_current": 0.0,
                "sentiment_trend": 0.0,
                "sentiment_min_so_far": 0.0,
            }

    def _compute_trend(self) -> float:
        """Compute the slope of the recent sentiment history using a degree-1 fit."""

        return self.compute_features_from_history(self._history)["sentiment_trend"]

    @staticmethod
    def _compute_trend_from_history(history: list[float]) -> float:
        """Compute the slope of the recent sentiment history using a degree-1 fit."""

        if len(history) <= 1:
            # A slope is not defined from a single point, so use neutral trend.
            return 0.0

        recent_history = history[-SentimentTracker.TREND_WINDOW :]
        try:
            x_values = np.arange(len(recent_history), dtype=float)
            y_values = np.asarray(recent_history, dtype=float)
            slope = float(np.polyfit(x_values, y_values, deg=1)[0])
            return slope
        except Exception:
            logger.exception("Failed to compute sentiment trend slope")
            return 0.0
