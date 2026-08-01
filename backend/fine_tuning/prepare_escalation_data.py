"""Generate synthetic escalation-risk training data for portfolio/demo purposes.

The labeling logic in this script is a designed heuristic, documented inline, and
is not derived from real call center data. It exists to create a plausible tabular
dataset for demos and experimentation before any real operational data is available.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_SAMPLES = 2000
INTENTS = [
    "billing_inquiry",
    "technical_support",
    "service_outage_status",
    "cancel_service",
    "account_management",
    "general_inquiry",
    "sales_inquiry",
    "fraud_report",
]
RNG = np.random.default_rng(42)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a floating-point value to the provided range."""

    return float(np.clip(value, minimum, maximum))


def _sample_row() -> dict[str, object]:
    """Generate one synthetic call-state feature row."""

    intent = str(RNG.choice(INTENTS))
    intent_confidence = float(RNG.beta(5, 2))
    low_confidence = int(intent_confidence < 0.35)

    sentiment_current = _clamp(float(RNG.normal(0.1, 0.4)), -1.0, 1.0)
    sentiment_trend = float(RNG.normal(-0.02, 0.15))

    # The minimum sentiment so far is modeled as the worse of the current moment
    # and a rough future-projection term that includes worsening trend plus extra
    # negative noise. This keeps the value generally at or below sentiment_current.
    turn_factor = float(RNG.uniform(1.0, 3.0))
    extra_negative_noise = -abs(float(RNG.normal(0.05, 0.05)))
    sentiment_projection = sentiment_current + sentiment_trend * turn_factor + extra_negative_noise
    sentiment_min_so_far = _clamp(min(sentiment_current, sentiment_projection), -1.0, 1.0)

    turn_count = int(max(1, RNG.poisson(8)))

    # Approximate transcript length by summing per-turn word counts. The sampled
    # words-per-turn distribution is clipped to a realistic minimum of 3 words.
    words_per_turn = np.maximum(3, np.round(RNG.normal(15, 5, size=turn_count)).astype(int))
    transcript_length = int(words_per_turn.sum())

    # Hold time proxy loosely scales with turn_count to reflect longer, more involved
    # calls, but still retains randomness from the exponential draw.
    hold_time_proxy = float(RNG.exponential(scale=8.0) * (1.0 + turn_count / 12.0))

    row = {
        "intent": intent,
        "intent_confidence": intent_confidence,
        "low_confidence": low_confidence,
        "sentiment_current": sentiment_current,
        "sentiment_trend": sentiment_trend,
        "sentiment_min_so_far": sentiment_min_so_far,
        "turn_count": turn_count,
        "transcript_length": transcript_length,
        "hold_time_proxy": hold_time_proxy,
    }
    return row


def compute_escalation_probability(row: dict[str, object]) -> float:
    """Compute a synthetic escalation probability from the engineered call-state row.

    The heuristic is intentionally transparent and reviewable:
    - Start from a modest base rate.
    - Add risk when the intent classifier is low confidence.
    - Add risk when sentiment is trending downward or has already become very negative.
    - Add modest risk for longer, more drawn-out calls and higher hold time.
    - Add a small bump for intents that are commonly escalation-prone.
    - Add Gaussian jitter so the label is not perfectly separable.
    """

    base_probability = 0.10

    probability = base_probability
    probability += 0.15 if int(row["low_confidence"]) == 1 else 0.0

    # More negative sentiment trend increases escalation risk, capped at 0.20.
    sentiment_trend = float(row["sentiment_trend"])
    trend_component = min(max(0.0, -sentiment_trend) / 0.50, 1.0) * 0.20
    probability += trend_component

    # A lower minimum sentiment observed during the call signals a more difficult
    # interaction, with a maximum contribution of 0.15.
    sentiment_min_so_far = float(row["sentiment_min_so_far"])
    sentiment_component = min(max(0.0, -sentiment_min_so_far) / 1.0, 1.0) * 0.15
    probability += sentiment_component

    # Longer calls generally leave more room for escalation, but the effect is mild.
    turn_count = int(row["turn_count"])
    turn_component = min(turn_count / 30.0, 1.0) * 0.15
    probability += turn_component

    # Hold time proxy contributes a smaller amount because latency alone is a weak
    # escalation signal compared with sentiment or low model confidence.
    hold_time_proxy = float(row["hold_time_proxy"])
    hold_component = min(hold_time_proxy / 60.0, 1.0) * 0.10
    probability += hold_component

    # Fraud and cancellation calls are slightly more likely to escalate, but this
    # bump is intentionally minor so it cannot dominate the behavioral signals.
    intent = str(row["intent"])
    if intent in {"fraud_report", "cancel_service"}:
        probability += 0.05

    probability += float(RNG.normal(0.0, 0.05))

    return _clamp(probability, 0.02, 0.95)


def generate_escalation_dataset(num_samples: int = NUM_SAMPLES) -> pd.DataFrame:
    """Generate the full synthetic escalation-risk dataset as a pandas DataFrame."""

    rows: list[dict[str, object]] = []
    for _ in range(num_samples):
        row = _sample_row()
        row["escalation_probability"] = compute_escalation_probability(row)
        row["escalated"] = int(RNG.random() < float(row["escalation_probability"]))
        rows.append(row)

    dataframe = pd.DataFrame(rows)

    # Keep intent as a string column. One-hot encoding is deferred to training time
    # so this file remains human-readable and easier to inspect.
    column_order = [
        "intent",
        "intent_confidence",
        "low_confidence",
        "sentiment_current",
        "sentiment_trend",
        "sentiment_min_so_far",
        "turn_count",
        "transcript_length",
        "hold_time_proxy",
        "escalation_probability",
        "escalated",
    ]
    return dataframe[column_order]


def save_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the generated dataset to disk as CSV."""

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
    except Exception:
        logger.exception("Failed to write escalation training data to %s", output_path)
        raise


def main() -> None:
    """Generate the synthetic dataset, save it, and print summary statistics."""

    dataframe = generate_escalation_dataset()
    output_path = Path(__file__).resolve().parents[2] / "data" / "training" / "escalation_training_data.csv"

    try:
        save_dataset(dataframe, output_path)
    except Exception:
        raise

    print("\nGeneration complete:")
    print(f"- row count: {len(dataframe)}")
    print(f"- mean escalation_probability: {dataframe['escalation_probability'].mean():.4f}")
    class_balance = dataframe["escalated"].value_counts(normalize=True).sort_index()
    print("- escalated class balance:")
    for label, share in class_balance.items():
        print(f"  - {label}: {share:.4f}")

    print("\nPer-feature describe():\n")
    print(dataframe.describe().T)
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
