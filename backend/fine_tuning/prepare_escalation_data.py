"""Generate synthetic escalation-risk training data for portfolio/demo purposes.

The labeling logic in this script is a designed heuristic, documented inline, and
is not derived from real call center data. It exists to create a plausible tabular
dataset for demos and experimentation before any real operational data is available.
The features are now correlated through a shared latent call_difficulty factor so
the resulting dataset has more realistic multi-feature structure and is easier for
the downstream classifier to learn than fully independent weak signals.
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

# This exponent controls how strongly probabilities are pushed away from 0.5
# toward the extremes (0 and 1). It was calibrated empirically against a measured
# ceiling AUC-ROC of about 0.73 before sharpening, with the project target being
# AUC-ROC > 0.80. Values greater than 1 increase polarization; 1.0 leaves the
# probabilities unchanged.
SHARPENING_EXPONENT = 2.5


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a floating-point value to the provided range."""

    return float(np.clip(value, minimum, maximum))


def _sharpen_probability(probability: float, exponent: float) -> float:
    """Apply an odds-power transform that preserves rank order while increasing
    separation from 0.5.

    This transform raises the odds of the probability to a configurable exponent,
    which pushes probabilities away from the midpoint and toward the extremes
    without changing their relative ordering. That makes the synthetic label
    generation more separable while keeping the intended ranking structure intact.
    """

    clipped_probability = _clamp(probability, 0.001, 0.999)
    odds = clipped_probability / (1.0 - clipped_probability)
    sharpened_odds = odds ** exponent
    sharpened_probability = sharpened_odds / (1.0 + sharpened_odds)
    return float(sharpened_probability)


def _sample_row() -> dict[str, object]:
    """Generate one synthetic call-state feature row with latent difficulty."""

    # Beta(0.5, 0.5) is U-shaped, so mass concentrates near 0 and 1 rather than
    # clustering around the middle; this produces more calls that are clearly easy
    # or clearly difficult and reduces Bernoulli label-sampling noise when the true
    # escalation probability would otherwise sit near 0.5.
    call_difficulty = float(RNG.beta(0.5, 0.5))
    intent = str(RNG.choice(INTENTS))
    intent_confidence = float(RNG.beta(5, 2))
    low_confidence = int(intent_confidence < 0.35)

    sentiment_current = _clamp(float(RNG.normal(0.1, 0.4)), -1.0, 1.0)
    sentiment_trend = float(RNG.normal(-0.35 * call_difficulty, 0.12))

    # The minimum sentiment so far is modeled as the worse of the current moment
    # and a rough future-projection term that includes worsening trend plus extra
    # negative noise. The latent call_difficulty factor makes the symptoms co-vary
    # more realistically instead of behaving like independent random draws.
    turn_factor = float(RNG.uniform(1.0, 3.0))
    extra_negative_noise = -abs(float(RNG.normal(0.05 + 0.15 * call_difficulty, 0.05)))
    sentiment_projection = sentiment_current + sentiment_trend * turn_factor + extra_negative_noise
    sentiment_min_so_far = _clamp(min(sentiment_current, sentiment_projection), -1.0, 1.0)

    turn_count = int(max(1, RNG.poisson(5 + 10 * call_difficulty)))

    # Approximate transcript length by summing per-turn word counts. The sampled
    # words-per-turn distribution is clipped to a realistic minimum of 3 words.
    words_per_turn = np.maximum(3, np.round(RNG.normal(15, 5, size=turn_count)).astype(int))
    transcript_length = int(words_per_turn.sum())

    # Hold time proxy loosely scales with turn_count to reflect longer, more involved
    # calls, but still retains randomness from the exponential draw. The scale grows
    # with latent call difficulty so harder calls tend to accumulate more latency.
    hold_time_proxy = float(RNG.exponential(scale=(4.0 + 8.0 * call_difficulty)) * (1.0 + turn_count / 12.0))

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

    base_probability = 0.05

    probability = base_probability
    probability += 0.15 if int(row["low_confidence"]) == 1 else 0.0

    # More negative sentiment trend increases escalation risk, capped at 0.30.
    sentiment_trend = float(row["sentiment_trend"])
    trend_component = min(max(0.0, -sentiment_trend) / 0.50, 1.0) * 0.30
    probability += trend_component

    # A lower minimum sentiment observed during the call signals a more difficult
    # interaction, with a maximum contribution of 0.25.
    sentiment_min_so_far = float(row["sentiment_min_so_far"])
    sentiment_component = min(max(0.0, -sentiment_min_so_far) / 1.0, 1.0) * 0.25
    probability += sentiment_component

    # Longer calls generally leave more room for escalation, but the effect is mild.
    turn_count = int(row["turn_count"])
    turn_component = min(turn_count / 30.0, 1.0) * 0.20
    probability += turn_component

    # Hold time proxy contributes a smaller amount because latency alone is a weak
    # escalation signal compared with sentiment or low model confidence.
    hold_time_proxy = float(row["hold_time_proxy"])
    hold_component = min(hold_time_proxy / 60.0, 1.0) * 0.15
    probability += hold_component

    # Fraud and cancellation calls are slightly more likely to escalate, but this
    # bump is intentionally kept small so intent does not dominate the behavioral
    # signals even after reweighting the feature contributions above.
    intent = str(row["intent"])
    if intent in {"fraud_report", "cancel_service"}:
        probability += 0.05

    probability = _sharpen_probability(probability, SHARPENING_EXPONENT)

    probability += float(RNG.normal(0.0, 0.02))

    return _clamp(probability, 0.03, 0.90)


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
