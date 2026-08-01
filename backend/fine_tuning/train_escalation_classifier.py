"""Train a local XGBoost escalation classifier on synthetic portfolio data.

This script evaluates against the project spec targets of AUC-ROC > 0.80 and a
false-positive rate below 15% at candidate thresholds, so the resulting model can
be used to inform a later ESCALATION_ALERT_THRESHOLD choice.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
THRESHOLDS = [0.3, 0.4, 0.5, 0.6]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""

    parser = argparse.ArgumentParser(description="Train an XGBoost escalation classifier.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "training" / "escalation_training_data.csv",
        help="Path to the synthetic escalation training CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "models" / "escalation_classifier",
        help="Directory where model artifacts will be saved.",
    )
    return parser.parse_args()


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the synthetic escalation dataset from disk."""

    try:
        return pd.read_csv(data_path)
    except Exception:
        logger.exception("Failed to load escalation training data from %s", data_path)
        raise


def prepare_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the model feature matrix and target vector.

    The exact post-encoding feature column order is persisted because inference
    must reproduce the same one-hot layout later; XGBoost does not infer this for us.
    """

    target = dataframe["escalated"].astype(int)
    feature_frame = dataframe[
        [
            "intent",
            "intent_confidence",
            "low_confidence",
            "sentiment_current",
            "sentiment_trend",
            "sentiment_min_so_far",
            "turn_count",
            "transcript_length",
            "hold_time_proxy",
        ]
    ].copy()

    encoded_features = pd.get_dummies(feature_frame, columns=["intent"], drop_first=False)
    feature_columns = list(encoded_features.columns)
    return encoded_features, target, feature_columns


def save_json(path: Path, payload: Any) -> None:
    """Write a JSON payload to disk with error handling."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2)
    except Exception:
        logger.exception("Failed to write JSON artifact to %s", path)
        raise


def compute_split_balance(y: pd.Series) -> dict[str, float]:
    """Return the binary class balance for a split as label proportions."""

    counts = y.value_counts(normalize=True).sort_index()
    return {str(int(label)): float(value) for label, value in counts.items()}


def compute_threshold_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Compute threshold-dependent metrics from the predicted probabilities."""

    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "false_positive_rate": float(false_positive_rate),
        "true_positive_rate": float(true_positive_rate),
        "precision": float(precision),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> XGBClassifier:
    """Train an XGBoost classifier with the requested hyperparameters."""

    model_kwargs = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
    }

    # `use_label_encoder` is only supported by some xgboost versions; keep the code
    # compatible without assuming the installed package accepts the parameter.
    try:
        model = XGBClassifier(use_label_encoder=False, **model_kwargs)
    except TypeError:
        model = XGBClassifier(**model_kwargs)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def run_pipeline(data_path: Path, output_dir: Path) -> dict[str, Any]:
    """Execute the full training and evaluation pipeline."""

    dataframe = load_data(data_path)
    features, target, feature_columns = prepare_features(dataframe)

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=target,
    )

    # Persist the exact encoded feature order because real inference must reproduce
    # this column layout exactly; otherwise the trained model will receive mismatched
    # inputs and make incorrect predictions.
    feature_columns_path = output_dir / "feature_columns.json"
    save_json(feature_columns_path, feature_columns)

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0
    logger.info("Computed scale_pos_weight for reference only: %.4f", scale_pos_weight)

    # This is a deliberate choice. The target is a calibrated probability score, not
    # just a class label, and the imbalance is mild (~24/76). Weighting would distort
    # calibration without meaningfully improving ranking (AUC-ROC), so the value is
    # computed for documentation/reference only.
    model = train_model(X_train, y_train, X_val, y_val)

    y_val_probabilities = model.predict_proba(X_val)[:, 1]
    auc_roc = float(roc_auc_score(y_val, y_val_probabilities))

    threshold_rows: list[dict[str, float]] = []
    for threshold in THRESHOLDS:
        metrics = compute_threshold_metrics(y_val.to_numpy(), y_val_probabilities, threshold)
        threshold_rows.append(metrics)

    default_threshold_metrics = next(item for item in threshold_rows if item["threshold"] == 0.5)
    print("\nValidation classification report at threshold 0.5:\n")
    print(classification_report(y_val, (y_val_probabilities >= 0.5).astype(int), digits=4, zero_division=0))

    print("\nThreshold summary table:\n")
    summary_df = pd.DataFrame(threshold_rows)[
        ["threshold", "false_positive_rate", "true_positive_rate", "precision", "tn", "fp", "fn", "tp"]
    ]
    print(summary_df.to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.json"
    try:
        model.save_model(str(model_path))
    except Exception:
        logger.exception("Failed to save XGBoost model to %s", model_path)
        raise

    metrics_payload = {
        "auc_roc": auc_roc,
        "threshold_metrics": threshold_rows,
        "scale_pos_weight": scale_pos_weight,
        "training_size": int(len(X_train)),
        "validation_size": int(len(X_val)),
        "train_class_balance": compute_split_balance(y_train),
        "validation_class_balance": compute_split_balance(y_val),
    }
    save_json(output_dir / "metrics.json", metrics_payload)

    return {
        "auc_roc": auc_roc,
        "threshold_metrics": threshold_rows,
        "default_threshold_metrics": default_threshold_metrics,
        "output_dir": output_dir,
        "model_path": model_path,
    }


def print_final_summary(results: dict[str, Any]) -> None:
    """Print a concise pass/fail summary for the project target metrics."""

    auc_achieved = results["auc_roc"] > 0.80
    fpr_below_15_thresholds = [
        item["threshold"]
        for item in results["threshold_metrics"]
        if item["false_positive_rate"] < 0.15
    ]

    print("\nFinal summary:\n")
    print(f"- AUC-ROC achieved: {results['auc_roc']:.4f}")
    print(f"- AUC-ROC > 0.80 target met: {'yes' if auc_achieved else 'no'}")
    if fpr_below_15_thresholds:
        formatted_thresholds = ", ".join(f"{threshold:.1f}" for threshold in fpr_below_15_thresholds)
        print(f"- Threshold(s) with FPR < 15%: {formatted_thresholds}")
    else:
        print("- Threshold(s) with FPR < 15%: none")


def main() -> None:
    """Run the full training pipeline and persist all artifacts."""

    args = parse_args()
    try:
        results = run_pipeline(args.data_path, args.output_dir)
    except Exception:
        logger.exception("Escalation classifier training failed")
        raise

    print_final_summary(results)


if __name__ == "__main__":
    main()
