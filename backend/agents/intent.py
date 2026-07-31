"""Intent Classification Agent for the call center automation system."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from config import settings


logger = logging.getLogger(__name__)


class IntentClassificationError(RuntimeError):
    """Raised when intent model loading or inference fails."""


class IntentClassifier:
    def __init__(self) -> None:
        self.model_dir = Path(settings.INTENT_MODEL_DIR)
        self.model_dir = self.model_dir.resolve()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = float(settings.INTENT_CONFIDENCE_THRESHOLD)
        self.label2id, self.id2label = self._load_label_mappings()

        try:
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            self.model.eval()
        except Exception as exc:
            logger.exception("Failed to load intent classification model from %s", self.model_dir)
            raise IntentClassificationError(
                f"Failed to load intent classification model from {self.model_dir}"
            ) from exc

    def _load_label_mappings(self) -> tuple[dict[str, int], dict[int, str]]:
        labels_path = self.model_dir / "labels.json"
        try:
            with labels_path.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except Exception as exc:
            logger.exception("Failed to load label mappings from %s", labels_path)
            raise IntentClassificationError(
                f"Failed to load label mappings from {labels_path}"
            ) from exc

        try:
            label2id = payload["label2id"]
            id2label = payload["id2label"]

            if not isinstance(label2id, dict) or not isinstance(id2label, dict):
                raise ValueError("labels.json must contain label2id and id2label dictionaries")

            normalized_label2id = {str(label): int(index) for label, index in label2id.items()}
            normalized_id2label = {int(index): str(label) for index, label in id2label.items()}
            return normalized_label2id, normalized_id2label
        except Exception as exc:
            logger.exception("Invalid label mapping structure in %s", labels_path)
            raise IntentClassificationError(
                f"Invalid label mapping structure in {labels_path}"
            ) from exc

    def classify(self, text: str) -> dict[str, Any]:
        try:
            inputs = self.tokenizer(
                text,
                max_length=128,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)[0]

            top_index = int(torch.argmax(probabilities).item())
            confidence = float(probabilities[top_index].item())
            intent = self.id2label[top_index]
            all_scores = {
                self.id2label[index]: round(float(probabilities[index].item()), 4)
                for index in range(len(self.id2label))
            }

            return {
                "intent": intent,
                "confidence": confidence,
                "low_confidence": confidence < self.confidence_threshold,
                "all_scores": all_scores,
            }
        except Exception as exc:
            logger.exception("Intent classification failed for text: %s", text)
            raise IntentClassificationError("Intent classification failed") from exc


intent_classifier = IntentClassifier()


if __name__ == "__main__":
    examples = [
        "Why was I charged twice this month?",
        "My app keeps crashing every time I open it.",
        "Is there an outage in my area right now?",
    ]

    for example in examples:
        try:
            result = intent_classifier.classify(example)
            print(f"Input: {example}")
            print(json.dumps(result, indent=2))
            print()
        except IntentClassificationError:
            logger.exception("Manual test classification failed for example: %s", example)
