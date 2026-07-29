"""Designed to run on Google Colab with GPU, but works identically on CPU (slower)."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


LABELS = [
    "billing_inquiry",
    "technical_support",
    "service_outage_status",
    "cancel_service",
    "account_management",
    "general_inquiry",
    "sales_inquiry",
    "fraud_report",
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
MAX_LENGTH = 128
RANDOM_SEED = 42


class EncodedIntentDataset(TorchDataset):
    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_jsonl_records(data_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    try:
        with data_path.open("r", encoding="utf-8") as file_handle:
            for line_number, line in enumerate(file_handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                record = json.loads(stripped)
                text = record.get("text")
                label = record.get("label")

                if not isinstance(text, str) or not isinstance(label, str):
                    raise ValueError(f"Invalid record at line {line_number}: expected text and label strings")

                if label not in LABEL2ID:
                    raise ValueError(f"Unknown label '{label}' at line {line_number}")

                records.append({"text": text, "label": label})
    except (OSError, json.JSONDecodeError, ValueError):
        logger.exception("Failed to load training data from %s", data_path)
        raise

    return records


def tokenize_dataset(dataset: Dataset, tokenizer: DistilBertTokenizerFast) -> tuple[list[list[int]], list[list[int]], list[int]]:
    texts = dataset["text"]
    labels = [LABEL2ID[label] for label in dataset["label"]]
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    return encodings["input_ids"], encodings["attention_mask"], labels


def compute_metrics(eval_pred: object) -> dict[str, float]:
    logits = getattr(eval_pred, "predictions")
    labels = getattr(eval_pred, "label_ids")
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT for intent classification.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("../data/training/intent_training_data.jsonl"),
        help="Path to the JSONL training data file.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("../models/intent_classifier"),
        help="Directory where the final model and tokenizer will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training and evaluation batch size.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_random_seeds(RANDOM_SEED)

    try:
        records = load_jsonl_records(args.data_path)
    except Exception:
        raise

    if not records:
        logger.error("No training records found in %s", args.data_path)
        raise ValueError(f"No training records found in {args.data_path}")

    dataset = Dataset.from_list(records)
    label_ids = [LABEL2ID[label] for label in dataset["label"]]
    indices = list(range(len(dataset)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=label_ids,
    )

    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    train_input_ids, train_attention_mask, train_labels = tokenize_dataset(train_dataset, tokenizer)
    val_input_ids, val_attention_mask, val_labels = tokenize_dataset(val_dataset, tokenizer)

    train_encodings = {
        "input_ids": train_input_ids,
        "attention_mask": train_attention_mask,
    }
    val_encodings = {
        "input_ids": val_input_ids,
        "attention_mask": val_attention_mask,
    }

    train_torch_dataset = EncodedIntentDataset(train_encodings, train_labels)
    val_torch_dataset = EncodedIntentDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(args.output_dir / "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_torch_dataset,
        eval_dataset=val_torch_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    try:
        trainer.train()
    except Exception:
        logger.exception("Training failed")
        raise

    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(args.output_dir))
        tokenizer.save_pretrained(str(args.output_dir))
        with (args.output_dir / "labels.json").open("w", encoding="utf-8") as file_handle:
            json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, file_handle, indent=2)
    except (OSError, TypeError):
        logger.exception("Failed to save model artifacts to %s", args.output_dir)
        raise

    try:
        predictions = trainer.predict(val_torch_dataset)
        predicted_labels = np.argmax(predictions.predictions, axis=-1)
        report = classification_report(
            val_labels,
            predicted_labels,
            target_names=LABELS,
            digits=4,
            zero_division=0,
        )
        print("\nValidation classification report:\n")
        print(report)
    except Exception:
        logger.exception("Failed to generate validation classification report")
        raise

    print("\nTraining complete.")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
