"""Evaluate a fine-tuned DistilBERT intent classifier on the held-out validation split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


RANDOM_SEED = 42
MAX_LENGTH = 128


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned DistilBERT intent classifier.")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("../models/intent_classifier"),
        help="Directory containing the fine-tuned model, tokenizer, and labels.json.",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path("../data/training/intent_training_data.jsonl"),
        help="Path to the JSONL dataset used for evaluation.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for evaluation.",
    )
    return parser.parse_args()


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

                records.append({"text": text, "label": label})
    except (OSError, json.JSONDecodeError, ValueError):
        logger.exception("Failed to load evaluation data from %s", data_path)
        raise

    return records


def load_label_mappings(model_dir: Path) -> tuple[dict[str, int], dict[int, str], list[str]]:
    labels_path = model_dir / "labels.json"
    try:
        with labels_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load labels.json from %s", labels_path)
        raise

    label2id = payload.get("label2id")
    id2label = payload.get("id2label")

    if not isinstance(label2id, dict) or not isinstance(id2label, dict):
        raise ValueError(f"Invalid labels.json structure in {labels_path}")

    normalized_label2id = {str(label): int(index) for label, index in label2id.items()}
    normalized_id2label = {int(index): str(label) for index, label in id2label.items()}
    labels = [normalized_id2label[index] for index in sorted(normalized_id2label)]
    return normalized_label2id, normalized_id2label, labels


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: DistilBertTokenizerFast,
    label2id: dict[str, int],
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    texts = list(dataset["text"])
    labels = [label2id[label] for label in dataset["label"]]
    encodings = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    return encodings["input_ids"], encodings["attention_mask"], labels


def print_confusion_matrix_table(matrix: np.ndarray, labels: list[str]) -> None:
    column_width = max(14, max(len(label) for label in labels) + 2)
    header = " " * column_width + "".join(f"{label:>{column_width}}" for label in labels)
    print("\nConfusion matrix (rows = true labels, columns = predicted labels):\n")
    print(header)
    for row_label, row_values in zip(labels, matrix):
        row = f"{row_label:>{column_width}}" + "".join(f"{int(value):>{column_width}d}" for value in row_values)
        print(row)


def main() -> None:
    args = parse_args()

    try:
        label2id, id2label, labels = load_label_mappings(args.model_dir)
        records = load_jsonl_records(args.data_path)
    except Exception:
        raise

    if not records:
        logger.error("No evaluation records found in %s", args.data_path)
        raise ValueError(f"No evaluation records found in {args.data_path}")

    dataset = Dataset.from_list(records)
    label_ids = [label2id[label] for label in dataset["label"]]
    indices = list(range(len(dataset)))

    _, val_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=RANDOM_SEED,
        stratify=label_ids,
    )

    val_dataset = dataset.select(val_indices)

    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    val_input_ids, val_attention_mask, val_labels = tokenize_dataset(val_dataset, tokenizer, label2id)
    val_encodings = {
        "input_ids": val_input_ids,
        "attention_mask": val_attention_mask,
    }
    val_torch_dataset = EncodedIntentDataset(val_encodings, val_labels)
    val_dataloader = DataLoader(val_torch_dataset, batch_size=args.batch_size, shuffle=False)

    predictions: list[int] = []
    true_labels: list[int] = []

    try:
        with torch.no_grad():
            for batch in val_dataloader:
                batch_labels = batch.pop("labels")
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)

                predictions.extend(batch_predictions.cpu().tolist())
                true_labels.extend(batch_labels.tolist())
    except Exception:
        logger.exception("Evaluation inference failed")
        raise

    try:
        print("\nValidation classification report:\n")
        print(
            classification_report(
                true_labels,
                predictions,
                target_names=labels,
                digits=4,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(true_labels, predictions, labels=list(range(len(labels))))
        print_confusion_matrix_table(matrix, labels)
    except Exception:
        logger.exception("Failed to compute evaluation metrics")
        raise


if __name__ == "__main__":
    main()
