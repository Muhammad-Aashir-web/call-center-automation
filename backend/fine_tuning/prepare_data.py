"""Generate synthetic training data for intent classification."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path

from services.llm import llm_client


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CATEGORIES: dict[str, str] = {
    "billing_inquiry": (
        "questions about charges, invoices, payment methods, why they were billed a certain amount"
    ),
    "technical_support": (
        "something isn't working (device, app, connection quality) and the caller wants help troubleshooting their own specific issue"
    ),
    "service_outage_status": (
        "caller is asking whether there's a known/wider outage and when service will be restored, not asking for individual troubleshooting"
    ),
    "cancel_service": "caller wants to cancel or downgrade their service/plan",
    "account_management": (
        "routine account changes — updating personal info, password resets, changing plan tier"
    ),
    "general_inquiry": (
        "general questions about hours, policies, locations, non-technical non-billing questions"
    ),
    "sales_inquiry": "caller is interested in new products, upgrades, or add-on services",
    "fraud_report": (
        "caller is reporting unauthorized charges, suspected account takeover, or a scam/fraud attempt against their account"
    ),
}


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


async def generate_batch(
    category: str,
    description: str,
    batch_size: int,
    existing_examples: list[str],
) -> list[str]:
    prompt = f"""
Generate {batch_size} realistic, varied customer utterances for the intent category '{category}'.

Category scope:
{description}

Requirements for the utterances:
- Write first-person natural speech as if transcribed from a real phone call
- Mix short and long utterances
- Mix formal and casual language
- Include some filler words, hesitations, and occasional typos
- Use different phrasings and perspectives
- Keep each utterance focused on this intent category
- Do not repeat similar phrasing from prior examples

Existing examples for this category to avoid repeating:
{json.dumps(existing_examples, ensure_ascii=False, indent=2) if existing_examples else "[]"}

Return ONLY a JSON array of strings.
Do not include markdown code fences, explanations, numbering, or any other text.
""".strip()

    system_prompt = (
        "You generate high-quality synthetic utterances for intent classification training. "
        "Respond with valid JSON only."
    )

    try:
        response = await llm_client.complete(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.9,
            max_tokens=1500,
        )
    except Exception:
        logger.exception("LLM request failed for category %s", category)
        return []

    try:
        parsed = json.loads(_strip_markdown_fences(response))
        if not isinstance(parsed, list):
            raise ValueError("LLM response was not a JSON array")

        examples: list[str] = []
        for item in parsed:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    examples.append(stripped)
        return examples
    except Exception:
        logger.exception("Failed to parse JSON response for category %s", category)
        return []


async def generate_category_examples(
    category: str,
    description: str,
    target_count: int = 80,
) -> list[str]:
    unique_examples: list[str] = []
    seen: set[str] = set()
    batch_attempts = 0

    while len(unique_examples) < target_count and batch_attempts < 15:
        batch_attempts += 1
        batch = await generate_batch(
            category=category,
            description=description,
            batch_size=15,
            existing_examples=unique_examples,
        )

        for example in batch:
            normalized = example.strip().lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                unique_examples.append(example)

        print(f"[{category}] {len(unique_examples)}/{target_count} examples generated")

    if len(unique_examples) < target_count:
        logger.warning(
            "Stopped short for %s: generated %s/%s examples after %s batch attempts",
            category,
            len(unique_examples),
            target_count,
            batch_attempts,
        )

    return unique_examples[:target_count]


async def main() -> None:
    all_examples: list[tuple[str, str]] = []
    per_category_counts: dict[str, int] = {}

    for category, description in CATEGORIES.items():
        examples = await generate_category_examples(category, description)
        per_category_counts[category] = len(examples)
        all_examples.extend((example, category) for example in examples)

    random.shuffle(all_examples)

    output_path = Path(__file__).resolve().parents[2] / "data" / "training" / "intent_training_data.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file_handle:
        for text, label in all_examples:
            json.dump({"text": text, "label": label}, file_handle, ensure_ascii=False)
            file_handle.write("\n")

    print("\nGeneration complete:")
    for category, count in per_category_counts.items():
        print(f"- {category}: {count}")
    print(f"- total: {len(all_examples)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
