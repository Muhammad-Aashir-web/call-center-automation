"""Index markdown knowledge-base articles into ChromaDB.

This script reads markdown articles from the repository's data/knowledge_base
directory and stores them through VectorStoreService, which is the sole ChromaDB
access point in the codebase.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import frontmatter

from services.vector_store import VectorStoreService


logger = logging.getLogger(__name__)
KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


def load_kb_articles(kb_dir: Path) -> list[dict[str, Any]]:
    """Load markdown KB articles from disk and normalize them for indexing.

    Each article is parsed with python-frontmatter. Malformed files are skipped so
    one bad article does not prevent the rest of the knowledge base from being
    indexed.
    """

    articles: list[dict[str, Any]] = []

    for kb_file in sorted(kb_dir.glob("*.md")):
        try:
            post = frontmatter.load(kb_file)
            metadata = post.metadata
            article_id = kb_file.stem

            articles.append(
                {
                    "id": article_id,
                    "document": post.content,
                    "metadata": {
                        "title": metadata.get("title"),
                        "category": metadata.get("category"),
                        "last_updated": metadata.get("last_updated"),
                        "status": metadata.get("status"),
                    },
                }
            )
        except Exception:
            logger.exception("Failed to parse KB article: %s", kb_file)
            continue

    return articles


def index_articles() -> None:
    """Load KB articles and index them into the configured Chroma collection."""

    articles = load_kb_articles(KB_DIR)
    logger.info("Found %s KB articles in %s", len(articles), KB_DIR)

    if not articles:
        logger.warning("No KB articles found; skipping indexing.")
        return

    service = VectorStoreService()
    before_count = service.collection_count()
    logger.info("Chroma collection count before indexing: %s", before_count)

    documents = [article["document"] for article in articles]
    metadatas = [article["metadata"] for article in articles]
    ids = [article["id"] for article in articles]

    service.add_documents(documents=documents, metadatas=metadatas, ids=ids)

    after_count = service.collection_count()
    logger.info("Chroma collection count after indexing: %s", after_count)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    index_articles()