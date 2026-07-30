"""Knowledge Retrieval Agent implementing hybrid keyword + semantic search.

This agent follows the project's design for continuous re-retrieval as topics shift
mid-call, while filtering superseded articles so live agents do not surface outdated
content that could confuse a customer interaction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import frontmatter
from rank_bm25 import BM25Okapi

from services.vector_store import VectorStoreService


logger = logging.getLogger(__name__)
KB_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge_base"


class KnowledgeRetrievalAgent:
    """Hybrid keyword + semantic retrieval over markdown knowledge-base articles."""

    def __init__(self, vector_store: VectorStoreService | None = None):
        """Initialize the vector store and build a BM25 index from the source KB."""

        self.vector_store = vector_store if vector_store is not None else VectorStoreService()
        self.bm25_corpus: list[dict[str, Any]] = []
        self.bm25_index: BM25Okapi | None = None

        try:
            self._build_bm25_index()
        except Exception:
            logger.exception("Failed to build BM25 index for knowledge retrieval")
            self.bm25_index = None
            self.bm25_corpus = []

    def _build_bm25_index(self) -> None:
        """Load KB markdown files from disk and build the BM25 corpus/index."""

        corpus: list[dict[str, Any]] = []
        tokenized_documents: list[list[str]] = []

        for kb_file in sorted(KB_DIR.glob("*.md")):
            try:
                post = frontmatter.load(kb_file)
                metadata = post.metadata
                title = str(metadata.get("title", kb_file.stem))
                category = str(metadata.get("category", ""))
                last_updated = str(metadata.get("last_updated", ""))
                status = str(metadata.get("status", ""))
                document = post.content
                tokenized_text = f"{title} {document}".lower().split()

                corpus.append(
                    {
                        "id": kb_file.stem,
                        "title": title,
                        "category": category,
                        "last_updated": last_updated,
                        "status": status,
                        "document": document,
                        "tokenized_text": tokenized_text,
                    }
                )
                tokenized_documents.append(tokenized_text)
            except Exception:
                logger.exception("Failed to parse KB article for BM25 indexing: %s", kb_file)
                continue

        self.bm25_corpus = corpus
        self.bm25_index = BM25Okapi(tokenized_documents) if tokenized_documents else None

    def _keyword_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run BM25 keyword search over the markdown KB corpus."""

        if self.bm25_index is None:
            return []

        try:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            ranked_indices = sorted(
                range(len(scores)),
                key=lambda index: scores[index],
                reverse=True,
            )[:top_k]

            results: list[dict[str, Any]] = []
            for index in ranked_indices:
                article = self.bm25_corpus[index]
                results.append(
                    {
                        "id": article["id"],
                        "title": article["title"],
                        "category": article["category"],
                        "last_updated": article["last_updated"],
                        "status": article["status"],
                        "document": article["document"],
                        "score": float(scores[index]),
                    }
                )

            return results
        except Exception:
            logger.exception("Keyword search failed for query: %s", query)
            return []

    def _semantic_search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run semantic search through the shared vector store wrapper."""

        try:
            results = self.vector_store.semantic_search(query, n_results=top_k)
            normalized_results: list[dict[str, Any]] = []

            for result in results:
                metadata = result.get("metadata") or {}
                # Lower distance means higher semantic similarity; this is not directly
                # comparable to BM25's higher-is-better score before fusion.
                normalized_results.append(
                    {
                        "id": result.get("id"),
                        "title": metadata.get("title"),
                        "category": metadata.get("category"),
                        "last_updated": metadata.get("last_updated"),
                        "status": metadata.get("status"),
                        "document": result.get("document"),
                        "score": result.get("distance"),
                    }
                )

            return normalized_results
        except Exception:
            logger.exception("Semantic search failed for query: %s", query)
            return []

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve the most relevant KB articles using hybrid keyword + semantic search."""

        try:
            semantic_results = self._semantic_search(query, top_k)
            keyword_results = self._keyword_search(query, top_k)

            combined_scores: dict[str, float] = {}
            result_lookup: dict[str, dict[str, Any]] = {}
            metadata_lookup: dict[str, dict[str, Any]] = {}

            for result_list in (semantic_results, keyword_results):
                for rank, result in enumerate(result_list, start=1):
                    article_id = result.get("id")
                    if not article_id:
                        continue

                    combined_scores[article_id] = combined_scores.get(article_id, 0.0) + 1.0 / (60.0 + rank)
                    result_lookup[article_id] = result
                    metadata_lookup[article_id] = {
                        "title": result.get("title"),
                        "category": result.get("category"),
                        "last_updated": result.get("last_updated"),
                        "status": result.get("status"),
                    }

            final_results: list[dict[str, Any]] = []
            for article_id, rrf_score in combined_scores.items():
                metadata = metadata_lookup.get(article_id, {})
                if metadata.get("status") == "superseded":
                    continue

                article = result_lookup[article_id]
                final_results.append(
                    {
                        "id": article_id,
                        "title": metadata.get("title"),
                        "category": metadata.get("category"),
                        "last_updated": metadata.get("last_updated"),
                        "document": article.get("document"),
                        "rrf_score": rrf_score,
                    }
                )

            final_results.sort(key=lambda item: item["rrf_score"], reverse=True)
            return final_results[:top_k]
        except Exception:
            logger.exception("Hybrid retrieval failed for query: %s", query)
            return []