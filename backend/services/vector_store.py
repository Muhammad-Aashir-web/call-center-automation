"""Thin ChromaDB access wrapper.

This module is the sole direct ChromaDB access point in the codebase. Keeping all
embedded vector-store interaction behind this abstraction makes it possible to
swap to a managed vector database later with changes isolated to this file.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import settings


logger = logging.getLogger(__name__)


class VectorStoreService:
    """Thin wrapper around embedded ChromaDB for document storage and retrieval."""

    def __init__(self) -> None:
        """Initialize the persistent Chroma client and collection.

        This is a startup-critical dependency. If initialization fails, the error is
        logged and re-raised so the application can fail fast rather than running
        without retrieval capability.
        """

        try:
            persist_dir = settings.CHROMA_PERSIST_DIR
            collection_name = settings.CHROMA_COLLECTION_NAME
            embedding_function = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
            )
        except Exception:
            logger.exception("Failed to initialize ChromaDB vector store")
            raise

    def add_documents(self, documents: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """Add documents, metadata, and ids to the backing Chroma collection.

        Raises:
            ValueError: If the provided lists are not all the same length.
        """

        if not (len(documents) == len(metadatas) == len(ids)):
            raise ValueError(
                "documents, metadatas, and ids must all have the same length"
            )

        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception:
            logger.exception("Failed to add documents to ChromaDB collection")
            raise

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[dict[str, Any]]:
        """Run a one-shot semantic search and normalize the Chroma response.

        Returns:
            A flat list of result dictionaries containing id, document, metadata, and
            distance. If retrieval fails, an empty list is returned so downstream call
            flows can degrade gracefully.
        """

        try:
            query_kwargs: dict[str, Any] = {
                "query_texts": [query],
                "n_results": n_results,
            }
            if where is not None:
                query_kwargs["where"] = where

            results = self.collection.query(**query_kwargs)

            ids = results.get("ids", [[]])
            documents = results.get("documents", [[]])
            metadatas = results.get("metadatas", [[]])
            distances = results.get("distances", [[]])

            first_ids = ids[0] if ids else []
            first_documents = documents[0] if documents else []
            first_metadatas = metadatas[0] if metadatas else []
            first_distances = distances[0] if distances else []

            normalized_results: list[dict[str, Any]] = []
            for item_id, document, metadata, distance in zip(
                first_ids,
                first_documents,
                first_metadatas,
                first_distances,
            ):
                normalized_results.append(
                    {
                        "id": item_id,
                        "document": document,
                        "metadata": metadata,
                        "distance": distance,
                    }
                )

            return normalized_results
        except Exception:
            logger.exception("Failed semantic search against ChromaDB collection")
            return []

    def collection_count(self) -> int:
        """Return the number of stored vectors in the collection.

        Returns 0 if the count operation fails so health checks can fail gracefully.
        """

        try:
            return self.collection.count()
        except Exception:
            logger.exception("Failed to count documents in ChromaDB collection")
            return 0
