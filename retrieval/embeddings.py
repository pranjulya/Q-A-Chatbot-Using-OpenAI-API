"""Embedding utilities backed by the OpenAI API."""
from __future__ import annotations

from typing import Iterable, List, Sequence

from openai import OpenAI


class OpenAIEmbedder:
    """Thin wrapper around the OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.client = OpenAI()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        embedding = self.embed_documents([text])
        return embedding[0] if embedding else []


def batch_embeddings(embedder: OpenAIEmbedder, texts: Iterable[str], batch_size: int = 16) -> List[List[float]]:
    batch: List[str] = []
    vectors: List[List[float]] = []
    for text in texts:
        batch.append(text)
        if len(batch) == batch_size:
            vectors.extend(embedder.embed_documents(batch))
            batch.clear()
    if batch:
        vectors.extend(embedder.embed_documents(batch))
    return vectors
