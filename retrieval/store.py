"""Lightweight local vector store for small projects and demos."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class StoreRecord:
    text: str
    embedding: List[float]
    metadata: Dict[str, str]


class LocalVectorStore:
    def __init__(self, records: Iterable[StoreRecord] | None = None) -> None:
        self._records: List[StoreRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self._records)

    def add(self, text: str, embedding: Sequence[float], metadata: Dict[str, str]) -> None:
        self._records.append(
            StoreRecord(text=text, embedding=list(embedding), metadata=dict(metadata))
        )

    def add_batch(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[Dict[str, str]],
    ) -> None:
        if not (len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError("texts, embeddings, and metadatas must have the same length")
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            self.add(text, embedding, metadata)

    def search(
        self, query_embedding: Sequence[float], top_k: int = 5
    ) -> List[Tuple[float, StoreRecord]]:
        if not self._records:
            return []
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")

        results: List[Tuple[float, StoreRecord]] = []
        for record in self._records:
            score = cosine_similarity(query_embedding, record.embedding)
            results.append((score, record))
        results.sort(key=lambda item: item[0], reverse=True)
        return results[:top_k]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = [
            {"text": rec.text, "embedding": rec.embedding, "metadata": rec.metadata}
            for rec in self._records
        ]
        path.write_text(json.dumps({"records": serializable}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalVectorStore":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        records = [
            StoreRecord(
                text=item["text"],
                embedding=list(item["embedding"]),
                metadata=dict(item["metadata"]),
            )
            for item in data.get("records", [])
        ]
        return cls(records=records)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimension for cosine similarity")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
