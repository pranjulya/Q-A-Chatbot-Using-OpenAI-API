"""Utilities for chunking text into overlapping windows."""
from __future__ import annotations

from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = normalize_whitespace(text).split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


def chunk_documents(texts: Iterable[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks: List[str] = []
    for text in texts:
        chunks.extend(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    return chunks
