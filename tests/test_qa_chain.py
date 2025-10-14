from __future__ import annotations

import pytest

from llm.qa_chain import QaChain
from retrieval.store import StoreRecord


class StubEmbedder:
    def __init__(self, embedding: list[float]) -> None:
        self._embedding = embedding

    def embed_query(self, _: str) -> list[float]:
        return list(self._embedding)


class StubVectorStore:
    def __init__(self, matches: list[tuple[float, StoreRecord]]) -> None:
        self._matches = matches
        self.received_queries: list[list[float]] = []

    def search(self, query_embedding: list[float], top_k: int = 3):
        self.received_queries.append(list(query_embedding))
        return list(self._matches)[:top_k]


class NoCallClient:
    class _Completions:
        def create(self, **kwargs):
            raise AssertionError("Client should not be called")

    class _Chat:
        def __init__(self):
            self.completions = NoCallClient._Completions()

    def __init__(self):
        self.chat = NoCallClient._Chat()


class RecordingClient:
    class _Response:
        def __init__(self, content: str):
            message = type("Message", (), {"content": content})
            choice = type("Choice", (), {"message": message()})
            self.choices = [choice()]

    class _Completions:
        def __init__(self, content: str):
            self._content = content
            self.calls: list[dict] = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return RecordingClient._Response(self._content)

    class _Chat:
        def __init__(self, content: str):
            self.completions = RecordingClient._Completions(content)

    def __init__(self, content: str):
        self.chat = RecordingClient._Chat(content)


def test_qa_chain_returns_fallback_when_no_matches():
    embedder = StubEmbedder([0.1, 0.2, 0.3])
    vector_store = StubVectorStore(matches=[])
    chain = QaChain(
        vector_store=vector_store,
        embedder=embedder,
        client=NoCallClient(),
    )

    result = chain.ask("What is the capital of France?")

    assert "I do not know" in result.answer
    assert result.sources == []


def test_qa_chain_calls_client_when_matches_exist():
    record = StoreRecord(
        text="Paris is the capital of France.",
        embedding=[0.9, 0.1, 0.0],
        metadata={"name": "facts.txt"},
    )
    embedder = StubEmbedder([0.5, 0.4, 0.1])
    vector_store = StubVectorStore(matches=[(0.87, record)])
    client = RecordingClient(content="Paris is the capital of France. (facts.txt)")

    chain = QaChain(
        vector_store=vector_store,
        embedder=embedder,
        client=client,
        chat_model="test-model",
    )

    result = chain.ask("What is the capital of France?")

    assert result.answer == "Paris is the capital of France. (facts.txt)"
    assert result.sources[0][1].metadata["name"] == "facts.txt"
    assert client.chat.completions.calls[0]["model"] == "test-model"
    assert client.chat.completions.calls[0]["messages"][1]["role"] == "user"


def test_qa_chain_raises_when_embedding_generation_fails():
    embedder = StubEmbedder([])
    vector_store = StubVectorStore(matches=[])
    chain = QaChain(
        vector_store=vector_store,
        embedder=embedder,
        client=NoCallClient(),
    )

    with pytest.raises(RuntimeError):
        chain.ask("Why is the sky blue?")
