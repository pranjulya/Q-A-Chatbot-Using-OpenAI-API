"""High-level orchestration for retrieval-augmented QA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from openai import OpenAI

from retrieval.embeddings import OpenAIEmbedder
from retrieval.store import LocalVectorStore, StoreRecord

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant who answers questions using only the provided "
    "context. Cite sources in parentheses using the metadata 'name' when relevant. If "
    "the answer is not in the context, say you do not know."
)


@dataclass
class QaResult:
    answer: str
    sources: List[tuple[float, StoreRecord]]


class QaChain:
    def __init__(
        self,
        vector_store: LocalVectorStore,
        embedder: OpenAIEmbedder,
        chat_model: str = "gpt-4o-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client: OpenAI | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.client = client or OpenAI()

    def _format_context(self, sources: Sequence[tuple[float, StoreRecord]]) -> str:
        formatted = []
        for idx, (score, record) in enumerate(sources, 1):
            formatted.append(
                f"Source {idx} | score={score:.2f} | {record.metadata.get('name', 'unknown')}\n{record.text}"
            )
        return "\n\n".join(formatted)

    def ask(self, question: str, top_k: int = 3) -> QaResult:
        query_embedding = self.embedder.embed_query(question)
        if not query_embedding:
            raise RuntimeError("Failed to generate embedding for the supplied question.")
        matches = self.vector_store.search(query_embedding, top_k=top_k)
        if not matches:
            return QaResult(
                answer="I do not know. No relevant context chunks were retrieved from the vector store.",
                sources=[],
            )
        context_block = self._format_context(matches)

        user_prompt = (
            "You must ground every answer in the provided sources."
            "\n\nContext:\n" + context_block + f"\n\nQuestion: {question}\nAnswer:"
        )

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return QaResult(answer=answer, sources=list(matches))
