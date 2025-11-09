"""High-level orchestration for retrieval-augmented QA."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Sequence

from openai import OpenAI

from retrieval.embeddings import OpenAIEmbedder
from retrieval.store import LocalVectorStore, StoreRecord

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful teaching assistant who answers questions using only the provided "
    "context. Use the conversation history to answer follow-up questions. Cite sources "
    "in parentheses using the metadata 'name' when relevant. If the answer is not in the "
    "context, say you do not know."
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
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.chat_model = chat_model
        self.system_prompt = system_prompt
        self.client = OpenAI()

    def _format_context(self, sources: Sequence[tuple[float, StoreRecord]]) -> str:
        formatted = []
        for idx, (score, record) in enumerate(sources, 1):
            formatted.append(
                f"Source {idx} | score={score:.2f} | {record.metadata.get('name', 'unknown')}\n{record.text}"
            )
        return "\n\n".join(formatted)

    def _rerank_documents(
        self, question: str, sources: List[tuple[float, StoreRecord]]
    ) -> List[tuple[float, StoreRecord]]:
        if not sources:
            return []

        # Prepare the documents for the re-ranking prompt
        doc_texts = [
            f"ID: {idx}\nText: {record.text}" for idx, (_, record) in enumerate(sources)
        ]
        docs_str = "\n\n".join(doc_texts)

        # Create a prompt for the LLM to score the documents
        prompt = (
            f"Given the question '{question}', please score the following documents for relevance "
            "from 1 (not relevant) to 10 (highly relevant). Respond with a JSON object where "
            "keys are the document IDs and values are their scores.\n\n"
            f"Documents:\n{docs_str}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            scores = json.loads(response.choices[0].message.content)

            # Sort the original sources based on the new scores
            reranked_sources = sorted(
                sources,
                key=lambda item: scores.get(str(sources.index(item)), 0),
                reverse=True,
            )
            return reranked_sources
        except (json.JSONDecodeError, KeyError):
            # Fallback to original order if re-ranking fails
            return sources

    def ask(
        self,
        question: str,
        chat_history: List[dict] | None = None,
        top_k: int = 3,
        rerank: bool = False,
    ) -> QaResult:
        chat_history = chat_history or []
        query_embedding = self.embedder.embed_query(question)

        # Retrieve a larger pool of documents if re-ranking is enabled
        initial_k = top_k * 3 if rerank else top_k
        matches = self.vector_store.search(query_embedding, top_k=initial_k)

        # Re-rank if enabled
        if rerank:
            matches = self._rerank_documents(question, list(matches))

        # Select the top_k documents after potential re-ranking
        final_matches = matches[:top_k]
        context_block = self._format_context(final_matches)

        user_prompt = (
            "You must ground every answer in the provided sources."
            "\n\nContext:\n" + context_block + f"\n\nQuestion: {question}\nAnswer:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            *chat_history,
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return QaResult(answer=answer, sources=list(final_matches))
