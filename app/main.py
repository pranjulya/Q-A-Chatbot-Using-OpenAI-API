"""Streamlit interface for the Q&A chatbot."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from llm.qa_chain import QaChain
from retrieval.embeddings import OpenAIEmbedder
from retrieval.store import LocalVectorStore


def load_chain(
    vector_path: Path, embedding_model: str, chat_model: str
) -> tuple[QaChain, LocalVectorStore]:
    store = LocalVectorStore.load(vector_path)
    embedder = OpenAIEmbedder(model=embedding_model)
    chain = QaChain(vector_store=store, embedder=embedder, chat_model=chat_model)
    return chain, store


def _render_sources(sources) -> None:
    for idx, (score, record) in enumerate(sources, 1):
        st.caption(
            f"Source {idx} • score={score:.2f} • {record.metadata.get('source', 'unknown')}"
        )


def run() -> None:
    load_dotenv()
    st.set_page_config(page_title="Open Source Q&A Chatbot", layout="wide")
    st.title("Q&A Chatbot (Open Source Edition)")
    st.write(
        "Ask questions grounded in your uploaded documents. Make sure you run the ingestion CLI "
        "to build a vector store before chatting."
    )

    sidebar = st.sidebar
    sidebar.header("Configuration")
    default_vector_path = "data/processed/index.json"
    vector_path_str = sidebar.text_input("Vector store path", value=default_vector_path)
    embedding_model = sidebar.text_input(
        "Embedding model", value=st.session_state.get("embedding_model", "text-embedding-3-small")
    )
    chat_model = sidebar.text_input(
        "Chat model", value=st.session_state.get("chat_model", "gpt-4o-mini")
    )
    top_k = sidebar.slider("Top K context chunks", min_value=1, max_value=10, value=3)

    vector_path = Path(vector_path_str)
    chain: Optional[QaChain] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if vector_path.exists():
        config_changed = (
            st.session_state.get("_config")
            != (str(vector_path.resolve()), embedding_model, chat_model)
        )
        if config_changed:
            try:
                chain, store = load_chain(vector_path, embedding_model, chat_model)
                st.session_state["chain"] = chain
                st.session_state["_store_size"] = len(store)
                st.session_state["embedding_model"] = embedding_model
                st.session_state["chat_model"] = chat_model
                st.session_state["_config"] = (
                    str(vector_path.resolve()),
                    embedding_model,
                    chat_model,
                )
            except FileNotFoundError:
                st.error("Vector store file not found. Run the ingestion script first.")
            except Exception as exc:  # pragma: no cover - UI feedback
                st.error(f"Failed to load vector store: {exc}")
        chain = st.session_state.get("chain")
    else:
        st.warning(
            "Vector store file not found. Run `python scripts/ingest.py data/raw --output data/processed/index.json`."
        )

    if st.session_state.get("_store_size"):
        st.sidebar.success(f"Loaded {st.session_state['_store_size']} chunks.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                _render_sources(message["sources"])

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not chain:
            st.warning("Load a vector store before asking questions.")
        else:
            with st.chat_message("assistant"):
                try:
                    result = chain.ask(prompt, top_k=top_k)
                    st.markdown(result.answer)
                    _render_sources(result.sources)
                    st.session_state["messages"].append(
                        {
                            "role": "assistant",
                            "content": result.answer,
                            "sources": result.sources,
                        }
                    )
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(f"Something went wrong: {exc}")


if __name__ == "__main__":  # pragma: no cover
    run()
