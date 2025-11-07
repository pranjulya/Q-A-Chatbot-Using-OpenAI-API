"""Streamlit interface for the Q&A chatbot."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

from ingestion.chunker import chunk_text
from ingestion.loaders import LOADERS
from llm.qa_chain import QaChain
from retrieval.embeddings import OpenAIEmbedder
from retrieval.store import LocalVectorStore, StoreRecord


def load_chain(
    vector_path: Path, embedding_model: str, chat_model: str
) -> tuple[QaChain, LocalVectorStore]:
    store = LocalVectorStore.load(vector_path)
    embedder = OpenAIEmbedder(model=embedding_model)
    chain = QaChain(vector_store=store, embedder=embedder, chat_model=chat_model)
    return chain, store


@st.cache_data
def process_uploaded_files(
    uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile],
    embedding_model: str,
    chunk_size: int,
    overlap: int,
) -> LocalVectorStore:
    with tempfile.TemporaryDirectory() as temp_dir:
        store = LocalVectorStore()
        records = []
        for uploaded_file in uploaded_files:
            path = Path(temp_dir) / uploaded_file.name
            path.write_bytes(uploaded_file.getvalue())

            loader = LOADERS.get(path.suffix.lower())
            if not loader:
                continue
            text = loader(path)
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in chunks:
                records.append(
                    StoreRecord(
                        text=chunk,
                        embedding=[],  # Will be embedded in batch
                        metadata={"source": uploaded_file.name},
                    )
                )

        # Batch embed and add to store
        texts = [rec.text for rec in records]
        embedder = OpenAIEmbedder(model=embedding_model)
        embeddings = embedder.embed_documents(texts)
        for rec, embedding in zip(records, embeddings):
            rec.embedding = embedding
        store.add_batch(
            texts=[rec.text for rec in records],
            embeddings=[rec.embedding for rec in records],
            metadatas=[rec.metadata for rec in records],
        )
        return store


def _render_sources(sources) -> None:
    for idx, (score, record) in enumerate(sources, 1):
        with st.expander(f"Source {idx} | score={score:.2f} | {record.metadata.get('source', 'unknown')}"):
            st.markdown(record.text)


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

    # Add a file uploader to the sidebar
    uploaded_files = sidebar.file_uploader(
        "Upload your documents",
        type=["txt", "md", "pdf", "docx"],
        accept_multiple_files=True,
        help="Supports .txt, .md, .pdf, and .docx files.",
    )

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

    # Handle document uploads and in-memory processing
    if uploaded_files:
        st.session_state["uploaded_files"] = uploaded_files
        chunk_size = sidebar.slider("Chunk size", min_value=100, max_value=2000, value=500)
        overlap = sidebar.slider("Overlap", min_value=0, max_value=500, value=50)

        if sidebar.button("Process Uploaded Files"):
            with st.spinner("Processing documents..."):
                store = process_uploaded_files(
                    uploaded_files,
                    embedding_model=embedding_model,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                embedder = OpenAIEmbedder(model=embedding_model)
                chain = QaChain(vector_store=store, embedder=embedder, chat_model=chat_model)
                st.session_state["chain"] = chain
                st.session_state["_store_size"] = len(store)
                st.sidebar.success(f"Processed {len(store)} chunks from uploaded files.")
        chain = st.session_state.get("chain")

    elif vector_path.exists():
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
            "Vector store file not found. Run `python scripts/ingest.py data/raw --output data/processed/index.json` "
            "or upload documents to start."
        )

    if st.session_state.get("_store_size"):
        st.sidebar.success(f"Loaded {st.session_state['_store_size']} chunks from disk.")

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
                    # History should include all messages up to the current one.
                    # The last message is the current user prompt, which is passed separately.
                    chat_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state["messages"][:-1]
                    ]
                    result = chain.ask(prompt, chat_history=chat_history, top_k=top_k)
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
