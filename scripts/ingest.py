"""CLI for ingesting documents into the local vector store."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

from ingestion.chunker import chunk_text
from ingestion.loaders import load_documents_from_directory
from retrieval.embeddings import OpenAIEmbedder, batch_embeddings
from retrieval.store import LocalVectorStore

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def run(
    input_dir: Path = typer.Argument(..., help="Directory containing source documents."),
    output_path: Path = typer.Option(
        Path("data/processed/index.json"), help="Path to write the vector store JSON file."
    ),
    embedding_model: str = typer.Option(
        "text-embedding-3-small", help="OpenAI embedding model to use."
    ),
    chunk_size: int = typer.Option(500, help="Number of words per chunk."),
    overlap: int = typer.Option(50, help="Number of overlapping words between chunks."),
    batch_size: int = typer.Option(16, help="Number of chunks to embed per API call."),
) -> None:
    load_dotenv()
    docs = load_documents_from_directory(input_dir)
    typer.echo(f"Loaded {len(docs)} documents from {input_dir}.")

    embedder = OpenAIEmbedder(model=embedding_model)
    store = LocalVectorStore()

    total_chunks = 0
    for doc in docs:
        chunks = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue
        embeddings = batch_embeddings(embedder, chunks, batch_size=batch_size)
        metadatas = []
        for idx, _chunk in enumerate(chunks):
            metadata = {
                "source": doc.metadata["source"],
                "name": doc.metadata["name"],
                "chunk_index": str(idx),
            }
            metadatas.append(metadata)
        store.add_batch(chunks, embeddings, metadatas)
        total_chunks += len(chunks)
        typer.echo(f"Processed {len(chunks)} chunks from {doc.metadata['name']}.")

    if total_chunks == 0:
        typer.echo("No chunks were generated; nothing to save.")
        raise typer.Exit(code=1)

    store.save(output_path)
    typer.echo(f"Vector store saved to {output_path} with {total_chunks} chunks.")


if __name__ == "__main__":  # pragma: no cover
    app()
