from pathlib import Path

import pytest

from ingestion.loaders import Document, iter_files, load_documents, load_documents_from_directory


def test_iter_files_filters_supported_extensions(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "ignore.jpg").write_text("world", encoding="utf-8")

    files = list(iter_files(tmp_path))

    assert [path.name for path in files] == ["keep.txt"]


def test_load_documents_populates_metadata(tmp_path: Path) -> None:
    txt_path = tmp_path / "note.txt"
    txt_path.write_text("content", encoding="utf-8")

    docs = load_documents([txt_path])

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, Document)
    assert doc.text == "content"
    assert doc.metadata["source"] == str(txt_path)
    assert doc.metadata["name"] == "note.txt"


def test_load_documents_from_directory_requires_supported_files(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_text("binary-ish", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_documents_from_directory(tmp_path)


def test_load_documents_from_directory_returns_documents(tmp_path: Path) -> None:
    (tmp_path / "doc.md").write_text("markdown text", encoding="utf-8")
    (tmp_path / "readme.TXT").write_text("case insensitive", encoding="utf-8")

    docs = load_documents_from_directory(tmp_path)

    assert len(docs) == 2
    names = sorted(doc.metadata["name"] for doc in docs)
    assert names == ["doc.md", "readme.TXT"]
