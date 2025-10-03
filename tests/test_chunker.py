from ingestion.chunker import chunk_text


def test_chunk_text_respects_overlap():
    text = " ".join([f"word{i}" for i in range(200)])
    chunks = chunk_text(text, chunk_size=40, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_chunk_text_handles_empty_text():
    assert chunk_text("", chunk_size=20, overlap=5) == []


def test_chunk_text_validates_overlap():
    try:
        chunk_text("hello", chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("chunk_text should have raised ValueError")
