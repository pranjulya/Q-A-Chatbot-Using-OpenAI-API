"""Smoke tests for the documented ingest command (no network calls)."""
import json

from typer.testing import CliRunner

from scripts import ingest


class FakeEmbedder:
    def __init__(self, model: str = "fake") -> None:
        self.model = model

    def embed_documents(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def test_documented_ingest_command_writes_index(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "notes.txt").write_text("alpha beta gamma delta " * 20)
    out = tmp_path / "index.json"

    monkeypatch.setattr(ingest, "OpenAIEmbedder", FakeEmbedder)

    # Same shape as the README: `python -m scripts.ingest run data/raw --output ...`
    result = CliRunner().invoke(ingest.app, ["run", str(raw), "--output", str(out)])

    assert result.exit_code == 0, result.output
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data["records"]) >= 1
