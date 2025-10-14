from pathlib import Path

import json
import pytest

from retrieval.store import LocalVectorStore


def test_search_returns_ranked_results():
    store = LocalVectorStore()
    store.add("cat", [1.0, 0.0], {"source": "cat.txt", "name": "cat.txt"})
    store.add("dog", [0.0, 1.0], {"source": "dog.txt", "name": "dog.txt"})

    results = store.search([0.9, 0.1], top_k=1)
    assert len(results) == 1
    score, record = results[0]
    assert record.metadata["name"] == "cat.txt"
    assert 0 <= score <= 1


def test_add_batch_validates_lengths():
    store = LocalVectorStore()
    with pytest.raises(ValueError):
        store.add_batch(["text"], [], [{"name": "doc.txt", "source": "doc.txt"}])


def test_save_and_load_round_trip(tmp_path: Path):
    store = LocalVectorStore()
    store.add("cat", [1.0, 0.0], {"source": "cat.txt", "name": "cat.txt"})
    store.add("dog", [0.0, 1.0], {"source": "dog.txt", "name": "dog.txt"})

    output_path = tmp_path / "index.json"
    store.save(output_path)

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(data["records"]) == 2

    loaded = LocalVectorStore.load(output_path)
    assert len(loaded.search([1.0, 0.0], top_k=1)) == 1
