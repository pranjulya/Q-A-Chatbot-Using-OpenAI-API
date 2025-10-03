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
