import json
from collections import Counter
from pathlib import Path

import faiss
import numpy as np

from app.retrieval import confidence_label, load_artifacts, retrieve_chunks, select_diverse_hits


class DummyEmbedder:
    def encode(self, texts):
        mapping = {
            "rest day rules": [1.0, 0.0],
            "rest days are protected": [1.0, 0.0],
            "agency commission is capped at 10 percent": [0.0, 1.0],
        }
        return [mapping[text] for text in texts]


def test_retrieve_chunks_returns_best_match(tmp_path: Path) -> None:
    records = [
        {"chunk_id": "c1", "text": "rest days are protected", "source_title": "EO", "source_url": "https://example.com/eo"},
        {"chunk_id": "c2", "text": "agency commission is capped at 10 percent", "source_title": "EA", "source_url": "https://example.com/ea"},
    ]
    (tmp_path / "chunks.json").write_text(json.dumps(records), encoding="utf-8")

    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "faiss.index"))

    loaded_records, loaded_index = load_artifacts(tmp_path / "chunks.json", tmp_path / "faiss.index")
    hits = retrieve_chunks("rest day rules", loaded_records, loaded_index, DummyEmbedder(), top_k=1)

    assert hits[0]["chunk_id"] == "c1"


def test_confidence_label_maps_scores_to_buckets() -> None:
    assert confidence_label([0.8]) == "High"
    assert confidence_label([1.5]) == "Medium"
    assert confidence_label([2.5]) == "Low"


def test_select_diverse_hits_enforces_source_cap_when_possible() -> None:
    hits = [
        {"chunk_id": "a1", "source_url": "s1", "score": 0.1},
        {"chunk_id": "a2", "source_url": "s1", "score": 0.2},
        {"chunk_id": "a3", "source_url": "s1", "score": 0.3},
        {"chunk_id": "b1", "source_url": "s2", "score": 0.4},
        {"chunk_id": "b2", "source_url": "s2", "score": 0.5},
        {"chunk_id": "c1", "source_url": "s3", "score": 0.6},
    ]
    selected = select_diverse_hits(hits, top_k=5, per_source_cap=2)
    counts = Counter(item["source_url"] for item in selected)
    assert len(selected) == 5
    assert counts["s1"] <= 2
    assert counts["s2"] <= 2


def test_select_diverse_hits_backfills_when_diversity_insufficient() -> None:
    hits = [
        {"chunk_id": "a1", "source_url": "same", "score": 0.1},
        {"chunk_id": "a2", "source_url": "same", "score": 0.2},
        {"chunk_id": "a3", "source_url": "same", "score": 0.3},
    ]
    selected = select_diverse_hits(hits, top_k=3, per_source_cap=1)
    assert len(selected) == 3
