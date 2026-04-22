from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def load_artifacts(chunks_path: Path, index_path: Path) -> tuple[list[dict[str, Any]], faiss.Index]:
    records = json.loads(chunks_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    return records, index


def retrieve_chunks(
    query: str,
    records: list[dict[str, Any]],
    index: faiss.Index,
    embedder: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    vector = np.asarray(embedder.encode([query]), dtype="float32")
    distances, indices = index.search(vector, top_k)
    hits: list[dict[str, Any]] = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        hit = dict(records[idx])
        hit["score"] = float(distances[0][rank])
        hits.append(hit)
    return hits


def select_diverse_hits(
    hits: list[dict[str, Any]],
    top_k: int,
    per_source_cap: int = 2,
    source_key: str = "source_url",
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    if per_source_cap <= 0:
        return hits[:top_k]

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    source_counts: dict[str, int] = defaultdict(int)

    # Pass 1: keep ranking order but cap repeated chunks from the same source.
    for hit in hits:
        source = str(hit.get(source_key, ""))
        if source_counts[source] < per_source_cap and len(selected) < top_k:
            selected.append(hit)
            source_counts[source] += 1
        else:
            deferred.append(hit)
        if len(selected) >= top_k:
            break

    # Pass 2: if there are not enough diverse hits, backfill by best remaining.
    if len(selected) < top_k:
        for hit in deferred:
            selected.append(hit)
            if len(selected) >= top_k:
                break
    return selected


def confidence_label(scores: list[float]) -> str:
    if not scores:
        return "Low"
    best = min(scores)
    # L2 distances from all-MiniLM-L6-v2 are typically much larger than 0.05.
    # Use practical cutoffs so good matches do not always fall back.
    if best <= 1.0:
        return "High"
    if best <= 2.0:
        return "Medium"
    return "Low"
