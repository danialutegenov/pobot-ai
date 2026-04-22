from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


def build_and_save_index(
    chunks: list[dict[str, Any]],
    embedder: Any,
    chunks_path: Path,
    index_path: Path,
) -> None:
    if not chunks:
        raise ValueError("No chunks available to index.")

    vectors = np.asarray(embedder.encode([chunk["text"] for chunk in chunks]), dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("Embedding model returned unexpected shape.")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(index_path))

