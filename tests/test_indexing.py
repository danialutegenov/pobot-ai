import json
from pathlib import Path

from app.indexing import build_and_save_index


class DummyEmbedder:
    def encode(self, texts):
        return [[float(len(text)), float(len(text.split()))] for text in texts]


def test_build_and_save_index_writes_chunks_and_faiss(tmp_path: Path) -> None:
    chunks = [
        {"chunk_id": "a", "text": "Rest days are protected.", "source_title": "EO", "source_url": "https://example.com/eo"},
        {"chunk_id": "b", "text": "Recruitment agencies may charge up to 10%.", "source_title": "EA", "source_url": "https://example.com/ea"},
    ]

    build_and_save_index(
        chunks=chunks,
        embedder=DummyEmbedder(),
        chunks_path=tmp_path / "chunks.json",
        index_path=tmp_path / "faiss.index",
    )

    saved = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert saved[0]["chunk_id"] == "a"
    assert (tmp_path / "faiss.index").exists()

