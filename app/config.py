from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    manifest_path: Path
    chunks_path: Path
    index_path: Path
    build_report_path: Path
    embedding_model: str
    chat_model: str
    top_k: int
    deepseek_api_key: str
    deepseek_base_url: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        project_root = Path.cwd()
        raw_dir = project_root / "data" / "raw"
        processed_dir = project_root / "data" / "processed"
        return cls(
            project_root=project_root,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            manifest_path=raw_dir / "manifest.json",
            chunks_path=processed_dir / "chunks.json",
            index_path=processed_dir / "faiss.index",
            build_report_path=processed_dir / "build_report.json",
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            top_k=int(os.getenv("TOP_K", "5")),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )

    def ensure_directories(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

