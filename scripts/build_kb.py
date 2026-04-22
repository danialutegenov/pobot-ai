from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urldefrag, urlsplit, urlunsplit, quote, unquote

import requests
from sentence_transformers import SentenceTransformer

# Make `python scripts/build_kb.py` work without extra PYTHONPATH setup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import AppConfig
from app.fetch import download_source
from app.indexing import build_and_save_index
from app.preprocess import chunk_text, clean_html_to_text, extract_pdf_text, extract_topic_links
from app.sources import SEED_SOURCES


def _normalize_url(url: str) -> str:
    stripped = url.strip()
    clean_no_fragment, _ = urldefrag(stripped)
    parts = urlsplit(clean_no_fragment)
    clean_path = quote(unquote(parts.path).strip(), safe="/:%._-~")
    return urlunsplit((parts.scheme, parts.netloc, clean_path, parts.query, ""))


def _download_all_sources(config: AppConfig) -> list[dict[str, Any]]:
    session = requests.Session()
    downloaded: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    topic_counter = 0

    for source in SEED_SOURCES:
        if source["url"] in seen_urls:
            continue
        seed_record = download_source(
            session=session,
            source=source,
            raw_dir=config.raw_dir,
            manifest_path=config.manifest_path,
        )
        downloaded.append(seed_record)
        seen_urls.add(source["url"])

        if source.get("expand_links"):
            html = Path(seed_record["local_path"]).read_text(encoding="utf-8", errors="ignore")
            for link in extract_topic_links(
                html=html,
                base_url=source["url"],
                allowed_prefix=source["allowed_prefix"],
            ):
                link = _normalize_url(link)
                if link in seen_urls:
                    continue
                topic_counter += 1
                topic_source = {
                    "id": f"{source['id']}-topic-{topic_counter}",
                    "title": f"{source.get('title', source['id'])} Topic {topic_counter}",
                    "url": link,
                    "kind": "html",
                }
                try:
                    topic_record = download_source(
                        session=session,
                        source=topic_source,
                        raw_dir=config.raw_dir,
                        manifest_path=config.manifest_path,
                    )
                    downloaded.append(topic_record)
                    seen_urls.add(link)
                except requests.RequestException:
                    # Skip broken topic links from index pages and continue.
                    continue
    return downloaded


def _prepare_chunks(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_chunks: list[dict[str, Any]] = []
    for record in records:
        local_path = Path(record["local_path"])
        if local_path.suffix.lower() == ".pdf":
            text = extract_pdf_text(local_path)
        else:
            text = clean_html_to_text(local_path.read_text(encoding="utf-8", errors="ignore"))

        chunks = chunk_text(text=text, chunk_size=850, overlap=140)
        for index, chunk in enumerate(chunks, start=1):
            all_chunks.append(
                {
                    "chunk_id": f"{record['id']}-{index}",
                    "source_title": record["title"],
                    "source_url": record["url"],
                    "local_raw_path": record["local_path"],
                    "text": chunk,
                }
            )
    return all_chunks


def main() -> None:
    config = AppConfig.from_env()
    config.ensure_directories()

    downloaded_records = _download_all_sources(config)
    chunks = _prepare_chunks(downloaded_records)

    embedder = SentenceTransformer(config.embedding_model)
    build_and_save_index(
        chunks=chunks,
        embedder=embedder,
        chunks_path=config.chunks_path,
        index_path=config.index_path,
    )

    report = {
        "downloaded_sources": len(downloaded_records),
        "chunk_count": len(chunks),
        "manifest_path": str(config.manifest_path),
        "chunks_path": str(config.chunks_path),
        "index_path": str(config.index_path),
    }
    config.build_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
