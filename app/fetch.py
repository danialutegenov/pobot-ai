from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_manifest(manifest_path: Path, records: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def download_source(
    session: requests.Session,
    source: dict[str, Any],
    raw_dir: Path,
    manifest_path: Path,
    timeout: int = 30,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = session.get(source["url"], timeout=timeout)
    response.raise_for_status()

    ext = ".pdf" if source.get("kind") == "pdf" else ".html"
    file_name = f"{_safe_id(source['id'])}{ext}"
    file_path = raw_dir / file_name
    payload = response.content if ext == ".pdf" else response.text.encode("utf-8", errors="ignore")
    file_path.write_bytes(payload)

    record = {
        "id": source["id"],
        "title": source.get("title", source["id"]),
        "url": source["url"],
        "kind": source.get("kind", "html"),
        "content_type": response.headers.get("content-type", ""),
        "local_path": str(file_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest = _load_manifest(manifest_path)
    manifest = [item for item in manifest if item["url"] != source["url"]]
    manifest.append(record)
    _write_manifest(manifest_path, manifest)
    return record

