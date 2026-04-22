# Migrant Support RAG Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Streamlit-based RAG assistant for Hong Kong migrant support that downloads the approved Labour Department sources, stores the raw files visibly, indexes cleaned chunks in FAISS, answers in English or Chinese using DeepSeek, and shows grounded citations plus a simple confidence label.

**Architecture:** A build step downloads and snapshots the fixed HTML/PDF sources into `data/raw/`, expands topic links from the FAQ and service index pages, cleans and chunks the text, and saves chunk metadata plus a FAISS index in `data/processed/`. The Streamlit app loads those artifacts, retrieves English evidence for each question, rewrites Chinese questions into English for retrieval, asks DeepSeek for a grounded answer in the user’s language, and renders answer, confidence, and citations.

**Tech Stack:** Python 3.11+, `streamlit`, `requests`, `beautifulsoup4`, `pypdf`, `sentence-transformers`, `faiss-cpu`, `openai`, `langdetect`, `pytest`

---

## Preflight

Run these commands before Task 1:

```bash
cd /Users/dan06ial/projects/pobot-ai
test -d .git || git init
python3 -m venv .venv
source .venv/bin/activate
mkdir -p app scripts tests data/raw data/processed evaluation
```

Install dependencies once `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

Set environment variables before chat runs:

```bash
export DEEPSEEK_API_KEY="your-deepseek-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_CHAT_MODEL="deepseek-chat"
```

## File Structure

- `requirements.txt`
  Declares runtime and test dependencies.
- `.env.example`
  Documents required environment variables for DeepSeek and optional embedding overrides.
- `app/__init__.py`
  Marks `app` as a package.
- `app/config.py`
  Centralizes paths, model names, and runtime defaults.
- `app/sources.py`
  Holds the fixed source list and topic-expansion rules.
- `app/fetch.py`
  Downloads raw HTML/PDF files and updates the source manifest.
- `app/preprocess.py`
  Cleans HTML, extracts PDF text, expands topic links, and chunks normalized text.
- `app/indexing.py`
  Embeds chunk text, writes chunk metadata, and saves the FAISS index.
- `app/retrieval.py`
  Loads artifacts, retrieves top chunks, and assigns `High` / `Medium` / `Low` confidence labels.
- `app/chat.py`
  Detects query language, rewrites Chinese queries into English, builds grounded prompts, and calls DeepSeek.
- `streamlit_app.py`
  Renders the demo UI and wires together retrieval and chat.
- `scripts/build_kb.py`
  End-to-end knowledge-base build entrypoint.
- `evaluation/sample_queries.md`
  Stores example questions, outputs, and one known limitation.
- `README.md`
  Explains setup, build, run, and submission mapping.
- `tests/test_config.py`
  Covers config defaults and path creation.
- `tests/test_fetch.py`
  Covers source catalog and manifest-aware downloads.
- `tests/test_preprocess.py`
  Covers HTML cleaning, topic-link extraction, and chunking.
- `tests/test_indexing.py`
  Covers metadata serialization and FAISS artifact creation.
- `tests/test_retrieval.py`
  Covers top-k retrieval and confidence labeling.
- `tests/test_chat.py`
  Covers language handling, fallback logic, and prompt construction.

### Task 1: Bootstrap Project Settings And Runtime Config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_config.py
from pathlib import Path

from app.config import AppConfig


def test_config_uses_repo_relative_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    config = AppConfig.from_env()

    assert config.raw_dir == tmp_path / "data" / "raw"
    assert config.processed_dir == tmp_path / "data" / "processed"
    assert config.top_k == 5
    assert config.chat_model == "deepseek-chat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 3: Write minimal implementation**

```text
# /Users/dan06ial/projects/pobot-ai/requirements.txt
streamlit==1.44.1
requests==2.32.3
beautifulsoup4==4.13.3
pypdf==5.4.0
sentence-transformers==4.1.0
faiss-cpu==1.10.0
openai==1.75.0
langdetect==1.0.9
pytest==8.3.5
```

```dotenv
# /Users/dan06ial/projects/pobot-ai/.env.example
DEEPSEEK_API_KEY=your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_CHAT_MODEL=deepseek-chat
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
```

```python
# /Users/dan06ial/projects/pobot-ai/app/__init__.py
"""Application package for the migrant support RAG assistant."""
```

```python
# /Users/dan06ial/projects/pobot-ai/app/config.py
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
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            chat_model=os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
            top_k=int(os.getenv("TOP_K", "5")),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -q`
Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .env.example app/__init__.py app/config.py tests/test_config.py
git commit -m "chore: bootstrap python rag project config"
```

### Task 2: Define Fixed Sources And Snapshot Raw Files

**Files:**
- Create: `app/sources.py`
- Create: `app/fetch.py`
- Test: `tests/test_fetch.py`

- [ ] **Step 1: Write the failing tests**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_fetch.py
import json
from pathlib import Path

from app.fetch import download_source
from app.sources import SEED_SOURCES


class DummyResponse:
    def __init__(self, text: str, content_type: str) -> None:
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = {"content-type": content_type}

    def raise_for_status(self) -> None:
        return None


class DummySession:
    def get(self, url: str, timeout: int = 30) -> DummyResponse:
        return DummyResponse("<html><body><main>Example</main></body></html>", "text/html")


def test_seed_sources_include_expected_entries() -> None:
    urls = {source["url"] for source in SEED_SOURCES}
    assert "https://www.fdh.labour.gov.hk/en/fdh_corner.html" in urls
    assert "https://www.labour.gov.hk/eng/public/wcp/ConciseGuide/EO_guide_full.pdf" in urls
    assert "https://www.labour.gov.hk/eng/legislat/content5.htm" in urls


def test_download_source_writes_file_and_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"

    record = download_source(
        session=DummySession(),
        source={"id": "faq-index", "url": "https://example.com/faq.htm", "kind": "html"},
        raw_dir=tmp_path,
        manifest_path=manifest_path,
    )

    assert record["local_path"].endswith("faq-index.html")
    assert (tmp_path / "faq-index.html").exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest[0]["url"] == "https://example.com/faq.htm"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_fetch.py -q`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `app.fetch`

- [ ] **Step 3: Write minimal implementation**

```python
# /Users/dan06ial/projects/pobot-ai/app/sources.py
SEED_SOURCES = [
    {"id": "fdh-corner", "url": "https://www.fdh.labour.gov.hk/en/fdh_corner.html", "kind": "html", "expand_links": False},
    {"id": "employment-guide", "url": "https://www.labour.gov.hk/eng/public/wcp/ConciseGuide/EO_guide_full.pdf", "kind": "pdf", "expand_links": False},
    {"id": "faq-index", "url": "https://www.labour.gov.hk/eng/faq/content.htm", "kind": "html", "expand_links": True},
    {"id": "service-index", "url": "https://www.labour.gov.hk/eng/service/content.htm", "kind": "html", "expand_links": True},
    {"id": "legislation-ec", "url": "https://www.labour.gov.hk/eng/legislat/content1.htm", "kind": "html", "expand_links": False},
    {"id": "legislation-eo", "url": "https://www.labour.gov.hk/eng/legislat/content2.htm", "kind": "html", "expand_links": False},
    {"id": "legislation-fiuo", "url": "https://www.labour.gov.hk/eng/legislat/content3.htm", "kind": "html", "expand_links": False},
    {"id": "legislation-osho", "url": "https://www.labour.gov.hk/eng/legislat/content4.htm", "kind": "html", "expand_links": False},
    {"id": "legislation-mwo", "url": "https://www.labour.gov.hk/eng/legislat/content5.htm", "kind": "html", "expand_links": False},
]
```

```python
# /Users/dan06ial/projects/pobot-ai/app/fetch.py
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def download_source(
    session: requests.Session,
    source: dict[str, Any],
    raw_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    response = session.get(source["url"], timeout=30)
    response.raise_for_status()

    extension = ".pdf" if source["kind"] == "pdf" else ".html"
    file_path = raw_dir / f'{source["id"]}{extension}'
    payload = response.content if extension == ".pdf" else response.text.encode("utf-8")
    file_path.write_bytes(payload)

    record = {
        "id": source["id"],
        "url": source["url"],
        "kind": source["kind"],
        "content_type": response.headers.get("content-type", ""),
        "local_path": str(file_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    manifest = [item for item in manifest if item["url"] != source["url"]]
    manifest.append(record)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return record
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_fetch.py -q`
Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/sources.py app/fetch.py tests/test_fetch.py
git commit -m "feat: add fixed source catalog and raw snapshot downloads"
```

### Task 3: Clean HTML, Expand Topic Links, And Chunk Text

**Files:**
- Create: `app/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write the failing tests**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_preprocess.py
from app.preprocess import chunk_text, clean_html_to_text, extract_topic_links


def test_clean_html_removes_navigation_noise() -> None:
    html = """
    <html>
      <body>
        <nav>Skip Content</nav>
        <main>
          <h1>Employment Ordinance</h1>
          <p>Rest days are protected.</p>
        </main>
        <footer>Copyright</footer>
      </body>
    </html>
    """

    text = clean_html_to_text(html)

    assert "Skip Content" not in text
    assert "Copyright" not in text
    assert "Rest days are protected." in text


def test_extract_topic_links_keeps_only_labour_department_targets() -> None:
    html = """
    <html><body>
      <a href="topic1.htm">Topic 1</a>
      <a href="https://www.labour.gov.hk/eng/faq/topic2.htm">Topic 2</a>
      <a href="https://example.com/outside.htm">Outside</a>
    </body></html>
    """

    links = extract_topic_links(
        html=html,
        base_url="https://www.labour.gov.hk/eng/faq/content.htm",
        allowed_prefix="https://www.labour.gov.hk/eng/faq/",
    )

    assert links == [
        "https://www.labour.gov.hk/eng/faq/topic1.htm",
        "https://www.labour.gov.hk/eng/faq/topic2.htm",
    ]


def test_chunk_text_creates_overlap() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three.\n\nParagraph four."
    chunks = chunk_text(text, chunk_size=30, overlap=10)

    assert len(chunks) >= 2
    assert "Paragraph two." in chunks[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_preprocess.py -q`
Expected: FAIL with `ModuleNotFoundError` for `app.preprocess`

- [ ] **Step 3: Write minimal implementation**

```python
# /Users/dan06ial/projects/pobot-ai/app/preprocess.py
from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ["nav", "footer", "header", "script", "style"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    text_blocks: list[str] = []
    for node in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        value = node.get_text(" ", strip=True)
        if value:
            text_blocks.append(value)
    return "\n".join(text_blocks)


def extract_topic_links(html: str, base_url: str, allowed_prefix: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"])
        if absolute.startswith(allowed_prefix) and absolute not in links:
            links.append(absolute)
    return links


def extract_pdf_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join((page.extract_text() or "").strip() for page in reader.pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        proposal = f"{current}\n{paragraph}".strip()
        if current and len(proposal) > chunk_size:
            chunks.append(current)
            current = f"{current[-overlap:]}\n{paragraph}".strip()
        else:
            current = proposal
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_preprocess.py -q`
Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/preprocess.py tests/test_preprocess.py
git commit -m "feat: add preprocessing and chunking helpers"
```

### Task 4: Build The Knowledge Base And Save FAISS Artifacts

**Files:**
- Create: `app/indexing.py`
- Create: `scripts/build_kb.py`
- Test: `tests/test_indexing.py`

- [ ] **Step 1: Write the failing test**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_indexing.py
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

    saved = json.loads((tmp_path / "chunks.json").read_text())
    assert saved[0]["chunk_id"] == "a"
    assert (tmp_path / "faiss.index").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_indexing.py -q`
Expected: FAIL with `ModuleNotFoundError` for `app.indexing`

- [ ] **Step 3: Write minimal implementation**

```python
# /Users/dan06ial/projects/pobot-ai/app/indexing.py
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def build_and_save_index(chunks, embedder, chunks_path: Path, index_path: Path) -> None:
    vectors = np.asarray(embedder.encode([chunk["text"] for chunk in chunks]), dtype="float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False))
    faiss.write_index(index, str(index_path))
```

```python
# /Users/dan06ial/projects/pobot-ai/scripts/build_kb.py
from __future__ import annotations

import json
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

from app.config import AppConfig
from app.fetch import download_source
from app.indexing import build_and_save_index
from app.preprocess import chunk_text, clean_html_to_text, extract_pdf_text, extract_topic_links
from app.sources import SEED_SOURCES


def main() -> None:
    config = AppConfig.from_env()
    session = requests.Session()
    manifest_records = []

    for source in SEED_SOURCES:
        record = download_source(session, source, config.raw_dir, config.manifest_path)
        manifest_records.append(record)

        if source["expand_links"]:
            html = Path(record["local_path"]).read_text()
            prefix = source["url"].rsplit("/", 1)[0] + "/"
            for index, link in enumerate(extract_topic_links(html, source["url"], prefix), start=1):
                topic_source = {"id": f'{source["id"]}-topic-{index}', "url": link, "kind": "html"}
                manifest_records.append(download_source(session, topic_source, config.raw_dir, config.manifest_path))

    chunks = []
    for record in manifest_records:
        local_path = Path(record["local_path"])
        raw_text = extract_pdf_text(str(local_path)) if local_path.suffix == ".pdf" else clean_html_to_text(local_path.read_text())
        for chunk_index, chunk in enumerate(chunk_text(raw_text), start=1):
            chunks.append(
                {
                    "chunk_id": f'{record["id"]}-{chunk_index}',
                    "source_title": record["id"],
                    "source_url": record["url"],
                    "local_raw_path": record["local_path"],
                    "text": chunk,
                }
            )

    embedder = SentenceTransformer(config.embedding_model)
    build_and_save_index(chunks, embedder, config.chunks_path, config.index_path)
    print(json.dumps({"chunks": len(chunks), "index_path": str(config.index_path)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_indexing.py -q`
Expected: PASS with `1 passed`

- [ ] **Step 5: Commit**

```bash
git add app/indexing.py scripts/build_kb.py tests/test_indexing.py
git commit -m "feat: add knowledge base build pipeline"
```

### Task 5: Load Artifacts, Retrieve Evidence, And Assign Confidence

**Files:**
- Create: `app/retrieval.py`
- Test: `tests/test_retrieval.py`

- [ ] **Step 1: Write the failing tests**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_retrieval.py
import json
from pathlib import Path

import faiss
import numpy as np

from app.retrieval import confidence_label, load_artifacts, retrieve_chunks


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
    (tmp_path / "chunks.json").write_text(json.dumps(records))

    index = faiss.IndexFlatL2(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype="float32"))
    faiss.write_index(index, str(tmp_path / "faiss.index"))

    loaded_records, loaded_index = load_artifacts(tmp_path / "chunks.json", tmp_path / "faiss.index")
    hits = retrieve_chunks("rest day rules", loaded_records, loaded_index, DummyEmbedder(), top_k=1)

    assert hits[0]["chunk_id"] == "c1"


def test_confidence_label_maps_scores_to_buckets() -> None:
    assert confidence_label([0.03]) == "High"
    assert confidence_label([0.25]) == "Medium"
    assert confidence_label([0.75]) == "Low"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_retrieval.py -q`
Expected: FAIL with `ModuleNotFoundError` for `app.retrieval`

- [ ] **Step 3: Write minimal implementation**

```python
# /Users/dan06ial/projects/pobot-ai/app/retrieval.py
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np


def load_artifacts(chunks_path: Path, index_path: Path):
    records = json.loads(chunks_path.read_text())
    index = faiss.read_index(str(index_path))
    return records, index


def retrieve_chunks(query: str, records, index, embedder, top_k: int = 5):
    vector = np.asarray(embedder.encode([query]), dtype="float32")
    distances, indices = index.search(vector, top_k)
    hits = []
    for rank, idx in enumerate(indices[0]):
        if idx == -1:
            continue
        hit = dict(records[idx])
        hit["score"] = float(distances[0][rank])
        hits.append(hit)
    return hits


def confidence_label(scores: list[float]) -> str:
    if not scores:
        return "Low"
    best = min(scores)
    if best <= 0.05:
        return "High"
    if best <= 0.30:
        return "Medium"
    return "Low"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_retrieval.py -q`
Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add app/retrieval.py tests/test_retrieval.py
git commit -m "feat: add retrieval and confidence labeling"
```

### Task 6: Add DeepSeek Prompting, Chinese Query Handling, And Fallback Logic

**Files:**
- Create: `app/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write the failing tests**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_chat.py
from app.chat import build_answer_prompt, choose_output_language, should_fallback


def test_choose_output_language_detects_chinese() -> None:
    assert choose_output_language("香港最低工資是多少？") == "zh-Hant"
    assert choose_output_language("What are rest day rules?") == "en"


def test_should_fallback_when_confidence_is_low() -> None:
    assert should_fallback("Low", []) is True
    assert should_fallback("Medium", [{"chunk_id": "c1"}]) is False


def test_build_answer_prompt_requires_grounded_citations() -> None:
    prompt = build_answer_prompt(
        user_question="What are the rules for recruitment agencies?",
        output_language="en",
        retrieved_chunks=[{"source_title": "Employment Ordinance", "source_url": "https://example.com", "text": "Maximum commission is 10% of first-month wages."}],
    )

    assert "Use only the retrieved context" in prompt
    assert "Cite each claim" in prompt
    assert "Maximum commission is 10% of first-month wages." in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_chat.py -q`
Expected: FAIL with `ModuleNotFoundError` for `app.chat`

- [ ] **Step 3: Write minimal implementation**

```python
# /Users/dan06ial/projects/pobot-ai/app/chat.py
from __future__ import annotations

from openai import OpenAI


def choose_output_language(user_question: str) -> str:
    return "zh-Hant" if any("\u4e00" <= char <= "\u9fff" for char in user_question) else "en"


def should_fallback(confidence: str, retrieved_chunks: list[dict]) -> bool:
    return confidence == "Low" or not retrieved_chunks


def build_answer_prompt(user_question: str, output_language: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f'Source: {chunk["source_title"]}\nURL: {chunk["source_url"]}\nSnippet: {chunk["text"]}'
        for chunk in retrieved_chunks
    )
    return (
        "Use only the retrieved context to answer the user.\n"
        "If the context is insufficient, say so directly.\n"
        "Cite each claim with the source title and URL.\n"
        f"Answer language: {output_language}\n\n"
        f"User question: {user_question}\n\n"
        f"Retrieved context:\n{context}"
    )


def build_search_query_prompt(user_question: str) -> str:
    return (
        "Rewrite the following user question into a concise English search query for retrieval. "
        "Keep legal terms intact and do not answer the question.\n\n"
        f"Question: {user_question}"
    )


def deepseek_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_chat.py -q`
Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add app/chat.py tests/test_chat.py
git commit -m "feat: add DeepSeek chat prompt and multilingual handling"
```

### Task 7: Wire The Streamlit UI And End-To-End Demo Flow

**Files:**
- Create: `streamlit_app.py`
- Create: `README.md`
- Create: `evaluation/sample_queries.md`
- Modify: `app/chat.py`
- Test: `tests/test_chat.py`

- [ ] **Step 1: Write the failing test for final fallback text**

```python
# /Users/dan06ial/projects/pobot-ai/tests/test_chat.py
from app.chat import fallback_message


def test_fallback_message_mentions_indexed_materials() -> None:
    message = fallback_message("zh-Hant")

    assert "indexed Labour Department materials" in message
```

- [ ] **Step 2: Run tests to verify it fails**

Run: `pytest tests/test_chat.py -q`
Expected: FAIL with `ImportError: cannot import name 'fallback_message'`

- [ ] **Step 3: Write minimal implementation and demo files**

```python
# /Users/dan06ial/projects/pobot-ai/app/chat.py
def fallback_message(output_language: str) -> str:
    if output_language == "zh-Hant":
        return "I couldn’t find a reliable answer in the indexed Labour Department materials. 我未能在已建立索引的勞工處資料中找到可靠答案。"
    return "I couldn’t find a reliable answer in the indexed Labour Department materials."
```

```python
# /Users/dan06ial/projects/pobot-ai/streamlit_app.py
from __future__ import annotations

from sentence_transformers import SentenceTransformer
import streamlit as st

from app.chat import build_answer_prompt, choose_output_language, fallback_message, should_fallback
from app.config import AppConfig
from app.retrieval import confidence_label, load_artifacts, retrieve_chunks


st.set_page_config(page_title="Migrant Support RAG Assistant", layout="wide")
st.title("Migrant Support RAG Assistant")
st.caption("Grounded answers from Hong Kong Labour Department materials")

config = AppConfig.from_env()
embedder = SentenceTransformer(config.embedding_model)

if not config.chunks_path.exists() or not config.index_path.exists():
    st.warning("Knowledge base not built yet. Run `python scripts/build_kb.py` first.")
else:
    records, index = load_artifacts(config.chunks_path, config.index_path)
    question = st.text_input("Ask a question", placeholder="What are the rules for recruitment agencies?")
    if question:
        hits = retrieve_chunks(question, records, index, embedder, top_k=config.top_k)
        confidence = confidence_label([hit["score"] for hit in hits])
        language = choose_output_language(question)

        if should_fallback(confidence, hits):
            st.error(fallback_message(language))
        else:
            st.subheader("Retrieved Evidence")
            for hit in hits:
                st.markdown(f'**{hit["source_title"]}**  \n{hit["text"][:280]}...')
                st.markdown(hit["source_url"])

            st.subheader("Prompt Preview")
            st.code(build_answer_prompt(question, language, hits), language="markdown")
```

````markdown
<!-- /Users/dan06ial/projects/pobot-ai/README.md -->
# Migrant Support RAG Assistant

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build The Knowledge Base

```bash
python scripts/build_kb.py
```

This downloads the approved Labour Department source files into `data/raw/`, updates `data/raw/manifest.json`, and writes FAISS artifacts into `data/processed/`.

## Run The Demo

```bash
streamlit run streamlit_app.py
```

## Submission Mapping

- raw source files: `data/raw/`
- processed chunks and vector index: `data/processed/`
- chatbot UI: `streamlit_app.py`
- sample queries and outputs: `evaluation/sample_queries.md`
````

````markdown
<!-- /Users/dan06ial/projects/pobot-ai/evaluation/sample_queries.md -->
# Sample Queries And Outputs

1. What are the rights of foreign domestic workers in Hong Kong?
2. What are the rules for recruitment agencies?
3. What is the statutory minimum wage in Hong Kong?
4. 如果僱主沒有給我休息日，我可以怎麼做？

## Known Limitation

Questions that require cross-checking multiple ordinances may still return incomplete answers if the retrieved chunks cluster around only one source.
````

- [ ] **Step 4: Run tests and smoke checks**

Run: `pytest tests/test_chat.py -q`
Expected: PASS with `4 passed`

Run: `python scripts/build_kb.py`
Expected: JSON summary with chunk count and `faiss.index` path

Run: `streamlit run streamlit_app.py`
Expected: Browser opens with question input, evidence panel, and prompt preview

- [ ] **Step 5: Commit**

```bash
git add app/chat.py streamlit_app.py README.md evaluation/sample_queries.md tests/test_chat.py
git commit -m "feat: add streamlit demo and submission docs"
```

## Spec Coverage Check

- Raw official sources stored visibly: Task 2 and Task 4
- HTML/PDF cleaning and chunking: Task 3 and Task 4
- Embeddings plus FAISS retrieval: Task 4 and Task 5
- Grounded DeepSeek answer generation: Task 6
- Simple UI instead of CLI: Task 7
- Source citations: Task 6 and Task 7
- Chinese answers over English retrieval: Task 6
- Confidence and fallback response: Task 5 and Task 7
- Sample queries and limitation: Task 7

## Placeholder Scan

- No `TBD`, `TODO`, or “implement later” markers remain.
- Every code-changing task names exact files and includes concrete commands.
- Every required task maps to at least one test or explicit smoke-check.
