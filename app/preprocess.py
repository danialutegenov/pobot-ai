from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pypdf import PdfReader


NOISE_PATTERN = re.compile(
    r"(last update|site map|share this|print|menu|skip to content|copyright)",
    re.IGNORECASE,
)
EVENT_PATTERN = re.compile(r"(seminar|event|workshop|schedule|registration)", re.IGNORECASE)


def clean_html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag_name in ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    container = soup.find("main") or soup.find(id="content") or soup.body or soup
    lines: list[str] = []
    for node in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if NOISE_PATTERN.search(text):
            continue
        # FDH page may include event announcements; filter to reduce retrieval pollution.
        if EVENT_PATTERN.search(text) and len(text) < 180:
            continue
        lines.append(text)
    return "\n".join(lines)


def extract_topic_links(html: str, base_url: str, allowed_prefix: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"])
        if absolute.startswith(allowed_prefix) and absolute not in links:
            links.append(absolute)
    return links


def extract_pdf_text(pdf_path: str | Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        value = re.sub(r"\s+", " ", (page.extract_text() or "").strip())
        if value:
            pages.append(value)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n") if part.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []

    def _joined(parts: list[str]) -> str:
        return "\n".join(parts).strip()

    for paragraph in paragraphs:
        proposal_parts = current_parts + [paragraph]
        proposal = _joined(proposal_parts)
        if current_parts and len(proposal) > chunk_size:
            chunks.append(_joined(current_parts))
            overlap_parts: list[str] = []
            overlap_len = 0
            for existing in reversed(current_parts):
                overlap_parts.insert(0, existing)
                overlap_len += len(existing)
                if overlap_len >= overlap:
                    break
            current_parts = overlap_parts + [paragraph]
        else:
            current_parts = proposal_parts
    if current_parts:
        chunks.append(_joined(current_parts))
    return chunks
