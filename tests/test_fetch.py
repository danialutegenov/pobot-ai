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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest[0]["url"] == "https://example.com/faq.htm"

