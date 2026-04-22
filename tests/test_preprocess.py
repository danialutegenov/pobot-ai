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

