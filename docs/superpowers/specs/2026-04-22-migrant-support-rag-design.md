# Migrant Support RAG Assistant Design

Date: 2026-04-22

## Goal

Build a simple Retrieval-Augmented Generation assistant for migrant support focused on Hong Kong labour and employment regulations. The deliverable should satisfy the technical task without expanding beyond it, while including three bonus features:

- source citations in responses
- Chinese queries answered in Chinese while retrieval remains English-grounded
- a simple confidence/fallback behavior

The system will use a simple UI instead of CLI and DeepSeek as the model provider.

## Scope

Included:

- fixed-source ingestion for the provided official English sources
- visible storage of collected raw source files for submission review
- text extraction and cleaning for HTML and PDF content
- topic-page expansion for index pages that link to subtopics
- chunking, embeddings, FAISS retrieval, and grounded answer generation
- Streamlit chat UI
- sample queries, outputs, and one documented limitation

Excluded:

- generic document upload or arbitrary source ingestion
- agent-based routing or tool-using agent workflows
- broad workflow automation beyond the requested RAG chatbot
- multilingual document collection beyond the provided English sources

## Fixed Sources

Primary sources:

- https://www.fdh.labour.gov.hk/en/fdh_corner.html
- https://www.labour.gov.hk/eng/public/wcp/ConciseGuide/EO_guide_full.pdf
- https://www.labour.gov.hk/eng/faq/content.htm and its linked topic pages
- https://www.labour.gov.hk/eng/service/content.htm and its linked topic pages
- https://www.labour.gov.hk/eng/legislat/content2.htm
- https://www.labour.gov.hk/eng/legislat/content3.htm
- https://www.labour.gov.hk/eng/legislat/content1.htm
- https://www.labour.gov.hk/eng/legislat/content4.htm
- https://www.labour.gov.hk/eng/legislat/content5.htm

The build step will download these sources directly rather than ingesting local files from a user-managed folder.
Downloaded source files should be kept in the project so the submission clearly shows which data was collected and used.

## Product Shape

The app is a small Python project with three layers:

1. knowledge-base build
2. retrieval and answer generation
3. Streamlit demo UI

The user opens the Streamlit app, asks a question in English or Chinese, and receives an answer grounded in retrieved English source content. If the user asked in Chinese, the answer is returned in Chinese, but retrieval still runs against English chunks.

## Architecture

### 1. Knowledge-Base Build

A build step downloads the fixed sources, extracts usable text, expands linked topic pages where required, removes boilerplate, chunks the text, creates embeddings, and writes a local FAISS index with chunk metadata.

Expected outputs:

- raw downloaded source snapshots
- source manifest with original URL, local file path, content type, and download timestamp
- cleaned and normalized chunk records
- FAISS vector index
- metadata store for citations and UI inspection

### 2. Retrieval Layer

The retrieval layer loads the saved FAISS index and chunk metadata, embeds the normalized search query, retrieves the top matching chunks, and returns both the chunk texts and a simple retrieval-strength signal used for fallback/confidence labeling.

### 3. Chat Layer

The chat layer builds the prompt sent to DeepSeek. It instructs the model to answer only from the retrieved context, cite supporting sources, avoid unsupported claims, and return the answer in the user’s language.

### 4. Streamlit UI

The UI remains simple and demo-oriented:

- chat input and message history
- answer block
- confidence label
- citations section with quoted supporting snippets and source links
- collapsible retrieved-evidence panel showing chunk excerpts and scores

This is enough to look polished without turning the project into a larger product build.

## Components

Suggested file/module split:

- `scripts/build_kb.py`
  - fetch fixed sources
  - extract text from HTML and PDF
  - expand FAQ and service topic links
  - clean and chunk text
  - generate embeddings
  - save FAISS index and metadata
- `data/raw/`
  - downloaded source files stored in HTML and PDF form
- `data/raw/manifest.json`
  - record of collected sources, origin URLs, and saved file paths
- `app/sources.py`
  - canonical source list and topic-expansion rules
- `app/preprocess.py`
  - HTML cleaning, PDF text normalization, chunking helpers
- `app/retrieval.py`
  - index loading, query embedding, top-k retrieval, score handling
- `app/chat.py`
  - language detection, optional Chinese-to-English retrieval query rewrite, grounded prompt construction, DeepSeek call, response shaping
- `streamlit_app.py`
  - Streamlit interface
- `evaluation/sample_queries.md`
  - example questions, outputs, and one limitation/failure case
- `data/processed/`
  - normalized chunks and metadata

This split keeps responsibilities clear and makes the pipeline inspectable for submission review.

## Data Flow

1. The build script downloads the fixed source pages and PDF.
2. The original downloaded files are stored locally in a usable format, preserving HTML for web pages and PDF for PDF sources.
3. A manifest records which URLs were downloaded, where they were saved, their content type, and when they were fetched.
4. HTML pages are cleaned to extract meaningful body content and follow relevant topic links from the FAQ and service index pages.
5. The PDF is extracted page by page and normalized.
6. Boilerplate such as navigation, repeated headers/footers, and irrelevant page furniture is removed.
7. The cleaned text is chunked into paragraph-aware segments with overlap.
8. Each chunk is stored with metadata:
   - source title
   - source URL
   - local raw file path
   - section or topic label when available
   - chunk text
9. Embeddings are created for each chunk and stored in a FAISS index.
10. At chat time, the system detects the question language.
11. If the query is in Chinese, the system creates an English retrieval query.
12. English chunks are retrieved from FAISS.
13. DeepSeek receives the user question, retrieved English evidence, and instructions to answer in the required output language with citations.
14. The UI displays the answer, confidence label, and quoted supporting snippets with links.

## Retrieval and Prompting

### Chunking

Use conservative, paragraph-aware chunking suitable for regulatory material:

- target chunk size around 500 to 900 characters
- small overlap to preserve section continuity
- avoid over-splitting lists and statutory conditions

### Embeddings and Vector Store

- embeddings: an API-based or local embedding model selected during implementation, provided it is simple and reliable for English retrieval
- vector store: FAISS

### Prompt Rules

The answer-generation prompt should enforce:

- answer only from the provided retrieved context
- if the context is insufficient, say so directly
- do not invent legal rights, procedures, or eligibility conditions
- cite the supporting source entries used
- answer in Chinese when the user asked in Chinese, otherwise answer in English

## Bonus Features

### 1. Source Citations

Each answer includes visible citations with:

- source title
- source URL
- short quoted snippet from the retrieved chunk

This is the strongest bonus feature for demonstrating grounded RAG behavior.

### 2. Chinese Query Support

The source corpus remains English-only. For Chinese questions:

- detect that the query is Chinese
- produce an English retrieval query
- retrieve English evidence
- generate the final answer in Chinese

This keeps the data pipeline simple while still demonstrating multilingual usability.

### 3. Confidence and Fallback

Confidence will be simple and interpretable, not a pseudo-probability. The system will expose:

- `High`
- `Medium`
- `Low`

The label is derived from retrieval strength and whether the top chunks clearly support the answer. When confidence is low or the evidence is incomplete, the system should fall back to a cautious response such as:

`I couldn’t find a reliable answer in the indexed Labour Department materials.`

It should still show the closest retrieved sources when possible.

### 4. Agent-Based Approach

Do not implement this bonus. It is unnecessary for the task and would add complexity without materially improving the demo.

## Source-Specific Handling

The FDH Corner page includes both rights guidance and time-sensitive event information. Preprocessing should remove or down-rank event and schedule content so it does not contaminate regulatory answers.

FAQ and service pages are index-like pages. Their linked topic pages should be expanded and included in the corpus so the system has enough substantive content to answer practical questions.

The additional legislation pages broaden coverage across employment, compensation, minimum wage, and workplace safety topics. They should be stored as downloaded HTML snapshots and processed like the other HTML sources.

## Error Handling

The system should prefer caution over fluency.

Expected behaviors:

- if retrieval finds weak or loosely related chunks, return a fallback answer
- if the answer is partial, say what is known and what is not clearly supported by the indexed material
- if a source page cannot be fetched during knowledge-base build, surface a build error clearly
- if the index has not been built, the UI should instruct the user to run the build step first

## Evaluation

Provide 3 to 5 example queries and outputs, covering topics such as:

- rights of foreign domestic workers
- recruitment agency rules and fees
- wages, rest days, or leave entitlements
- where to seek help or file complaints

Also provide one limitation or failure case, such as:

- ambiguous questions spanning multiple ordinances
- incomplete coverage for questions that are broader than the selected sources
- noisy retrieval from non-regulatory page fragments if a source page structure changes

## Deliverables Mapping To Task

This design maps directly to the expected output:

- Python implementation of the RAG pipeline
- stored raw and processed documents, with the raw source files visible in the project
- working chatbot with a simple UI
- sample queries and outputs
- short written summary of approach and findings

## Non-Goals

To avoid surpassing the assignment, do not add:

- user accounts
- admin dashboards
- analytics
- feedback loops
- editable source management UI
- multi-agent orchestration
- external tool integrations not required for the core demo

## Open Implementation Choice

One implementation detail remains flexible during build: the embedding model. The design intentionally leaves this open as long as it stays simple, works well for English retrieval, and does not complicate the submission.
