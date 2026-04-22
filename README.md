# Migrant Support RAG Assistant

This project implements a simple Retrieval-Augmented Generation assistant for Hong Kong labour and employment questions.

## Live Demo

- https://pobot-ai.streamlit.app/

## What It Includes

- fixed-source data collection from approved Labour Department links
- raw data snapshots stored in `data/raw/` with manifest metadata
- preprocessing and chunking for HTML and PDF sources
- FAISS retrieval over embedding vectors
- DeepSeek-based grounded answer generation
- Streamlit UI with confidence and source citations

## Setup

```bash
cd /Users/dan06ial/projects/pobot-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure Environment

```bash
cp .env.example .env
export DEEPSEEK_API_KEY="your-deepseek-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_CHAT_MODEL="deepseek-chat"
```

## Build Knowledge Base

```bash
python scripts/build_kb.py
```

Outputs:

- raw files: `data/raw/`
- source manifest: `data/raw/manifest.json`
- processed chunks: `data/processed/chunks.json`
- vector index: `data/processed/faiss.index`
- build report: `data/processed/build_report.json`

## Run App

```bash
streamlit run streamlit_app.py
```

## Run Non-UI Test Query (Requirement 5)

Simple testing function without UI:

```bash
python scripts/test_query.py "What are the rules for recruitment agencies in Hong Kong?"
```

Structured output with retrieval metadata:

```bash
python scripts/test_query.py "What are the rules for recruitment agencies in Hong Kong?" --json
```

List the built-in sample question set:

```bash
python scripts/test_query.py --list-sample-questions
```

## Run 5 Example Test Queries (Batch)

Run the predefined 5 sample questions and save all responses in one report:

```bash
python scripts/run_sample_queries.py
```

Where to check the 5 example query outputs:

- `evaluation/sample_queries.md`

## Submission Mapping

- RAG pipeline scripts/modules: `app/` and `scripts/build_kb.py`
- collected and processed documents: `data/raw/` and `data/processed/`
- working chatbot UI: `streamlit_app.py`
- sample queries and outputs: `evaluation/sample_queries.md`
- approach summary: `evaluation/summary.md`
