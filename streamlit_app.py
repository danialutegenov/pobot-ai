from __future__ import annotations

from pathlib import Path

import streamlit as st
from sentence_transformers import SentenceTransformer

from app.chat import (
    choose_output_language,
    generate_grounded_answer,
    rewrite_query_for_retrieval,
)
from app.config import AppConfig
from app.retrieval import confidence_label, load_artifacts, retrieve_chunks, select_diverse_hits


@st.cache_resource
def load_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource
def load_kb(chunks_path: Path, index_path: Path):
    return load_artifacts(chunks_path=chunks_path, index_path=index_path)


def render() -> None:
    st.set_page_config(page_title="Migrant Support RAG Assistant", layout="wide")
    st.title("Migrant Support RAG Assistant")
    st.caption("Grounded Q&A from Hong Kong Labour Department sources")

    config = AppConfig.from_env()
    if not config.chunks_path.exists() or not config.index_path.exists():
        st.warning("Knowledge base not built yet. Run `python scripts/build_kb.py` first.")
        return

    records, index = load_kb(config.chunks_path, config.index_path)
    embedder = load_embedder(config.embedding_model)

    question = st.text_input(
        "Ask a question",
        placeholder="What are the rules for recruitment agencies in Hong Kong?",
    )
    if not question:
        return

    output_language = choose_output_language(question)
    retrieval_query = rewrite_query_for_retrieval(
        user_question=question,
        output_language=output_language,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        chat_model=config.chat_model,
    )
    candidate_top_k = max(config.top_k * 3, config.top_k + 5)
    candidate_hits = retrieve_chunks(
        query=retrieval_query,
        records=records,
        index=index,
        embedder=embedder,
        top_k=candidate_top_k,
    )
    hits = select_diverse_hits(
        hits=candidate_hits,
        top_k=config.top_k,
        per_source_cap=2,
    )
    confidence = confidence_label([hit["score"] for hit in hits])
    answer = generate_grounded_answer(
        user_question=question,
        output_language=output_language,
        retrieved_chunks=hits,
        confidence=confidence,
        api_key=config.deepseek_api_key,
        base_url=config.deepseek_base_url,
        chat_model=config.chat_model,
    )

    st.subheader("Answer")
    st.write(answer)
    st.badge(f"Confidence: {confidence}")

    st.subheader("Citations")
    for hit in hits:
        st.markdown(f"**{hit['source_title']}**")
        st.markdown(f"> {hit['text'][:280]}...")
        st.markdown(f"[Source Link]({hit['source_url']})")

    with st.expander("Retrieved Evidence (Debug)"):
        st.write(
            {
                "retrieval_query": retrieval_query,
                "candidate_top_k": candidate_top_k,
                "final_top_k": config.top_k,
                "per_source_cap": 2,
            }
        )
        for hit in hits:
            st.write(
                {
                    "chunk_id": hit["chunk_id"],
                    "score": hit["score"],
                    "source_url": hit["source_url"],
                    "local_raw_path": hit["local_raw_path"],
                }
            )


if __name__ == "__main__":
    render()
