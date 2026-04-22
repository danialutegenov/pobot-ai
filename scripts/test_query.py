from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Allow `python scripts/test_query.py` without extra PYTHONPATH setup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chat import (
    choose_output_language,
    generate_grounded_answer,
    rewrite_query_for_retrieval,
)
from app.config import AppConfig
from app.retrieval import confidence_label, load_artifacts, retrieve_chunks, select_diverse_hits

SAMPLE_QUESTIONS = [
    "What are the rights of domestic workers in Hong Kong?",
    "What are the rules for recruitment agencies in Hong Kong?",
    "What is the statutory minimum wage in Hong Kong?",
    "如果僱主沒有給我休息日，我可以怎麼做？",
    "Where can migrant workers file a labour complaint in Hong Kong?",
]


def run_query(question: str, top_k: int | None = None) -> dict:
    config = AppConfig.from_env()
    if top_k is not None:
        config.top_k = top_k

    if not config.chunks_path.exists() or not config.index_path.exists():
        raise FileNotFoundError(
            "Knowledge base artifacts missing. Run `python scripts/build_kb.py` first."
        )

    records, index = load_artifacts(config.chunks_path, config.index_path)
    embedder = SentenceTransformer(config.embedding_model)

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
    citations = [
        {
            "source_title": hit["source_title"],
            "source_url": hit["source_url"],
            "score": hit["score"],
            "snippet": hit["text"][:220],
        }
        for hit in hits
    ]
    return {
        "question": question,
        "output_language": output_language,
        "retrieval_query": retrieval_query,
        "candidate_top_k": candidate_top_k,
        "final_top_k": config.top_k,
        "per_source_cap": 2,
        "confidence": confidence,
        "answer": answer,
        "citations": citations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single non-UI RAG query.")
    parser.add_argument("question", nargs="?", help="Question to ask the RAG assistant.")
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k.")
    parser.add_argument(
        "--list-sample-questions",
        action="store_true",
        help="Print the built-in sample question set and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full structured output as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_sample_questions:
        print(json.dumps(SAMPLE_QUESTIONS, indent=2, ensure_ascii=False))
        return
    if not args.question:
        raise SystemExit("Question is required unless --list-sample-questions is used.")
    result = run_query(args.question, args.top_k)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print(f"Question: {result['question']}")
    print(f"Output language: {result['output_language']}")
    print(f"Retrieval query: {result['retrieval_query']}")
    print(f"Confidence: {result['confidence']}")
    print("")
    print("Answer:")
    print(result["answer"])
    print("")
    print("Citations:")
    for item in result["citations"]:
        print(f"- {item['source_title']} ({item['source_url']}) score={item['score']:.4f}")


if __name__ == "__main__":
    # Preserve compatibility with `.env` values when users source env in shell.
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
