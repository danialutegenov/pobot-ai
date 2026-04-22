from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from test_query import SAMPLE_QUESTIONS, run_query


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def render_markdown(results: list[dict], top_k: int | None) -> str:
    lines = [
        "# Sample Query Batch Results",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Question count: {len(results)}",
        f"Top-k override: {top_k if top_k is not None else 'default'}",
        "",
    ]
    for index, item in enumerate(results, start=1):
        lines.extend(
            [
                f"## Query {index}",
                "",
                f"**Question**: {item['question']}",
                f"**Output language**: {item['output_language']}",
                f"**Confidence**: {item['confidence']}",
                f"**Retrieval query**: `{item['retrieval_query']}`",
                "",
                "### Answer",
                "",
                item["answer"],
                "",
                "### Citations",
                "",
            ]
        )
        for citation in item.get("citations", []):
            lines.append(
                f"- {citation['source_title']} ({citation['source_url']}) score={citation['score']:.4f}"
            )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 5 predefined sample queries and save full responses."
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path for full JSON results. Omit to skip JSON output.",
    )
    parser.add_argument(
        "--md-out",
        default="evaluation/sample_queries.md",
        help="Where to write markdown results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k override for all sample queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    results: list[dict] = []
    for question in SAMPLE_QUESTIONS[:5]:
        results.append(run_query(question=question, top_k=args.top_k))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "question_count": len(results),
        "questions": SAMPLE_QUESTIONS[:5],
        "results": results,
    }

    md_path = (project_root / args.md_out).resolve()
    md_path.parent.mkdir(parents=True, exist_ok=True)

    md_path.write_text(render_markdown(results=results, top_k=args.top_k), encoding="utf-8")

    if args.json_out:
        json_path = (project_root / args.json_out).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote JSON: {json_path}")
    print(f"Wrote Markdown: {md_path}")


if __name__ == "__main__":
    main()
