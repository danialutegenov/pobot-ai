from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "what",
    "how",
    "when",
    "where",
    "why",
    "who",
    "which",
    "can",
    "could",
    "should",
    "would",
    "do",
    "does",
    "did",
    "about",
    "hong",
    "kong",
}

DOMAIN_HINTS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\b(recruitment|employment agency|agency|commission)\b", re.IGNORECASE),
        "employment agency licence commission code of practice",
    ),
    (
        re.compile(r"\b(minimum wage|smw|wage)\b", re.IGNORECASE),
        "minimum wage ordinance hourly rate cap 608",
    ),
    (
        re.compile(r"\b(rest day|leave|holiday)\b", re.IGNORECASE),
        "employment ordinance rest day statutory holiday paid leave",
    ),
    (
        re.compile(r"\b(domestic worker|foreign domestic helper|fdh)\b", re.IGNORECASE),
        "foreign domestic helper contract rights",
    ),
]


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in value)


def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{2,}", text.lower())
    seen: set[str] = set()
    results: list[str] = []
    for token in tokens:
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        results.append(token)
    return results[:8]


def build_structured_retrieval_query(english_question: str) -> str:
    base = re.sub(r"\s+", " ", english_question).strip()
    if not base:
        return english_question

    keywords = _extract_keywords(base)
    hints: list[str] = []
    lowered = base.lower()
    for pattern, hint in DOMAIN_HINTS:
        if pattern.search(lowered):
            hints.append(hint)

    parts = [base, "hong kong labour law employment ordinance"]
    if keywords:
        parts.append("keywords " + " ".join(keywords))
    if hints:
        parts.append("focus " + " ".join(dict.fromkeys(hints)))
    return " | ".join(parts)


def choose_output_language(user_question: str) -> str:
    return "zh-Hant" if any("\u4e00" <= char <= "\u9fff" for char in user_question) else "en"


def should_fallback(confidence: str, retrieved_chunks: list[dict[str, Any]]) -> bool:
    return confidence == "Low" or not retrieved_chunks


def fallback_message(output_language: str) -> str:
    if output_language == "zh-Hant":
        return "我未能在已建立索引的勞工處資料中找到可靠答案。"
    return "I couldn't find a reliable answer in the indexed Labour Department materials."


def build_search_query_prompt(user_question: str) -> str:
    return (
        "Rewrite the following question as a concise English retrieval query. "
        "Keep legal terms and entities. Do not answer the question.\n\n"
        f"Question: {user_question}"
    )


def build_retrieval_plan_prompt(user_question: str) -> str:
    return (
        "Create an English retrieval plan for Hong Kong labour-regulation RAG search.\n"
        "Return JSON only with keys: english_query, keywords, legal_terms, must_include, topic.\n"
        "Rules:\n"
        "- english_query must be concise English, legal-term-preserving, no answer text.\n"
        "- If question is Chinese, translate intent into English in english_query.\n"
        "- keywords/legal_terms/must_include must be short lists (0-8 items each).\n"
        "- topic is one short label.\n\n"
        f"Question: {user_question}"
    )


def _extract_first_json_object(raw: str) -> str:
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object start found.")
    depth = 0
    for index in range(start, len(raw)):
        char = raw[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]
    raise ValueError("No complete JSON object found.")


def _sanitize_plan_list(values: Any, max_items: int = 8) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def parse_retrieval_plan(raw_content: str, fallback_query: str) -> dict[str, Any]:
    plan = {
        "english_query": fallback_query,
        "keywords": [],
        "legal_terms": [],
        "must_include": [],
        "topic": "",
    }
    try:
        json_text = _extract_first_json_object(raw_content)
        payload = json.loads(json_text)
    except (ValueError, json.JSONDecodeError, TypeError):
        return plan

    if not isinstance(payload, dict):
        return plan

    english_query = str(payload.get("english_query", "")).strip()
    if english_query:
        plan["english_query"] = english_query
    plan["keywords"] = _sanitize_plan_list(payload.get("keywords"))
    plan["legal_terms"] = _sanitize_plan_list(payload.get("legal_terms"))
    plan["must_include"] = _sanitize_plan_list(payload.get("must_include"))
    plan["topic"] = str(payload.get("topic", "")).strip()
    return plan


def compose_retrieval_query_from_plan(plan: dict[str, Any]) -> str:
    english_query = str(plan.get("english_query", "")).strip()
    if not english_query:
        return ""
    parts = [english_query, "hong kong labour law employment ordinance"]
    keywords = plan.get("keywords", [])
    legal_terms = plan.get("legal_terms", [])
    must_include = plan.get("must_include", [])
    topic = str(plan.get("topic", "")).strip()
    if keywords:
        parts.append("keywords " + " ".join(keywords))
    if legal_terms:
        parts.append("legal_terms " + " ".join(legal_terms))
    if must_include:
        parts.append("must_include " + " ".join(must_include))
    if topic:
        parts.append("topic " + topic)
    return " | ".join(parts)


def build_answer_prompt(
    user_question: str,
    output_language: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    context = "\n\n".join(
        (
            f"Source Title: {chunk['source_title']}\n"
            f"Source URL: {chunk['source_url']}\n"
            f"Evidence Snippet: {chunk['text']}"
        )
        for chunk in retrieved_chunks
    )
    return (
        "Use only the retrieved context to answer the user.\n"
        "If context is insufficient, explicitly say you are unsure.\n"
        "Cite each factual claim with source title and URL.\n"
        "Do not invent legal rights or procedures.\n"
        f"Output language: {output_language}\n\n"
        f"User question: {user_question}\n\n"
        f"Retrieved context:\n{context}"
    )


def deepseek_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def _llm_retrieval_plan(
    user_question: str,
    api_key: str,
    base_url: str,
    chat_model: str,
) -> dict[str, Any]:
    client = deepseek_client(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You produce retrieval planning JSON only."},
            {"role": "user", "content": build_retrieval_plan_prompt(user_question)},
        ],
    )
    content = (completion.choices[0].message.content or "").strip()
    return parse_retrieval_plan(raw_content=content, fallback_query=user_question)


def _rewrite_to_english_with_llm(
    user_question: str,
    api_key: str,
    base_url: str,
    chat_model: str,
) -> str:
    client = deepseek_client(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": "You convert user questions into English retrieval queries."},
            {"role": "user", "content": build_search_query_prompt(user_question)},
        ],
    )
    return (completion.choices[0].message.content or "").strip() or user_question


def rewrite_query_for_retrieval(
    user_question: str,
    output_language: str,
    api_key: str,
    base_url: str,
    chat_model: str,
) -> str:
    if not api_key:
        if output_language == "zh-Hant":
            return user_question
        return build_structured_retrieval_query(user_question)

    # Preferred path: LLM emits structured retrieval plan JSON.
    try:
        plan = _llm_retrieval_plan(
            user_question=user_question,
            api_key=api_key,
            base_url=base_url,
            chat_model=chat_model,
        )
        planned_query = compose_retrieval_query_from_plan(plan)
        if planned_query and not _contains_cjk(planned_query):
            return planned_query
    except Exception:
        pass

    # Fallback path: keep deterministic behavior and optional Chinese->English rewrite.
    base_question = user_question
    if output_language == "zh-Hant":
        try:
            base_question = _rewrite_to_english_with_llm(
                user_question=user_question,
                api_key=api_key,
                base_url=base_url,
                chat_model=chat_model,
            )
        except Exception:
            return user_question

    if _contains_cjk(base_question):
        return user_question
    return build_structured_retrieval_query(base_question)


def generate_grounded_answer(
    user_question: str,
    output_language: str,
    retrieved_chunks: list[dict[str, Any]],
    confidence: str,
    api_key: str,
    base_url: str,
    chat_model: str,
) -> str:
    if should_fallback(confidence, retrieved_chunks):
        return fallback_message(output_language)

    if not api_key:
        # Deterministic fallback so app still runs without model key.
        first = retrieved_chunks[0]
        if output_language == "zh-Hant":
            return (
                "未設定 DeepSeek API 金鑰。以下是最相關證據摘要："
                f"{first['text'][:240]}..."
            )
        return (
            "DeepSeek API key is not configured. Most relevant evidence: "
            f"{first['text'][:240]}..."
        )

    client = deepseek_client(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=chat_model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a cautious labour regulations assistant."},
            {
                "role": "user",
                "content": build_answer_prompt(
                    user_question=user_question,
                    output_language=output_language,
                    retrieved_chunks=retrieved_chunks,
                ),
            },
        ],
    )
    return (completion.choices[0].message.content or "").strip() or fallback_message(output_language)
