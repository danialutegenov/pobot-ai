from app.chat import (
    build_answer_prompt,
    compose_retrieval_query_from_plan,
    build_structured_retrieval_query,
    choose_output_language,
    fallback_message,
    parse_retrieval_plan,
    rewrite_query_for_retrieval,
    should_fallback,
)


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
        retrieved_chunks=[
            {
                "source_title": "Employment Ordinance",
                "source_url": "https://example.com",
                "text": "Maximum commission is 10% of first-month wages.",
            }
        ],
    )

    assert "Use only the retrieved context" in prompt
    assert "Cite each factual claim" in prompt
    assert "Maximum commission is 10% of first-month wages." in prompt


def test_fallback_message_mentions_indexed_materials() -> None:
    message = fallback_message("zh-Hant")
    assert "勞工處資料" in message


def test_structured_query_adds_domain_context() -> None:
    query = build_structured_retrieval_query("What are the rules for recruitment agencies in Hong Kong?")
    lowered = query.lower()
    assert "hong kong labour law employment ordinance" in lowered
    assert "recruitment" in lowered
    assert "employment agency licence commission code of practice" in lowered


def test_rewrite_query_for_english_uses_structured_format_without_api() -> None:
    query = rewrite_query_for_retrieval(
        user_question="What is the statutory minimum wage in Hong Kong?",
        output_language="en",
        api_key="",
        base_url="https://api.deepseek.com",
        chat_model="deepseek-chat",
    )
    lowered = query.lower()
    assert "hong kong labour law employment ordinance" in lowered
    assert "minimum wage ordinance hourly rate cap 608" in lowered


def test_parse_retrieval_plan_accepts_markdown_wrapped_json() -> None:
    raw = """```json
{
  "english_query": "employment agency licensing rules hong kong",
  "keywords": ["employment agency", "licensing", "hong kong"],
  "legal_terms": ["employment ordinance", "employment agency regulations"],
  "must_include": ["commission"],
  "topic": "recruitment_agency_rules"
}
```"""
    plan = parse_retrieval_plan(raw_content=raw, fallback_query="fallback query")
    assert plan["english_query"] == "employment agency licensing rules hong kong"
    assert "employment agency" in plan["keywords"]
    assert "employment ordinance" in plan["legal_terms"]
    assert plan["topic"] == "recruitment_agency_rules"


def test_parse_retrieval_plan_falls_back_on_invalid_json() -> None:
    plan = parse_retrieval_plan(raw_content="not json at all", fallback_query="fallback query")
    assert plan["english_query"] == "fallback query"
    assert plan["keywords"] == []


def test_compose_retrieval_query_from_plan_includes_structured_fields() -> None:
    plan = {
        "english_query": "minimum wage in hong kong",
        "keywords": ["minimum", "wage"],
        "legal_terms": ["minimum wage ordinance", "cap 608"],
        "must_include": ["hourly rate"],
        "topic": "minimum_wage",
    }
    query = compose_retrieval_query_from_plan(plan).lower()
    assert "minimum wage in hong kong" in query
    assert "keywords minimum wage" in query
    assert "legal_terms minimum wage ordinance cap 608" in query
    assert "must_include hourly rate" in query
    assert "topic minimum_wage" in query
