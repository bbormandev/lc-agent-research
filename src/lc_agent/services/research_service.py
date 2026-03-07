import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from lc_agent.infra.run_bundler import RunBundler
from lc_agent.infra.run_context import RunContext
from lc_agent.prompts.decomposition import (
    COMPLEXITY_JSON_SCHEMA,
    COMPLEXITY_SYSTEM_PROMPT,
    COMPLEXITY_USER_PROMPT_TEMPLATE,
)
from lc_agent.prompts.research import ANSWER_PROMPT, GATE_PROMPT, QUERY_PROMPT
from lc_agent.services.decomposition_service import (
    DecompositionResult,
    DecompositionServiceConfig,
    Subtopic,
    decompose_question as run_decomposition,
)
from lc_agent.tools.extract import extract_passages
from lc_agent.tools.fetch import fetch_url
from lc_agent.tools.search_tavily import SearchResult, search_web

load_dotenv()


@dataclass
class ResearchServiceConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_sources: int = 5
    max_chars_per_source: int = 6000
    total_context_chars: int = 12000  # not currently enforced; leave for now
    max_queries: int = 3


@dataclass
class ComplexityResult:
    is_complex: bool
    reason: str


@dataclass
class SubtopicResearchResult:
    subtopic: Subtopic
    result: dict[str, Any]
    did_search: bool
    search_queries: list[str]


@dataclass
class TopicResearchResult:
    question: str
    decomposition: DecompositionResult
    subtopics: list[SubtopicResearchResult]


def decide_should_search(llm: ChatOpenAI, question: str, ctx: RunContext) -> bool:
    decision = llm.invoke(GATE_PROMPT.format(today=ctx.today, question=question)).content.strip().upper()
    return decision == "YES"


def generate_queries(llm: ChatOpenAI, question: str, max_queries: int, ctx: RunContext) -> list[str]:
    raw = llm.invoke(QUERY_PROMPT.format(question=question, today=ctx.today)).content
    data = json.loads(raw)
    queries = data.get("queries", [])
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return queries[:max_queries]


def classify_question_complexity(llm: ChatOpenAI, question: str) -> ComplexityResult:
    bound = llm.bind(
        response_format={
            "type": "json_schema",
            "json_schema": COMPLEXITY_JSON_SCHEMA,
        }
    )
    raw = bound.invoke(
        [
            ("system", COMPLEXITY_SYSTEM_PROMPT),
            ("user", COMPLEXITY_USER_PROMPT_TEMPLATE.format(question=question)),
        ]
    ).content
    if not isinstance(raw, str):
        raise RuntimeError("Complexity classifier returned non-text content")

    parsed = json.loads(raw)
    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        reason = "No reason provided."
    return ComplexityResult(is_complex=bool(parsed.get("is_complex")), reason=reason)


def decompose_question(
    question: str,
    config: ResearchServiceConfig | None = None,
) -> DecompositionResult:
    config = config or ResearchServiceConfig()
    decomposition_config = DecompositionServiceConfig(
        model=config.model,
        temperature=config.temperature,
    )
    return run_decomposition(question, decomposition_config)


def validate_citations(answer_bullets: list[str], sources: list[str]) -> None:
    # Sources formatted like "S1: Title - URL"
    source_ids = set()
    for source in sources:
        match = re.match(r"^(S\d+):", source.strip())
        if match:
            source_ids.add(match.group(1))

    if not source_ids:
        return  # nothing to validate

    for bullet in answer_bullets:
        match = re.search(r"\[([^\]]+)\]\s*$", bullet.strip())
        if not match:
            raise RuntimeError(f"Bullet missing ending citations: {bullet}")

        cited = {citation.strip() for citation in match.group(1).split(",")}
        bad = [citation for citation in cited if citation not in source_ids]
        if bad:
            raise RuntimeError(f"Bullet cites unknown sources {bad}: {bullet}")


def validate_summary(summary: str) -> None:
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("Missing or empty summary")
    if "[" in summary or "]" in summary:
        raise RuntimeError(f"Summary must not contain citations/brackets: {summary}")


def validate_note_title(note_title: object) -> str:
    if not isinstance(note_title, str):
        raise RuntimeError("Response missing required field: note_title")
    cleaned = note_title.strip()
    if not cleaned:
        raise RuntimeError("Response note_title must be non-empty")
    return cleaned


def serialize_search_result(result: SearchResult) -> dict[str, Any]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.snippet,
    }


def serialize_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [serialize_search_result(result) for result in results]


def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _run_single_topic_research(
    llm: ChatOpenAI,
    question: str,
    config: ResearchServiceConfig,
    ctx: RunContext,
    bundler: RunBundler,
    artifact_prefix: str = "",
    force_search: bool = False,
) -> tuple[dict[str, Any], bool, list[str]]:
    def path_for(name: str) -> str:
        if not artifact_prefix:
            return name
        return f"{artifact_prefix.rstrip('/')}/{name}"

    did_search = force_search or decide_should_search(llm, question, ctx)

    context = ""
    sources_list: list[str] = []
    search_queries: list[str] = []

    if did_search:
        search_queries = generate_queries(llm, question, config.max_queries, ctx)
        if not search_queries:
            search_queries = [question]
        bundler.write_json(path_for("search_queries.json"), search_queries)

        per_query_limit = max(config.max_sources, 5)
        buckets: list[list[SearchResult]] = []
        search_dump = []
        for query in search_queries:
            try:
                results = search_web(query)[:per_query_limit]
            except Exception:
                results = []

            buckets.append(results)
            search_dump.append(
                {
                    "query": query,
                    "results": [
                        {"title": result.title, "url": result.url, "snippet": result.snippet}
                        for result in results
                    ],
                }
            )
        bundler.write_json(path_for("search_results.json"), search_dump)

        selected_results: list[SearchResult] = []
        seen_urls: set[str] = set()

        # Round-robin pick 1 from each bucket until max_sources
        index = 0
        while len(selected_results) < config.max_sources:
            progressed = False
            for bucket in buckets:
                if index < len(bucket):
                    result = bucket[index]
                    url = (result.url or "").strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        selected_results.append(result)
                        progressed = True
                        if len(selected_results) >= config.max_sources:
                            break
            if not progressed:
                break
            index += 1
        bundler.write_json(path_for("selected_sources.json"), serialize_results(selected_results))

        passage_blocks: list[str] = []
        for source_idx, result in enumerate(selected_results, start=1):
            sources_list.append(f"S{source_idx}: {result.title} - {result.url}")

            try:
                doc = fetch_url(result.url, max_chars=config.max_chars_per_source)
                bundler.write_json(
                    path_for(f"fetch/{url_hash(result.url)}.json"),
                    {
                        "url": doc.url,
                        "title": doc.title,
                        "text": doc.text,
                    },
                )

                passages = extract_passages(
                    llm,
                    question,
                    title=result.title or (doc.title or "Untitled"),
                    url=result.url,
                    text=doc.text,
                )
                bundler.write_json(
                    path_for(f"extracts/{url_hash(result.url)}.json"),
                    {
                        "source_id": f"S{source_idx}",
                        "title": result.title,
                        "url": result.url,
                        "passages": passages,
                    },
                )

                block_lines = [
                    f"SOURCE_ID: S{source_idx}",
                    f"TITLE: {result.title}",
                    f"URL: {result.url}",
                    "PASSAGES:",
                ]
                for passage in passages:
                    block_lines.append(f"- {passage['quote']}  (why: {passage['why']})")
                passage_blocks.append("\n".join(block_lines))
            except Exception as exc:
                passage_blocks.append(
                    f"SOURCE_ID: S{source_idx}\n"
                    f"TITLE: {result.title}\n"
                    f"URL: {result.url}\n"
                    "PASSAGES:\n"
                    f"- (EXTRACTION FAILED: {exc})\n"
                    f"- SNIPPET: {result.snippet}"
                )

        context = "\n\n".join(passage_blocks)
        bundler.write_text(path_for("context.txt"), context)

    prompt = ANSWER_PROMPT.format(
        question=question,
        context=context,
        did_search=str(did_search).lower(),
        search_queries=json.dumps(search_queries if did_search else []),
        sources_json=json.dumps(sources_list if did_search else []),
    )

    raw = llm.invoke(prompt).content
    data = json.loads(raw)

    data["note_title"] = validate_note_title(data.get("note_title"))
    validate_citations(data.get("answer_bullets", []), data.get("sources", []))
    summary = data.get("summary")
    if summary is None:
        raise RuntimeError("Response missing required field: summary")
    validate_summary(summary)

    if did_search and not data.get("sources"):
        raise RuntimeError("Expected sources when did_search=true, got empty sources.")

    return data, did_search, search_queries


def research_subtopic(
    subtopic: Subtopic,
    *,
    llm: ChatOpenAI,
    config: ResearchServiceConfig,
    ctx: RunContext,
    bundler: RunBundler,
) -> SubtopicResearchResult:
    subtopic_prefix = f"subtopics/{subtopic.id}"
    result, did_search, search_queries = _run_single_topic_research(
        llm=llm,
        question=subtopic.question,
        config=config,
        ctx=ctx,
        bundler=bundler,
        artifact_prefix=subtopic_prefix,
        force_search=True,
    )
    bundler.write_json(f"{subtopic_prefix}/final.json", result)
    return SubtopicResearchResult(
        subtopic=subtopic,
        result=result,
        did_search=did_search,
        search_queries=search_queries,
    )


def ask(question: str, config: ResearchServiceConfig, ctx: RunContext) -> dict:
    bundler = RunBundler(base_dir="runs")
    run_id = bundler.start()
    llm = ChatOpenAI(model=config.model, temperature=config.temperature)

    meta = {
        "run_id": run_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "today": ctx.today,
        "model": config.model,
        "config": {
            "max_sources": config.max_sources,
            "max_queries": config.max_queries,
            "max_chars_per_source": config.max_chars_per_source,
        },
    }
    bundler.write_json("meta.json", meta)

    complexity = classify_question_complexity(llm, question)
    meta["complexity"] = asdict(complexity)
    bundler.write_json("meta.json", meta)

    decomposition_result: DecompositionResult | None = None
    if complexity.is_complex:
        decomposition_result = decompose_question(question, config=config)
        if len(decomposition_result.subtopics) < 4 or len(decomposition_result.subtopics) > 7:
            raise RuntimeError("Complex questions must decompose into 4-7 subtopics")
        bundler.write_json("decomposition.json", asdict(decomposition_result))

    data, did_search, search_queries = _run_single_topic_research(
        llm=llm,
        question=question,
        config=config,
        ctx=ctx,
        bundler=bundler,
    )
    subtopic_results: list[SubtopicResearchResult] = []
    if decomposition_result is not None:
        data["decomposition"] = asdict(decomposition_result)
        for subtopic in decomposition_result.subtopics:
            subtopic_results.append(
                research_subtopic(
                    subtopic,
                    llm=llm,
                    config=config,
                    ctx=ctx,
                    bundler=bundler,
                )
            )
        topic_result = TopicResearchResult(
            question=question,
            decomposition=decomposition_result,
            subtopics=subtopic_results,
        )
        topic_result_payload = asdict(topic_result)
        data["topic_research"] = topic_result_payload
        bundler.write_json("topic_research.json", topic_result_payload)

    data["_meta"] = {
        "is_complex": complexity.is_complex,
        "complexity_reason": complexity.reason,
        "did_search": did_search,
        "search_queries": search_queries,
        "subtopics_researched": len(subtopic_results),
        "max_sources": config.max_sources,
        "model": config.model,
        "run_id": run_id,
        "run_dir": str(bundler.path()),
    }
    meta["did_search"] = did_search

    bundler.write_json("final.json", data)
    meta = bundler.finish_meta(meta)
    bundler.write_json("meta.json", meta)

    return data
