import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from lc_agent.prompts import GATE_PROMPT, QUERY_PROMPT, ANSWER_PROMPT
from lc_agent.tools.search_tavily import search_web
from lc_agent.tools.fetch import fetch_url
from lc_agent.tools.extract import extract_passages

load_dotenv()


@dataclass
class PipelineConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_sources: int = 5
    max_chars_per_source: int = 6000
    total_context_chars: int = 12000  # not currently enforced; leave for now
    max_queries: int = 3


def decide_should_search(llm: ChatOpenAI, question: str) -> bool:
    decision = llm.invoke(GATE_PROMPT.format(question=question)).content.strip().upper()
    return decision == "YES"


def generate_queries(llm: ChatOpenAI, question: str, max_queries: int) -> list[str]:
    raw = llm.invoke(QUERY_PROMPT.format(question=question)).content
    data = json.loads(raw)
    queries = data.get("queries", [])
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
    return queries[:max_queries]


def validate_citations(answer_bullets: list[str], sources: list[str]) -> None:
    # Sources formatted like "S1: Title - URL"
    source_ids = set()
    for s in sources:
        m = re.match(r"^(S\d+):", s.strip())
        if m:
            source_ids.add(m.group(1))

    if not source_ids:
        return  # nothing to validate

    for b in answer_bullets:
        m = re.search(r"\[([^\]]+)\]\s*$", b.strip())
        if not m:
            raise RuntimeError(f"Bullet missing ending citations: {b}")

        cited = {c.strip() for c in m.group(1).split(",")}
        bad = [c for c in cited if c not in source_ids]
        if bad:
            raise RuntimeError(f"Bullet cites unknown sources {bad}: {b}")


def ask_question(question: str, config: PipelineConfig) -> dict:
    llm = ChatOpenAI(model=config.model, temperature=config.temperature)

    did_search = decide_should_search(llm, question)

    context = ""
    sources_list: list[str] = []
    search_queries: list[str] = []

    if did_search:
        search_queries = generate_queries(llm, question, config.max_queries)
        if not search_queries:
            search_queries = [question]

        query = search_queries[0]
        results = search_web(query)
        results = results[: config.max_sources]

        passage_blocks: list[str] = []

        for idx, r in enumerate(results, start=1):
            sources_list.append(f"S{idx}: {r.title} - {r.url}")

            try:
                doc = fetch_url(r.url, max_chars=config.max_chars_per_source)

                passages = extract_passages(
                    llm,
                    question,
                    title=r.title or (doc.title or "Untitled"),
                    url=r.url,
                    text=doc.text,
                )

                block_lines = [
                    f"SOURCE_ID: S{idx}",
                    f"TITLE: {r.title}",
                    f"URL: {r.url}",
                    "PASSAGES:",
                ]
                for p in passages:
                    block_lines.append(f"- {p['quote']}  (why: {p['why']})")

                passage_blocks.append("\n".join(block_lines))

            except Exception as e:
                passage_blocks.append(
                    f"SOURCE_ID: S{idx}\n"
                    f"TITLE: {r.title}\n"
                    f"URL: {r.url}\n"
                    "PASSAGES:\n"
                    f"- (EXTRACTION FAILED: {e})\n"
                    f"- SNIPPET: {r.snippet}"
                )

        context = "\n\n".join(passage_blocks)

    prompt = ANSWER_PROMPT.format(
        question=question,
        context=context,
        did_search=str(did_search).lower(),
        search_queries=json.dumps(search_queries if did_search else []),
        sources_json=json.dumps(sources_list if did_search else []),
    )

    raw = llm.invoke(prompt).content
    data = json.loads(raw)

    # Your existing guardrails
    validate_citations(data.get("answer_bullets", []), data.get("sources", []))

    if did_search and not data.get("sources"):
        raise RuntimeError("Expected sources when did_search=true, got empty sources.")

    # Helpful debug fields for CLI output (optional)
    data["_meta"] = {
        "did_search": did_search,
        "search_queries": search_queries,
        "max_sources": config.max_sources,
        "model": config.model,
    }

    return data
