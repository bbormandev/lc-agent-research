import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from lc_agent.prompts import GATE_PROMPT, QUERY_PROMPT, ANSWER_PROMPT
from lc_agent.run_context import RunContext
from lc_agent.tools.search_tavily import search_web, SearchResult
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


def decide_should_search(llm: ChatOpenAI, question: str, ctx: RunContext) -> bool:
	decision = llm.invoke(GATE_PROMPT.format(today=ctx.today, question=question)).content.strip().upper()
	return decision == "YES"


def generate_queries(llm: ChatOpenAI, question: str, max_queries: int, ctx: RunContext) -> list[str]:
	raw = llm.invoke(QUERY_PROMPT.format(question=question, today=ctx.today)).content
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
		
def validate_summary(summary: str) -> None:
    if not isinstance(summary, str) or not summary.strip():
        raise RuntimeError("Missing or empty summary")
    if "[" in summary or "]" in summary:
        raise RuntimeError(f"Summary must not contain citations/brackets: {summary}")


def ask_question(question: str, config: PipelineConfig, ctx: RunContext) -> dict:
	llm = ChatOpenAI(model=config.model, temperature=config.temperature)

	did_search = decide_should_search(llm, question, ctx)

	context = ""
	sources_list: list[str] = []
	search_queries: list[str] = []

	if did_search:
		search_queries = generate_queries(llm, question, config.max_queries, ctx)
		if not search_queries:
			search_queries = [question]

		PER_QUERY_LIMIT = max(config.max_sources, 5)

		buckets: list[list[SearchResult]] = []
		for q in search_queries:
			try:
				buckets.append(search_web(q)[:PER_QUERY_LIMIT])
			except Exception:
				buckets.append([])

		results: list[SearchResult] = []
		seen_urls: set[str] = set()

		# round-robin pick 1 from each bucket until max_sources
		i = 0
		while len(results) < config.max_sources:
			progressed = False
			for b in buckets:
				if i < len(b):
					r = b[i]
					url = (r.url or "").strip()
					if url and url not in seen_urls:
						seen_urls.add(url)
						results.append(r)
						progressed = True
						if len(results) >= config.max_sources:
							break
			if not progressed:
				break
			i += 1


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
	summary = data.get("summary")
	if summary is None:
		raise RuntimeError("Response missing required field: summary")
	validate_summary(summary)

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
