import hashlib
import json
import re
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from datetime import datetime, timezone

from lc_agent.prompts import GATE_PROMPT, QUERY_PROMPT, ANSWER_PROMPT
from lc_agent.run_context import RunContext
from lc_agent.tools.search_tavily import search_web, SearchResult
from lc_agent.tools.fetch import fetch_url
from lc_agent.tools.extract import extract_passages
from lc_agent.run_bundle import RunBundler

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
	
def serialize_search_result(r: SearchResult) -> dict:
    return {
        "title": r.title,
        "url": r.url,
        "snippet": r.snippet,
    }

def serialize_results(results: list[SearchResult]) -> list[dict]:
    return [serialize_search_result(r) for r in results]

def url_hash(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def ask_question(question: str, config: PipelineConfig, ctx: RunContext) -> dict:
	# Set up bundler and LLM
	bundler = RunBundler(base_dir="runs")
	run_id = bundler.start()
	llm = ChatOpenAI(model=config.model, temperature=config.temperature)

	# Save initial data to bundler
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

	# Decide if we need to perform a web search
	did_search = decide_should_search(llm, question, ctx)
	meta["did_search"] = did_search
	bundler.write_json("meta.json", meta)

	context = ""
	sources_list: list[str] = []
	search_queries: list[str] = []

	if did_search:
		# Generate list of search queries
		search_queries = generate_queries(llm, question, config.max_queries, ctx)
		if not search_queries:
			search_queries = [question]
		bundler.write_json("search_queries.json", search_queries)

		# Perform searches and store in buckets
		PER_QUERY_LIMIT = max(config.max_sources, 5)
		buckets: list[list[SearchResult]] = []
		search_dump = []
		for q in search_queries:
			try:
				res = search_web(q)[:PER_QUERY_LIMIT]
			except Exception:
				res = []

			buckets.append(res)
			search_dump.append({
				"query": q,
				"results": [
					{"title": r.title, "url": r.url, "snippet": r.snippet}
					for r in res
				],
			})
		bundler.write_json("search_results.json", search_dump)

		results: list[SearchResult] = []
		seen_urls: set[str] = set()

		# Round-robin pick 1 from each bucket until max_sources
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
		# results not contains a spread of search results from each query
		bundler.write_json("selected_sources.json", serialize_results(results))

		# Pull passages from our built out list of results
		passage_blocks: list[str] = []
		for idx, r in enumerate(results, start=1):
			sources_list.append(f"S{idx}: {r.title} - {r.url}")

			try:
				doc = fetch_url(r.url, max_chars=config.max_chars_per_source)
				bundler.write_json(f"fetch/{url_hash(r.url)}.json", {
					"url": doc.url,
					"title": doc.title,
					"text": doc.text
				})

				passages = extract_passages(
					llm,
					question,
					title=r.title or (doc.title or "Untitled"),
					url=r.url,
					text=doc.text,
				)
				bundler.write_json(f"extracts/{url_hash(r.url)}.json", {
					"source_id": f"S{idx}",
					"title": r.title,
					"url": r.url,
					"passages": passages,
				})

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
		bundler.write_text("context.txt", context)

	prompt = ANSWER_PROMPT.format(
		question=question,
		context=context,
		did_search=str(did_search).lower(),
		search_queries=json.dumps(search_queries if did_search else []),
		sources_json=json.dumps(sources_list if did_search else []),
	)

	raw = llm.invoke(prompt).content
	data = json.loads(raw)

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
		"run_id": run_id,
		"run_dir": str(bundler.path())
	}
	bundler.write_json("final.json", data)
	meta = bundler.finish_meta(meta)
	bundler.write_json("meta.json", meta)

	return data
