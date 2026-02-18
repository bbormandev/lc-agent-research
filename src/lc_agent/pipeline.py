import json
from dataclasses import dataclass
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from lc_agent.tools.extract import extract_passages
from lc_agent.tools.search_tavily import search_web, SearchResult
from lc_agent.prompts import (GATE_PROMPT, QUERY_PROMPT, ANSWER_PROMPT)
from lc_agent.tools.fetch import fetch_url

load_dotenv()

def decide_should_search(llm: ChatOpenAI, question: str) -> bool:
	decision = llm.invoke(GATE_PROMPT.format(question=question)).content.strip().upper()
	return decision == "YES"

def generate_queries(llm: ChatOpenAI, question: str) -> list[str]:
	raw = llm.invoke(QUERY_PROMPT.format(question=question)).content
	data = json.loads(raw)
	queries = data.get("queries", [])
	# defensive cleanup
	queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
	return queries[:3]

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
        # Must end with [...] and contain S# refs
        m = re.search(r"\[([^\]]+)\]\s*$", b.strip())
        if not m:
            raise RuntimeError(f"Bullet missing ending citations: {b}")

        cited = {c.strip() for c in m.group(1).split(",")}
        bad = [c for c in cited if c not in source_ids]
        if bad:
            raise RuntimeError(f"Bullet cites unknown sources {bad}: {b}")

def main() -> None:
	llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

	# Try these:
	# question = "What changed in Python 3.13 compared to 3.12?"
	question = "Compare Redis vs RabbitMQ for background jobs."
	# question = "Explain what a queue is in simple terms."

	did_search = decide_should_search(llm, question)

	context = ""
	sources_list: list[str] = []
	search_queries: list[str] = []

	if did_search:
		search_queries = generate_queries(llm, question)
		if not search_queries:
			# fallback: use the question
			search_queries = [question]

		query = search_queries[0]
		results = search_web(query)

		MAX_SOURCES = 2
		MAX_CHARS_PER_SOURCE = 6000
		TOTAL_CONTEXT_CHARS = 12000

		results = results[:MAX_SOURCES]
		passage_blocks = []

		for idx, r in enumerate(results, start=1):
			sources_list.append(f"S{idx}: {r.title} - {r.url}")

			try:
				doc = fetch_url(r.url, max_chars=MAX_CHARS_PER_SOURCE)
				passages = extract_passages(
					llm,
					question,
					title=r.title or (doc.title or "Untitled"),
					url=r.url,
					text=doc.text,
				)

				# Build a compact block per source
				block_lines = [
					f"SOURCE_ID: S{idx}\n"
					f"TITLE: {r.title}\n"
					f"URL: {r.url}\n"
					"PASSAGES:",
				]
				for p in passages:
					block_lines.append(f"- {p['quote']}  (why: {p['why']})")

				passage_blocks.append("\n".join(block_lines))

			except Exception as e:
				# fallback: keep snippet only
				passage_blocks.append(
					f"SOURCE: {r.title}\nURL: {r.url}\nPASSAGES:\n- (EXTRACTION FAILED: {e})\n- SNIPPET: {r.snippet}"
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
	validate_citations(data["answer_bullets"], data["sources"])

	# sanity check
	if did_search and not data["sources"]:
		raise RuntimeError("Expected sources when did_search=true, got empty sources.")

	print(json.dumps(data, indent=2))

if __name__ == "__main__":
	main()
