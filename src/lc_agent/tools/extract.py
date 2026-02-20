import json
from langchain_openai import ChatOpenAI
from lc_agent.prompts import EXTRACT_PROMPT

def extract_passages(llm: ChatOpenAI, question: str, *, title: str, url: str, text: str) -> list[dict]:
	raw = llm.invoke(EXTRACT_PROMPT.format(
		question=question,
		title=title,
		url=url,
		text=text,
	)).content

	data = json.loads(raw)
	passages = data.get("passages", [])
	# defensive cleanup
	cleaned = []
	for p in passages:
		quote = (p.get("quote") or "").strip()
		why = (p.get("why") or "").strip()
		if quote:
			cleaned.append({"quote": quote[:300], "why": why[:120]})
	return cleaned[:5]