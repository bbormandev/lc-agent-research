import os
from dataclasses import dataclass
from tavily import TavilyClient


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


def search_web(query: str, *, max_results: int = 5) -> list[SearchResult]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set. Add it to your .env file.")

    client = TavilyClient(api_key=api_key)

    # Tavily returns a dict containing "results" (list of dicts with url/title/content)
    resp = client.search(
        query=query,
        max_results=max_results,
        search_depth="basic",  # keep MVP simple/cheap
        include_answer=False,
        include_raw_content=False,
    )

    results = []
    for r in resp.get("results", []):
        results.append(
            SearchResult(
                title=(r.get("title") or "").strip(),
                url=(r.get("url") or "").strip(),
                snippet=(r.get("content") or "").strip(),
            )
        )

    # Basic safety: drop empties
    return [r for r in results if r.url and (r.title or r.snippet)]
