import re
from dataclasses import dataclass
from typing import Optional

import requests
from bs4 import BeautifulSoup


@dataclass
class FetchedDoc:
    url: str
    title: Optional[str]
    text: str


def _clean_text(text: str) -> str:
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_url(url: str, *, timeout_s: int = 15, max_chars: int = 12_000) -> FetchedDoc:
    """
    Fetch a URL and return cleaned, best-effort main text.
    MVP approach: pull all visible text from the page.
    Later we can improve with readability extraction.
    """
    headers = {
        "User-Agent": "lc-agent/0.1 (research-assistant; +https://example.local)",
    }

    resp = requests.get(url, headers=headers, timeout=timeout_s)
    resp.raise_for_status()

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # Remove junk
    for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else None

    text = soup.get_text(separator=" ")
    text = _clean_text(text)

    # Truncate so you don't blow up context size
    if len(text) > max_chars:
        text = text[:max_chars] + "…"

    return FetchedDoc(url=url, title=title, text=text)
