# LC Agent

A lightweight research assistant built to explore agent-style workflows with LangChain. The app can optionally perform live web searches, retrieve and extract relevant passages from source pages, and synthesize results into a concise, bulleted summary with citations.

The current implementation is intentionally simple and focused on the core pipeline: query → search → fetch → parse → summarize. It is designed as a playground for experimenting with retrieval, tool usage, and source attribution rather than a production-ready system.

## Features

- Optional web search to augment the model’s responses with fresh, external context
- Passage extraction from fetched web pages
- Source-aware summarization with bulleted responses and citations
- Modular agent workflow built on LangChain
- Pluggable search layer using Tavily
- HTML parsing and content extraction via Beautiful Soup

## Tech Stack

- **LangChain** – Agent orchestration and workflow framework
- **Tavily** – Web search API
- **Beautiful Soup (beautifulsoup4)** – HTML parsing and content extraction
- **OpenAI API** – LLM backend

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit the .env file and provide the required API keys:

- OPENAI_API_KEY
- TAVILY_API_KEY

## Usage

At the moment, the query is hardcoded directly in the pipeline file. To change what the agent researches, update the question in that file and re-run the script. This will likely be replaced with a CLI or API interface in a future iteration.

`python -m lc_agent.pipeline`

## Notes

This project is primarily a research and experimentation sandbox. Expect rough edges, evolving structure, and breaking changes as the pipeline and tooling are refined.
