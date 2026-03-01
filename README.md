# LC Agent

A lightweight research assistant built to explore agent-style workflows with LangChain. The app can optionally perform live web searches, retrieve and extract relevant passages from source pages, and synthesize results into a concise, bulleted summary with citations.

The current implementation is intentionally simple and focused on the core pipeline: query → search → fetch → parse → summarize. It is designed as a playground for experimenting with retrieval, tool usage, and source attribution rather than a production-ready system.

## Features

- Optional web search to augment the model’s responses with fresh, external context
- Passage extraction from fetched web pages
- Source-aware summarization with bulleted responses and citations
- Run artifact bundling per execution (`meta.json`, `final.json`, search/fetch/extract artifacts)
- Topic categorization artifact generation (`categories.json`) when using `--vault`
- Optional Obsidian vault publishing to Markdown notes via `--vault`
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
pip install -e .
cp .env.example .env
```

Edit the .env file and provide the required API keys:

- OPENAI_API_KEY
- TAVILY_API_KEY

## Usage

You can ask a question through the CLI using the `research` command and the `ask` subcommand while passing in the question.

### Basic Ask

```bash
research ask "{question}"
```

### Ask with Source Limit

```bash
research ask "{question}" --max-sources 5
```

### Ask and Publish to Obsidian Vault

When `--vault` is provided, the CLI now runs:
1. research (`final.json`)
2. categorization (if `Index/Category Tree.md` exists in the vault)
3. Obsidian note publish

Categorization metadata is surfaced in `_meta.categorization`, and publish metadata remains under `_meta.obsidian`.

```bash
research ask "{question}" --vault /path/to/your/obsidian-vault
```

Generated note behavior:
- Output directory: `Topics/` under the provided vault path (created automatically if missing)
- Filename format: `<slug(note_title)>-<YYYY-MM-DD>.md` (falls back to question slug if `note_title` is unavailable)
- Collision handling: appends numeric suffix (`-2`, `-3`, ...)
- Note template: YAML frontmatter + `Summary`, `Key Points`, `Sources`, optional `Links`, `Run Metadata`
- If run `categories.json` exists, frontmatter includes `broad`, `refined`, `subrefined`, and `tags`

### Category Registry in Obsidian

For categorization, create this file in your vault:

- `Index/Category Tree.md`

Use YAML frontmatter as the source of truth:

```yaml
---
version: 1
broad_categories:
  - name: technology
    refined_categories:
      - name: machine-learning
        subrefined_categories: [llm-deployment, evaluation]
canonical_tags: [privacy, open-source, cost-optimization]
rules:
  max_depth: 3
  max_tags: 8
  max_new_tags: 2
---
```

Markdown body content below the frontmatter is ignored in v1.

## Running Tests

This project currently uses Python's built-in `unittest` test runner.

Recommended setup (ensures dependencies are installed in a local venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run all tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v
```

Run a single test module:

```bash
PYTHONPATH=src python -m unittest tests.test_obsidian_service -v
```

Run a single test case:

```bash
PYTHONPATH=src python -m unittest tests.test_cli_obsidian.TestCliObsidianPublish.test_cli_with_vault_orders_ask_categorize_publish -v
```

## Notes

This project is primarily a research and experimentation sandbox. Expect rough edges, evolving structure, and breaking changes as the pipeline and tooling are refined.
