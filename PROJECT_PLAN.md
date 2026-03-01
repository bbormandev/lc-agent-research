# Personal Research Assistant – Project Plan (v3)

This plan reflects the current architecture and naming:

- Controller: CLI (`cli.py`)
- Services: `services/research_service.py`, `services/categorization_service.py`, `services/obsidian_service.py`
- Tools: `tools/search_tavily.py`, `tools/fetch.py`, `tools/extract.py`
- Infra: `infra/run_bundler.py`, `infra/run_context.py`
- Prompts: `prompts/` (research, categorization, decomposition)

The goal is to balance **motivating, user-visible progress** with **just enough structure** to keep the system debuggable and evolvable.

---

## Current Status Snapshot (as of 2026-02-22)

- **Overall status:** Phase 0 complete, Phase 1 complete, Phase 2 complete.
- **Completed recently:**
  - Added Obsidian publishing service at `services/obsidian_service.py`
  - Added CLI flag: `research ask "..." --vault /path/to/vault`
  - Implemented Markdown note rendering from `final.json`-shaped output
  - Updated publishing target to `Topics/` inside the vault (auto-created if missing)
  - Updated filename strategy to prefer model-provided `note_title` (fallback to question slug), with date suffix and collision handling
  - Added unit + CLI integration-style tests for Obsidian publishing flow
  - Implemented `services/categorization_service.py` with single-call LLM categorization and strict JSON schema response format
  - Added category registry parsing from Obsidian markdown frontmatter (`Index/Category Tree.md`)
  - Added CLI orchestration order: research → categorize → publish
  - Added `categories.json` run artifact persistence and categorization metadata in `_meta`
  - Updated Obsidian note rendering to consume `categories.json` (frontmatter categories/tags + optional wikilink Links section)
  - Added tests for categorization service + updated CLI/Obsidian tests
  - Updated `README.md` with category registry format and categorization flow
- **Current gap to close indexing roadmap:** Study Index writing (`Index/Study Index.md`) remains deferred.

---

## Vision

Build a personal research assistant that:

- Decomposes complex topics into subquestions
- Performs iterative research with source diversity
- Decides when research is “done” based on coverage and confidence
- Produces structured, cited outputs
- Writes categorized notes into an Obsidian vault
- Maintains a study index for ongoing learning
- Reuses prior research to avoid duplication (second brain)

---

## Architecture (Current Mental Model)

- **Controller (CLI)**: Orchestrates the workflow based on user input
  - Current order for `ask`: research (`ask`) → categorize (`categorize`) → publish (`publish_note`)
- **Services**:
  - `ResearchService`: research + synthesis only
  - `CategorizationService`: parse vault category registry, classify into domain/category/subcategory, normalize tags, emit links + proposals
  - `ObsidianService`: render Markdown + publish note, optionally enrich from `categories.json`
- **Tools**: web search, fetch, extract (pure helpers)
- **Infra**: run bundler, run context, caching
- **Prompts**: centralized prompt definitions

---

## Phase 0: Keep the Core Working (Baseline)

**Status:** ✅ Complete

**Goal:** Ensure the refactor did not break functionality.

### Deliverables

- CLI `ask` command still works end-to-end
- Run bundler writes artifacts per run
- Pretty JSON output to CLI

### Definition of Done

- `research ask "..."` completes successfully
- A run folder is created with `final.json` and `meta.json`

---

## Phase 1: Obsidian Writer v1 (Fun, Visible Win)

**Status:** ✅ Complete (indexing deferred by design)

**Goal:** Make the tool personally useful and motivating.

### Deliverables

- `services/obsidian_service.py`
  - Render a Markdown note from `final.json`
  - Write note into Obsidian vault
- Study Index v1
  - `Index/Study Index.md`
  - Append links under domain categories (Technology, Finance, Politics)
- CLI flag: `--vault /path/to/vault` to enable publishing

### Implemented in v1 (current)

- ✅ `services/obsidian_service.py` created
- ✅ Markdown note rendering from `final.json` data (`summary`, `answer_bullets`, `sources`, `_meta`)
- ✅ Vault publishing behind `--vault` (opt-in only)
- ✅ Notes are published to `Topics/` (created on demand)
- ✅ Deterministic filename format: `<slug(note_title|question)>-<YYYY-MM-DD>.md`
- ✅ Collision handling via numeric suffix (`-2`, `-3`, ...)
- ✅ `_meta.obsidian` metadata returned in CLI JSON when publish succeeds
- ✅ Test coverage for rendering, slugging, collisions, and CLI publishing

### Deferred after this phase

- Study Index write/update (`Index/Study Index.md`)
- Broad-category grouping and links

### Definition of Done

- Running:
  - `research ask "..." --vault /path/to/vault`
- Produces:
  - A new note in Obsidian
  - An entry in `Study Index.md` (still pending)

---

## Phase 2: Topic Categorization v1 (Fun + Structure)

**Status:** ✅ Complete

**Goal:** Enable domain → category → subcategory classification for study guides.

### Deliverables Implemented

- `services/categorization_service.py`
  - LLM-based categorizer (single call) for domain/category/subcategory + tags + links + proposals
  - Strict JSON schema response format bound at the model call
  - Lightweight validation/normalization:
    - depth <= 3
    - non-empty strings
    - confidence in [0..1]
    - tags normalized to lowercase kebab-case
    - cap enforcement (max tags: 8, max new tags: 2)
- Registry source-of-truth in Obsidian:
  - `Index/Category Tree.md` YAML frontmatter
  - markdown body ignored in v1
- CLI orchestration updated:
  - categorization is called from `cli.py` after `ask` and before `publish_note`
  - `ResearchService` remains unchanged for categorization orchestration
- Run artifact persistence:
  - `categories.json` written into run directory from CLI layer
  - `_meta.categorization` includes enablement, artifact path, registry path, and skipped reason when applicable
- Obsidian writer integration:
  - consumes run `categories.json` when present
  - frontmatter includes `domain`, `category`, `subcategory`, and `tags`
  - note body includes optional `Links` section with `[[wikilinks]]` from entities/concepts
- Tests:
  - new `tests/test_categorization_service.py`
  - updated `tests/test_cli_obsidian.py`
  - updated `tests/test_obsidian_service.py`

### Definition of Done Status

- ✅ Notes include frontmatter categories/tags when `categories.json` exists
- ✅ Categories can be re-rendered from run artifacts without re-running research
- ⏳ Study Index domain-category grouping remains deferred to later phase

---

## Phase 3: Topic Decomposition v1 (Fun, Smarter Research)

**Status:** Not Started

**Goal:** Break complex questions into subtopics.

### Deliverables

- `prompts/decomposition.py`
- Decomposition step in `ResearchService`
  - Generate subquestions
  - Research each subquestion once
- Final note includes subtopic links

### Definition of Done

- Complex questions result in multiple subtopic sections
- Obsidian notes link to subtopic notes (or sections)

---

## Phase 4: Iterative Search v1 (Smarter Research, Controlled)

**Status:** Not Started

**Goal:** Add a second research pass for missing gaps.

### Deliverables

- Gap analysis prompt
- Second-pass research for uncovered subtopics
- Hard cap on iterations (e.g., max 2 rounds)

### Definition of Done

- The system can identify missing areas
- Performs one additional research round
- Stops deterministically

---

## Phase 5: Evals v1 (Lightweight Quality Gate)

**Status:** Not Started

**Goal:** Lock in quality before complexity grows.

### Deliverables

- `research eval` command
- 5–10 fixed eval cases
- Checks:
  - JSON schema
  - Citation formatting
  - Gate correctness
  - Run bundle completeness

### Definition of Done

- Eval report produced
- Prompt/model changes can be compared

---

## Phase 6: Caching v1 (Structural, Boring but Helpful)

**Status:** Not Started

**Goal:** Speed up iteration and reduce cost.

### Deliverables

- Fetch cache (URL → content)
- Extract cache (URL hash + prompt version → passages)
- Cache stats command

### Definition of Done

- Re-running the same question hits cache
- Extracts are reused when inputs are unchanged

---

## Phase 7: Vault-Aware Research (Second Brain)

**Status:** Not Started

**Goal:** Avoid re-researching topics.

### Deliverables

- Search Obsidian vault before web search
- Use prior notes as context
- “Refresh topic” command to update existing notes

### Definition of Done

- Existing notes influence new research
- Refresh updates notes instead of duplicating them

---

## Task Ordering (Motivation-Balanced)

1. Obsidian Writer v1 (visible payoff)
2. Categorization v1 (makes notes feel structured)
3. Topic Decomposition v1 (agent feels smarter)
4. Iterative Search v1 (research feels real)
5. Evals v1 (quality guardrails)
6. Caching v1 (iteration speed)
7. Vault-aware research (second brain)

---

## Non-Goals (For Now)

- Multi-user support
- Hosted dashboards
- Heavy agent frameworks (LangGraph) before iteration loops exist
- Perfect confidence metrics

---

## Notes for Future Extensions

- Add per-category indexes (e.g., `Technology.md`, `Finance.md`)
- Add spaced repetition or study plans
- Add scheduled refresh jobs for tracked topics
- Consider LangGraph only after iterative loops stabilize

---

## Success Criteria

The project is successful when:

- You use it weekly for real research
- Your Obsidian vault becomes a trusted study resource
- The agent stops re-researching the same topics
- Changes to prompts/models are guided by evals

---
