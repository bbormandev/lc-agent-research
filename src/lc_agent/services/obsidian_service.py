import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lc_agent.services.category_registry_service import (
    CategoryRegistryServiceConfig,
    load_category_tree,
)

@dataclass(frozen=True)
class ObsidianServiceConfig:
    vault_path: Path


def slugify_for_filename(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "note"


def note_title_seed(question: str, result: dict) -> str:
    candidate = str(result.get("note_title", "")).strip()
    return candidate if candidate else question


def build_note_filename(question: str, result: dict, today: str) -> str:
    return f"{slugify_for_filename(note_title_seed(question, result))}-{today}.md"


def resolve_note_path(topics_path: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = Path(filename).suffix or ".md"
    candidate = topics_path / f"{base}{ext}"
    if not candidate.exists():
        return candidate

    i = 2
    while True:
        candidate = topics_path / f"{base}-{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


def _sanitize_note_title(title: str) -> str:
    cleaned = str(title).strip()
    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    cleaned = cleaned.replace(":", " ").replace('"', "").replace("'", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or "Untitled"


def _note_path_for_title(topics_path: Path, title: str) -> Path:
    safe_title = _sanitize_note_title(title)
    return topics_path / f"{safe_title}.md"


def _validate_vault(vault_path: str) -> tuple[Path, Path]:
    config = ObsidianServiceConfig(vault_path=Path(vault_path).expanduser())
    vault = config.vault_path
    if not vault.exists():
        raise ValueError(f"Vault path does not exist: {vault}")
    if not vault.is_dir():
        raise ValueError(f"Vault path is not a directory: {vault}")

    topics = vault / "Topics"
    topics.mkdir(parents=True, exist_ok=True)
    return vault, topics


def _extract_hub_title(question: str, topic_result: dict) -> str:
    candidate = str(topic_result.get("note_title", "")).strip()
    if candidate:
        return _sanitize_note_title(candidate)
    return _sanitize_note_title(question)


def _extract_subtopic_title(subtopic_payload: dict[str, Any], hub_title: str) -> str:
    subtopic = subtopic_payload.get("subtopic", {}) if isinstance(subtopic_payload, dict) else {}
    note_title = str((subtopic or {}).get("note_title", "")).strip()
    title = str((subtopic or {}).get("title", "")).strip()

    if note_title:
        if " - " in note_title:
            return _sanitize_note_title(note_title)
        if hub_title:
            return _sanitize_note_title(f"{hub_title} - {note_title}")
    if title:
        return _sanitize_note_title(f"{hub_title} - {title}")
    return _sanitize_note_title(f"{hub_title} - Subtopic")


def _safe_meta_value(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    return text if text else "unknown"


def _load_categories_from_run(result: dict) -> dict | None:
    meta = result.get("_meta", {}) if isinstance(result.get("_meta"), dict) else {}
    run_dir = str(meta.get("run_dir", "")).strip()
    if not run_dir:
        return None

    categories_path = Path(run_dir) / "categories.json"
    if not categories_path.exists():
        return None

    try:
        data = json.loads(categories_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _wikilink(value: str) -> str:
    text = str(value).strip()
    return f"[[{text}]]" if text else ""


def _slug_to_readable(slug: str) -> str:
    parts = [p for p in str(slug).strip().split("-") if p]
    return " ".join(part.capitalize() for part in parts)


def _build_category_title_maps(registry: dict) -> dict[str, dict[str, str]]:
    domain_titles: dict[str, str] = {}
    category_titles: dict[str, str] = {}
    subcategory_titles: dict[str, str] = {}

    for domain in registry.get("domains", []) or []:
        domain_slug = str(domain.get("slug", "")).strip()
        domain_title = str(domain.get("title", "")).strip()
        if domain_slug and domain_title:
            domain_titles[domain_slug] = domain_title

        for category in domain.get("categories", []) or []:
            category_slug = str(category.get("slug", "")).strip()
            category_title = str(category.get("title", "")).strip()
            if category_slug and category_title:
                category_titles[category_slug] = category_title

            for subcategory in category.get("subcategories", []) or []:
                subcategory_slug = str(subcategory.get("slug", "")).strip()
                subcategory_title = str(subcategory.get("title", "")).strip()
                if subcategory_slug and subcategory_title:
                    subcategory_titles[subcategory_slug] = subcategory_title

    return {
        "domain": domain_titles,
        "category": category_titles,
        "subcategory": subcategory_titles,
    }


def _load_category_title_maps(vault: Path) -> dict[str, dict[str, str]]:
    try:
        registry = load_category_tree(str(vault), CategoryRegistryServiceConfig())
    except Exception:
        return {"domain": {}, "category": {}, "subcategory": {}}
    return _build_category_title_maps(registry)


def _render_category_path_line(
    categories: dict | None,
    category_title_maps: dict[str, dict[str, str]] | None,
) -> str | None:
    if not categories:
        return None

    domain = str(categories.get("domain", "")).strip()
    category = str(categories.get("category", "")).strip()
    subcategory_raw = categories.get("subcategory")
    subcategory = str(subcategory_raw).strip() if subcategory_raw is not None else ""

    if not domain or not category:
        return None

    title_maps = category_title_maps or {"domain": {}, "category": {}, "subcategory": {}}
    domain_title = title_maps.get("domain", {}).get(domain, _slug_to_readable(domain))
    category_title = title_maps.get("category", {}).get(category, _slug_to_readable(category))

    links = [_wikilink(domain_title), _wikilink(category_title)]
    if subcategory:
        subcategory_title = title_maps.get("subcategory", {}).get(
            subcategory, _slug_to_readable(subcategory)
        )
        links.append(_wikilink(subcategory_title))

    chain = [link for link in links if link]
    if len(chain) < 2:
        return None
    return f"Categories: {' → '.join(chain)}"


def render_note_markdown(
    question: str,
    result: dict,
    today: str,
    *,
    category_title_maps: dict[str, dict[str, str]] | None = None,
) -> str:
    meta = result.get("_meta", {}) if isinstance(result.get("_meta"), dict) else {}
    categories = _load_categories_from_run(result)
    summary = str(result.get("summary", "")).strip()
    answer_bullets = result.get("answer_bullets", []) or []
    sources = result.get("sources", []) or []

    run_id = _safe_meta_value(meta.get("run_id"))
    model = _safe_meta_value(meta.get("model"))
    run_dir = _safe_meta_value(meta.get("run_dir"))
    did_search = bool(meta.get("did_search", False))

    frontmatter = [
        "---",
        f'title: "{question.replace(chr(34), chr(39))}"',
        f"created: {today}",
        f'run_id: "{run_id}"',
        f'model: "{model}"',
        f"did_search: {'true' if did_search else 'false'}",
    ]
    if categories:
        frontmatter.append(f'domain: "{str(categories.get("domain", "")).strip()}"')
        frontmatter.append(f'category: "{str(categories.get("category", "")).strip()}"')
        subcategory = categories.get("subcategory")
        if subcategory is None:
            frontmatter.append("subcategory: null")
        else:
            frontmatter.append(f'subcategory: "{str(subcategory).strip()}"')
        tags = categories.get("tags", []) or []
        frontmatter.append(f"tags: {json.dumps(tags, ensure_ascii=False)}")

    frontmatter.extend(["---", ""])

    body = [
        f"# {question}",
        "",
        "## Summary",
        summary,
    ]
    category_path_line = _render_category_path_line(categories, category_title_maps)
    if category_path_line:
        body.extend(["", category_path_line])

    body.extend(["", "## Key Points"])
    for bullet in answer_bullets:
        body.append(f"- {bullet}")

    body.extend(["", "## Sources"])
    for source in sources:
        body.append(f"- {source}")

    links = categories.get("links", {}) if categories else {}
    entities = [str(v).strip() for v in links.get("entities", []) or [] if str(v).strip()]
    concepts = [str(v).strip() for v in links.get("concepts", []) or [] if str(v).strip()]
    if entities or concepts:
        body.extend(["", "## Links"])
        for entity in entities:
            link = _wikilink(entity)
            if link:
                body.append(f"- {link}")
        for concept in concepts:
            link = _wikilink(concept)
            if link:
                body.append(f"- {link}")

    body.extend(
        [
            "",
            "## Run Metadata",
            f"- Run ID: {run_id}",
            f"- Run Dir: {run_dir}",
            f"- Date: {today}",
            "",
        ]
    )

    return "\n".join(frontmatter + body)


def render_hub_note_markdown(
    hub_title: str,
    topic_result: dict[str, Any],
    subtopic_titles: list[str],
    today: str,
    *,
    category_title_maps: dict[str, dict[str, str]] | None = None,
) -> str:
    meta = topic_result.get("_meta", {}) if isinstance(topic_result.get("_meta"), dict) else {}
    categories = _load_categories_from_run(topic_result)
    summary = str(topic_result.get("summary", "")).strip() or "No overview available."
    key_points = topic_result.get("answer_bullets", []) or []
    sources = topic_result.get("sources", []) or []
    run_id = _safe_meta_value(meta.get("run_id"))
    model = _safe_meta_value(meta.get("model"))
    did_search = bool(meta.get("did_search", False))

    frontmatter = [
        "---",
        f'title: "{hub_title.replace(chr(34), chr(39))}"',
        f"created: {today}",
        f'run_id: "{run_id}"',
        f'model: "{model}"',
        f"did_search: {'true' if did_search else 'false'}",
    ]
    if categories:
        frontmatter.append(f'domain: "{str(categories.get("domain", "")).strip()}"')
        frontmatter.append(f'category: "{str(categories.get("category", "")).strip()}"')
        subcategory = categories.get("subcategory")
        if subcategory is None:
            frontmatter.append("subcategory: null")
        else:
            frontmatter.append(f'subcategory: "{str(subcategory).strip()}"')
        tags = categories.get("tags", []) or []
        frontmatter.append(f"tags: {json.dumps(tags, ensure_ascii=False)}")
    frontmatter.extend(["---", ""])

    lines = [
        f"# {hub_title}",
        "",
        "## Core Topics",
    ]
    for title in subtopic_titles:
        lines.append(f"- [[{title}]]")

    lines.extend(["", "## Overview", summary])
    category_path_line = _render_category_path_line(categories, category_title_maps)
    if category_path_line:
        lines.extend(["", category_path_line])
    lines.extend(["", "## Key Points"])

    for bullet in key_points:
        lines.append(f"- {bullet}")

    lines.extend(["", "## Sources"])
    for source in sources:
        lines.append(f"- {source}")
    lines.append("")
    return "\n".join(frontmatter + lines)


def render_subtopic_note_markdown(
    subtopic_title: str,
    subtopic_result: dict[str, Any],
    hub_title: str,
) -> str:
    result = subtopic_result.get("result", {}) if isinstance(subtopic_result, dict) else {}
    summary = str(result.get("summary", "")).strip() or "No summary available."
    key_points = result.get("answer_bullets", []) or []
    sources = result.get("sources", []) or []

    lines = [
        f"# {subtopic_title}",
        "",
        "## Summary",
        summary,
        "",
        "## Key Points",
    ]
    for bullet in key_points:
        lines.append(f"- {bullet}")

    lines.extend(["", "## Sources"])
    for source in sources:
        lines.append(f"- {source}")

    lines.extend(["", "## Related", f"[[{hub_title}]]", ""])
    return "\n".join(lines)


def write_hub_note(topic_result: dict[str, Any], *, vault_path: str, today: str) -> dict[str, Any]:
    vault, topics = _validate_vault(vault_path)

    question = str(topic_result.get("question", "")).strip() or "Topic"
    hub_title = _extract_hub_title(question, topic_result)

    subtopic_items = (
        topic_result.get("topic_research", {}).get("subtopics", [])
        if isinstance(topic_result.get("topic_research"), dict)
        else []
    )
    subtopic_titles = [
        _extract_subtopic_title(item, hub_title)
        for item in subtopic_items
    ]

    note_path = _note_path_for_title(topics, hub_title)
    if note_path.exists():
        return {
            "title": hub_title,
            "note_path": str(note_path.resolve()),
            "note_filename": note_path.name,
            "status": "skipped_existing",
        }

    markdown = render_hub_note_markdown(
        hub_title,
        topic_result,
        subtopic_titles,
        today,
        category_title_maps=_load_category_title_maps(vault),
    )
    note_path.write_text(markdown, encoding="utf-8")
    return {
        "title": hub_title,
        "note_path": str(note_path.resolve()),
        "note_filename": note_path.name,
        "status": "created",
    }


def write_subtopic_note(
    subtopic_result: dict[str, Any],
    *,
    hub_title: str,
    vault_path: str,
    today: str,
) -> dict[str, Any]:
    _ = today  # reserved for future metadata stamping
    _, topics = _validate_vault(vault_path)

    title = _extract_subtopic_title(subtopic_result, hub_title)
    note_path = _note_path_for_title(topics, title)
    if note_path.exists():
        return {
            "title": title,
            "note_path": str(note_path.resolve()),
            "note_filename": note_path.name,
            "status": "skipped_existing",
        }

    markdown = render_subtopic_note_markdown(title, subtopic_result, hub_title)
    note_path.write_text(markdown, encoding="utf-8")
    return {
        "title": title,
        "note_path": str(note_path.resolve()),
        "note_filename": note_path.name,
        "status": "created",
    }


def publish_research(question: str, topic_result: dict[str, Any], *, vault_path: str, today: str) -> dict[str, Any]:
    vault, _ = _validate_vault(vault_path)

    enriched_result = dict(topic_result)
    enriched_result["question"] = question
    hub = write_hub_note(enriched_result, vault_path=vault_path, today=today)
    hub_title = str(hub.get("title", "")).strip() or _extract_hub_title(question, topic_result)

    subtopic_items = (
        topic_result.get("topic_research", {}).get("subtopics", [])
        if isinstance(topic_result.get("topic_research"), dict)
        else []
    )
    subtopic_notes = [
        write_subtopic_note(item, hub_title=hub_title, vault_path=vault_path, today=today)
        for item in subtopic_items
    ]

    created = sum(1 for item in [hub, *subtopic_notes] if item.get("status") == "created")
    skipped = sum(1 for item in [hub, *subtopic_notes] if item.get("status") == "skipped_existing")

    return {
        "mode": "topic_research",
        "hub": hub,
        "subtopic_notes": subtopic_notes,
        "created_count": created,
        "skipped_existing_count": skipped,
        "note_path": hub["note_path"],
        "note_filename": hub["note_filename"],
        "vault_path": str(vault.resolve()),
    }


def publish_note(question: str, result: dict, *, vault_path: str, today: str) -> dict:
    if isinstance(result.get("topic_research"), dict):
        return publish_research(question, result, vault_path=vault_path, today=today)

    vault, topics = _validate_vault(vault_path)

    filename = build_note_filename(question, result, today)
    note_path = resolve_note_path(topics, filename)
    category_title_maps = _load_category_title_maps(vault)
    markdown = render_note_markdown(
        question,
        result,
        today,
        category_title_maps=category_title_maps,
    )
    note_path.write_text(markdown, encoding="utf-8")

    return {
        "note_path": str(note_path.resolve()),
        "note_filename": note_path.name,
        "vault_path": str(vault.resolve()),
    }
