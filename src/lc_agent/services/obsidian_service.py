import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObsidianServiceConfig:
    vault_path: Path


def slugify_question(question: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
    return slug or "note"


def build_note_filename(question: str, today: str) -> str:
    return f"{slugify_question(question)}-{today}.md"


def resolve_note_path(vault_path: Path, filename: str) -> Path:
    base = Path(filename).stem
    ext = Path(filename).suffix or ".md"
    candidate = vault_path / f"{base}{ext}"
    if not candidate.exists():
        return candidate

    i = 2
    while True:
        candidate = vault_path / f"{base}-{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1


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


def render_note_markdown(question: str, result: dict, today: str) -> str:
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
        frontmatter.append(f'broad: "{str(categories.get("broad", "")).strip()}"')
        frontmatter.append(f'refined: "{str(categories.get("refined", "")).strip()}"')
        subrefined = categories.get("subrefined")
        if subrefined is None:
            frontmatter.append("subrefined: null")
        else:
            frontmatter.append(f'subrefined: "{str(subrefined).strip()}"')
        tags = categories.get("tags", []) or []
        frontmatter.append(f"tags: {json.dumps(tags, ensure_ascii=False)}")

    frontmatter.extend(["---", ""])

    body = [
        f"# {question}",
        "",
        "## Summary",
        summary,
        "",
        "## Key Points",
    ]
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


def publish_note(question: str, result: dict, *, vault_path: str, today: str) -> dict:
    config = ObsidianServiceConfig(vault_path=Path(vault_path).expanduser())
    vault = config.vault_path
    if not vault.exists():
        raise ValueError(f"Vault path does not exist: {vault}")
    if not vault.is_dir():
        raise ValueError(f"Vault path is not a directory: {vault}")

    filename = build_note_filename(question, today)
    note_path = resolve_note_path(vault, filename)
    markdown = render_note_markdown(question, result, today)
    note_path.write_text(markdown, encoding="utf-8")

    return {
        "note_path": str(note_path.resolve()),
        "note_filename": note_path.name,
        "vault_path": str(vault.resolve()),
    }
