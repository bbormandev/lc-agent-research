import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_openai import ChatOpenAI

from lc_agent.prompts.categorization import (
    CATEGORIZATION_JSON_SCHEMA,
    CATEGORIZATION_SYSTEM_PROMPT,
    CATEGORIZATION_USER_PROMPT_TEMPLATE,
)


@dataclass
class CategorizationServiceConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    registry_rel_path: str = "Index/Category Tree.md"
    max_tags: int = 8
    max_new_tags: int = 2


def resolve_registry_path(vault_path: str, config: CategorizationServiceConfig) -> Path:
    return Path(vault_path).expanduser() / config.registry_rel_path


def _extract_frontmatter(markdown_text: str) -> dict[str, Any]:
    if not markdown_text.startswith("---"):
        raise ValueError("Category registry markdown is missing YAML frontmatter")

    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", markdown_text, flags=re.DOTALL)
    if not match:
        raise ValueError("Unable to parse YAML frontmatter from category registry markdown")

    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("Category registry frontmatter must be a mapping/object")
    return data


def load_registry_from_markdown(vault_path: str, config: CategorizationServiceConfig) -> tuple[dict[str, Any], str]:
    registry_path = resolve_registry_path(vault_path, config)
    if not registry_path.exists():
        raise FileNotFoundError(f"Category registry markdown not found: {registry_path}")

    markdown_text = registry_path.read_text(encoding="utf-8")
    registry = _extract_frontmatter(markdown_text)

    if not isinstance(registry.get("broad_categories"), list):
        raise ValueError("Registry frontmatter missing broad_categories list")
    if not isinstance(registry.get("canonical_tags", []), list):
        raise ValueError("Registry frontmatter canonical_tags must be a list")

    return registry, str(registry_path.resolve())


def normalize_tag(tag: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", tag.lower()).strip("-")
    return normalized


def _normalize_nonempty_strings(values: list[Any]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
    return cleaned


def _registry_sets(registry: dict[str, Any]) -> tuple[set[str], set[str], set[str], set[str]]:
    broad_set: set[str] = set()
    refined_set: set[str] = set()
    subrefined_set: set[str] = set()

    for broad in registry.get("broad_categories", []):
        broad_name = normalize_tag(str((broad or {}).get("name", "")))
        if broad_name:
            broad_set.add(broad_name)

        for refined in (broad or {}).get("refined_categories", []) or []:
            refined_name = normalize_tag(str((refined or {}).get("name", "")))
            if refined_name:
                refined_set.add(refined_name)

            for sub in (refined or {}).get("subrefined_categories", []) or []:
                sub_name = normalize_tag(str(sub))
                if sub_name:
                    subrefined_set.add(sub_name)

    canonical_tags = {
        normalize_tag(str(tag))
        for tag in (registry.get("canonical_tags", []) or [])
        if normalize_tag(str(tag))
    }
    return broad_set, refined_set, subrefined_set, canonical_tags


def validate_and_normalize(result: dict[str, Any], registry: dict[str, Any], config: CategorizationServiceConfig) -> dict[str, Any]:
    broad = normalize_tag(str(result.get("broad", "")))
    refined = normalize_tag(str(result.get("refined", "")))
    sub_raw = result.get("subrefined")
    subrefined = None
    if sub_raw is not None:
        sub_text = normalize_tag(str(sub_raw))
        subrefined = sub_text if sub_text else None

    if not broad:
        raise ValueError("Categorization output broad must be non-empty")
    if not refined:
        raise ValueError("Categorization output refined must be non-empty")

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError("Categorization output confidence must be numeric")
    confidence = float(confidence)
    if confidence < 0 or confidence > 1:
        raise ValueError("Categorization output confidence must be in [0,1]")

    raw_tags = result.get("tags", []) or []
    if not isinstance(raw_tags, list):
        raise ValueError("Categorization output tags must be a list")

    normalized_tags: list[str] = []
    for tag in raw_tags:
        nt = normalize_tag(str(tag))
        if nt and nt not in normalized_tags:
            normalized_tags.append(nt)

    broad_set, _, _, canonical_tags = _registry_sets(registry)

    canonical_selected = [t for t in normalized_tags if t in canonical_tags]
    novel_selected = [t for t in normalized_tags if t not in canonical_tags]
    novel_selected = novel_selected[: config.max_new_tags]
    tags = (canonical_selected + novel_selected)[: config.max_tags]

    links = result.get("links", {}) or {}
    entities = _normalize_nonempty_strings(links.get("entities", []) or [])
    concepts = _normalize_nonempty_strings(links.get("concepts", []) or [])

    proposals_in = result.get("proposed_new_categories", {}) or {}
    proposed = {
        "broad": [normalize_tag(s) for s in _normalize_nonempty_strings(proposals_in.get("broad", []) or [])],
        "refined": [normalize_tag(s) for s in _normalize_nonempty_strings(proposals_in.get("refined", []) or [])],
        "subrefined": [normalize_tag(s) for s in _normalize_nonempty_strings(proposals_in.get("subrefined", []) or [])],
    }

    # Broad categories are fixed for adoption; if model selected a novel broad, keep selection
    # but ensure it is explicitly logged as a proposal.
    if broad not in broad_set and broad not in proposed["broad"]:
        proposed["broad"].append(broad)

    cleaned_proposed = {
        key: value
        for key, value in proposed.items()
        if value
    }

    return {
        "broad": broad,
        "refined": refined,
        "subrefined": subrefined,
        "tags": tags,
        "links": {
            "entities": entities,
            "concepts": concepts,
        },
        "confidence": confidence,
        "proposed_new_categories": cleaned_proposed,
    }


def categorize(
    question: str,
    final_result: dict[str, Any],
    *,
    vault_path: str,
    config: CategorizationServiceConfig,
) -> dict[str, Any]:
    registry, _ = load_registry_from_markdown(vault_path, config)

    llm = ChatOpenAI(model=config.model, temperature=config.temperature)
    llm = llm.bind(
        response_format={
            "type": "json_schema",
            "json_schema": CATEGORIZATION_JSON_SCHEMA,
        }
    )

    final_payload = {
        "summary": final_result.get("summary"),
        "answer_bullets": final_result.get("answer_bullets", []),
        "sources": final_result.get("sources", []),
    }

    user_prompt = CATEGORIZATION_USER_PROMPT_TEMPLATE.format(
        question=question,
        final_json=json.dumps(final_payload, ensure_ascii=False),
        registry_json=json.dumps(registry, ensure_ascii=False),
    )

    raw = llm.invoke(
        [
            ("system", CATEGORIZATION_SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    ).content

    if not isinstance(raw, str):
        raise RuntimeError("Categorization model returned non-text content")

    parsed = json.loads(raw)
    return validate_and_normalize(parsed, registry, config)


def write_categories_artifact(run_dir: str, categories: dict[str, Any]) -> str:
    path = Path(run_dir) / "categories.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(categories, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
