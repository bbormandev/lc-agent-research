import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI

from lc_agent.services.category_registry_service import (
    CategoryRegistryServiceConfig,
    load_category_tree,
    resolve_registry_path as resolve_compiled_registry_path,
)
from lc_agent.prompts.categorization import (
    CATEGORIZATION_JSON_SCHEMA,
    CATEGORIZATION_SYSTEM_PROMPT,
    CATEGORIZATION_USER_PROMPT_TEMPLATE,
)


@dataclass
class CategorizationServiceConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tags: int = 8
    max_new_tags: int = 2


def resolve_registry_path(vault_path: str, config: CategorizationServiceConfig):
    registry_config = CategoryRegistryServiceConfig(
        max_tags=config.max_tags,
        max_new_tags=config.max_new_tags,
    )
    return resolve_compiled_registry_path(vault_path, registry_config)


def load_registry_from_json(vault_path: str, config: CategorizationServiceConfig) -> tuple[dict[str, Any], str]:
    registry_config = CategoryRegistryServiceConfig(
        max_tags=config.max_tags,
        max_new_tags=config.max_new_tags,
    )
    registry = load_category_tree(vault_path, registry_config)
    path = resolve_compiled_registry_path(vault_path, registry_config)
    return registry, str(path.resolve())


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
    domain_set: set[str] = set()
    category_set: set[str] = set()
    subcategory_set: set[str] = set()

    for domain in registry.get("domains", []):
        domain_name = normalize_tag(str((domain or {}).get("slug", "")))
        if domain_name:
            domain_set.add(domain_name)

        for category in (domain or {}).get("categories", []) or []:
            category_name = normalize_tag(str((category or {}).get("slug", "")))
            if category_name:
                category_set.add(category_name)

            for sub in (category or {}).get("subcategories", []) or []:
                sub_name = normalize_tag(str((sub or {}).get("slug", "")))
                if sub_name:
                    subcategory_set.add(sub_name)

    canonical_tags = {
        normalize_tag(str(tag))
        for tag in (registry.get("canonical_tags", []) or [])
        if normalize_tag(str(tag))
    }
    return domain_set, category_set, subcategory_set, canonical_tags


def validate_and_normalize(result: dict[str, Any], registry: dict[str, Any], config: CategorizationServiceConfig) -> dict[str, Any]:
    domain = normalize_tag(str(result.get("domain", "")))
    category = normalize_tag(str(result.get("category", "")))
    sub_raw = result.get("subcategory")
    subcategory = None
    if sub_raw is not None:
        sub_text = normalize_tag(str(sub_raw))
        subcategory = sub_text if sub_text else None

    if not domain:
        raise ValueError("Categorization output domain must be non-empty")
    if not category:
        raise ValueError("Categorization output category must be non-empty")

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

    domain_set, _, _, canonical_tags = _registry_sets(registry)

    canonical_selected = [t for t in normalized_tags if t in canonical_tags]
    novel_selected = [t for t in normalized_tags if t not in canonical_tags]
    novel_selected = novel_selected[: config.max_new_tags]
    tags = (canonical_selected + novel_selected)[: config.max_tags]

    links = result.get("links", {}) or {}
    entities = _normalize_nonempty_strings(links.get("entities", []) or [])
    concepts = _normalize_nonempty_strings(links.get("concepts", []) or [])

    proposals_in = result.get("proposed_new_categories", {}) or {}
    proposed = {
        "domain": [normalize_tag(s) for s in _normalize_nonempty_strings(proposals_in.get("domain", []) or [])],
        "category": [normalize_tag(s) for s in _normalize_nonempty_strings(proposals_in.get("category", []) or [])],
        "subcategory": [
            normalize_tag(s)
            for s in _normalize_nonempty_strings(proposals_in.get("subcategory", []) or [])
        ],
    }

    if domain not in domain_set and domain not in proposed["domain"]:
        proposed["domain"].append(domain)

    cleaned_proposed = {
        key: value
        for key, value in proposed.items()
        if value
    }

    return {
        "domain": domain,
        "category": category,
        "subcategory": subcategory,
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
    registry, _ = load_registry_from_json(vault_path, config)

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
