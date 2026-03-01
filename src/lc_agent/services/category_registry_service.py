import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CategoryRegistryServiceConfig:
    categories_rel_dir: str = "Index/Categories"
    registry_rel_path: str = "Index/category_tree.json"
    meta_rel_path: str = "Index/category_tree.meta.json"
    max_depth: int = 3
    max_tags: int = 8
    max_new_tags: int = 2


@dataclass(frozen=True)
class SourceFileSnapshot:
    path: str
    mtime_ns: int
    size: int


def resolve_categories_dir(vault_path: str, config: CategoryRegistryServiceConfig) -> Path:
    return Path(vault_path).expanduser() / config.categories_rel_dir


def resolve_registry_path(vault_path: str, config: CategoryRegistryServiceConfig) -> Path:
    return Path(vault_path).expanduser() / config.registry_rel_path


def resolve_meta_path(vault_path: str, config: CategoryRegistryServiceConfig) -> Path:
    return Path(vault_path).expanduser() / config.meta_rel_path


def _normalize_slug(value: object) -> str:
    text = str(value).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return normalized


def _extract_frontmatter(markdown_text: str) -> dict[str, Any]:
    import yaml

    if not markdown_text.startswith("---"):
        raise ValueError("Category note is missing YAML frontmatter")

    match = re.match(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", markdown_text, flags=re.DOTALL)
    if not match:
        raise ValueError("Unable to parse YAML frontmatter from category note")

    data = yaml.safe_load(match.group(1))
    if not isinstance(data, dict):
        raise ValueError("Category note frontmatter must be a mapping/object")
    return data


def _build_source_snapshot(vault: Path, categories_dir: Path) -> list[SourceFileSnapshot]:
    if not categories_dir.exists():
        return []

    snapshots: list[SourceFileSnapshot] = []
    for path in sorted(categories_dir.glob("*.md"), key=lambda p: p.name.lower()):
        stat = path.stat()
        snapshots.append(
            SourceFileSnapshot(
                path=str(path.relative_to(vault)),
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
            )
        )
    return snapshots


def _snapshot_hash(snapshots: list[SourceFileSnapshot]) -> str:
    encoded = json.dumps(
        [{"path": s.path, "mtime_ns": s.mtime_ns, "size": s.size} for s in snapshots],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_registry_stale(
    vault_path: str,
    config: CategoryRegistryServiceConfig,
) -> tuple[bool, str]:
    vault = Path(vault_path).expanduser()
    categories_dir = resolve_categories_dir(vault_path, config)
    registry_path = resolve_registry_path(vault_path, config)
    meta_path = resolve_meta_path(vault_path, config)

    snapshots = _build_source_snapshot(vault, categories_dir)
    if not registry_path.exists():
        return True, "registry_missing"
    if not meta_path.exists():
        return True, "meta_missing"

    if snapshots:
        newest_source = max(s.mtime_ns for s in snapshots)
        registry_mtime = registry_path.stat().st_mtime_ns
        if newest_source > registry_mtime:
            return True, "source_newer_than_registry"

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return True, "meta_unreadable"

    expected_hash = _snapshot_hash(snapshots)
    if str(meta.get("snapshot_hash", "")) != expected_hash:
        return True, "snapshot_mismatch"

    return False, "fresh"


def compile_category_tree(
    vault_path: str,
    config: CategoryRegistryServiceConfig,
) -> dict[str, Any]:
    vault = Path(vault_path).expanduser()
    categories_dir = resolve_categories_dir(vault_path, config)
    if not categories_dir.exists():
        categories_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for path in sorted(categories_dir.glob("*.md"), key=lambda p: p.name.lower()):
        frontmatter = _extract_frontmatter(path.read_text(encoding="utf-8"))

        if str(frontmatter.get("type", "")).strip() != "category":
            raise ValueError(f"Category note {path} must have type: category")

        level = str(frontmatter.get("level", "")).strip()
        if level not in {"domain", "category", "subcategory"}:
            raise ValueError(f"Category note {path} has invalid level: {level}")

        slug = _normalize_slug(frontmatter.get("slug", ""))
        if not slug:
            raise ValueError(f"Category note {path} must define a non-empty slug")

        parent = frontmatter.get("parent")
        parent_slug = _normalize_slug(parent) if parent is not None else None
        if level in {"category", "subcategory"} and not parent_slug:
            raise ValueError(f"Category note {path} must define parent for level={level}")

        records.append(
            {
                "level": level,
                "slug": slug,
                "title": path.stem.strip(),
                "parent": parent_slug,
                "path": str(path),
            }
        )

    domain_records = [r for r in records if r["level"] == "domain"]
    category_records = [r for r in records if r["level"] == "category"]
    subcategory_records = [r for r in records if r["level"] == "subcategory"]

    domains: dict[str, dict[str, Any]] = {}
    for rec in domain_records:
        slug = rec["slug"]
        if slug in domains:
            raise ValueError(f"Duplicate domain slug: {slug}")
        domains[slug] = {"slug": slug, "title": rec["title"], "categories": []}

    categories: dict[str, dict[str, Any]] = {}
    for rec in category_records:
        slug = rec["slug"]
        if slug in categories:
            raise ValueError(f"Duplicate category slug: {slug}")
        parent = rec["parent"]
        if parent not in domains:
            raise ValueError(f"Category {slug} references missing domain parent: {parent}")
        node = {"slug": slug, "title": rec["title"], "subcategories": []}
        categories[slug] = node
        domains[parent]["categories"].append(node)

    for rec in subcategory_records:
        slug = rec["slug"]
        parent = rec["parent"]
        if parent not in categories:
            raise ValueError(f"Subcategory {slug} references missing category parent: {parent}")
        siblings = categories[parent]["subcategories"]
        if any(existing["slug"] == slug for existing in siblings):
            raise ValueError(f"Duplicate subcategory slug under {parent}: {slug}")
        siblings.append({"slug": slug, "title": rec["title"]})

    sorted_domains = sorted(domains.values(), key=lambda d: d["title"].lower())
    for domain in sorted_domains:
        domain["categories"] = sorted(domain["categories"], key=lambda c: c["title"].lower())
        for category in domain["categories"]:
            category["subcategories"] = sorted(
                category["subcategories"], key=lambda s: s["title"].lower()
            )

    generated_at = datetime.now(timezone.utc).isoformat()
    return {
        "version": 1,
        "generated_at": generated_at,
        "domains": sorted_domains,
        "canonical_tags": [],
        "rules": {
            "max_depth": config.max_depth,
            "max_tags": config.max_tags,
            "max_new_tags": config.max_new_tags,
        },
    }


def write_compiled_registry(
    vault_path: str,
    registry: dict[str, Any],
    config: CategoryRegistryServiceConfig,
) -> tuple[str, str]:
    vault = Path(vault_path).expanduser()
    categories_dir = resolve_categories_dir(vault_path, config)
    snapshots = _build_source_snapshot(vault, categories_dir)
    snapshot_hash = _snapshot_hash(snapshots)

    registry_path = resolve_registry_path(vault_path, config)
    meta_path = resolve_meta_path(vault_path, config)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "version": 1,
        "generated_at": registry.get("generated_at"),
        "categories_dir": str(categories_dir),
        "source_files": [
            {"path": s.path, "mtime_ns": s.mtime_ns, "size": s.size}
            for s in snapshots
        ],
        "snapshot_hash": snapshot_hash,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(registry_path.resolve()), str(meta_path.resolve())


def ensure_category_registry_fresh(
    vault_path: str,
    config: CategoryRegistryServiceConfig,
) -> dict[str, Any]:
    stale, reason = is_registry_stale(vault_path, config)
    registry_path = resolve_registry_path(vault_path, config)
    meta_path = resolve_meta_path(vault_path, config)

    if stale:
        registry = compile_category_tree(vault_path, config)
        registry_path_resolved, meta_path_resolved = write_compiled_registry(
            vault_path, registry, config
        )
        return {
            "registry": registry,
            "registry_path": registry_path_resolved,
            "meta_path": meta_path_resolved,
            "rebuilt": True,
            "stale_reason": reason,
        }

    registry = load_category_tree(vault_path, config)
    return {
        "registry": registry,
        "registry_path": str(registry_path.resolve()),
        "meta_path": str(meta_path.resolve()),
        "rebuilt": False,
        "stale_reason": None,
    }


def load_category_tree(vault_path: str, config: CategoryRegistryServiceConfig) -> dict[str, Any]:
    registry_path = resolve_registry_path(vault_path, config)
    if not registry_path.exists():
        raise FileNotFoundError(f"Compiled category registry not found: {registry_path}")

    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Compiled category registry must be a JSON object")
    if not isinstance(data.get("domains"), list):
        raise ValueError("Compiled category registry is missing domains list")
    return data
