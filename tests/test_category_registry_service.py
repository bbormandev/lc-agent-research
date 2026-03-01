import json
import tempfile
import unittest
from pathlib import Path

from lc_agent.services.category_registry_service import (
    CategoryRegistryServiceConfig,
    ensure_category_registry_fresh,
)


def _write_note(path: Path, frontmatter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frontmatter, encoding="utf-8")


class TestCategoryRegistryService(unittest.TestCase):
    def test_compiles_expected_tree_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            categories = vault / "Index" / "Categories"

            _write_note(
                categories / "Technology.md",
                "---\n"
                "type: category\n"
                "level: domain\n"
                "slug: technology\n"
                "---\n",
            )
            _write_note(
                categories / "Artificial Intelligence.md",
                "---\n"
                "type: category\n"
                "level: category\n"
                "parent: technology\n"
                "slug: artificial-intelligence\n"
                "---\n",
            )
            _write_note(
                categories / "LLM Agents.md",
                "---\n"
                "type: category\n"
                "level: subcategory\n"
                "parent: artificial-intelligence\n"
                "slug: llm-agents\n"
                "---\n",
            )

            state = ensure_category_registry_fresh(str(vault), CategoryRegistryServiceConfig())

            self.assertTrue(state["rebuilt"])
            registry = state["registry"]
            self.assertEqual(registry["version"], 1)
            self.assertEqual(registry["rules"]["max_depth"], 3)
            self.assertEqual(registry["domains"][0]["slug"], "technology")
            self.assertEqual(registry["domains"][0]["title"], "Technology")
            self.assertEqual(
                registry["domains"][0]["categories"][0]["slug"],
                "artificial-intelligence",
            )
            self.assertEqual(
                registry["domains"][0]["categories"][0]["subcategories"][0]["slug"],
                "llm-agents",
            )

    def test_stale_check_rebuilds_when_source_newer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            categories = vault / "Index" / "Categories"
            note = categories / "Technology.md"
            _write_note(
                note,
                "---\n"
                "type: category\n"
                "level: domain\n"
                "slug: technology\n"
                "---\n",
            )

            first = ensure_category_registry_fresh(str(vault), CategoryRegistryServiceConfig())
            self.assertTrue(first["rebuilt"])

            note.write_text(
                "---\n"
                "type: category\n"
                "level: domain\n"
                "slug: technology\n"
                "---\n"
                "Updated content.\n",
                encoding="utf-8",
            )
            second = ensure_category_registry_fresh(str(vault), CategoryRegistryServiceConfig())
            self.assertTrue(second["rebuilt"])
            self.assertIn(
                second["stale_reason"],
                {"source_newer_than_registry", "snapshot_mismatch"},
            )

    def test_stale_check_skips_rebuild_when_fresh(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            categories = vault / "Index" / "Categories"
            _write_note(
                categories / "Technology.md",
                "---\n"
                "type: category\n"
                "level: domain\n"
                "slug: technology\n"
                "---\n",
            )

            first = ensure_category_registry_fresh(str(vault), CategoryRegistryServiceConfig())
            self.assertTrue(first["rebuilt"])

            second = ensure_category_registry_fresh(str(vault), CategoryRegistryServiceConfig())
            self.assertFalse(second["rebuilt"])
            self.assertIsNone(second["stale_reason"])

            registry_path = Path(second["registry_path"])
            meta_path = Path(second["meta_path"])
            self.assertTrue(registry_path.exists())
            self.assertTrue(meta_path.exists())
            parsed = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["domains"][0]["slug"], "technology")


if __name__ == "__main__":
    unittest.main()
