import json
import tempfile
import unittest
from pathlib import Path

from lc_agent.services.obsidian_service import (
    build_note_filename,
    note_title_seed,
    publish_note,
    render_note_markdown,
    resolve_note_path,
    slugify_for_filename,
)


class TestObsidianService(unittest.TestCase):
    def test_render_markdown_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "2026-02-22" / "run_123"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "categories.json").write_text(
                json.dumps(
                    {
                        "domain": "technology",
                        "category": "machine-learning",
                        "subcategory": "llm-deployment",
                        "tags": ["privacy", "open-source"],
                        "links": {"entities": ["Ollama"], "concepts": ["local-inference"]},
                        "confidence": 0.81,
                        "proposed_new_categories": {},
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "summary": "A concise summary.",
                "answer_bullets": ["Point A [S1]", "Point B [S2]"],
                "sources": ["S1: One - https://one", "S2: Two - https://two"],
                "_meta": {
                    "run_id": "run_123",
                    "model": "gpt-4o-mini",
                    "run_dir": str(run_dir),
                    "did_search": True,
                },
            }
            md = render_note_markdown("What is local LLM tooling?", result, "2026-02-22")

            self.assertIn('title: "What is local LLM tooling?"', md)
            self.assertIn("created: 2026-02-22", md)
            self.assertIn('run_id: "run_123"', md)
            self.assertIn('model: "gpt-4o-mini"', md)
            self.assertIn("did_search: true", md)
            self.assertIn('domain: "technology"', md)
            self.assertIn('category: "machine-learning"', md)
            self.assertIn('subcategory: "llm-deployment"', md)
            self.assertIn('tags: ["privacy", "open-source"]', md)
            self.assertIn("## Summary", md)
            self.assertIn("A concise summary.", md)
            self.assertIn(
                "Categories: [[Technology]] → [[Machine Learning]] → [[Llm Deployment]]",
                md,
            )
            self.assertIn("- Point A [S1]", md)
            self.assertIn("- S1: One - https://one", md)
            self.assertIn(f"- Run Dir: {run_dir}", md)
            self.assertIn("## Links", md)
            self.assertIn("- [[Ollama]]", md)
            self.assertIn("- [[local-inference]]", md)
            self.assertTrue(md.endswith("\n"))

    def test_render_markdown_missing_optional_meta(self) -> None:
        result = {
            "summary": "Summary only.",
            "answer_bullets": ["Only one point [S1]"],
            "sources": ["S1: One - https://one"],
        }
        md = render_note_markdown("Question", result, "2026-02-22")
        self.assertIn('run_id: "unknown"', md)
        self.assertIn('model: "unknown"', md)
        self.assertIn("- Run Dir: unknown", md)
        self.assertIn("did_search: false", md)
        self.assertNotIn("## Links", md)
        self.assertNotIn("Categories:", md)

    def test_render_markdown_omits_subcategory_link_when_null(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "2026-02-22" / "run_123"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "categories.json").write_text(
                json.dumps(
                    {
                        "domain": "technology",
                        "category": "artificial-intelligence",
                        "subcategory": None,
                        "tags": ["privacy"],
                        "links": {"entities": [], "concepts": []},
                        "confidence": 0.81,
                        "proposed_new_categories": {},
                    }
                ),
                encoding="utf-8",
            )
            result = {
                "summary": "A concise summary.",
                "answer_bullets": ["Point A [S1]"],
                "sources": ["S1: One - https://one"],
                "_meta": {
                    "run_id": "run_123",
                    "model": "gpt-4o-mini",
                    "run_dir": str(run_dir),
                    "did_search": True,
                },
            }
            md = render_note_markdown("Question", result, "2026-02-22")
            self.assertIn(
                "Categories: [[Technology]] → [[Artificial Intelligence]]",
                md,
            )
            self.assertNotIn("→ [[Llm Agents]]", md)

    def test_slugify_and_filename(self) -> None:
        self.assertEqual(
            slugify_for_filename("Best local LLM tools?! 2026"),
            "best-local-llm-tools-2026",
        )
        self.assertEqual(slugify_for_filename("!!!"), "note")
        self.assertEqual(
            note_title_seed(
                "fallback question",
                {"note_title": "Local LLM tooling for privacy-conscious teams"},
            ),
            "Local LLM tooling for privacy-conscious teams",
        )
        self.assertEqual(
            build_note_filename(
                "Best local LLM tools?! 2026",
                {"note_title": "Practical local LLM tools"},
                "2026-02-22",
            ),
            "practical-local-llm-tools-2026-02-22.md",
        )

    def test_collision_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            topics = Path(tmp)
            (topics / "topic-2026-02-22.md").write_text("one", encoding="utf-8")
            (topics / "topic-2026-02-22-2.md").write_text("two", encoding="utf-8")

            path = resolve_note_path(topics, "topic-2026-02-22.md")
            self.assertEqual(path.name, "topic-2026-02-22-3.md")

    def test_publish_note_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "2026-02-22" / "run_1"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "categories.json").write_text(
                json.dumps(
                    {
                        "domain": "technology",
                        "category": "artificial-intelligence",
                        "subcategory": "llm-agents",
                        "tags": ["privacy"],
                        "links": {"entities": [], "concepts": []},
                        "confidence": 0.81,
                        "proposed_new_categories": {},
                    }
                ),
                encoding="utf-8",
            )
            index_dir = Path(tmp) / "Index"
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "category_tree.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "generated_at": "2026-03-01T00:00:00+00:00",
                        "domains": [
                            {
                                "slug": "technology",
                                "title": "Technology",
                                "categories": [
                                    {
                                        "slug": "artificial-intelligence",
                                        "title": "Artificial Intelligence",
                                        "subcategories": [
                                            {
                                                "slug": "llm-agents",
                                                "title": "LLM Agents",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                        "canonical_tags": [],
                        "rules": {"max_depth": 3, "max_tags": 8, "max_new_tags": 2},
                    }
                ),
                encoding="utf-8",
            )

            result = {
                "note_title": "Short title",
                "summary": "Summary",
                "answer_bullets": ["Point [S1]"],
                "sources": ["S1: Source - https://example.com"],
                "_meta": {
                    "did_search": True,
                    "run_id": "r1",
                    "model": "m1",
                    "run_dir": str(run_dir),
                },
            }
            meta = publish_note(
                "Q",
                result,
                vault_path=tmp,
                today="2026-02-22",
            )
            note_path = Path(meta["note_path"])
            self.assertTrue(note_path.exists())
            self.assertEqual(note_path.parent.resolve(), (Path(tmp) / "Topics").resolve())
            content = note_path.read_text(encoding="utf-8")
            self.assertIn("## Key Points", content)
            self.assertIn("## Sources", content)
            self.assertIn(
                "Categories: [[Technology]] → [[Artificial Intelligence]] → [[LLM Agents]]",
                content,
            )


if __name__ == "__main__":
    unittest.main()
