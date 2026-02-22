import tempfile
import unittest
from pathlib import Path

from lc_agent.services.obsidian_service import (
    build_note_filename,
    publish_note,
    render_note_markdown,
    resolve_note_path,
    slugify_question,
)


class TestObsidianService(unittest.TestCase):
    def test_render_markdown_happy_path(self) -> None:
        result = {
            "summary": "A concise summary.",
            "answer_bullets": ["Point A [S1]", "Point B [S2]"],
            "sources": ["S1: One - https://one", "S2: Two - https://two"],
            "_meta": {
                "run_id": "run_123",
                "model": "gpt-4o-mini",
                "run_dir": "runs/2026-02-22/run_123",
                "did_search": True,
            },
        }
        md = render_note_markdown("What is local LLM tooling?", result, "2026-02-22")

        self.assertIn('title: "What is local LLM tooling?"', md)
        self.assertIn("created: 2026-02-22", md)
        self.assertIn('run_id: "run_123"', md)
        self.assertIn('model: "gpt-4o-mini"', md)
        self.assertIn("did_search: true", md)
        self.assertIn("## Summary", md)
        self.assertIn("A concise summary.", md)
        self.assertIn("- Point A [S1]", md)
        self.assertIn("- S1: One - https://one", md)
        self.assertIn("- Run Dir: runs/2026-02-22/run_123", md)
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

    def test_slugify_and_filename(self) -> None:
        self.assertEqual(
            slugify_question("Best local LLM tools?! 2026"),
            "best-local-llm-tools-2026",
        )
        self.assertEqual(slugify_question("!!!"), "note")
        self.assertEqual(
            build_note_filename("Best local LLM tools?! 2026", "2026-02-22"),
            "best-local-llm-tools-2026-2026-02-22.md",
        )

    def test_collision_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / "topic-2026-02-22.md").write_text("one", encoding="utf-8")
            (vault / "topic-2026-02-22-2.md").write_text("two", encoding="utf-8")

            path = resolve_note_path(vault, "topic-2026-02-22.md")
            self.assertEqual(path.name, "topic-2026-02-22-3.md")

    def test_publish_note_writes_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "summary": "Summary",
                "answer_bullets": ["Point [S1]"],
                "sources": ["S1: Source - https://example.com"],
                "_meta": {"did_search": True, "run_id": "r1", "model": "m1"},
            }
            meta = publish_note(
                "Q",
                result,
                vault_path=tmp,
                today="2026-02-22",
            )
            note_path = Path(meta["note_path"])
            self.assertTrue(note_path.exists())
            self.assertEqual(note_path.parent.resolve(), Path(tmp).resolve())
            content = note_path.read_text(encoding="utf-8")
            self.assertIn("## Key Points", content)
            self.assertIn("## Sources", content)


if __name__ == "__main__":
    unittest.main()
