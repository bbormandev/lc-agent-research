import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from lc_agent.cli import main
from lc_agent.infra.run_context import RunContext


class TestCliObsidianPublish(unittest.TestCase):
    def test_cli_without_vault_skips_categorization_and_publish(self) -> None:
        fake_result = {
            "summary": "Summary",
            "answer_bullets": ["Point [S1]"],
            "sources": ["S1: Source - https://example.com"],
            "_meta": {"run_id": "r1", "model": "m1", "did_search": False, "run_dir": "runs/2026-02-22/r1"},
        }
        with patch("lc_agent.cli.make_run_context", return_value=RunContext(today="2026-02-22", current_year=2026)), patch(
            "lc_agent.cli.ask", return_value=fake_result
        ) as ask_mock, patch("lc_agent.cli.publish_note") as publish_mock, patch("lc_agent.cli.categorize") as categorize_mock:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["ask", "question"])

        self.assertEqual(code, 0)
        ask_mock.assert_called_once()
        publish_mock.assert_not_called()
        categorize_mock.assert_not_called()
        parsed = json.loads(out.getvalue())
        self.assertNotIn("obsidian", parsed.get("_meta", {}))
        self.assertEqual(parsed["_meta"]["categorization"]["enabled"], False)
        self.assertEqual(parsed["_meta"]["categorization"]["skipped_reason"], "vault_not_provided")

    def test_cli_with_vault_orders_ask_categorize_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / "vault"
            run_dir = Path(tmp) / "runs" / "2026-02-22" / "run_1"
            vault.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)

            fake_result = {
                "summary": "Summary",
                "answer_bullets": ["Point [S1]"],
                "sources": ["S1: Source - https://example.com"],
                "_meta": {
                    "run_id": "run_1",
                    "model": "gpt-4o-mini",
                    "run_dir": str(run_dir),
                    "did_search": True,
                },
            }
            fake_categories = {
                "broad": "technology",
                "refined": "machine-learning",
                "subrefined": "llm-deployment",
                "tags": ["privacy"],
                "links": {"entities": ["Ollama"], "concepts": ["local-inference"]},
                "confidence": 0.77,
                "proposed_new_categories": {},
            }

            calls: list[str] = []
            with patch("lc_agent.cli.make_run_context", return_value=RunContext(today="2026-02-22", current_year=2026)), patch(
                "lc_agent.cli.ask", return_value=fake_result
            ) as ask_mock, patch(
                "lc_agent.cli.ensure_category_registry_fresh",
                return_value={
                    "registry": {},
                    "registry_path": str(vault / "Index" / "category_tree.json"),
                    "meta_path": str(vault / "Index" / "category_tree.meta.json"),
                    "rebuilt": False,
                    "stale_reason": None,
                },
            ) as ensure_mock, patch("lc_agent.cli.categorize", return_value=fake_categories) as categorize_mock, patch(
                "lc_agent.cli.publish_note",
                side_effect=lambda *args, **kwargs: calls.append("publish") or {
                    "note_path": str(vault / "note.md"),
                    "note_filename": "note.md",
                    "vault_path": str(vault),
                },
            ) as publish_mock:
                categorize_mock.side_effect = lambda *args, **kwargs: calls.append("categorize") or fake_categories
                ask_mock.side_effect = lambda *args, **kwargs: calls.append("ask") or fake_result
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["ask", "question", "--vault", str(vault)])

            self.assertEqual(code, 0)
            self.assertEqual(calls, ["ask", "categorize", "publish"])
            parsed = json.loads(out.getvalue())
            obsidian = parsed["_meta"]["obsidian"]
            self.assertTrue(obsidian["enabled"])
            self.assertTrue(obsidian["note_path"].endswith(".md"))
            self.assertEqual(Path(obsidian["vault_path"]).resolve(), vault.resolve())
            self.assertEqual(parsed["_meta"]["categorization"]["enabled"], True)
            self.assertTrue((run_dir / "categories.json").exists())
            ensure_mock.assert_called_once()
            categorize_mock.assert_called_once()
            publish_mock.assert_called_once()

    def test_cli_with_vault_registry_refresh_failure_skips_categorization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp) / "vault"
            run_dir = Path(tmp) / "runs" / "2026-02-22" / "run_1"
            vault.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            fake_result = {
                "summary": "Summary",
                "answer_bullets": ["Point [S1]"],
                "sources": ["S1: Source - https://example.com"],
                "_meta": {
                    "run_id": "run_1",
                    "model": "gpt-4o-mini",
                    "run_dir": str(run_dir),
                    "did_search": True,
                },
            }
            with patch("lc_agent.cli.make_run_context", return_value=RunContext(today="2026-02-22", current_year=2026)), patch(
                "lc_agent.cli.ask", return_value=fake_result
            ), patch(
                "lc_agent.cli.ensure_category_registry_fresh",
                side_effect=RuntimeError("bad registry"),
            ), patch("lc_agent.cli.publish_note", return_value={
                "note_path": str(vault / "note.md"),
                "note_filename": "note.md",
                "vault_path": str(vault),
            }), patch("lc_agent.cli.categorize") as categorize_mock:
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["ask", "question", "--vault", str(vault)])

            self.assertEqual(code, 0)
            categorize_mock.assert_not_called()
            parsed = json.loads(out.getvalue())
            self.assertEqual(parsed["_meta"]["categorization"]["enabled"], False)
            self.assertEqual(parsed["_meta"]["categorization"]["skipped_reason"], "registry_refresh_failed")


if __name__ == "__main__":
    unittest.main()
