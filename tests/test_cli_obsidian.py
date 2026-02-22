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
    def test_cli_without_vault_no_publish(self) -> None:
        fake_result = {
            "summary": "Summary",
            "answer_bullets": ["Point [S1]"],
            "sources": ["S1: Source - https://example.com"],
            "_meta": {"run_id": "r1", "model": "m1", "did_search": False},
        }
        with patch("lc_agent.cli.make_run_context", return_value=RunContext(today="2026-02-22", current_year=2026)), patch(
            "lc_agent.cli.ask", return_value=fake_result
        ) as ask_mock, patch("lc_agent.cli.publish_note") as publish_mock:
            out = io.StringIO()
            with redirect_stdout(out):
                code = main(["ask", "question"])

        self.assertEqual(code, 0)
        ask_mock.assert_called_once()
        publish_mock.assert_not_called()
        parsed = json.loads(out.getvalue())
        self.assertNotIn("obsidian", parsed.get("_meta", {}))

    def test_cli_with_vault_writes_note_and_sets_meta(self) -> None:
        fake_result = {
            "summary": "Summary",
            "answer_bullets": ["Point [S1]"],
            "sources": ["S1: Source - https://example.com"],
            "_meta": {
                "run_id": "run_1",
                "model": "gpt-4o-mini",
                "run_dir": "runs/2026-02-22/run_1",
                "did_search": True,
            },
        }

        with tempfile.TemporaryDirectory() as vault:
            with patch("lc_agent.cli.make_run_context", return_value=RunContext(today="2026-02-22", current_year=2026)), patch(
                "lc_agent.cli.ask", return_value=fake_result
            ):
                out = io.StringIO()
                with redirect_stdout(out):
                    code = main(["ask", "question", "--vault", vault])

            self.assertEqual(code, 0)
            parsed = json.loads(out.getvalue())
            obsidian = parsed["_meta"]["obsidian"]
            self.assertTrue(obsidian["enabled"])
            self.assertTrue(obsidian["note_path"].endswith(".md"))
            self.assertEqual(Path(obsidian["vault_path"]).resolve(), Path(vault).resolve())


if __name__ == "__main__":
    unittest.main()
