import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lc_agent.services.categorization_service import (
    CategorizationServiceConfig,
    categorize,
    normalize_tag,
)


def _make_registry(vault: Path) -> None:
    registry = """---
version: 1
broad_categories:
  - name: technology
    refined_categories:
      - name: machine-learning
        subrefined_categories: [llm-deployment, evaluation]
  - name: finance
    refined_categories:
      - name: markets
        subrefined_categories: [macro]
canonical_tags: [privacy, open-source, cost-optimization]
rules:
  max_depth: 3
  max_tags: 8
  max_new_tags: 2
---
# Category Tree
"""
    index_dir = vault / "Index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "Category Tree.md").write_text(registry, encoding="utf-8")


class TestCategorizationService(unittest.TestCase):
    def test_selects_existing_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            _make_registry(vault)

            llm_response = {
                "broad": "technology",
                "refined": "machine-learning",
                "subrefined": "llm-deployment",
                "tags": ["privacy", "open-source"],
                "links": {"entities": ["Ollama"], "concepts": ["local inference"]},
                "confidence": 0.82,
                "proposed_new_categories": {},
            }

            mock_bound = MagicMock()
            mock_bound.invoke.return_value = MagicMock(content=json.dumps(llm_response))
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_bound

            with patch("lc_agent.services.categorization_service.ChatOpenAI", return_value=mock_llm):
                out = categorize(
                    "Best tools for local LLMs?",
                    {
                        "summary": "Summary",
                        "answer_bullets": ["A [S1]"],
                        "sources": ["S1: src"],
                    },
                    vault_path=str(vault),
                    config=CategorizationServiceConfig(),
                )

            self.assertEqual(out["broad"], "technology")
            self.assertEqual(out["refined"], "machine-learning")
            self.assertEqual(out["subrefined"], "llm-deployment")
            self.assertEqual(out["tags"], ["privacy", "open-source"])
            self.assertEqual(out["proposed_new_categories"], {})

    def test_proposes_new_refined_when_none_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            _make_registry(vault)

            llm_response = {
                "broad": "technology",
                "refined": "agent-orchestration",
                "subrefined": None,
                "tags": ["open-source"],
                "links": {"entities": [], "concepts": ["multi-agent systems"]},
                "confidence": 0.66,
                "proposed_new_categories": {"refined": ["agent-orchestration"]},
            }

            mock_bound = MagicMock()
            mock_bound.invoke.return_value = MagicMock(content=json.dumps(llm_response))
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_bound

            with patch("lc_agent.services.categorization_service.ChatOpenAI", return_value=mock_llm):
                out = categorize(
                    "How should I orchestrate multiple agents?",
                    {
                        "summary": "Summary",
                        "answer_bullets": ["A [S1]"],
                        "sources": ["S1: src"],
                    },
                    vault_path=str(vault),
                    config=CategorizationServiceConfig(),
                )

            self.assertEqual(out["broad"], "technology")
            self.assertEqual(out["refined"], "agent-orchestration")
            self.assertIn("refined", out["proposed_new_categories"])
            self.assertEqual(out["proposed_new_categories"]["refined"], ["agent-orchestration"])

    def test_normalizes_tags_and_enforces_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            _make_registry(vault)

            llm_response = {
                "broad": "Technology",
                "refined": "Machine Learning",
                "subrefined": "",
                "tags": [
                    "Privacy",
                    "Open Source",
                    "Cost Optimization",
                    "RAG Patterns",
                    "Edge AI",
                    "Benchmarking",
                    "Long Context",
                    "Agent Memory",
                    "Extra Novel Tag",
                    "One More Novel",
                ],
                "links": {"entities": ["  Ollama  ", "Ollama"], "concepts": ["RAG", " "]},
                "confidence": 0.73,
                "proposed_new_categories": {},
            }

            mock_bound = MagicMock()
            mock_bound.invoke.return_value = MagicMock(content=json.dumps(llm_response))
            mock_llm = MagicMock()
            mock_llm.bind.return_value = mock_bound

            with patch("lc_agent.services.categorization_service.ChatOpenAI", return_value=mock_llm):
                out = categorize(
                    "What are good RAG practices?",
                    {
                        "summary": "Summary",
                        "answer_bullets": ["A [S1]"],
                        "sources": ["S1: src"],
                    },
                    vault_path=str(vault),
                    config=CategorizationServiceConfig(max_tags=8, max_new_tags=2),
                )

            self.assertEqual(out["broad"], "technology")
            self.assertEqual(out["refined"], "machine-learning")
            self.assertIsNone(out["subrefined"])
            self.assertLessEqual(len(out["tags"]), 8)
            for tag in out["tags"]:
                self.assertEqual(tag, normalize_tag(tag))

            canonical = {"privacy", "open-source", "cost-optimization"}
            novel = [tag for tag in out["tags"] if tag not in canonical]
            self.assertLessEqual(len(novel), 2)
            self.assertEqual(out["links"]["entities"], ["Ollama"])
            self.assertEqual(out["links"]["concepts"], ["RAG"])


if __name__ == "__main__":
    unittest.main()
