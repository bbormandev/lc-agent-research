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
    registry = {
        "version": 1,
        "generated_at": "2026-03-01T00:00:00+00:00",
        "domains": [
            {
                "slug": "technology",
                "title": "Technology",
                "categories": [
                    {
                        "slug": "machine-learning",
                        "title": "Machine Learning",
                        "subcategories": [
                            {"slug": "llm-deployment", "title": "LLM Deployment"},
                            {"slug": "evaluation", "title": "Evaluation"},
                        ],
                    }
                ],
            },
            {
                "slug": "finance",
                "title": "Finance",
                "categories": [
                    {
                        "slug": "markets",
                        "title": "Markets",
                        "subcategories": [{"slug": "macro", "title": "Macro"}],
                    }
                ],
            },
        ],
        "canonical_tags": ["privacy", "open-source", "cost-optimization"],
        "rules": {"max_depth": 3, "max_tags": 8, "max_new_tags": 2},
    }
    index_dir = vault / "Index"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "category_tree.json").write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )


class TestCategorizationService(unittest.TestCase):
    def test_selects_existing_categories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            _make_registry(vault)

            llm_response = {
                "domain": "technology",
                "category": "machine-learning",
                "subcategory": "llm-deployment",
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

            self.assertEqual(out["domain"], "technology")
            self.assertEqual(out["category"], "machine-learning")
            self.assertEqual(out["subcategory"], "llm-deployment")
            self.assertEqual(out["tags"], ["privacy", "open-source"])
            self.assertEqual(out["proposed_new_categories"], {})

    def test_proposes_new_category_when_none_fit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            _make_registry(vault)

            llm_response = {
                "domain": "technology",
                "category": "agent-orchestration",
                "subcategory": None,
                "tags": ["open-source"],
                "links": {"entities": [], "concepts": ["multi-agent systems"]},
                "confidence": 0.66,
                "proposed_new_categories": {"category": ["agent-orchestration"]},
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

            self.assertEqual(out["domain"], "technology")
            self.assertEqual(out["category"], "agent-orchestration")
            self.assertIn("category", out["proposed_new_categories"])
            self.assertEqual(out["proposed_new_categories"]["category"], ["agent-orchestration"])

    def test_normalizes_tags_and_enforces_caps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            _make_registry(vault)

            llm_response = {
                "domain": "Technology",
                "category": "Machine Learning",
                "subcategory": "",
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

            self.assertEqual(out["domain"], "technology")
            self.assertEqual(out["category"], "machine-learning")
            self.assertIsNone(out["subcategory"])
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
