import json
import tempfile
import unittest
from pathlib import Path

from lc_agent.services.obsidian_service import (
    build_note_filename,
    note_title_seed,
    publish_note,
    publish_research,
    render_note_markdown,
    resolve_note_path,
    slugify_for_filename,
    write_hub_note,
    write_subtopic_note,
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

    def test_publish_research_writes_hub_and_subtopic_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "note_title": "Proxmox",
                "summary": "Proxmox is a virtualization platform.",
                "answer_bullets": ["Supports VMs and containers [S1]"],
                "sources": ["S1: Docs - https://example.com/proxmox"],
                "topic_research": {
                    "subtopics": [
                        {
                            "subtopic": {
                                "id": "S1",
                                "title": "Architecture",
                                "question": "How is Proxmox architected?",
                                "note_title": "Proxmox - Architecture",
                            },
                            "result": {
                                "summary": "Clustered nodes and services.",
                                "answer_bullets": ["Corosync coordinates nodes [S1]"],
                                "sources": ["S1: Cluster - https://example.com/cluster"],
                            },
                        },
                        {
                            "subtopic": {
                                "id": "S2",
                                "title": "Networking",
                                "question": "How does Proxmox networking work?",
                                "note_title": "Proxmox - Networking",
                            },
                            "result": {
                                "summary": "Linux bridge and SDN options.",
                                "answer_bullets": ["Bridge mode is common [S1]"],
                                "sources": ["S1: Net - https://example.com/net"],
                            },
                        },
                    ]
                },
            }
            meta = publish_research("What is Proxmox?", result, vault_path=tmp, today="2026-03-07")
            self.assertEqual(meta["mode"], "topic_research")
            self.assertEqual(meta["created_count"], 3)
            self.assertEqual(meta["skipped_existing_count"], 0)

            hub_path = Path(tmp) / "Topics" / "Proxmox.md"
            sub1_path = Path(tmp) / "Topics" / "Proxmox - Architecture.md"
            sub2_path = Path(tmp) / "Topics" / "Proxmox - Networking.md"
            self.assertTrue(hub_path.exists())
            self.assertTrue(sub1_path.exists())
            self.assertTrue(sub2_path.exists())

            hub_md = hub_path.read_text(encoding="utf-8")
            self.assertIn("# Proxmox", hub_md)
            self.assertIn("## Core Topics", hub_md)
            self.assertIn("- [[Proxmox - Architecture]]", hub_md)
            self.assertIn("- [[Proxmox - Networking]]", hub_md)

            sub_md = sub1_path.read_text(encoding="utf-8")
            self.assertIn("## Related", sub_md)
            self.assertIn("[[Proxmox]]", sub_md)

    def test_write_functions_skip_existing_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            topics = Path(tmp) / "Topics"
            topics.mkdir(parents=True, exist_ok=True)
            (topics / "Proxmox.md").write_text("existing", encoding="utf-8")
            (topics / "Proxmox - Security Hardening.md").write_text("existing", encoding="utf-8")

            topic_result = {
                "question": "What is Proxmox?",
                "note_title": "Proxmox",
                "topic_research": {"subtopics": []},
            }
            hub = write_hub_note(topic_result, vault_path=tmp, today="2026-03-07")
            self.assertEqual(hub["status"], "skipped_existing")

            subtopic = {
                "subtopic": {
                    "id": "S1",
                    "title": "Security Hardening",
                    "question": "...",
                    "note_title": "Proxmox - Security Hardening",
                },
                "result": {
                    "summary": "Summary",
                    "answer_bullets": ["Point [S1]"],
                    "sources": ["S1: Source - https://example.com"],
                },
            }
            sub = write_subtopic_note(subtopic, hub_title="Proxmox", vault_path=tmp, today="2026-03-07")
            self.assertEqual(sub["status"], "skipped_existing")

    def test_publish_note_uses_topic_research_mode_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = {
                "note_title": "Raft",
                "summary": "Consensus protocol.",
                "answer_bullets": ["Uses replicated logs [S1]"],
                "sources": ["S1: Paper - https://example.com/raft"],
                "topic_research": {
                    "subtopics": [
                        {
                            "subtopic": {
                                "id": "S1",
                                "title": "Leader Election",
                                "question": "...",
                                "note_title": "Raft - Leader Election",
                            },
                            "result": {
                                "summary": "Terms and votes.",
                                "answer_bullets": ["Candidates request votes [S1]"],
                                "sources": ["S1: Paper - https://example.com/raft"],
                            },
                        }
                    ]
                },
            }
            meta = publish_note("How does Raft work?", result, vault_path=tmp, today="2026-03-07")
            self.assertEqual(meta["mode"], "topic_research")
            self.assertTrue((Path(tmp) / "Topics" / "Raft.md").exists())
            self.assertTrue((Path(tmp) / "Topics" / "Raft - Leader Election.md").exists())

    def test_hub_note_includes_categories_frontmatter_and_breadcrumb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "runs" / "2026-03-07" / "run_42"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "categories.json").write_text(
                json.dumps(
                    {
                        "domain": "technology",
                        "category": "hardware-computing",
                        "subcategory": None,
                        "tags": ["virtualization"],
                        "links": {"entities": [], "concepts": []},
                        "confidence": 0.9,
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
                        "generated_at": "2026-03-07T00:00:00+00:00",
                        "domains": [
                            {
                                "slug": "technology",
                                "title": "Technology",
                                "categories": [
                                    {
                                        "slug": "hardware-computing",
                                        "title": "Hardware & Computing",
                                        "subcategories": [],
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
                "note_title": "Proxmox",
                "summary": "Overview",
                "answer_bullets": ["Point [S1]"],
                "sources": ["S1: Source - https://example.com"],
                "_meta": {
                    "run_id": "run_42",
                    "model": "gpt-4o-mini",
                    "run_dir": str(run_dir),
                    "did_search": True,
                },
                "topic_research": {
                    "subtopics": [
                        {
                            "subtopic": {
                                "id": "S1",
                                "title": "Architecture",
                                "question": "...",
                                "note_title": "Proxmox - Architecture",
                            },
                            "result": {
                                "summary": "Architecture summary",
                                "answer_bullets": ["A [S1]"],
                                "sources": ["S1: A - https://example.com"],
                            },
                        }
                    ]
                },
            }
            publish_research("What is Proxmox?", result, vault_path=tmp, today="2026-03-07")
            hub_md = (Path(tmp) / "Topics" / "Proxmox.md").read_text(encoding="utf-8")
            self.assertIn('domain: "technology"', hub_md)
            self.assertIn('category: "hardware-computing"', hub_md)
            self.assertIn('tags: ["virtualization"]', hub_md)
            self.assertIn("Categories: [[Technology]] → [[Hardware & Computing]]", hub_md)


if __name__ == "__main__":
    unittest.main()
