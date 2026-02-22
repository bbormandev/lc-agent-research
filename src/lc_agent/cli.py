import argparse
import json
import sys

from lc_agent.services.research_service import ResearchServiceConfig, ask
from lc_agent.services.obsidian_service import publish_note
from lc_agent.infra.run_context import make_run_context


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lc_agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask a question")
    ask.add_argument("question", type=str, help="The question to research/answer")
    ask.add_argument(
        "--max-sources",
        type=int,
        default=5,
        help="Maximum number of web sources to use (default: 5)",
    )
    ask.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)",
    )
    ask.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON (default prints pretty JSON anyway)",
    )
    ask.add_argument(
        "--vault",
        type=str,
        default=None,
        help="Optional path to Obsidian vault root. If provided, writes a markdown note.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        config = ResearchServiceConfig(
            max_sources=args.max_sources,
            model=args.model,
        )
        ctx = make_run_context()
        result = ask(args.question, config, ctx)

        if args.vault:
            publish_meta = publish_note(
                args.question,
                result,
                vault_path=args.vault,
                today=ctx.today,
            )
            if not isinstance(result.get("_meta"), dict):
                result["_meta"] = {}
            result["_meta"]["obsidian"] = {
                "enabled": True,
                "note_path": publish_meta["note_path"],
                "note_filename": publish_meta["note_filename"],
                "vault_path": publish_meta["vault_path"],
            }

        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
