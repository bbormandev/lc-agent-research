import argparse
import json
import sys

from lc_agent.pipeline import PipelineConfig, ask_question


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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        config = PipelineConfig(
            max_sources=args.max_sources,
            model=args.model,
        )
        result = ask_question(args.question, config)

        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
