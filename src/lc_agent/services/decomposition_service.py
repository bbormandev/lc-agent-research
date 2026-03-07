import json
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from lc_agent.prompts.decomposition import (
    DECOMPOSITION_JSON_SCHEMA,
    DECOMPOSITION_SYSTEM_PROMPT,
    DECOMPOSITION_USER_PROMPT_TEMPLATE,
)

load_dotenv()


@dataclass
class Subtopic:
    id: str
    title: str
    question: str
    note_title: str


@dataclass
class DecompositionResult:
    strategy: str
    subtopics: list[Subtopic]


@dataclass
class DecompositionServiceConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


_FILLER_TITLES = {
    "overview",
    "conclusion",
    "example",
    "examples",
    "introduction",
    "summary",
}


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _clean_note_fragment(value: str) -> str:
    cleaned = _clean_text(value)
    cleaned = cleaned.replace("/", "-").replace("\\", "-")
    cleaned = cleaned.replace(":", " ").replace('"', "").replace("'", "")
    return re.sub(r"\s+", " ", cleaned).strip()


def _validate_and_build(raw: dict, parent_topic: str) -> DecompositionResult:
    strategy = _clean_text(raw.get("strategy"))
    if strategy != "conceptual_map":
        raise RuntimeError("Decomposition output strategy must be 'conceptual_map'")

    raw_subtopics = raw.get("subtopics")
    if not isinstance(raw_subtopics, list):
        raise RuntimeError("Decomposition output subtopics must be a list")
    if len(raw_subtopics) < 4 or len(raw_subtopics) > 7:
        raise RuntimeError("Decomposition must return 4-7 subtopics")

    safe_parent = _clean_note_fragment(parent_topic)
    if not safe_parent:
        raise RuntimeError("Decomposition parent_topic must be non-empty")

    subtopics: list[Subtopic] = []
    for idx, item in enumerate(raw_subtopics, start=1):
        if not isinstance(item, dict):
            raise RuntimeError("Each subtopic must be an object")

        title = _clean_text(item.get("title"))
        question = _clean_text(item.get("question"))

        if not title or not question:
            raise RuntimeError("Each subtopic must include non-empty title and question")
        if title.lower() in _FILLER_TITLES:
            raise RuntimeError(f"Subtopic title is filler and not allowed: {title}")

        subtopic_id = f"S{idx}"
        note_title = f"{safe_parent} - {_clean_note_fragment(title)}"
        subtopics.append(
            Subtopic(
                id=subtopic_id,
                title=title,
                question=question,
                note_title=note_title,
            )
        )

    return DecompositionResult(strategy=strategy, subtopics=subtopics)


def decompose_question(
    question: str,
    config: DecompositionServiceConfig | None = None,
) -> DecompositionResult:
    cleaned_question = _clean_text(question)
    if not cleaned_question:
        raise ValueError("question must be non-empty")
    config = config or DecompositionServiceConfig()

    llm = ChatOpenAI(model=config.model, temperature=config.temperature)
    llm = llm.bind(
        response_format={
            "type": "json_schema",
            "json_schema": DECOMPOSITION_JSON_SCHEMA,
        }
    )

    user_prompt = DECOMPOSITION_USER_PROMPT_TEMPLATE.format(question=cleaned_question)
    raw = llm.invoke(
        [
            ("system", DECOMPOSITION_SYSTEM_PROMPT),
            ("user", user_prompt),
        ]
    ).content

    if not isinstance(raw, str):
        raise RuntimeError("Decomposition model returned non-text content")

    parsed = json.loads(raw)
    parent_topic = _clean_text(parsed.get("parent_topic"))
    return _validate_and_build(parsed, parent_topic=parent_topic)
