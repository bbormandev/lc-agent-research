DECOMPOSITION_SYSTEM_PROMPT = """You decompose research questions into reusable conceptual subtopics.
Return only JSON that matches the required schema.

Rules:
- strategy must be "conceptual_map".
- Generate 4 to 7 subtopics.
- Subtopics must be concept-level building blocks, not document sections.
- Avoid filler sections like overview, conclusion, examples, introduction, summary.
- Each title should be concise (2-6 words), specific, and reusable across related topics.
- Each question should be a focused research question tied to the subtopic.
- parent_topic should be a concise canonical name (1-6 words) for the main topic.
"""

COMPLEXITY_SYSTEM_PROMPT = """You classify whether a research question is complex.
Return only JSON that matches the required schema.

Rules:
- is_complex is true when the question likely requires multiple conceptual areas/subproblems.
- is_complex is false when the question is a single focused topic.
- reason must be concise and concrete (<= 140 chars).
"""

COMPLEXITY_USER_PROMPT_TEMPLATE = """Classify this research question.

Question:
{question}

Output schema fields:
- is_complex: boolean
- reason: string
"""

DECOMPOSITION_USER_PROMPT_TEMPLATE = """Decompose this research question.

Question:
{question}

Output schema fields:
- strategy: "conceptual_map"
- parent_topic: string
- subtopics: array of 4-7 objects
  - title: string
  - question: string

Constraints:
- Use conceptual components, mechanisms, guarantees, tradeoffs, or failure modes.
- Keep subtopics reusable across other related systems/topics.
- Do not include generic sections (overview/conclusion/examples).
"""

DECOMPOSITION_JSON_SCHEMA = {
    "name": "decomposition_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "strategy": {
                "type": "string",
                "enum": ["conceptual_map"],
            },
            "parent_topic": {
                "type": "string",
                "minLength": 1,
            },
            "subtopics": {
                "type": "array",
                "minItems": 4,
                "maxItems": 7,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "title": {"type": "string", "minLength": 1},
                        "question": {"type": "string", "minLength": 1},
                    },
                    "required": ["title", "question"],
                },
            },
        },
        "required": ["strategy", "parent_topic", "subtopics"],
    },
}

COMPLEXITY_JSON_SCHEMA = {
    "name": "complexity_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "is_complex": {"type": "boolean"},
            "reason": {"type": "string", "minLength": 1, "maxLength": 140},
        },
        "required": ["is_complex", "reason"],
    },
}
