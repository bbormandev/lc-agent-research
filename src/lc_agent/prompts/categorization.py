CATEGORIZATION_SYSTEM_PROMPT = """You are a taxonomy classifier for a personal research assistant.
Return only JSON that matches the required schema.

Rules:
- Prefer selecting categories from the provided category tree.
- Keep taxonomy stable and low-growth; only propose new categories when no existing fit is reasonable.
- broad/refined must be non-empty strings.
- subrefined may be null.
- tags must be concise, lower-case kebab-case labels.
- Use at most 8 tags.
- Suggest at most 2 new tags that are not in canonical_tags.
- links.entities are concrete people/orgs/products/places.
- links.concepts are abstract topics/frameworks/methods.
- confidence must be a number between 0 and 1.
- proposed_new_categories is for logging only; do not assume automatic adoption.
"""

CATEGORIZATION_USER_PROMPT_TEMPLATE = """Classify this research result.

Question:
{question}

Research Final JSON:
{final_json}

Category Registry:
{registry_json}

Output schema fields:
- broad: string
- refined: string
- subrefined: string|null
- tags: string[]
- links: {{ entities: string[], concepts: string[] }}
- confidence: number (0..1)
- proposed_new_categories: {{ broad?: string[], refined?: string[], subrefined?: string[] }}

Selection constraints:
- Pick from registry categories/tags when possible.
- If no refined/subrefined category fits, choose best available path and add proposals.
- If broad category does not fit existing broad options, still choose best current broad and log proposed broad.
- Keep tags cross-cutting and deduplicated.
"""

CATEGORIZATION_JSON_SCHEMA = {
    "name": "categorization_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "broad": {"type": "string", "minLength": 1},
            "refined": {"type": "string", "minLength": 1},
            "subrefined": {"type": ["string", "null"]},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": 8,
            },
            "links": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "entities": {"type": "array", "items": {"type": "string"}},
                    "concepts": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["entities", "concepts"],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "proposed_new_categories": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "broad": {"type": "array", "items": {"type": "string"}},
                    "refined": {"type": "array", "items": {"type": "string"}},
                    "subrefined": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["broad", "refined", "subrefined"],
            },
        },
        "required": [
            "broad",
            "refined",
            "subrefined",
            "tags",
            "links",
            "confidence",
            "proposed_new_categories",
        ],
    },
}
