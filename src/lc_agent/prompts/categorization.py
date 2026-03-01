CATEGORIZATION_SYSTEM_PROMPT = """You are a taxonomy classifier for a personal research assistant.
Return only JSON that matches the required schema.

Rules:
- Prefer selecting categories from the provided category tree.
- Keep taxonomy stable and low-growth; only propose new categories when no existing fit is reasonable.
- domain/category must be non-empty strings.
- subcategory may be null.
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
- domain: string
- category: string
- subcategory: string|null
- tags: string[]
- links: {{ entities: string[], concepts: string[] }}
- confidence: number (0..1)
- proposed_new_categories: {{ domain?: string[], category?: string[], subcategory?: string[] }}

Selection constraints:
- Pick from registry categories/tags when possible.
- If no category/subcategory fit, choose best available path and add proposals.
- If domain does not fit existing domain options, still choose best current domain and log proposed domain.
- Keep tags cross-cutting and deduplicated.
"""

CATEGORIZATION_JSON_SCHEMA = {
    "name": "categorization_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "domain": {"type": "string", "minLength": 1},
            "category": {"type": "string", "minLength": 1},
            "subcategory": {"type": ["string", "null"]},
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
                    "domain": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "array", "items": {"type": "string"}},
                    "subcategory": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["domain", "category", "subcategory"],
            },
        },
        "required": [
            "domain",
            "category",
            "subcategory",
            "tags",
            "links",
            "confidence",
            "proposed_new_categories",
        ],
    },
}
