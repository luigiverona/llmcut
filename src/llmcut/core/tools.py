from __future__ import annotations

from typing import Any


def retrieval_tool_definitions(integration_mode: str) -> list[dict[str, Any]]:
    """Return opt-in managed schemas; transparent mode never alters application tools."""
    if integration_mode == "transparent":
        return []
    if integration_mode != "managed":
        raise ValueError("integration mode must be transparent or managed")
    operations = {
        "llmcut_evidence_get": "Retrieve verified evidence by reference.",
        "llmcut_source_range": "Retrieve a verified 1-based source line range.",
        "llmcut_symbol_get": "Retrieve a symbol and its enclosing source range.",
        "llmcut_dependencies_get": "Retrieve direct dependencies for selected source.",
        "llmcut_log_search": "Search stored command output with a bounded pattern.",
        "llmcut_context_expand": "Monotonically add requested recoverable context.",
    }
    return [
        {
            "name": name,
            "description": description,
            "input_schema": {
                "type": "object",
                "properties": {"reference": {"type": "string", "maxLength": 128}},
                "required": ["reference"],
                "additionalProperties": False,
            },
        }
        for name, description in operations.items()
    ]
