from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmcut.model import CanonicalRequest


@dataclass(slots=True)
class CorpusCase:
    task_id: str
    request: CanonicalRequest
    expected_invariants: dict[str, Any]
    evaluator_command: list[str] | None = None
    expected_files: list[str] | None = None
    timeout: float = 60
    provider_config: str | None = None
    recorded_response: dict[str, Any] | None = None
    expected_output: dict[str, Any] | None = None
    normalize_response: bool = False


def read_corpus(path: Path) -> Iterator[CorpusCase]:
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        try:
            yield CorpusCase(
                value["task_id"],
                CanonicalRequest.from_dict(value["input_request"]),
                value.get("expected_invariants", {}),
                value.get("evaluator_command"),
                value.get("expected_files"),
                value.get("timeout", 60),
                value.get("provider_configuration_reference"),
                value.get("recorded_response"),
                value.get("expected_output"),
                bool(value.get("normalize_response", False)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid corpus record at line {number}: {exc}") from exc
