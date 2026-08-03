from __future__ import annotations

import re
from dataclasses import dataclass

from llmcut.index.repository import FileRecord


@dataclass(frozen=True, slots=True)
class Score:
    value: int
    reasons: tuple[str, ...]


def score(record: FileRecord, task: str) -> Score:
    terms = {term.lower() for term in re.findall(r"[A-Za-z_][\w.-]+", task) if len(term) > 2}
    haystack = " ".join([record.path, *record.symbols, *record.imports]).lower()
    matches = sorted(term for term in terms if term in haystack)
    value = len(matches) * 10
    reasons = [f"task term: {term}" for term in matches]
    if record.status != "tracked":
        value += 30
        reasons.append("changed file")
    if PathLike(record.path).instruction:
        value += 100
        reasons.append("repository instruction/configuration")
    if record.tests and matches:
        value += 5
        reasons.append("associated tests")
    return Score(value, tuple(reasons or ["no deterministic relevance evidence"]))


class PathLike:
    def __init__(self, value: str) -> None:
        self.name = value.rsplit("/", 1)[-1]

    @property
    def instruction(self) -> bool:
        return self.name in {
            "AGENTS.md",
            "README.md",
            "pyproject.toml",
            "package.json",
            "tsconfig.json",
        }
