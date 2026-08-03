from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from llmcut.model import EvidenceReference
from llmcut.store.evidence import EvidenceStore


@dataclass(slots=True)
class VirtualCommandOutput:
    command: list[str]
    working_directory: str
    exit_status: int
    duration_seconds: float
    summary: str
    warnings: list[str]
    failures: list[str]
    source_locations: list[str]
    reference: EvidenceReference

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def virtualize_output(
    store: EvidenceStore,
    raw: str,
    command: list[str],
    cwd: str,
    exit_status: int,
    duration: float,
    max_summary_lines: int = 30,
) -> VirtualCommandOutput:
    reference = store.put(raw, "command:" + " ".join(command), metadata={"cwd": cwd})
    lines = raw.splitlines()
    warnings = [line for line in lines if re.search(r"\bwarning\b|\bskipped\b", line, re.I)]
    failures = [line for line in lines if re.search(r"\bfail(?:ed|ure)?\b|\berror\b", line, re.I)]
    locations = [
        match.group(0) for line in lines if (match := re.search(r"(?:[\w./-]+):\d+(?::\d+)?", line))
    ]
    important = list(dict.fromkeys([*failures, *warnings, *lines[-max_summary_lines:]]))
    summary = "\n".join(important)
    if len(lines) > max_summary_lines:
        summary += (
            f"\n[{len(lines) - max_summary_lines} earlier lines recoverable as {reference.digest}]"
        )
    return VirtualCommandOutput(
        command, cwd, exit_status, duration, summary, warnings, failures, locations, reference
    )
