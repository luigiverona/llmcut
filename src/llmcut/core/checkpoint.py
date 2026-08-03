from __future__ import annotations

import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from llmcut.errors import IntegrityError
from llmcut.store.evidence import EvidenceStore


@dataclass(slots=True)
class Checkpoint:
    objective: str
    constraints: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    rejected_alternatives: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    validation_commands: list[str] = field(default_factory=list)
    validation_results: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    repository_revision: str | None = None
    remaining_work: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


def repository_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


class CheckpointStore:
    def __init__(self, evidence: EvidenceStore) -> None:
        self.evidence = evidence

    def save(self, checkpoint: Checkpoint) -> str:
        for digest in checkpoint.evidence:
            self.evidence.get(digest)
            self.evidence.reference(f"checkpoint:{checkpoint.id}", digest)
        with self.evidence.db.connect() as db:
            db.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?)",
                (
                    checkpoint.id,
                    json.dumps(asdict(checkpoint), sort_keys=True),
                    checkpoint.repository_revision,
                    int(time.time()),
                ),
            )
        return checkpoint.id

    def load(self, identifier: str, repo: Path | None = None) -> Checkpoint:
        with self.evidence.db.connect() as db:
            row = db.execute("SELECT payload FROM checkpoints WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise KeyError(identifier)
        checkpoint = Checkpoint(**json.loads(row[0]))
        for digest in checkpoint.evidence:
            self.evidence.get(digest)
        if repo is not None and checkpoint.repository_revision != repository_revision(repo):
            raise IntegrityError("checkpoint repository revision is stale")
        return checkpoint
