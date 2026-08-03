from __future__ import annotations

import json
import re
import time
from pathlib import Path

from llmcut.errors import IntegrityError
from llmcut.model import EvidenceReference, digest_bytes
from llmcut.store.database import Database

SECRET_PATTERNS = [
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,]+"),
    re.compile(r"(?i)((?:api[_-]?key|secret|token|password)\s*[:=]\s*)[^\s,]+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
]


def redact_for_persistence(value: str) -> str:
    for pattern in SECRET_PATTERNS:
        value = pattern.sub(
            lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", value
        )
    return value


class EvidenceStore:
    def __init__(self, root: Path, *, persist_content: bool = True) -> None:
        self.root = root
        self.persist_content = persist_content
        self.db = Database(root / "state.db")
        self.db.initialize()

    def put(
        self,
        content: str,
        source: str,
        revision: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> EvidenceReference:
        persisted = (
            redact_for_persistence(content) if self.persist_content else "[CONTENT NOT PERSISTED]"
        )
        raw = persisted.encode()
        digest = digest_bytes(raw)
        now = int(time.time())
        with self.db.connect() as db:
            existing = db.execute(
                "SELECT content FROM evidence WHERE digest=?", (digest,)
            ).fetchone()
            if existing is not None and bytes(existing[0]) != raw:
                raise IntegrityError(
                    "cryptographic digest collision detected; evidence not overwritten"
                )
            db.execute(
                "INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?,?,?)",
                (
                    digest,
                    raw,
                    source,
                    revision,
                    now,
                    now,
                    len(raw),
                    json.dumps(metadata or {}, sort_keys=True),
                ),
            )
        return EvidenceReference(digest, source, revision)

    def get(self, digest: str) -> str:
        with self.db.connect() as db:
            row = db.execute("SELECT content FROM evidence WHERE digest=?", (digest,)).fetchone()
            if row is None:
                raise KeyError(digest)
            db.execute(
                "UPDATE evidence SET last_accessed=? WHERE digest=?", (int(time.time()), digest)
            )
        raw = bytes(row[0])
        if digest_bytes(raw) != digest:
            raise IntegrityError(f"evidence digest verification failed: {digest}")
        return raw.decode()

    def reference(self, owner: str, digest: str) -> None:
        with self.db.connect() as db:
            db.execute("INSERT OR IGNORE INTO evidence_refs VALUES(?,?)", (owner, digest))

    def collect(self, older_than: int) -> int:
        with self.db.connect() as db:
            cursor = db.execute(
                "DELETE FROM evidence WHERE last_accessed < ? AND digest NOT IN "
                "(SELECT digest FROM evidence_refs)",
                (older_than,),
            )
            return cursor.rowcount

    def list(self) -> list[dict[str, object]]:
        with self.db.connect() as db:
            rows = db.execute(
                "SELECT digest,source,revision,size,created_at FROM evidence ORDER BY digest"
            )
            return [
                dict(zip(("digest", "source", "revision", "size", "created_at"), row, strict=True))
                for row in rows
            ]
