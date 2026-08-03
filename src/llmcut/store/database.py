from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from llmcut.errors import IntegrityError

MIGRATIONS = [
    """
    CREATE TABLE evidence(
      digest TEXT PRIMARY KEY, content BLOB NOT NULL, source TEXT NOT NULL,
      revision TEXT, created_at INTEGER NOT NULL, last_accessed INTEGER NOT NULL,
      size INTEGER NOT NULL, metadata TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE evidence_refs(owner TEXT NOT NULL,
      digest TEXT NOT NULL REFERENCES evidence(digest),
      PRIMARY KEY(owner,digest));
    CREATE TABLE checkpoints(id TEXT PRIMARY KEY, payload TEXT NOT NULL, revision TEXT,
      created_at INTEGER NOT NULL);
    CREATE TABLE runs(id TEXT PRIMARY KEY, mode TEXT NOT NULL, original_tokens INTEGER,
      optimized_tokens INTEGER, count_quality TEXT, recovery_tokens INTEGER NOT NULL DEFAULT 0,
      retries INTEGER NOT NULL DEFAULT 0, created_at INTEGER NOT NULL);
    CREATE TABLE usage(id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(id), provider TEXT,
      input_tokens INTEGER, output_tokens INTEGER, cached_tokens INTEGER, reasoning_tokens INTEGER,
      billed_tokens INTEGER, raw TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE recovery_events(id INTEGER PRIMARY KEY, run_id TEXT REFERENCES runs(id),
      digest TEXT REFERENCES evidence(digest), reason TEXT NOT NULL, created_at INTEGER NOT NULL);
    CREATE TABLE evaluations(id TEXT PRIMARY KEY, payload TEXT NOT NULL,
      created_at INTEGER NOT NULL);
    CREATE INDEX evidence_access_idx ON evidence(last_accessed);
    CREATE INDEX usage_run_idx ON usage(run_id);
    """,
    """
    CREATE TABLE repository_index(
      repository_id TEXT NOT NULL, path TEXT NOT NULL, blob_oid TEXT,
      parser_version TEXT NOT NULL, record_json TEXT NOT NULL,
      updated_at INTEGER NOT NULL,
      PRIMARY KEY(repository_id,path)
    );
    CREATE INDEX repository_index_blob_idx
      ON repository_index(repository_id,blob_oid,parser_version);
    """,
    """
    CREATE TABLE request_metrics(
      id TEXT PRIMARY KEY, provider TEXT NOT NULL, endpoint_format TEXT NOT NULL,
      mode TEXT NOT NULL, integration_mode TEXT NOT NULL,
      original_tokens INTEGER NOT NULL, attempted_tokens INTEGER NOT NULL,
      effective_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL DEFAULT 0,
      cached_tokens INTEGER NOT NULL DEFAULT 0, reasoning_tokens INTEGER NOT NULL DEFAULT 0,
      count_quality TEXT NOT NULL, optimization_seconds REAL NOT NULL,
      upstream_seconds REAL NOT NULL, fallback INTEGER NOT NULL,
      fallback_reason TEXT, omitted_blocks INTEGER NOT NULL,
      recovery_events INTEGER NOT NULL DEFAULT 0, retries INTEGER NOT NULL DEFAULT 0,
      evaluated_parity INTEGER, created_at INTEGER NOT NULL
    );
    CREATE INDEX request_metrics_created_idx ON request_metrics(created_at);
    """,
    """
    CREATE TABLE managed_metrics(
      id TEXT PRIMARY KEY, integration_mode TEXT NOT NULL, optimization_mode TEXT NOT NULL,
      provider TEXT NOT NULL, model TEXT NOT NULL, baseline_tokens INTEGER,
      initial_tokens INTEGER NOT NULL, retrieval_request_tokens INTEGER NOT NULL,
      retrieval_result_tokens INTEGER NOT NULL, continuation_tokens INTEGER NOT NULL,
      total_effective_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL,
      reasoning_tokens INTEGER NOT NULL, cached_tokens INTEGER NOT NULL,
      count_quality TEXT NOT NULL, planning_seconds REAL NOT NULL,
      provider_seconds REAL NOT NULL, retrieval_count INTEGER NOT NULL,
      fallback INTEGER NOT NULL, quality_state TEXT NOT NULL, completed INTEGER NOT NULL,
      created_at INTEGER NOT NULL
    );
    CREATE INDEX managed_metrics_created_idx ON managed_metrics(created_at);
    """,
]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY)")
            current = db.execute("SELECT COALESCE(MAX(version), 0) FROM schema_version").fetchone()[
                0
            ]
            for version, script in enumerate(MIGRATIONS, 1):
                if version > current:
                    db.executescript(script)
                    db.execute("INSERT INTO schema_version VALUES(?)", (version,))
        os.chmod(self.path, 0o600)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def integrity_check(self) -> None:
        with self.connect() as db:
            result = db.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise IntegrityError(f"SQLite integrity check failed: {result}")
