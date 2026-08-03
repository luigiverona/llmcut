import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llmcut.errors import IntegrityError
from llmcut.model import digest_bytes
from llmcut.store.database import Database
from llmcut.store.evidence import EvidenceStore, redact_for_persistence
from llmcut.store.metrics import MetricsStore


def test_permissions_migrations_and_integrity(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "state")
    assert (store.root.stat().st_mode & 0o777) == 0o700
    assert (store.db.path.stat().st_mode & 0o777) == 0o600
    store.db.integrity_check()
    with store.db.connect() as db:
        assert db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 3
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_secret_redaction_persistence_boundary(tmp_path: Path) -> None:
    raw = "Authorization: Bearer topsecret\napi_key=abc123\nprompt remains"
    store = EvidenceStore(tmp_path)
    ref = store.put(raw, "request")
    persisted = store.get(ref.digest)
    assert "topsecret" not in persisted and "abc123" not in persisted
    assert "prompt remains" in persisted
    assert raw == "Authorization: Bearer topsecret\napi_key=abc123\nprompt remains"
    assert "topsecret" not in redact_for_persistence(raw)


def test_digest_tampering_detected(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    ref = store.put("safe", "test")
    with store.db.connect() as db:
        db.execute("UPDATE evidence SET content=? WHERE digest=?", (b"tampered", ref.digest))
    with pytest.raises(IntegrityError):
        store.get(ref.digest)


def test_gc_preserves_referenced_evidence(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    kept = store.put("keep", "test")
    removed = store.put("remove", "test")
    store.reference("checkpoint:x", kept.digest)
    with store.db.connect() as db:
        db.execute("UPDATE evidence SET last_accessed=0")
    assert store.collect(int(time.time())) == 1
    assert store.get(kept.digest) == "keep"
    with pytest.raises(KeyError):
        store.get(removed.digest)


def test_concurrent_idempotent_writes(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        refs = list(pool.map(lambda _: store.put("same", "worker"), range(20)))
    assert len({item.digest for item in refs}) == 1
    assert store.list()[0]["size"] == 4


def test_transaction_rolls_back(tmp_path: Path) -> None:
    database = Database(tmp_path / "state.db")
    database.initialize()
    with pytest.raises(sqlite3.IntegrityError), database.connect() as db:
        db.execute("INSERT INTO evidence_refs VALUES('x','missing')")
    with database.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM evidence_refs").fetchone()[0] == 0


def test_metadata_only_mode_is_explicit(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path, persist_content=False)
    ref = store.put("sensitive", "test")
    assert store.get(ref.digest) == "[CONTENT NOT PERSISTED]"
    assert ref.digest == digest_bytes(b"[CONTENT NOT PERSISTED]")


def test_request_metric_percentiles_and_no_prompt_storage(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    metrics = MetricsStore(store.db)
    base = {
        "provider": "openai",
        "endpoint_format": "chat-completions",
        "mode": "extreme",
        "integration_mode": "transparent",
        "count_quality": "estimated",
        "optimization_seconds": 0.01,
        "upstream_seconds": 0.02,
        "omitted_blocks": 1,
        "output_tokens": 2,
        "cached_tokens": 0,
        "reasoning_tokens": 0,
        "recovery_events": 0,
        "retries": 0,
        "evaluated_parity": None,
    }
    metrics.record_request(
        base
        | {
            "original_tokens": 100,
            "attempted_tokens": 60,
            "effective_tokens": 60,
            "fallback": 0,
            "fallback_reason": None,
        }
    )
    metrics.record_request(
        base
        | {
            "original_tokens": 100,
            "attempted_tokens": 110,
            "effective_tokens": 100,
            "fallback": 1,
            "fallback_reason": "not smaller",
        }
    )
    summary = metrics.summary()
    assert summary["median_effective_reduction"] in {0.0, 40.0}
    assert summary["no_savings_rate"] == 50.0 and summary["fallback_rate"] == 50.0
    with store.db.connect() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(request_metrics)")}
    assert "prompt" not in columns and "content" not in columns
