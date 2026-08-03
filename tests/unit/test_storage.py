import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llmcut.errors import IntegrityError
from llmcut.model import digest_bytes
from llmcut.store.database import Database
from llmcut.store.evidence import EvidenceStore, redact_for_persistence


def test_permissions_migrations_and_integrity(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "state")
    assert (store.root.stat().st_mode & 0o777) == 0o700
    assert (store.db.path.stat().st_mode & 0o777) == 0o600
    store.db.integrity_check()
    with store.db.connect() as db:
        assert db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0] == 1
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
