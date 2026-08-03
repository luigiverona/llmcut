from __future__ import annotations

import time
import uuid

from llmcut.model import TokenCount
from llmcut.store.database import Database


class MetricsStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def record_run(self, mode: str, original: TokenCount, optimized: TokenCount) -> str:
        run_id = str(uuid.uuid4())
        quality = original.quality.value if original.quality == optimized.quality else "mixed"
        with self.db.connect() as conn:
            conn.execute(
                "INSERT INTO runs(id,mode,original_tokens,optimized_tokens,"
                "count_quality,created_at) "
                "VALUES(?,?,?,?,?,?)",
                (run_id, mode, original.value, optimized.value, quality, int(time.time())),
            )
        return run_id

    def summary(self) -> dict[str, int | float | None]:
        with self.db.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*),SUM(original_tokens),SUM(optimized_tokens),"
                "SUM(recovery_tokens),SUM(retries) FROM runs"
            ).fetchone()
            cache = conn.execute("SELECT COALESCE(SUM(cached_tokens),0) FROM usage").fetchone()[0]
        original, optimized = row[1] or 0, row[2] or 0
        reduction = ((original - optimized) / original * 100) if original else None
        return {
            "runs": row[0],
            "original_input_tokens": original,
            "optimized_input_tokens": optimized,
            "logical_reduction_percent": reduction,
            "cached_input_tokens": cache,
            "recovery_tokens": row[3] or 0,
            "retries": row[4] or 0,
        }
