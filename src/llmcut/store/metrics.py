from __future__ import annotations

import time
import uuid
from typing import Any

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
            request_rows = conn.execute(
                "SELECT original_tokens,effective_tokens,fallback,evaluated_parity "
                "FROM request_metrics"
            ).fetchall()
        original, optimized = row[1] or 0, row[2] or 0
        reduction = ((original - optimized) / original * 100) if original else None
        reductions = sorted(
            ((item[0] - item[1]) / item[0] * 100) if item[0] else 0.0 for item in request_rows
        )
        return {
            "runs": row[0],
            "original_input_tokens": original,
            "optimized_input_tokens": optimized,
            "logical_reduction_percent": reduction,
            "cached_input_tokens": cache,
            "recovery_tokens": row[3] or 0,
            "retries": row[4] or 0,
            "median_effective_reduction": _percentile(reductions, 0.5),
            "p25_effective_reduction": _percentile(reductions, 0.25),
            "p75_effective_reduction": _percentile(reductions, 0.75),
            "no_savings_rate": _rate(sum(value <= 0 for value in reductions), len(reductions)),
            "fallback_rate": _rate(sum(item[2] for item in request_rows), len(request_rows)),
            "evaluated_parity_rate": _rate(
                sum(item[3] == 1 for item in request_rows),
                sum(item[3] is not None for item in request_rows),
            ),
            "unvalidated_requests": sum(item[3] is None for item in request_rows),
        }

    def record_request(self, values: dict[str, Any]) -> str:
        identifier = str(uuid.uuid4())
        columns = (
            "provider",
            "endpoint_format",
            "mode",
            "integration_mode",
            "original_tokens",
            "attempted_tokens",
            "effective_tokens",
            "output_tokens",
            "cached_tokens",
            "reasoning_tokens",
            "count_quality",
            "optimization_seconds",
            "upstream_seconds",
            "fallback",
            "fallback_reason",
            "omitted_blocks",
            "recovery_events",
            "retries",
            "evaluated_parity",
        )
        sql = (
            f"INSERT INTO request_metrics(id,{','.join(columns)},created_at) "  # noqa: S608
            f"VALUES({','.join('?' for _ in range(len(columns) + 2))})"
        )
        with self.db.connect() as conn:
            conn.execute(
                sql, (identifier, *(values.get(key, 0) for key in columns), int(time.time()))
            )
        return identifier


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    return values[round((len(values) - 1) * fraction)]


def _rate(count: int, total: int) -> float | None:
    return count / total * 100 if total else None
