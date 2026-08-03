from __future__ import annotations

import json
from dataclasses import asdict

from llmcut.core.checkpoint import Checkpoint
from llmcut.errors import IntegrityError
from llmcut.model import BlockKind, ContextBlock, Retention
from llmcut.store.evidence import EvidenceStore


def compact_history(
    blocks: list[ContextBlock],
    checkpoint: Checkpoint,
    evidence: EvidenceStore,
    *,
    repository_revision: str | None = None,
) -> tuple[list[ContextBlock], bool]:
    """Replace a closed history prefix only with a verified, smaller checkpoint."""
    if not checkpoint.objective or not checkpoint.repository_revision:
        raise IntegrityError("checkpoint objective and repository revision are required")
    if repository_revision is not None and checkpoint.repository_revision != repository_revision:
        raise IntegrityError("checkpoint repository revision is stale")
    for digest in checkpoint.evidence:
        evidence.get(digest)
    open_calls: set[str] = set()
    for block in blocks:
        call_id = block.metadata.get("tool_call_id")
        if block.kind is BlockKind.TOOL_CALL and isinstance(call_id, str):
            open_calls.add(call_id)
        if block.kind is BlockKind.TOOL_RESULT and isinstance(call_id, str):
            open_calls.discard(call_id)
    if open_calls:
        raise IntegrityError("unresolved tool call crosses checkpoint boundary")
    durable_instructions = [
        item for item in blocks if item.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}
    ]
    payload = asdict(checkpoint)
    # Recovery manifests remain local. Only facts required to continue are model-bound.
    for internal in ("evidence", "id"):
        payload.pop(internal, None)
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    checkpoint_block = ContextBlock(
        f"checkpoint:{checkpoint.id}",
        BlockKind.CHECKPOINT,
        rendered,
        "managed:checkpoint",
        retention=Retention.REQUIRED,
    )
    candidate = [*durable_instructions, checkpoint_block]
    original_size = sum(len(item.content.encode()) for item in blocks)
    candidate_size = sum(len(item.content.encode()) for item in candidate)
    return (candidate, True) if candidate_size < original_size else (blocks, False)
