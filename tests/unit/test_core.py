import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from llmcut.core.checkpoint import Checkpoint, CheckpointStore
from llmcut.core.command import virtualize_output
from llmcut.core.dedupe import deduplicate
from llmcut.core.optimize import Optimizer
from llmcut.core.recover import Recovery
from llmcut.errors import IntegrityError, UnsupportedModeError
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, ModelConfiguration
from llmcut.policy import OptimizationMode, Policy
from llmcut.store.evidence import EvidenceStore


def block(identifier: str, content: str, kind: BlockKind = BlockKind.USER) -> ContextBlock:
    return ContextBlock(identifier, kind, content, "test")


def request(blocks: list[ContextBlock]) -> CanonicalRequest:
    return CanonicalRequest(
        blocks, ModelConfiguration("fake", "same", {"temperature": 0}, {"effort": "high"})
    )


def test_exact_dedup_preserves_order_role_and_reports() -> None:
    blocks = [block("a", "same"), block("b", "same"), block("c", "same", BlockKind.ASSISTANT)]
    kept, removed = deduplicate(blocks)
    assert [item.id for item in kept] == ["a", "c"]
    assert removed[0].removed_id == "b" and removed[0].retained_id == "a"


@pytest.mark.parametrize("left,right", [("x = 1", "x=1"), ("hello", "Hello"), ("x ", "x")])
def test_similar_content_is_not_deduplicated(left: str, right: str) -> None:
    assert len(deduplicate([block("a", left), block("b", right)])[0]) == 2


def test_optimizer_deterministic_and_restorable(tmp_path: Path) -> None:
    original = request([block("a", "alpha"), block("b", "alpha"), block("c", "beta")])
    store = EvidenceStore(tmp_path / ".llmcut")
    first, report1 = Optimizer(store).optimize(original)
    second, report2 = Optimizer(store).optimize(original)
    assert first.to_json() == second.to_json()
    assert report1.stable_prefix_digest == report2.stable_prefix_digest
    restored = Recovery(store).restore_request(first)
    assert restored.to_json() == original.to_json()
    assert restored.model.reasoning == {"effort": "high"}


def test_low_confidence_fails_open(tmp_path: Path) -> None:
    original = request([ContextBlock("a", BlockKind.REPOSITORY, "uncertain", "x", priority=0)])
    optimized, report = Optimizer(EvidenceStore(tmp_path)).optimize(original)
    assert optimized.blocks[0].content == "uncertain"
    assert "low confidence" in report.decisions[0].reason


def test_explicit_proven_redundancy_is_recoverable(tmp_path: Path) -> None:
    original = request(
        [
            ContextBlock(
                "a", BlockKind.REPOSITORY, "redundant", "x", metadata={"proven_redundant": True}
            )
        ]
    )
    optimized, report = Optimizer(EvidenceStore(tmp_path)).optimize(original)
    assert optimized.blocks == []
    assert (
        Recovery(EvidenceStore(tmp_path)).restore_request(optimized).blocks[0].content
        == "redundant"
    )
    assert report.decisions[0].confidence == 1


def test_economy_refuses_nonfunctional_behavior() -> None:
    with pytest.raises(UnsupportedModeError, match="not implemented"):
        Policy(mode=OptimizationMode.ECONOMY).validate()


def test_modes_make_distinct_non_lossy_decisions(tmp_path: Path) -> None:
    original = request(
        [
            ContextBlock(
                "r",
                BlockKind.REPOSITORY,
                "old",
                "repo",
                metadata={"task_irrelevant": True, "confidence": "high"},
            ),
            ContextBlock(
                "c",
                BlockKind.CHECKPOINT,
                "old checkpoint",
                "history",
                metadata={"superseded": True},
            ),
        ]
    )
    store = EvidenceStore(tmp_path)
    strict, _ = Optimizer(store).optimize(original, Policy(mode=OptimizationMode.STRICT))
    parity, _ = Optimizer(store).optimize(original, Policy(mode=OptimizationMode.PARITY))
    extreme, _ = Optimizer(store).optimize(original, Policy(mode=OptimizationMode.EXTREME))
    assert [x.id for x in strict.blocks] == ["r", "c"]
    assert [x.id for x in parity.blocks] == ["r"]
    assert extreme.blocks == []
    assert strict.model == parity.model == extreme.model == original.model
    assert Recovery(store).restore_request(extreme).to_json() == original.to_json()


def test_recovery_ranges_and_matching(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    ref = store.put("one\ntwo error\nthree\nfour", "log")
    recovery = Recovery(store)
    assert recovery.source_range(ref.digest, 2, 3) == "two error\nthree"
    assert "2: two error" in recovery.matching(ref.digest, "error")
    assert "two error" in recovery.matching(ref.digest, "err.*", regex=True)


def test_command_virtualization_never_hides_failure_warning_or_skip(tmp_path: Path) -> None:
    raw = "start\nWARNING: risky\nFAILED test_a\n2 skipped\nend"
    value = virtualize_output(EvidenceStore(tmp_path), raw, ["pytest"], "/repo", 1, 0.5, 1)
    assert value.exit_status == 1
    assert "FAILED" in value.summary and "WARNING" in value.summary and "skipped" in value.summary
    assert EvidenceStore(tmp_path).get(value.reference.digest) == raw


def test_checkpoint_validates_evidence_and_stale_revision(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path / "state")
    ref = store.put("proof", "test")
    checkpoints = CheckpointStore(store)
    identifier = checkpoints.save(
        Checkpoint("goal", evidence=[ref.digest], repository_revision="old")
    )
    loaded = checkpoints.load(identifier)
    assert loaded.objective == "goal"
    with pytest.raises(IntegrityError, match="stale"):
        checkpoints.load(identifier, tmp_path)


@given(st.text())
def test_canonical_roundtrip_property(content: str) -> None:
    original = request([block("id", content)])
    assert (
        CanonicalRequest.from_dict(json.loads(original.to_json())).to_json() == original.to_json()
    )
