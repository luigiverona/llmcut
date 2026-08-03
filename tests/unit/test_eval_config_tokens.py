import json
from pathlib import Path
from typing import Any

import pytest

from llmcut.config import load_config
from llmcut.core.optimize import Optimizer
from llmcut.eval.corpus import CorpusCase, read_corpus
from llmcut.eval.runner import run_case
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, CountQuality, ModelConfiguration
from llmcut.store.evidence import EvidenceStore
from llmcut.tokens.estimate import ConservativeEstimator


def test_config_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    user = tmp_path / "user"
    (user / "llmcut").mkdir(parents=True)
    (user / "llmcut/config.toml").write_text('mode="strict"\n[proxy]\nport=1\n')
    project = tmp_path / "project"
    (project / ".llmcut").mkdir(parents=True)
    (project / ".llmcut/config.toml").write_text('mode="parity"\n[proxy]\nport=2\n')
    monkeypatch.setenv("XDG_CONFIG_HOME", str(user))
    monkeypatch.setenv("LLMCUT_PORT", "3")
    config = load_config(project, {"mode": "extreme", "port": 4})
    assert config.mode == "extreme" and config.port == 4


def test_estimate_is_labeled_and_conservative() -> None:
    count = ConservativeEstimator().count("hello")
    assert count.value == 2 and count.quality is CountQuality.ESTIMATED


def test_eval_same_settings_savings_and_regression(tmp_path: Path) -> None:
    request = CanonicalRequest(
        [ContextBlock("a", BlockKind.USER, "x", "t"), ContextBlock("b", BlockKind.USER, "x", "t")],
        ModelConfiguration("fake", "m", {"temperature": 0}, {"effort": "high"}),
    )
    case = CorpusCase("one", request, {"answer": 42})
    seen = []

    def executor(value: CanonicalRequest) -> tuple[dict[str, Any], dict[str, int]]:
        seen.append(value.model)
        result = {"answer": 42, "complete": True}
        return result, {
            "input_tokens": len(value.blocks) * 10,
            "output_tokens": 2,
            "cached_tokens": 1,
            "recovery_tokens": 0,
            "retries": 0,
        }

    result = run_case(case, Optimizer(EvidenceStore(tmp_path)), executor)
    assert result.settings_identical and not result.regression
    assert result.baseline_input_tokens == 20 and result.optimized_input_tokens == 10
    assert result.fallback_reason is None
    assert seen[0] == seen[1]


def test_eval_detects_regression(tmp_path: Path) -> None:
    request = CanonicalRequest(
        [ContextBlock("a", BlockKind.USER, "x", "t")], ModelConfiguration("fake", "m")
    )
    calls = 0

    def executor(_: CanonicalRequest) -> tuple[dict[str, Any], dict[str, int]]:
        nonlocal calls
        calls += 1
        return ({"answer": 1 if calls == 1 else 2}, {"input_tokens": 1})

    result = run_case(
        CorpusCase("x", request, {"answer": 1}), Optimizer(EvidenceStore(tmp_path)), executor
    )
    assert result.regression


def test_jsonl_corpus(tmp_path: Path) -> None:
    request = CanonicalRequest([], ModelConfiguration("fake", "m"))
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "task_id": "x",
                "input_request": request.to_dict(),
                "expected_invariants": {"ok": True},
            }
        )
        + "\n"
    )
    assert list(read_corpus(path))[0].task_id == "x"
