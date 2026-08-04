from pathlib import Path

from llmcut.eval.executable import evaluate_suite


def test_executable_release_suite_uses_isolated_worktrees_and_payload_counts() -> None:
    results, statistics = evaluate_suite(Path("tests/fixtures/benchmarks/suite.toml"))
    assert statistics["passed"]
    assert statistics["eligible_cases"] >= 5
    assert statistics["saving_cases"] >= 4
    assert statistics["median_reduction_all_eligible"] >= 20
    assert all(item.validation_passed and not item.unrelated_files for item in results)
    assert all(item.baseline.request_digest != item.optimized.request_digest for item in results)
    no_saving = next(item for item in results if item.task_id == "no-savings-control")
    retrieval_heavy = next(item for item in results if item.task_id == "retrieval-heavy-control")
    assert no_saving.reduction_percent == 0
    assert retrieval_heavy.reduction_percent <= 0
