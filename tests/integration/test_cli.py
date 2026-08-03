import json
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from llmcut.cli import app
from llmcut.model import BlockKind, CanonicalRequest, ContextBlock, ModelConfiguration

runner = CliRunner()


def test_help_and_version() -> None:
    assert runner.invoke(app, ["--help"]).exit_code == 0
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0 and "0.2.0" in result.stdout
    for command in (
        "init",
        "inspect",
        "pack",
        "optimize",
        "proxy",
        "checkpoint",
        "evidence",
        "stats",
        "eval",
        "doctor",
    ):
        assert command in runner.invoke(app, ["--help"]).stdout


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    (repo / "oauth.py").write_text("def callback():\n    return True\n")
    subprocess.run(["git", "-C", str(repo), "add", "oauth.py"], check=True)
    return repo


def test_init_idempotent_doctor_inspect_and_pack(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    assert runner.invoke(app, ["init", "--repo", str(repo)]).exit_code == 0
    original = (repo / ".llmcut/config.toml").read_text()
    assert runner.invoke(app, ["init", "--repo", str(repo)]).exit_code == 0
    assert (repo / ".llmcut/config.toml").read_text() == original
    assert (repo / ".llmcut").stat().st_mode & 0o777 == 0o700
    assert runner.invoke(app, ["doctor", "--repo", str(repo)]).exit_code == 0
    inspected = runner.invoke(app, ["inspect", "--repo", str(repo), "--format", "json"])
    assert inspected.exit_code == 0 and json.loads(inspected.stdout)["files"] == 1
    packed = runner.invoke(
        app, ["pack", "--repo", str(repo), "--task", "OAuth callback", "--mode", "extreme"]
    )
    assert packed.exit_code == 0
    assert json.loads(packed.stdout)["report"]["mode"] == "extreme"


def test_optimize_file_stdin_report_only_and_errors(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    runner.invoke(app, ["init", "--repo", str(repo)])
    canonical = CanonicalRequest(
        [ContextBlock("u", BlockKind.USER, "hello", "test")], ModelConfiguration("fake", "m")
    )
    path = tmp_path / "request.json"
    path.write_text(canonical.to_json())
    from_file = runner.invoke(app, ["optimize", "--repo", str(repo), "--input", str(path)])
    assert from_file.exit_code == 0 and "request" in json.loads(from_file.stdout)
    from_stdin = runner.invoke(
        app, ["optimize", "--repo", str(repo), "--report-only"], input=canonical.to_json()
    )
    assert from_stdin.exit_code == 0 and "request" not in json.loads(from_stdin.stdout)
    bad = runner.invoke(app, ["pack", "--repo", str(repo)])
    assert bad.exit_code != 0 and "required" in bad.output


def test_evidence_checkpoint_and_stats(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    runner.invoke(app, ["init", "--repo", str(repo)])
    canonical = CanonicalRequest(
        [ContextBlock("u", BlockKind.USER, "hello", "test")], ModelConfiguration("fake", "m")
    )
    runner.invoke(app, ["optimize", "--repo", str(repo)], input=canonical.to_json())
    listed = json.loads(runner.invoke(app, ["evidence", "list", "--repo", str(repo)]).stdout)
    digest = next(item["digest"] for item in listed if item["source"] == "test")
    assert (
        runner.invoke(app, ["evidence", "get", digest, "--repo", str(repo)]).stdout.strip()
        == "hello"
    )
    created = runner.invoke(
        app,
        ["checkpoint", "create", "--objective", "goal", "--evidence", digest, "--repo", str(repo)],
    )
    assert created.exit_code == 0
    shown = runner.invoke(app, ["checkpoint", "show", created.stdout.strip(), "--repo", str(repo)])
    assert json.loads(shown.stdout)["objective"] == "goal"
    assert json.loads(runner.invoke(app, ["stats", "--repo", str(repo)]).stdout)["runs"] == 1


def test_eval_executes_and_fails_regressions(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    request = CanonicalRequest([], ModelConfiguration("fake", "m"))
    passing = tmp_path / "pass.jsonl"
    passing.write_text(
        json.dumps(
            {
                "task_id": "ok",
                "input_request": request.to_dict(),
                "expected_invariants": {"answer": 1},
                "recorded_response": {"answer": 1},
                "provider_configuration_reference": "fake",
            }
        )
        + "\n"
    )
    result = runner.invoke(app, ["eval", "--corpus", str(passing), "--repo", str(repo)])
    assert result.exit_code == 0 and json.loads(result.stdout)["passed"] is True
    failing = tmp_path / "fail.jsonl"
    failing.write_text(
        json.dumps(
            {
                "task_id": "bad",
                "input_request": request.to_dict(),
                "expected_invariants": {"answer": 2},
                "recorded_response": {"answer": 1},
                "provider_configuration_reference": "fake",
            }
        )
        + "\n"
    )
    result = runner.invoke(app, ["eval", "--corpus", str(failing), "--repo", str(repo)])
    assert result.exit_code == 1 and json.loads(result.stdout)["passed"] is False
