from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import typer

from llmcut import __version__
from llmcut.config import DEFAULT_CONFIG, load_config
from llmcut.core.checkpoint import Checkpoint, CheckpointStore, repository_revision
from llmcut.core.optimize import Optimizer
from llmcut.core.recover import Recovery
from llmcut.errors import LlmcutError
from llmcut.index import RepositoryIndex, pack_repository
from llmcut.model import CanonicalRequest, ModelConfiguration
from llmcut.policy import OptimizationMode, Policy
from llmcut.proxy.app import create_app
from llmcut.proxy.security import external_bind_warning
from llmcut.store.evidence import EvidenceStore
from llmcut.store.metrics import MetricsStore

app = typer.Typer(
    help="Provider-neutral, recoverable LLM context optimization.",
    no_args_is_help=True,
    invoke_without_command=True,
)
checkpoint_app = typer.Typer(help="Create, inspect, verify, and restore checkpoints.")
evidence_app = typer.Typer(help="Inspect and retrieve content-addressed evidence.")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(evidence_app, name="evidence")


def _store(repo: Path) -> EvidenceStore:
    config = load_config(repo)
    return EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False,
) -> None:
    if version:
        typer.echo(f"llmcut {__version__}")
        raise typer.Exit()


@app.command("init")
def initialize(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    """Create safe local configuration and state, idempotently."""
    root = repo.resolve() / ".llmcut"
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    config = root / "config.toml"
    if not config.exists():
        config.write_text(DEFAULT_CONFIG)
        os.chmod(config, 0o600)
    EvidenceStore(root)
    typer.echo(f"Initialized {root}")


@app.command()
def inspect(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    include_untracked: Annotated[bool, typer.Option()] = False,
    output: Annotated[str, typer.Option("--format")] = "text",
) -> None:
    """Inspect repository scope, providers, counters, and security conditions."""
    config = load_config(repo)
    records = RepositoryIndex(repo).build(include_untracked)
    data = {
        "repository": str(repo.resolve()),
        "files": len(records),
        "changed_files": sum(item.status != "tracked" for item in records),
        "providers": sorted(config.providers),
        "token_counter": "conservative estimate",
        "state_directory": str(config.state_dir),
        "external_bind_warning": external_bind_warning(config.host),
    }
    typer.echo(
        json.dumps(data, indent=2)
        if output == "json"
        else "\n".join(f"{k}: {v}" for k, v in data.items())
    )


@app.command()
def pack(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    task: Annotated[str, typer.Option("--task")] = "",
    mode: Annotated[OptimizationMode, typer.Option("--mode")] = OptimizationMode.EXTREME,
    output: Annotated[str, typer.Option("--format")] = "json",
) -> None:
    """Build a deterministic task-specific repository context pack."""
    if not task:
        raise typer.BadParameter("--task is required")
    store = _store(repo)
    records = RepositoryIndex(repo).build()
    blocks = pack_repository(repo.resolve(), records, task, store)
    request = CanonicalRequest(blocks, ModelConfiguration("unconfigured", "unchanged"))
    optimized, report = Optimizer(store).optimize(request, Policy(mode=mode))
    _record_optimization(store, mode, report)
    data = {
        "request": optimized.to_dict(),
        "report": report.to_dict(),
        "evidence_manifest": store.list(),
    }
    if output == "markdown":
        typer.echo(
            f"# Context pack\n\nTask: {task}\n\nMode: {mode.value}\n\n"
            + "\n\n".join(
                f"## {block.source}\n\n```\n{block.content}\n```" for block in optimized.blocks
            )
        )
    elif output == "json":
        typer.echo(json.dumps(data, indent=2))
    else:
        raise typer.BadParameter("format must be json or markdown")


@app.command()
def optimize(
    input_file: Annotated[Path | None, typer.Option("--input")] = None,
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    mode: Annotated[OptimizationMode, typer.Option("--mode")] = OptimizationMode.EXTREME,
    report_only: Annotated[bool, typer.Option("--report-only", "--dry-run")] = False,
) -> None:
    """Optimize a canonical JSON request from a file or stdin."""
    raw = input_file.read_text() if input_file else sys.stdin.read()
    request = CanonicalRequest.from_dict(json.loads(raw))
    store = _store(repo)
    optimized, report = Optimizer(store).optimize(request, Policy(mode=mode))
    _record_optimization(store, mode, report)
    result: dict[str, Any] = {"report": report.to_dict(), "evidence_manifest": store.list()}
    if not report_only:
        result["request"] = optimized.to_dict()
    typer.echo(json.dumps(result, indent=2))


@app.command()
def proxy(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Start the bounded local ASGI proxy."""
    import uvicorn

    config = load_config(
        repo,
        {key: value for key, value in {"host": host, "port": port}.items() if value is not None},
    )
    warning = external_bind_warning(config.host)
    if warning:
        typer.echo(warning, err=True)
    uvicorn.run(create_app(config), host=config.host, port=config.port)


@checkpoint_app.command("create")
def checkpoint_create(
    objective: Annotated[str, typer.Option()],
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    evidence: Annotated[list[str] | None, typer.Option()] = None,
) -> None:
    checkpoint = Checkpoint(
        objective, evidence=evidence or [], repository_revision=repository_revision(repo.resolve())
    )
    typer.echo(CheckpointStore(_store(repo)).save(checkpoint))


@checkpoint_app.command("show")
def checkpoint_show(
    identifier: str,
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    verify_revision: Annotated[bool, typer.Option()] = False,
) -> None:
    value = CheckpointStore(_store(repo)).load(
        identifier, repo.resolve() if verify_revision else None
    )
    typer.echo(json.dumps(asdict(value), indent=2))


@checkpoint_app.command("verify")
def checkpoint_verify(
    identifier: str, repo: Annotated[Path, typer.Option("--repo")] = Path(".")
) -> None:
    CheckpointStore(_store(repo)).load(identifier, repo.resolve())
    typer.echo("Checkpoint evidence and repository revision verified")


@checkpoint_app.command("restore")
def checkpoint_restore(
    identifier: str, repo: Annotated[Path, typer.Option("--repo")] = Path(".")
) -> None:
    value = CheckpointStore(_store(repo)).load(identifier)
    typer.echo(
        json.dumps({digest: _store(repo).get(digest) for digest in value.evidence}, indent=2)
    )


@evidence_app.command("list")
def evidence_list(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    typer.echo(json.dumps(_store(repo).list(), indent=2))


@evidence_app.command("get")
def evidence_get(
    digest: str,
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    start: Annotated[int | None, typer.Option()] = None,
    end: Annotated[int | None, typer.Option()] = None,
) -> None:
    recovery = Recovery(_store(repo))
    typer.echo(
        recovery.source_range(digest, start, end) if start and end else recovery.evidence(digest)
    )


@app.command()
def stats(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    typer.echo(json.dumps(MetricsStore(_store(repo).db).summary(), indent=2))


@app.command("eval")
def evaluate(
    corpus: Annotated[Path, typer.Option("--corpus")],
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    minimum_reduction: Annotated[float, typer.Option("--minimum-reduction")] = 0.0,
) -> None:
    """Execute deterministic baseline-versus-optimized offline evaluations."""
    from llmcut.eval.corpus import read_corpus
    from llmcut.eval.runner import run_case
    from llmcut.tokens.estimate import ConservativeEstimator

    cases = list(read_corpus(corpus))
    store = _store(repo)
    estimator = ConservativeEstimator()
    results: list[dict[str, Any]] = []
    failed = False
    for case in cases:
        if case.provider_config not in {None, "fake", "recorded"} or case.recorded_response is None:
            results.append(
                {
                    "task_id": case.task_id,
                    "passed": False,
                    "error": "offline case requires fake/recorded provider and recorded_response",
                }
            )
            failed = True
            continue

        from functools import partial

        execute = partial(
            _execute_recorded, response=dict(case.recorded_response), estimator=estimator
        )

        result = run_case(case, Optimizer(store), execute)
        evaluator_passed, evaluator_error = _run_evaluator(case, result, corpus.parent)
        reduction = (
            (result.baseline_input_tokens - result.effective_input_tokens)
            / result.baseline_input_tokens
            * 100
            if result.baseline_input_tokens
            else 0.0
        )
        passed = not result.regression and evaluator_passed and reduction >= minimum_reduction
        failed = failed or not passed
        results.append(
            {
                "task_id": case.task_id,
                "passed": passed,
                "quality_parity": not result.regression,
                "original_tokens": result.baseline_input_tokens,
                "attempted_tokens": result.attempted_input_tokens,
                "effective_tokens": result.effective_input_tokens,
                "reduction_percent": round(reduction, 4),
                "count_quality": "estimated",
                "cached_tokens": result.cached_tokens,
                "fallback_reason": result.fallback_reason,
                "evaluator_error": evaluator_error,
            }
        )
    typer.echo(
        json.dumps({"cases": len(results), "passed": not failed, "results": results}, indent=2)
    )
    if failed:
        raise typer.Exit(1)


@app.command()
def doctor(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    """Validate configuration, permissions, database, credentials, index, and binding."""
    config = load_config(repo)
    issues: list[str] = []
    store = EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)
    store.db.integrity_check()
    mode = config.state_dir.stat().st_mode & 0o777
    if mode & 0o077:
        issues.append(f"state directory permissions are {mode:o}; expected 700")
    for provider in config.providers.values():
        if provider.credential_env and not os.environ.get(provider.credential_env):
            issues.append(
                f"provider {provider.name}: environment variable {provider.credential_env} is unset"
            )
    warning = external_bind_warning(config.host)
    if warning:
        issues.append(warning)
    RepositoryIndex(repo).build()
    if issues:
        typer.echo("\n".join(issues), err=True)
        raise typer.Exit(1)
    typer.echo(
        "Configuration, state permissions, database, migrations, evidence, index, "
        "and proxy binding: OK"
    )


@app.command()
def benchmark(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    """Run local indexing and evidence retrieval microbenchmarks."""
    import time

    started = time.perf_counter()
    records = RepositoryIndex(repo).build()
    elapsed = time.perf_counter() - started
    typer.echo(json.dumps({"files": len(records), "index_seconds": round(elapsed, 6)}, indent=2))


def _record_optimization(store: EvidenceStore, mode: OptimizationMode, report: Any) -> None:
    from llmcut.model import CountQuality, TokenCount

    quality = CountQuality(report.count_quality)
    MetricsStore(store.db).record_run(
        mode.value,
        TokenCount(report.original_tokens, quality, "optimizer"),
        TokenCount(report.optimized_tokens, quality, "optimizer"),
    )


def _run_evaluator(case: Any, result: Any, cwd: Path) -> tuple[bool, str | None]:
    if not case.evaluator_command:
        return True, None
    import subprocess

    try:
        completed = subprocess.run(
            case.evaluator_command,
            cwd=cwd,
            input=json.dumps(asdict(result)),
            text=True,
            capture_output=True,
            timeout=case.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "evaluator timed out"
    if completed.returncode != 0:
        return False, f"evaluator exited {completed.returncode}"
    return True, None


def _execute_recorded(
    value: CanonicalRequest, *, response: dict[str, Any], estimator: Any
) -> tuple[dict[str, Any], dict[str, int]]:
    return dict(response), {
        "input_tokens": estimator.count(value.to_json(), model=value.model.model).value,
        "output_tokens": estimator.count(json.dumps(response)).value,
        "cached_tokens": 0,
        "recovery_tokens": 0,
        "retries": 0,
    }


def run() -> None:
    try:
        app()
    except (LlmcutError, json.JSONDecodeError, KeyError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from exc
