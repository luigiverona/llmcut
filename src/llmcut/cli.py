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
from llmcut.managed.protocol import ManagedRequest
from llmcut.managed.runtime import ManagedRuntime
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
capture_app = typer.Typer(help="Inspect, verify, redact, replay, and delete captures.")
tokens_app = typer.Typer(help="Count and verify exact provider-bound payloads.")
mcp_app = typer.Typer(help="Serve and inspect the llmcut MCP integration.")
agent_app = typer.Typer(help="Evaluate supported coding-agent integrations.")
codex_app = typer.Typer(help="Inspect and configure the experimental Codex integration.")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(evidence_app, name="evidence")
app.add_typer(capture_app, name="capture")
app.add_typer(tokens_app, name="tokens")
app.add_typer(mcp_app, name="mcp")
app.add_typer(agent_app, name="agent")
agent_app.add_typer(codex_app, name="codex")


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
    repository_index = RepositoryIndex(repo)
    records = repository_index.build(include_untracked)
    data = {
        "repository": str(repo.resolve()),
        "files": len(records),
        "changed_files": sum(item.status != "tracked" for item in records),
        "providers": sorted(config.providers),
        "token_counter": "conservative estimate",
        "state_directory": str(config.state_dir),
        "external_bind_warning": external_bind_warning(config.host),
        "index_cache": repository_index.stats(),
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
    repository_index = RepositoryIndex(repo)
    records = repository_index.build()
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


@app.command("run")
def managed_run(
    request_file: Annotated[Path, typer.Option("--request")],
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    mode: Annotated[OptimizationMode, typer.Option("--mode")] = OptimizationMode.EXTREME,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Plan or execute a versioned provider-neutral managed request."""
    request = ManagedRequest.from_dict(json.loads(request_file.read_text()))
    request.execution.optimization = mode
    store = _store(repo)
    if dry_run:
        import asyncio

        result = asyncio.run(ManagedRuntime(store).run(request, dry_run=True))
    else:
        from llmcut.client import Client

        result = Client.from_config(repo).run(request, mode.value)
    typer.echo(json.dumps(result.to_dict(), indent=2))


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
    if corpus.suffix == ".toml":
        from llmcut.eval.executable import evaluate_suite

        executable_results, statistics = evaluate_suite(corpus)
        typer.echo(
            json.dumps(
                {
                    "cases": [item.to_dict() for item in executable_results],
                    "statistics": statistics,
                },
                indent=2,
            )
        )
        if not statistics["passed"]:
            raise typer.Exit(1)
        return
    first = next((line for line in corpus.read_text().splitlines() if line.strip()), "")
    if first and "managed_request" in json.loads(first):
        import asyncio

        from llmcut.eval.managed import evaluate_recorded_corpus, release_targets

        store = _store(repo)
        managed_results = asyncio.run(
            evaluate_recorded_corpus(corpus, lambda provider: ManagedRuntime(store, provider))
        )
        targets = release_targets(managed_results)
        rendered = [
            {
                "task_id": item.task_id,
                "baseline_tokens": item.baseline_tokens,
                "initial_tokens": item.initial_tokens,
                "retrieval_tokens": item.retrieval_tokens,
                "continuation_tokens": item.continuation_tokens,
                "total_tokens": item.total_tokens,
                "output_tokens": item.output_tokens,
                "cached_tokens": item.cached_tokens,
                "reduction_percent": round(item.reduction_percent, 4),
                "retrievals": item.retrievals,
                "turns": item.turns,
                "quality_passed": item.quality_passed,
                "saving": item.saving,
                "fallback": item.fallback,
            }
            for item in managed_results
        ]
        typer.echo(
            json.dumps(
                {
                    "cases": rendered,
                    "targets": targets,
                    "measurement_trust": "untrusted_fixture",
                    "release_eligible": False,
                    "purpose": "adapter parsing and managed replay only",
                },
                indent=2,
            )
        )
        return
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
    repository_index = RepositoryIndex(repo)
    records = repository_index.build()
    elapsed = time.perf_counter() - started
    typer.echo(
        json.dumps(
            {"files": len(records), "index_seconds": round(elapsed, 6), **repository_index.stats()},
            indent=2,
        )
    )


@capture_app.command("inspect")
def capture_inspect(path: Path) -> None:
    from llmcut.captures import load_capture

    value = load_capture(path)
    safe = {
        key: value.get(key)
        for key in (
            "schema_version",
            "capture_id",
            "provider",
            "model",
            "endpoint",
            "persistence",
            "redaction",
        )
    }
    safe["turns"] = len(value["turns"])
    typer.echo(json.dumps(safe, indent=2))


@capture_app.command("verify")
def capture_verify(path: Path) -> None:
    from llmcut.captures import verify_capture

    typer.echo(json.dumps(asdict(verify_capture(path)), indent=2))


@capture_app.command("redact")
def capture_redact(path: Path) -> None:
    from llmcut.captures import redact_capture

    typer.echo(json.dumps({"redacted_fields": redact_capture(path)}, indent=2))


@capture_app.command("replay")
def capture_replay(path: Path) -> None:
    """Verify a capture and describe its offline replay; never contact a provider."""
    from llmcut.captures import load_capture, verify_capture

    verification = verify_capture(path)
    value = load_capture(path)
    typer.echo(json.dumps({"verified": asdict(verification), "turns": value["turns"]}, indent=2))


@capture_app.command("delete")
def capture_delete(path: Path) -> None:
    from llmcut.captures import delete_capture

    delete_capture(path)
    typer.echo("Capture removed after reference and digest verification")


@tokens_app.command("count")
def tokens_count(
    provider: Annotated[str, typer.Option("--provider")],
    model: Annotated[str, typer.Option("--model")],
    input_file: Annotated[Path, typer.Option("--input")],
) -> None:
    from llmcut.measurement import count_payload
    from llmcut.tokens.registry import CounterRegistry

    payload = json.loads(input_file.read_text())
    typer.echo(
        json.dumps(count_payload(CounterRegistry(), provider, model, payload).to_dict(), indent=2)
    )


@tokens_app.command("verify")
def tokens_verify(path: Path) -> None:
    from llmcut.captures import verify_capture

    typer.echo(json.dumps(asdict(verify_capture(path)), indent=2))


@tokens_app.command("compare")
def tokens_compare(
    baseline: Path,
    optimized: Path,
    provider: Annotated[str, typer.Option("--provider")],
    model: Annotated[str, typer.Option("--model")],
) -> None:
    from llmcut.measurement import count_payload
    from llmcut.tokens.registry import CounterRegistry

    registry = CounterRegistry()
    first = count_payload(registry, provider, model, json.loads(baseline.read_text()))
    second = count_payload(registry, provider, model, json.loads(optimized.read_text()))
    reduction = (first.value - second.value) / first.value * 100 if first.value else 0.0
    typer.echo(
        json.dumps(
            {
                "baseline": first.to_dict(),
                "optimized": second.to_dict(),
                "reduction_percent": reduction,
            },
            indent=2,
        )
    )


@mcp_app.command("serve")
def mcp_serve(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    integration: Annotated[str, typer.Option("--integration")] = "optimized",
) -> None:
    from llmcut.mcp.server import serve

    if integration not in {"baseline", "optimized"}:
        raise typer.BadParameter("integration must be baseline or optimized")
    serve(repo)


@mcp_app.command("inspect")
def mcp_inspect(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    from llmcut.mcp.server import RepositoryContext

    context = RepositoryContext(repo)
    typer.echo(
        json.dumps(
            {
                "transport": "stdio",
                "repository": str(context.root),
                "indexed_files": len(context.records),
                "tools": 8,
                "resources": ["llmcut://repository/map", "llmcut://context/<id>"],
            },
            indent=2,
        )
    )


@mcp_app.command("doctor")
def mcp_doctor(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    from llmcut.mcp.server import RepositoryContext

    context = RepositoryContext(repo)
    typer.echo(
        "MCP stdio, repository allowlist, secret exclusion, and index: OK "
        f"({len(context.records)} files)"
    )


@mcp_app.command("config")
def mcp_config(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    typer.echo(
        json.dumps({"command": "llmcut", "args": ["mcp", "serve", "--repo", str(repo.resolve())]})
    )


@codex_app.command("doctor")
def codex_doctor() -> None:
    from llmcut.integrations.codex import detect_codex

    typer.echo(json.dumps(detect_codex().to_dict(), indent=2))


@codex_app.command("config")
def codex_config(repo: Annotated[Path, typer.Option("--repo")] = Path(".")) -> None:
    from llmcut.integrations.codex import configuration_snippet

    typer.echo(configuration_snippet(repo))


@codex_app.command("init")
def codex_init(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    remove: Annotated[bool, typer.Option("--remove")] = False,
) -> None:
    from llmcut.integrations.codex.config import configure_codex
    from llmcut.integrations.codex.doctor import default_config_path

    change = configure_codex(
        config_path or default_config_path(), repo, remove=remove, dry_run=dry_run
    )
    typer.echo(
        json.dumps(
            {
                "changed": change.changed,
                "path": str(change.path),
                "backup": str(change.backup) if change.backup else None,
                "before": change.before,
                "after": change.after,
            },
            indent=2,
        )
    )


@codex_app.command("run")
def codex_run(
    task: Annotated[str, typer.Option("--task")],
    model: Annotated[str, typer.Option("--model")],
    reasoning: Annotated[str, typer.Option("--reasoning")],
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    sandbox: Annotated[str, typer.Option("--sandbox")] = "workspace-write",
    approvals: Annotated[str, typer.Option("--approvals")] = "on-request",
) -> None:
    import asyncio

    from llmcut.integrations.codex import CodexAppServer

    result = asyncio.run(
        CodexAppServer().run(
            task=task,
            cwd=repo,
            model=model,
            reasoning=reasoning,
            sandbox=sandbox,
            approval_policy=approvals,
        )
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@codex_app.command("eval")
def codex_eval_alias(
    suite: Annotated[Path, typer.Option("--suite")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    agent_evaluate("codex", suite, dry_run)


@agent_app.command("eval")
def agent_evaluate(
    agent: Annotated[str, typer.Option("--agent")],
    suite: Annotated[Path, typer.Option("--suite")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    repetitions: Annotated[int | None, typer.Option("--repetitions")] = None,
    order: Annotated[str | None, typer.Option("--order")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    keep_worktrees: Annotated[bool, typer.Option("--keep-worktrees")] = False,
    output_format: Annotated[str, typer.Option("--format")] = "text",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    timeout: Annotated[float | None, typer.Option("--timeout")] = None,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    capture: Annotated[Path | None, typer.Option("--capture")] = None,
) -> None:
    import asyncio
    import tempfile

    if agent != "codex":
        raise typer.BadParameter("only the isolated codex integration is supported")
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("format must be text or json")
    from llmcut.captures import write_agent_capture
    from llmcut.integrations.codex.executor import CodexEvaluator
    from llmcut.integrations.codex.suite import load_suite

    evaluator = CodexEvaluator(
        load_suite(suite),
        repetitions=repetitions,
        order=order,
        seed=seed,
        timeout=timeout,
        keep_worktrees=keep_worktrees,
        fail_fast=fail_fast,
    )
    try:
        evaluation = evaluator.plan() if dry_run else asyncio.run(evaluator.run())
    except RuntimeError as exc:
        typer.echo(f"unsupported environment: {exc}", err=True)
        raise typer.Exit(3) from exc
    report = evaluation.to_dict()
    if capture is not None:
        write_agent_capture(report, capture)
        report["capture"] = str(capture.resolve())
    rendered = (
        json.dumps(report, indent=2) if output_format == "json" else _agent_text_report(report)
    )
    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(descriptor, "w") as handle:
                handle.write(rendered + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, output)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    else:
        typer.echo(rendered)
    if not dry_run and not bool(report["summary"].get("passed")):
        raise typer.Exit(1)


def _agent_text_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    lines = [
        f"Codex agent evaluation {report.get('run_id')}",
        f"dry run: {report.get('dry_run')}",
        f"runs: {summary.get('runs', summary.get('planned_runs', 0))}",
        f"quality successes: {summary.get('quality_successes', 'not executed')}",
        f"eligible comparisons: {summary.get('eligible_comparisons', 'not executed')}",
        f"median payload reduction: {summary.get('median_payload_reduction_percent')}",
        f"agent usage comparisons: {summary.get('agent_usage_comparisons', 0)}",
        "subscription usage: unavailable",
    ]
    return "\n".join(lines)


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
        "input_tokens": estimator.count(value.model_bound_json(), model=value.model.model).value,
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
