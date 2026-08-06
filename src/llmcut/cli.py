from __future__ import annotations

import contextlib
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
context_app = typer.Typer(help="Plan trusted repository orientation for coding agents.")
agent_app = typer.Typer(help="Evaluate supported coding-agent integrations.")
codex_app = typer.Typer(help="Inspect and configure the experimental Codex integration.")
codex_hooks_app = typer.Typer(help="Inspect and configure Codex lifecycle hooks.")
codex_exec_app = typer.Typer(help="Probe the structured codex exec evaluation surface.")
hook_app = typer.Typer(help="Handle hooks and recover exact compacted tool output.")
app.add_typer(checkpoint_app, name="checkpoint")
app.add_typer(evidence_app, name="evidence")
app.add_typer(capture_app, name="capture")
app.add_typer(tokens_app, name="tokens")
app.add_typer(mcp_app, name="mcp")
app.add_typer(context_app, name="context")
app.add_typer(agent_app, name="agent")
app.add_typer(hook_app, name="hook")
agent_app.add_typer(codex_app, name="codex")
codex_app.add_typer(codex_hooks_app, name="hooks")
codex_app.add_typer(codex_exec_app, name="exec")


def _store(repo: Path) -> EvidenceStore:
    config = load_config(repo)
    return EvidenceStore(config.state_dir, persist_content=config.persist_prompt_content)


def _hook_state_root() -> Path:
    configured = os.environ.get("LLMCUT_HOOK_STATE")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return (base / "llmcut" / "hooks").resolve()


def _hook_config_path() -> Path:
    return Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "hooks.json"


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
    integration: Annotated[str, typer.Option("--integration")] = "guided",
    run_state: Annotated[Path | None, typer.Option("--run-state", hidden=True)] = None,
) -> None:
    from llmcut.mcp.server import serve

    try:
        serve(repo, integration, run_state)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@mcp_app.command("inspect")
def mcp_inspect(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    task: Annotated[str, typer.Option("--task")] = "Inspect repository context",
    integration: Annotated[str, typer.Option("--integration")] = "guided",
) -> None:
    from llmcut.integrations.codex.context import plan_codex_context
    from llmcut.mcp.server import GUIDED_INSTRUCTIONS, LEGACY_INSTRUCTIONS, tool_schema_bytes

    try:
        plan = plan_codex_context(repo, task, integration)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    selected = plan.selected_strategy.value
    tools = (
        ["llmcut_context"]
        if selected == "guided"
        else (
            [
                "llmcut_plan",
                "llmcut_context_get",
                "llmcut_source_range",
                "llmcut_symbol_get",
                "llmcut_dependencies",
                "llmcut_log_search",
                "llmcut_checkpoint_get",
                "llmcut_tool_discover",
            ]
            if selected == "legacy-passive"
            else []
        )
    )
    instructions = (
        GUIDED_INSTRUCTIONS
        if selected == "guided"
        else (LEGACY_INSTRUCTIONS if selected == "legacy-passive" else "")
    )
    typer.echo(
        json.dumps(
            {
                "transport": "stdio",
                "repository": str(repo.resolve()),
                "requested_strategy": integration,
                "selected_strategy": selected,
                "indexed_files": plan.repository_file_count,
                "exposed_tools": tools,
                "schema_bytes": tool_schema_bytes(selected),
                "schema_token_estimate": plan.mcp_schema_estimate,
                "instruction_token_estimate": (
                    max(1, (len(instructions.encode()) + 2) // 3) if instructions else 0
                ),
                "orientation_token_estimate": plan.orientation_token_estimate,
                "selected_files": [item.path for item in plan.selected_files],
                "deferred_files": list(plan.deferred_files),
                "adaptive_reasons": list(plan.decision_reasons),
            },
            indent=2,
        )
    )


@context_app.command("plan")
def context_plan(
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    task: Annotated[str, typer.Option("--task")] = "",
    strategy: Annotated[str, typer.Option("--strategy")] = "adaptive",
    orientation_budget: Annotated[int, typer.Option("--orientation-budget")] = 200,
    retrieval_budget: Annotated[int, typer.Option("--retrieval-budget")] = 4_096,
) -> None:
    """Inspect a task-aware plan; source contents are never emitted."""
    from llmcut.integrations.codex.context import plan_codex_context

    try:
        plan = plan_codex_context(
            repo,
            task,
            strategy,
            orientation_budget=orientation_budget,
            retrieval_budget=retrieval_budget,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


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
def codex_doctor(
    backend: Annotated[str, typer.Option("--backend")] = "sdk",
) -> None:
    import asyncio

    from llmcut.integrations.codex import detect_codex
    from llmcut.integrations.codex.auth import authentication_preflight
    from llmcut.integrations.codex.backend import create_backend

    report = detect_codex().to_dict()
    report["selected_backend"] = backend
    capabilities = asyncio.run(create_backend(backend).doctor())
    report["backend_capabilities"] = asdict(capabilities)
    report["authentication"] = authentication_preflight(mode="existing-session").to_dict()
    report["evaluation_ready"] = bool(capabilities.installed and capabilities.usage_events)
    typer.echo(json.dumps(report, indent=2))


@codex_exec_app.command("probe")
def codex_exec_probe(
    output_format: Annotated[str, typer.Option("--format")] = "text",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    allow_hook_trust_bypass: Annotated[bool, typer.Option("--allow-hook-trust-bypass")] = False,
) -> None:
    """Run one bounded authenticated JSONL/hook contract probe in a disposable Git fixture."""
    import asyncio
    import shutil
    import subprocess
    import tempfile

    from llmcut.integrations.codex.auth import authentication_preflight
    from llmcut.integrations.codex.backend import codex_agent_environment, create_backend
    from llmcut.integrations.codex.hooks.capabilities import capabilities_for
    from llmcut.integrations.codex.hooks.config import (
        definition_digest,
        inline_overrides,
        proposed_document,
    )

    if output_format not in {"text", "json"}:
        raise typer.BadParameter("format must be text or json")
    if not allow_hook_trust_bypass:
        raise typer.BadParameter("--allow-hook-trust-bypass is required for the hook probe")
    auth = authentication_preflight(mode="existing-session")
    if not auth.automation_ready:
        raise typer.BadParameter(auth.diagnostic or "Codex authentication is unavailable")
    root = Path(tempfile.mkdtemp(prefix="llmcut-exec-probe-"))
    os.chmod(root, 0o700)
    try:
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "probe@llmcut.invalid"], cwd=root, check=True
        )
        subprocess.run(["git", "config", "user.name", "llmcut probe"], cwd=root, check=True)
        (root / "README.md").write_text("disposable llmcut exec probe\n")
        (root / "test_probe.py").write_text(
            "import pytest\n\n@pytest.mark.parametrize('value', range(240))\n"
            "def test_value(value):\n    assert value >= 0\n"
        )
        subprocess.run(["git", "add", "README.md", "test_probe.py"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "probe fixture"], cwd=root, check=True)
        hook_dir = root / ".codex"
        hook_dir.mkdir(mode=0o700)
        hook_file = hook_dir / "hooks.json"
        hook_file.write_text(json.dumps(proposed_document(), separators=(",", ":")) + "\n")
        os.chmod(hook_file, 0o600)
        state = root.parent / f".{root.name}-state"
        metrics = root.parent / f".{root.name}-metrics.jsonl"
        environment = codex_agent_environment((), "probe", "existing-session", None)
        environment.update(
            {
                "LLMCUT_HOOK_REPO": str(root),
                "LLMCUT_HOOK_STATE": str(state),
                "LLMCUT_HOOK_METRICS": str(metrics),
                "LLMCUT_HOOK_DEFINITION_DIGEST": definition_digest(),
            }
        )
        backend = create_backend("exec", allow_hook_trust_bypass=True)
        capabilities = asyncio.run(backend.doctor())
        result = asyncio.run(
            backend.run(
                task=(
                    "Use Bash once to run `python -m pytest -vv` and then report whether its exact "
                    "result was available. Do not edit files."
                ),
                cwd=root,
                model="gpt-5.6-terra",
                reasoning="low",
                sandbox="workspace-write",
                approval_policy="never",
                timeout=180,
                max_turns=1,
                environment=environment,
                config_overrides=inline_overrides(),
                validation_callback=None,
                cancellation=None,
            )
        )
        metric_count = len(metrics.read_text().splitlines()) if metrics.is_file() else 0
        compacted = 0
        if metrics.is_file():
            compacted = sum(
                json.loads(line).get("applied") is True for line in metrics.read_text().splitlines()
            )
        runtime = capabilities.runtime_version or "unavailable"
        report = {
            "runtime_version": runtime,
            "jsonl_contract": "observed",
            "terminal_state": result.status,
            "usage_fields": sorted(result.usage or {}),
            "usage": result.usage,
            "command_events": sum(event.kind == "command_execution" for event in result.events),
            "hook_events": metric_count,
            "compacted_events": compacted,
            "exclusive_replacement_capability": capabilities_for(runtime).post_replacement,
            "resume_capability": capabilities.resumable_turns,
            "resolved_model_observation": "unavailable",
            "process_cleanup": "observed"
            if getattr(backend, "_process", None) is None
            else "failed",
            "hook_definition_digest": definition_digest(),
            "trust_bypass": True,
            "result_state": (
                "passed"
                if result.status == "completed" and metric_count > 0 and compacted > 0
                else "failed_hook_activation"
            ),
        }
    finally:
        shutil.rmtree(root, ignore_errors=True)
        if "state" in locals():
            shutil.rmtree(state, ignore_errors=True)
        if "metrics" in locals():
            metrics.unlink(missing_ok=True)
    rendered = (
        json.dumps(report, indent=2)
        if output_format == "json"
        else "\n".join(f"{key}: {value}" for key, value in report.items())
    )
    if output is not None:
        target = output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_text(rendered + "\n")
        os.chmod(target, 0o600)
    else:
        typer.echo(rendered)
    if report["result_state"] != "passed":
        raise typer.Exit(1)


@codex_app.command("auth")
def codex_auth(
    mode: Annotated[str, typer.Option("--mode")] = "existing-session",
    env_var: Annotated[str | None, typer.Option("--env-var")] = None,
) -> None:
    from llmcut.integrations.codex.auth import authentication_preflight

    typer.echo(json.dumps(authentication_preflight(mode=mode, env_var=env_var).to_dict(), indent=2))


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


@codex_hooks_app.command("doctor")
def codex_hooks_doctor() -> None:
    """Report runtime hook support without reading credentials."""
    import shutil
    import subprocess

    from llmcut.integrations.codex.hooks.capabilities import capabilities_for
    from llmcut.integrations.codex.hooks.config import definition_digest

    executable = shutil.which("codex")
    version = None
    bypass = False
    if executable:
        result = subprocess.run(
            [executable, "--version"], text=True, capture_output=True, timeout=5, check=False
        )
        version = result.stdout.strip() if result.returncode == 0 else None
        help_result = subprocess.run(
            [executable, "--help"], text=True, capture_output=True, timeout=5, check=False
        )
        bypass = "--dangerously-bypass-hook-trust" in help_result.stdout
    target = _hook_config_path()
    capabilities = capabilities_for(version or "unavailable")
    typer.echo(
        json.dumps(
            {
                "supported": bool(executable and bypass),
                "runtime_version": version,
                "hooks_feature": "enabled_by_default; configuration may override",
                "supported_events": ["SessionStart", "PostToolUse"],
                "configured_source": str(target),
                "configured": target.is_file(),
                "hook_definition_digest": definition_digest(),
                "trust_state": "observable through Codex /hooks only",
                "exclusive_model_replacement_verified": (
                    capabilities.post_replacement == "supported"
                ),
                "direct_exec_probe_ready": bool(
                    executable and bypass and capabilities.post_replacement == "supported"
                ),
                "evaluation_ready": False,
                "evaluation_blocker": (
                    "SDK App Server hook activation was not observed; direct-exec conformance "
                    "does not establish evaluation-surface support"
                    if capabilities.post_replacement == "supported"
                    else "exact runtime version has no committed direct-exec replacement probe"
                ),
                "one_off_trust_bypass": bypass,
            },
            indent=2,
        )
    )


@codex_hooks_app.command("config")
def codex_hooks_config() -> None:
    from llmcut.integrations.codex.hooks.config import definition_digest, proposed_document

    document = proposed_document()
    typer.echo(
        json.dumps(
            {
                "target": str(_hook_config_path()),
                "definition_digest": definition_digest(document),
                "configuration": document,
                "mutates_files": False,
            },
            indent=2,
        )
    )


@codex_hooks_app.command("install")
def codex_hooks_install(
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    from llmcut.integrations.codex.hooks.config import install_hooks

    result = install_hooks(config_path or _hook_config_path(), dry_run=dry_run)
    result["persistent_trust_installed"] = False
    result["review"] = "Review and trust the exact definition using Codex /hooks."
    typer.echo(json.dumps(result, indent=2))


@codex_hooks_app.command("remove")
def codex_hooks_remove(
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    from llmcut.integrations.codex.hooks.config import remove_hooks

    typer.echo(
        json.dumps(remove_hooks(config_path or _hook_config_path(), dry_run=dry_run), indent=2)
    )


@codex_hooks_app.command("capabilities")
def codex_hooks_capabilities() -> None:
    """Report only version-bound, committed hook conformance evidence."""
    import shutil
    import subprocess

    executable = shutil.which("codex")
    version = "unavailable"
    if executable:
        result = subprocess.run(
            [executable, "--version"], text=True, capture_output=True, timeout=5, check=False
        )
        if result.returncode == 0:
            version = result.stdout.strip()
    from llmcut.integrations.codex.hooks.capabilities import capabilities_for

    report = capabilities_for(version).to_dict()
    report["detail"] = "capabilities are valid only for an exact probed runtime version"
    typer.echo(json.dumps(report, indent=2))


@codex_hooks_app.command("probe")
def codex_hooks_probe(
    post_tool_use: Annotated[bool, typer.Option("--post-tool-use")] = False,
    pre_tool_use: Annotated[bool, typer.Option("--pre-tool-use")] = False,
    output_format: Annotated[str, typer.Option("--format")] = "text",
    output: Annotated[Path | None, typer.Option("--output")] = None,
    allow_hook_trust_bypass: Annotated[bool, typer.Option("--allow-hook-trust-bypass")] = False,
    variant: Annotated[str | None, typer.Option("--variant", hidden=True)] = None,
) -> None:
    """Run an explicit, version-bound hook conformance probe."""
    from llmcut.integrations.codex.hooks.conformance import PostVariant, run_live_post_matrix

    if post_tool_use == pre_tool_use:
        raise typer.BadParameter("select exactly one conformance probe")
    if not allow_hook_trust_bypass:
        raise typer.BadParameter("--allow-hook-trust-bypass is required for automation")
    if pre_tool_use:
        raise typer.BadParameter(
            "PreToolUse probe is unavailable until the PostToolUse matrix runs"
        )
    if output_format not in {"text", "json"}:
        raise typer.BadParameter("format must be text or json")
    variants = (PostVariant(variant),) if variant else None
    executable = os.environ.get("LLMCUT_PROBE_CODEX", "codex")
    results = [
        asdict(item) for item in run_live_post_matrix(executable=executable, variants=variants)
    ]
    rendered = (
        json.dumps(results, indent=2)
        if output_format == "json"
        else "\n".join(f"{item['variant']}: {item['state']}" for item in results)
    )
    if output is not None:
        target = output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_text(rendered + "\n")
        os.chmod(target, 0o600)
    else:
        typer.echo(rendered)


@hook_app.command("handle")
def hook_handle() -> None:
    """Handle one bounded Codex hook event from stdin; malformed input passes through."""
    from llmcut.integrations.codex.hooks.config import MAX_HOOK_INPUT, HookConfig
    from llmcut.integrations.codex.hooks.handler import append_metrics, handle_hook

    raw = sys.stdin.buffer.read(MAX_HOOK_INPUT + 1)
    repository = Path(os.environ.get("LLMCUT_HOOK_REPO", os.getcwd())).resolve()
    response, metrics = handle_hook(raw, HookConfig(repository, _hook_state_root()))
    metrics_path = os.environ.get("LLMCUT_HOOK_METRICS")
    if metrics_path:
        with contextlib.suppress(OSError):
            append_metrics(Path(metrics_path), metrics)
    if response is not None:
        typer.echo(json.dumps(response, separators=(",", ":")))


@hook_app.command("conformance-handle", hidden=True)
def hook_conformance_handle(
    state: Annotated[Path, typer.Option("--state")],
) -> None:
    from llmcut.integrations.codex.hooks.config import MAX_HOOK_INPUT
    from llmcut.integrations.codex.hooks.conformance import handle_conformance_hook

    response, exit_code, stderr = handle_conformance_hook(
        sys.stdin.buffer.read(MAX_HOOK_INPUT + 1), state
    )
    if stderr:
        typer.echo(stderr, err=True)
    if response is not None:
        typer.echo(json.dumps(response, separators=(",", ":")))
    if exit_code:
        raise typer.Exit(exit_code)


def _hook_store() -> Any:
    from llmcut.integrations.codex.hooks.state import HookEvidenceStore

    return HookEvidenceStore(_hook_state_root())


@hook_app.command("show")
def hook_show(evidence_id: str) -> None:
    from llmcut.integrations.codex.hooks.state import render_exact

    typer.echo(render_exact(_hook_store().get(evidence_id)), nl=False)


@hook_app.command("info")
def hook_info(evidence_id: str) -> None:
    typer.echo(json.dumps(_hook_store().info(evidence_id), indent=2, sort_keys=True))


@hook_app.command("range")
def hook_range(
    evidence_id: str,
    start: Annotated[int, typer.Option("--start")],
    end: Annotated[int, typer.Option("--end")],
) -> None:
    from llmcut.integrations.codex.hooks.state import exact_lines

    if start < 1 or end < start or end - start > 2_000:
        raise typer.BadParameter("range must be ordered, one-based, and at most 2001 lines")
    typer.echo("".join(exact_lines(_hook_store().get(evidence_id))[start - 1 : end]), nl=False)


@hook_app.command("search")
def hook_search(
    evidence_id: str,
    pattern: Annotated[str, typer.Option("--pattern")],
) -> None:
    from llmcut.integrations.codex.hooks.state import exact_lines

    if not pattern or len(pattern) > 512:
        raise typer.BadParameter("literal pattern must contain 1 to 512 characters")
    matches = [line for line in exact_lines(_hook_store().get(evidence_id)) if pattern in line]
    typer.echo("".join(matches[:2_000]), nl=False)


@hook_app.command("gc")
def hook_gc(
    maximum_age: Annotated[int, typer.Option("--maximum-age")] = 604_800,
    maximum_bytes: Annotated[int, typer.Option("--maximum-bytes")] = 268_435_456,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    if maximum_age < 0 or maximum_bytes < 0:
        raise typer.BadParameter("GC bounds must be nonnegative")
    typer.echo(
        json.dumps(
            _hook_store().collect(
                maximum_age_seconds=maximum_age,
                maximum_total_bytes=maximum_bytes,
                dry_run=dry_run,
            ),
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
    backend: Annotated[str, typer.Option("--backend")] = "sdk",
    timeout: Annotated[float, typer.Option("--timeout")] = 900,
    auth_mode: Annotated[str, typer.Option("--auth-mode")] = "existing-session",
    auth_env_var: Annotated[str | None, typer.Option("--auth-env-var")] = None,
) -> None:
    import asyncio

    from llmcut.integrations.codex.auth import authentication_preflight
    from llmcut.integrations.codex.backend import codex_agent_environment, create_backend

    auth = authentication_preflight(mode=auth_mode, env_var=auth_env_var)
    if not auth.automation_ready:
        typer.echo(f"unsupported environment: {auth.diagnostic}", err=True)
        raise typer.Exit(3)
    result = asyncio.run(
        create_backend(backend).run(
            task=task,
            cwd=repo,
            model=model,
            reasoning=reasoning,
            sandbox=sandbox,
            approval_policy=approvals,
            timeout=timeout,
            max_turns=1,
            environment=codex_agent_environment((), "optimized", auth_mode, auth_env_var),
            config_overrides=(),
            validation_callback=None,
            cancellation=None,
        )
    )
    typer.echo(json.dumps(asdict(result), indent=2))


@codex_app.command("eval")
def codex_eval_alias(
    suite: Annotated[Path, typer.Option("--suite")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    backend: Annotated[str, typer.Option("--backend")] = "sdk",
) -> None:
    agent_evaluate("codex", suite, dry_run, backend=backend)


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
    backend: Annotated[str | None, typer.Option("--backend")] = None,
    auth_mode: Annotated[str | None, typer.Option("--auth-mode")] = None,
    auth_env_var: Annotated[str | None, typer.Option("--auth-env-var")] = None,
    context_strategy: Annotated[str | None, typer.Option("--context-strategy")] = None,
    pilot: Annotated[bool, typer.Option("--pilot")] = False,
    allow_hook_trust_bypass: Annotated[bool, typer.Option("--allow-hook-trust-bypass")] = False,
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
        backend=backend,
        auth_mode=auth_mode,
        auth_env_var=auth_env_var,
        context_strategy=context_strategy,
        pilot=pilot,
        allow_hook_trust_bypass=allow_hook_trust_bypass,
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
