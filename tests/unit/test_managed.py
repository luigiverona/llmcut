from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from llmcut import AsyncClient, Client, Context, ManagedRequest, ToolDefinition
from llmcut.core.checkpoint import Checkpoint
from llmcut.core.history import compact_history
from llmcut.errors import ExecutionError, IntegrityError, ProtocolError, RetrievalError
from llmcut.eval.managed import ManagedEvaluation, release_targets
from llmcut.managed.planner import ContextPlanner
from llmcut.managed.protocol import ExecutionSettings
from llmcut.managed.retrieval import RetrievalService
from llmcut.managed.runtime import ManagedRuntime
from llmcut.managed.tools import ToolRegistry
from llmcut.model import (
    BlockKind,
    CanonicalRequest,
    ContextBlock,
    ModelConfiguration,
    Retention,
)
from llmcut.policy import IntegrationMode, OptimizationMode
from llmcut.store.evidence import EvidenceStore
from llmcut.store.metrics import MetricsStore
from llmcut.tokens.registry import CounterRegistry


def managed_request() -> ManagedRequest:
    return ManagedRequest(
        "openai",
        "same-model",
        "Fix callback.py timeout",
        [
            Context(
                "policy",
                BlockKind.SYSTEM,
                "Never reduce validation.",
                Retention.STABLE,
                100,
            ),
            Context.source("src/callback.py", "def callback():\n    return timeout\n" * 20),
            Context.document("docs/unrelated.md", "unrelated evidence\n" * 100),
        ],
        settings={"temperature": 0, "reasoning": {"effort": "high"}},
    )


def test_protocol_round_trip_preserves_extensions() -> None:
    value = managed_request().to_dict()
    value["future"] = {"kept": True}
    value["context"][1]["future_block"] = 7
    parsed = ManagedRequest.from_dict(value)
    rendered = parsed.to_dict()
    assert rendered["future"] == {"kept": True}
    assert rendered["context"][1]["future_block"] == 7
    assert rendered["schema_version"] == "1"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value.update(schema_version="2"), "unsupported"),
        (lambda value: value["context"].append(dict(value["context"][0])), "duplicate"),
        (lambda value: value["context"][1].update(dependencies=["missing"]), "dependencies"),
        (lambda value: value.update(api_key="bad"), "credential"),
        (lambda value: value["execution"].update(integration="transparent"), "managed"),
    ],
)
def test_protocol_rejects_invalid_values(mutate: Any, match: str) -> None:
    value = managed_request().to_dict()
    mutate(value)
    with pytest.raises(ProtocolError, match=match):
        ManagedRequest.from_dict(value)


def test_protocol_cycles_and_tool_continuity() -> None:
    value = managed_request().to_dict()
    value["context"][1]["dependencies"] = ["docs-unrelated.md"]
    value["context"][2]["dependencies"] = ["src-callback.py"]
    with pytest.raises(ProtocolError, match="cyclic"):
        ManagedRequest.from_dict(value)
    value = managed_request().to_dict()
    value["context"].append(
        {
            "id": "orphan",
            "kind": "tool_result",
            "content": "x",
            "retention": "required",
            "tool_call_id": "no-call",
        }
    )
    with pytest.raises(ProtocolError, match="preceding"):
        ManagedRequest.from_dict(value)


def test_serialization_boundaries_exclude_internal_state() -> None:
    block = ContextBlock(
        "a",
        BlockKind.USER,
        "model text",
        "/secret/local/path",
        metadata={"confidence": "high", "proven_redundant": True},
    )
    block.tokens = CounterRegistry().estimate.count("model text")
    request = CanonicalRequest(
        [block], ModelConfiguration("openai", "m"), passthrough={"llmcut_original": "private"}
    )
    state = request.to_json()
    model = request.model_bound_json()
    assert "confidence" in state and "llmcut_original" in state
    assert "confidence" not in model and "llmcut_original" not in model
    assert "/secret/local/path" not in model and "tokens" not in model
    restored = CanonicalRequest.from_dict(json.loads(state))
    assert restored.blocks[0].content == "model text"


def test_planner_dependency_stable_prefix_and_noop(tmp_path: Path) -> None:
    request = managed_request()
    request.context[1].dependencies = ("docs-unrelated.md",)
    request.context[2].retention = Retention.STABLE
    planner = ContextPlanner(EvidenceStore(tmp_path))
    first = planner.plan(request)
    second = planner.plan(request)
    assert "docs-unrelated.md" not in first.deferred
    assert first.stable_prefix == second.stable_prefix
    assert first.stable_prefix_digest == second.stable_prefix_digest
    small = ManagedRequest("openai", "m", "tiny", [])
    plan = planner.plan(small)
    assert plan.fallback and plan.initial_tokens == plan.baseline_tokens


def test_planner_omits_recoverable_and_virtualizes_tools(tmp_path: Path) -> None:
    request = managed_request()
    request.tools = [
        ToolDefinition(f"tool_{number}", "realistic tool", {"type": "object"}, "database")
        for number in range(60)
    ]
    request.tools[3].required = True
    plan = ContextPlanner(EvidenceStore(tmp_path)).plan(request)
    assert plan.initial_tokens < plan.baseline_tokens
    assert plan.deferred == ("docs-unrelated.md",)
    assert plan.selected_tools == ("tool_3",)
    assert len(plan.deferred_tools) == 59 and "tool.discover" in plan.retrieval_operations


def test_retrieval_exact_ranges_search_cache_and_secrets(tmp_path: Path) -> None:
    request = managed_request()
    request.context[2] = Context(
        "log", BlockKind.COMMAND_OUTPUT, "one\nERROR bad\nthree", Retention.RECOVERABLE
    )
    planner = ContextPlanner(EvidenceStore(tmp_path))
    plan = planner.plan(request)
    service = RetrievalService(planner.evidence, request, plan)
    ranged = service.execute("log.range", {"id": "log", "start": 2, "end": 2})
    assert ranged.content == "ERROR bad" and ranged.source == "managed:log"
    searched = service.execute("log.search", {"id": "log", "pattern": "ERROR"})
    assert searched.content == "ERROR bad"
    assert service.execute("log.search", {"id": "log", "pattern": "ERROR"}).cached
    with pytest.raises(RetrievalError, match="unsafe"):
        service.execute("log.search", {"id": "log", "pattern": "(a+)+$", "regex": True})
    request.context[2].source_path = ".env"
    plan = planner.plan(request)
    with pytest.raises(RetrievalError, match="secret"):
        RetrievalService(planner.evidence, request, plan).execute("log.range", {"id": "log"})


def test_retrieval_source_symbol_dependency_repository_and_tool(tmp_path: Path) -> None:
    request = ManagedRequest(
        "openai",
        "m",
        "generic task",
        [
            Context(
                "source",
                BlockKind.SOURCE,
                "def target():\n    return 1\n\ndef other():\n    return 2\n",
                Retention.RECOVERABLE,
                dependencies=("config",),
            ),
            Context(
                "config",
                BlockKind.CONFIGURATION,
                "TIMEOUT=30",
                Retention.RECOVERABLE,
            ),
            Context(
                "map",
                BlockKind.REPOSITORY_MAP,
                "src/a.py -> config",
                Retention.RECOVERABLE,
            ),
        ],
        [ToolDefinition("deferred_tool", "Deferred", {"type": "object"})]
        + [ToolDefinition(f"other_{i}", "Other", {"type": "object"}) for i in range(8)],
    )
    planner = ContextPlanner(EvidenceStore(tmp_path))
    plan = planner.plan(request)
    service = RetrievalService(planner.evidence, request, plan)
    assert (
        "def target" in service.execute("symbol.get", {"id": "source", "symbol": "target"}).content
    )
    assert (
        service.execute("dependency.get", {"id": "source", "dependency": "config"}).content
        == "TIMEOUT=30"
    )
    assert service.execute("repository.map", {"id": "map"}).content == "src/a.py -> config"
    discovered = service.execute("tool.discover", {"name": "deferred_tool"})
    assert json.loads(discovered.content)["name"] == "deferred_tool"
    with pytest.raises(RetrievalError, match="declared"):
        service.execute("dependency.get", {"id": "source", "dependency": "map"})
    with pytest.raises(RetrievalError, match="invalid"):
        service.execute("source.range", {"id": "source", "start": 3, "end": 2})
    with pytest.raises(RetrievalError, match="not found"):
        service.execute("symbol.get", {"id": "source", "symbol": "missing"})
    with pytest.raises(RetrievalError, match="unavailable"):
        service.execute("tool.discover", {"name": "missing"})
    with pytest.raises(RetrievalError, match="not available"):
        service.execute("log.search", {"id": "source", "pattern": "x"})


@pytest.mark.asyncio
async def test_runtime_retrieval_continuation_and_accounting(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    async def provider(_: str, body: dict[str, Any], __: float) -> dict[str, Any]:
        captured.append(body)
        if len(captured) == 1:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "evidence.get",
                                        "arguments": '{"id":"docs-unrelated.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 4},
            }
        return {
            "choices": [{"message": {"role": "assistant", "content": '{"complete":true}'}}],
            "usage": {
                "prompt_tokens": 140,
                "completion_tokens": 6,
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        }

    result = await ManagedRuntime(EvidenceStore(tmp_path), provider).run(managed_request())
    assert result.status == "completed" and result.turns == 2
    assert result.usage.initial_input_tokens == 100
    assert result.usage.continuation_input_tokens == 140
    assert result.usage.total_input_tokens == 240
    assert result.usage.retrieval_result_tokens > 0 and len(result.retrievals) == 1
    assert "digest" not in json.dumps(captured[0])
    assert "unrelated evidence" not in json.dumps(captured[0])
    assert "unrelated evidence" in json.dumps(captured[1])
    assert captured[0]["model"] == captured[1]["model"] == "same-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,response",
    [
        (
            "openai",
            {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            },
        ),
        (
            "anthropic",
            {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 20, "output_tokens": 2},
            },
        ),
        (
            "gemini",
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 2},
            },
        ),
    ],
)
async def test_runtime_existing_provider_transports_exclude_internal_fields(
    tmp_path: Path, provider: str, response: dict[str, Any]
) -> None:
    bodies: list[dict[str, Any]] = []

    async def call(_: str, body: dict[str, Any], __: float) -> dict[str, Any]:
        bodies.append(body)
        return response

    request = managed_request()
    request.provider = provider
    result = await ManagedRuntime(EvidenceStore(tmp_path), call).run(request)
    assert result.output == "ok"
    rendered = json.dumps(bodies[0])
    for internal in ("digest", "reference", "retention", "confidence", "source_path"):
        assert internal not in rendered
    assert request.model in rendered and "unrelated evidence" not in rendered


@pytest.mark.asyncio
async def test_runtime_repeated_retrieval_turn_limit_and_cancel(tmp_path: Path) -> None:
    async def repeated(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c",
                                "function": {
                                    "name": "evidence.get",
                                    "arguments": {"id": "docs-unrelated.md"},
                                },
                            }
                        ]
                    }
                }
            ]
        }

    runtime = ManagedRuntime(EvidenceStore(tmp_path), repeated)
    with pytest.raises(ExecutionError, match="repeated"):
        await runtime.run(managed_request())
    cancelled = asyncio.Event()
    cancelled.set()
    with pytest.raises(asyncio.CancelledError):
        await runtime.run(managed_request(), cancellation=cancelled)
    request = managed_request()
    request.execution.max_turns = 1
    with pytest.raises(ExecutionError, match="turn limit"):
        await runtime.run(request)


@pytest.mark.asyncio
async def test_runtime_provider_errors_malformed_calls_and_bounds(tmp_path: Path) -> None:
    async def failed(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        raise OSError("network detail must not escape")

    with pytest.raises(ExecutionError, match="provider request failed"):
        await ManagedRuntime(EvidenceStore(tmp_path / "failed"), failed).run(managed_request())

    async def malformed(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call",
                                "function": {"name": "evidence.get", "arguments": "{"},
                            }
                        ]
                    }
                }
            ]
        }

    with pytest.raises(ExecutionError, match="malformed"):
        await ManagedRuntime(EvidenceStore(tmp_path / "malformed"), malformed).run(
            managed_request()
        )

    async def slow(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        await asyncio.sleep(0.05)
        return {}

    timed = managed_request()
    timed.execution.timeout_seconds = 0.001
    with pytest.raises(ExecutionError, match="timed out"):
        await ManagedRuntime(EvidenceStore(tmp_path / "timeout"), slow).run(timed)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
async def test_runtime_native_retrieval_shapes(tmp_path: Path, provider: str) -> None:
    bodies: list[dict[str, Any]] = []

    async def call(_: str, body: dict[str, Any], __: float) -> dict[str, Any]:
        bodies.append(body)
        if len(bodies) == 1 and provider == "anthropic":
            return {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-a",
                        "name": "evidence.get",
                        "input": {"id": "docs-unrelated.md"},
                    }
                ]
            }
        if len(bodies) == 1:
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "functionCall": {
                                        "id": "call-g",
                                        "name": "evidence.get",
                                        "args": {"id": "docs-unrelated.md"},
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        if provider == "anthropic":
            return {"content": [{"type": "text", "text": "done"}]}
        return {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}

    request = managed_request()
    request.provider = provider
    result = await ManagedRuntime(EvidenceStore(tmp_path), call).run(request)
    assert result.output == "done"
    rendered = json.dumps(bodies[1])
    assert "unrelated evidence" in rendered
    assert "tool_result" in rendered if provider == "anthropic" else "functionResponse" in rendered


def test_sync_async_clients_and_dry_run(tmp_path: Path) -> None:
    runtime = ManagedRuntime(EvidenceStore(tmp_path))
    sync = Client(runtime).run(managed_request(), dry_run=True)
    async_result = asyncio.run(AsyncClient(runtime).run(managed_request(), dry_run=True))
    assert sync.status == async_result.status == "planned"


def test_clients_from_config_transport_and_safe_repr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = tmp_path / ".llmcut"
    state.mkdir()
    (state / "config.toml").write_text(
        '[provider.mock]\nkind="openai"\nbase_url="https://mock.invalid/v1"\n'
        'credential_env="MOCK_API_KEY"\n'
    )
    monkeypatch.setenv("MOCK_API_KEY", "never-represent-this")
    captured: list[dict[str, Any]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2},
            }

    class HttpClient:
        def __init__(self, **_: Any) -> None:
            pass

        async def __aenter__(self) -> HttpClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def post(self, _: str, **kwargs: Any) -> Response:
            captured.append(kwargs)
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", HttpClient)
    client = Client.from_config(tmp_path)
    result = client.run(managed_request(), timeout=5)
    assert result.output == "ok"
    assert captured[0]["headers"]["authorization"] == "Bearer never-represent-this"
    assert "never-represent-this" not in repr(client)
    async_client = AsyncClient.from_config(tmp_path)
    planned = asyncio.run(async_client.run(managed_request(), mode="parity", dry_run=True))
    assert planned.status == "planned"


@pytest.mark.asyncio
async def test_sync_client_rejects_running_loop(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="AsyncClient"):
        Client(ManagedRuntime(EvidenceStore(tmp_path))).run(managed_request(), dry_run=True)


def test_counter_registry_priority_cache_and_fallback() -> None:
    calls = 0

    def endpoint(_: dict[str, Any]) -> int:
        nonlocal calls
        calls += 1
        return 12

    registry = CounterRegistry()
    registry.register_endpoint("openai", endpoint)
    assert registry.count_transport("openai", "m", {"x": 1}).value == 12
    assert registry.count_transport("openai", "m", {"x": 1}).value == 12 and calls == 1
    assert registry.count_transport("unknown", "m", {"x": 1}).quality.value == "estimated"

    class FixedCounter:
        def count(self, text: str, model: str | None = None) -> Any:
            from llmcut.model import CountQuality, TokenCount

            return TokenCount(9, CountQuality.TOKENIZER_DERIVED, "fixed")

    official = CounterRegistry()
    official.register_official_tokenizer("openai", "m", FixedCounter())
    assert official.count_transport("openai", "m", {}).value == 9
    compatible = CounterRegistry()
    compatible.register_compatible("other", FixedCounter())
    assert compatible.count_transport("other", "m", {}).value == 9
    invalid = CounterRegistry()
    invalid.register_endpoint("openai", lambda _: -1)
    assert invalid.count_transport("openai", "m", {}).quality.value == "estimated"


def test_checkpoint_compaction_verifies_and_saves(tmp_path: Path) -> None:
    store = EvidenceStore(tmp_path)
    reference = store.put("original evidence", "test")
    checkpoint = Checkpoint(
        "finish task",
        decisions=["keep API"],
        evidence=[reference.digest],
        repository_revision="abc",
    )
    blocks = [ContextBlock("history", BlockKind.USER, "long history " * 100, "conversation")]
    compacted, selected = compact_history(blocks, checkpoint, store, repository_revision="abc")
    assert selected and compacted[-1].kind is BlockKind.CHECKPOINT
    assert reference.digest not in compacted[-1].content
    with pytest.raises(IntegrityError, match="stale"):
        compact_history(blocks, checkpoint, store, repository_revision="other")
    incomplete = Checkpoint("goal", repository_revision=None)
    with pytest.raises(IntegrityError, match="required"):
        compact_history(blocks, incomplete, store)
    open_call = ContextBlock(
        "call",
        BlockKind.TOOL_CALL,
        "call",
        "history",
        metadata={"tool_call_id": "open"},
    )
    with pytest.raises(IntegrityError, match="unresolved"):
        compact_history([open_call], checkpoint, store)


def test_tool_registry_and_release_targets() -> None:
    registry = ToolRegistry((ToolDefinition("git_status", "status", {}, "git"),))
    assert registry.categories() == ("git",)
    assert registry.discover("status")[0].name == "git_status"
    with pytest.raises(RetrievalError):
        registry.load("missing")
    results = [
        ManagedEvaluation(str(i), 100, 50, 5, 10, 65, 2, 0, 1, 2, True, None) for i in range(4)
    ]
    targets = release_targets(results)
    assert targets["positive_saving_cases"] == 4
    assert not targets["passed"]  # v0.4 requires at least five eligible release tasks.
    with pytest.raises(ValueError, match="unique"):
        ToolRegistry(
            (
                ToolDefinition("same", "one", {}),
                ToolDefinition("same", "two", {}),
            )
        )


def test_managed_metrics_are_prompt_free(tmp_path: Path) -> None:
    metrics = MetricsStore(EvidenceStore(tmp_path).db)
    identifier = metrics.record_managed(
        {
            "integration_mode": "managed",
            "optimization_mode": "extreme",
            "provider": "openai",
            "model": "m",
            "baseline_tokens": 100,
            "initial_tokens": 40,
            "retrieval_request_tokens": 2,
            "retrieval_result_tokens": 3,
            "continuation_tokens": 10,
            "total_effective_tokens": 50,
            "output_tokens": 4,
            "reasoning_tokens": 1,
            "cached_tokens": 0,
            "count_quality": "estimated",
            "planning_seconds": 0.1,
            "provider_seconds": 0.2,
            "retrieval_count": 1,
            "fallback": 0,
            "quality_state": "passed",
            "completed": 1,
        }
    )
    assert len(identifier) == 32
    summary = metrics.summary()
    assert summary["median_total_reduction"] == 50
    assert summary["managed_completion_rate"] == 100


def test_execution_settings_bounds() -> None:
    request = managed_request()
    request.execution = ExecutionSettings(IntegrationMode.MANAGED, OptimizationMode.EXTREME, 0)
    with pytest.raises(ProtocolError, match="max_turns"):
        request.validate()
    request.execution = ExecutionSettings(timeout_seconds=-1)
    with pytest.raises(ProtocolError, match="bounds"):
        request.validate()


def test_protocol_helpers_json_and_malformed_shapes() -> None:
    request = ManagedRequest(
        "openai",
        "m",
        "task",
        [Context.test("tests/test_x.py", "def test_x(): pass")],
    )
    assert (
        ManagedRequest.from_dict(json.loads(request.to_json())).model_configuration().model == "m"
    )
    with pytest.raises(ProtocolError, match="object"):
        ManagedRequest.from_dict([])  # type: ignore[arg-type]
    value = request.to_dict()
    value["context"] = "bad"
    with pytest.raises(ProtocolError, match="list"):
        ManagedRequest.from_dict(value)
    value = request.to_dict()
    value["tools"] = [{"name": "x"}]
    with pytest.raises(ProtocolError, match="tool definition"):
        ManagedRequest.from_dict(value)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda value: value.update(provider=""), "required"),
        (lambda value: value["context"][0].update(priority=101), "priority"),
        (
            lambda value: value["context"][0].update(kind="system", retention="recoverable"),
            "critical",
        ),
        (
            lambda value: value["context"].append(
                {"id": "call", "kind": "tool_call", "content": "x"}
            ),
            "unique tool_call_id",
        ),
        (
            lambda value: value.update(
                tools=[
                    {"name": "same", "description": "a", "input_schema": {}},
                    {"name": "same", "description": "b", "input_schema": {}},
                ]
            ),
            "duplicate tool",
        ),
        (lambda value: value.update(execution=[]), "execution"),
    ],
)
def test_additional_protocol_validation(change: Any, match: str) -> None:
    value = ManagedRequest(
        "openai", "m", "task", [Context.test("tests/test_x.py", "pass")]
    ).to_dict()
    change(value)
    with pytest.raises(ProtocolError, match=match):
        ManagedRequest.from_dict(value)


@pytest.mark.asyncio
async def test_runtime_miscellaneous_safety_bounds(tmp_path: Path) -> None:
    request = managed_request()
    runtime = ManagedRuntime(EvidenceStore(tmp_path / "none"))
    assert (await runtime.plan(request)).request.model.model == "same-model"
    with pytest.raises(ExecutionError, match="transport"):
        await runtime.run(request)

    unknown = managed_request()
    unknown.provider = "unknown"

    async def response(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {}

    with pytest.raises(ExecutionError, match="unsupported"):
        await ManagedRuntime(EvidenceStore(tmp_path / "unknown"), response).run(unknown)

    async def retrieve(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "c",
                                "function": {
                                    "name": "evidence.get",
                                    "arguments": {"id": "docs-unrelated.md"},
                                },
                            }
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 100},
        }

    volume = managed_request()
    volume.execution.max_retrieval_bytes = 1
    with pytest.raises(ExecutionError, match="volume bound"):
        await ManagedRuntime(EvidenceStore(tmp_path / "volume"), retrieve).run(volume)

    token_bound = managed_request()
    token_bound.execution.max_total_tokens = 1
    with pytest.raises(ExecutionError, match="token bound"):
        await ManagedRuntime(EvidenceStore(tmp_path / "tokens"), retrieve).run(token_bound)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "gemini"])
async def test_managed_evaluation_provider_outputs(tmp_path: Path, provider: str) -> None:
    from llmcut.eval.managed import evaluate_managed

    if provider == "anthropic":
        responses = [
            {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 80}},
            {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 40}},
        ]
    else:
        responses = [
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 80},
            },
            {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 40},
            },
        ]

    async def call(_: str, __: dict[str, Any], ___: float) -> dict[str, Any]:
        return responses.pop(0)

    request = managed_request()
    request.provider = provider
    result = await evaluate_managed(
        "provider-output",
        request,
        ManagedRuntime(EvidenceStore(tmp_path), call),
        call,
        expected_output="ok",
        required_facts=("ok",),
    )
    assert result.quality_passed and result.saving
