import shutil
from pathlib import Path

import pytest

from llmcut.integrations.codex.suite import (
    _commands,
    _object,
    _required,
    _strings,
    _within,
    load_suite,
)

SUITE = Path("tests/fixtures/agent/suite.toml")


def test_suite_loads_versioned_settings_and_commands() -> None:
    suite = load_suite(SUITE)
    assert suite.schema_version == "1" and suite.agent == "codex"
    assert suite.repetitions == 3 and suite.order == "random"
    assert suite.tasks[0].validation == (("python", "tests/validate_callback.py"),)
    assert suite.digest.startswith("sha256:")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("schema_version='2'\nagent='codex'\n", "schema_version"),
        ("schema_version='1'\nagent='other'\n", "agent=codex"),
        (
            "schema_version='1'\nagent='codex'\nrepetitions=0\n[execution]\n"
            "model='m'\nreasoning_effort='h'\n[[tasks]]\nid='a'\n",
            "repetitions",
        ),
    ],
)
def test_suite_rejects_invalid_top_level(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "suite.toml"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        load_suite(path)


def _valid_suite(tmp_path: Path, overrides: str = "") -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree("tests/fixtures/agent/repositories/python-timeout", tmp_path / "repository")
    fake = tmp_path / "fake.py"
    fake.write_text("#!/usr/bin/env python3\n")
    path = tmp_path / "suite.toml"
    path.write_text(
        "schema_version='1'\nagent='codex'\nrepetitions=1\norder='baseline-first'\n"
        "timeout_seconds=30\ncodex_executable='fake.py'\n"
        "[execution]\nmodel='same'\nreasoning_effort='high'\n"
        "sandbox='workspace-write'\napproval_policy='never'\n"
        "[[tasks]]\nid='task'\nrepository='repository'\nstarting_ref='HEAD'\n"
        "prompt='fix it'\nvalidation=[['python','validate.py']]\n"
        "allowed_changes=['app/callback.py']\nrequired_files=['app/callback.py']\n" + overrides
    )
    return path


@pytest.mark.parametrize(
    ("old", "replacement", "message"),
    [
        ("order='baseline-first'", "order='bad'", "ordering"),
        ("timeout_seconds=30", "timeout_seconds=0", "timeout_seconds"),
        ("sandbox='workspace-write'", "sandbox='bad'", "sandbox"),
        ("approval_policy='never'", "approval_policy='bad'", "approval"),
        ("prompt='fix it'", "prompt='fix it'\nmax_turns=0", "max_turns"),
        (
            "validation=[['python','validate.py']]",
            "validation='shell command'",
            "argv arrays",
        ),
        (
            "allowed_changes=['app/callback.py']",
            "allowed_changes=['../escape']",
            "unsafe task path",
        ),
    ],
)
def test_suite_rejects_invalid_settings(
    tmp_path: Path, old: str, replacement: str, message: str
) -> None:
    path = _valid_suite(tmp_path)
    path.write_text(path.read_text().replace(old, replacement))
    with pytest.raises(ValueError, match=message):
        load_suite(path)


def test_suite_rejects_missing_repository_duplicate_task_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        load_suite(tmp_path / "missing.toml")
    path = _valid_suite(tmp_path)
    content = path.read_text()
    path.write_text(content + "[[tasks]]\nid='task'\nrepository='repository'\n")
    with pytest.raises(ValueError, match="duplicate task id"):
        load_suite(path)


def test_suite_rejects_missing_tasks_repository_and_executable(tmp_path: Path) -> None:
    path = _valid_suite(tmp_path)
    path.write_text(path.read_text().split("[[tasks]]", 1)[0])
    with pytest.raises(ValueError, match="at least one task"):
        load_suite(path)

    path = _valid_suite(tmp_path / "missing-repo")
    shutil.rmtree(path.parent / "repository")
    with pytest.raises(ValueError, match="repository is unavailable"):
        load_suite(path)

    path = _valid_suite(tmp_path / "missing-executable")
    path.write_text(path.read_text().replace("fake.py", "bin/missing-codex"))
    with pytest.raises(ValueError, match="executable is unavailable"):
        load_suite(path)


def test_suite_helper_bounds_and_types(tmp_path: Path) -> None:
    assert _commands(["python", "-V"]) == (("python", "-V"),)
    with pytest.raises(ValueError, match="non-empty argv"):
        _commands([[]])
    with pytest.raises(ValueError, match="configured bounds"):
        _commands([["x" * 4_097]])
    with pytest.raises(ValueError, match="escapes"):
        _within(tmp_path, "../escape")
    with pytest.raises(ValueError, match="string array"):
        _strings("value", "items")
    with pytest.raises(ValueError, match="table"):
        _object([], "item")
    with pytest.raises(ValueError, match="bounded"):
        _required({}, "name")
