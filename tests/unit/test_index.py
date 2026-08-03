import subprocess
from pathlib import Path

from llmcut.index.repository import RepositoryIndex
from llmcut.index.select import pack_repository
from llmcut.store.evidence import EvidenceStore


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    (repo / ".gitignore").write_text(".env\nignored.py\n")
    (repo / "app.py").write_text("import os\nclass OAuth:\n    pass\n")
    (repo / "web.ts").write_text("import { x } from './dep';\nexport function callback() {}\n")
    (repo / "test_app.py").write_text("from app import OAuth\ndef test_oauth(): pass\n")
    (repo / "AGENTS.md").write_text("instructions")
    (repo / ".env").write_text("SECRET=x")
    (repo / "ignored.py").write_text("password='x'")
    (repo / "binary.bin").write_bytes(b"x\0y")
    for name in (".gitignore", "app.py", "web.ts", "test_app.py", "AGENTS.md", "binary.bin"):
        git(repo, "add", name)
    git(
        repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "fixture"
    )
    return repo


def test_git_scope_secrets_binary_symbols_imports_tests_and_changes(tmp_path: Path) -> None:
    repo = fixture_repo(tmp_path)
    (repo / "app.py").write_text("import os\nclass OAuth:\n    pass\ndef timeout(): pass\n")
    records = RepositoryIndex(repo).build()
    paths = {item.path for item in records}
    assert ".env" not in paths and "ignored.py" not in paths
    assert next(x for x in records if x.path == "binary.bin").binary
    app = next(x for x in records if x.path == "app.py")
    assert app.parser == "python-ast-v2" and "OAuth" in app.symbols and "os" in app.imports
    assert app.status != "tracked" and "test_app.py" in app.tests
    web = next(x for x in records if x.path == "web.ts")
    assert web.parser == "tree-sitter-typescript-0.25" and "callback" in web.symbols


def test_untracked_requires_explicit_and_symlink_skipped(tmp_path: Path) -> None:
    repo = fixture_repo(tmp_path)
    (repo / "new.py").write_text("x=1")
    (repo / "outside-link.py").symlink_to(tmp_path / "secret")
    assert "new.py" not in {x.path for x in RepositoryIndex(repo).build()}
    paths = {x.path for x in RepositoryIndex(repo).build(include_untracked=True)}
    assert "new.py" in paths and "outside-link.py" not in paths


def test_pack_is_deterministic_and_recoverable(tmp_path: Path) -> None:
    repo = fixture_repo(tmp_path)
    records = RepositoryIndex(repo).build()
    store = EvidenceStore(repo / ".llmcut")
    first = pack_repository(repo, records, "OAuth callback", store)
    second = pack_repository(repo, records, "OAuth callback", store)
    assert [x.id for x in first] == [x.id for x in second]
    assert all(x.reference and store.get(x.reference.digest) for x in first)


def test_symbol_range_pack_reduces_realistic_irrelevant_module_content(tmp_path: Path) -> None:
    repo = fixture_repo(tmp_path)
    (repo / "app.py").write_text(
        "import os\n\ndef oauth_callback():\n    return 'ok'\n\n"
        + "def unrelated():\n    return 'noise'\n" * 200
    )
    git(repo, "add", "app.py")
    records = RepositoryIndex(repo).build()
    store = EvidenceStore(repo / ".llmcut")
    blocks = pack_repository(repo, records, "Fix oauth_callback", store)
    app = next(item for item in blocks if item.source == "app.py")
    assert "oauth_callback" in app.content
    assert app.metadata["range_selected"] is True
    assert len(app.content) < len((repo / "app.py").read_text()) / 4
    assert app.reference and "unrelated" in store.get(app.reference.digest)


def test_incremental_cache_update_delete_and_rename(tmp_path: Path) -> None:
    repo = fixture_repo(tmp_path)
    first = RepositoryIndex(repo)
    first.build()
    assert first.cache_misses > 0
    second = RepositoryIndex(repo)
    second.build()
    assert second.cache_hits > 0 and second.cache_misses == 0
    (repo / "app.py").write_text("def changed(): pass\n")
    changed = RepositoryIndex(repo)
    changed_records = changed.build()
    assert changed.cache_misses == 1
    assert "changed" in next(item for item in changed_records if item.path == "app.py").symbols
    git(repo, "mv", "web.ts", "renamed.ts")
    renamed = RepositoryIndex(repo)
    paths = {item.path for item in renamed.build()}
    assert "renamed.ts" in paths and "web.ts" not in paths
