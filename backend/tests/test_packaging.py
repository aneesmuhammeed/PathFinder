"""The zip builder is the last line of defence against shipping secrets or junk, so it gets tests."""

import importlib.util
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "make_zip.py"


@pytest.fixture()
def mz(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("make_zip", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ROOT", tmp_path)  # operate on a throwaway project tree
    return mod


def write(root: Path, rel: str, text: str = "x") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_junk_and_local_state_are_excluded_but_source_is_kept(mz, tmp_path):
    for keep in ("README.md", "backend/app/main.py", "frontend/src/App.tsx", ".env.example", "data/seed_jobs.jsonl"):
        write(tmp_path, keep)
    for junk in (".env", "backend/.venv/lib/x.py", "backend/.pgdata/PG_VERSION", "frontend/node_modules/react/index.js",
                 "frontend/dist/index.html", "backend/app/__pycache__/m.pyc", "backend/.pytest_cache/v", ".git/config",
                 ".claude/settings.json", "old.zip", "EVALS.generated.md", "run.log"):
        write(tmp_path, junk)
    names = {p.relative_to(tmp_path).as_posix() for p in mz.collect_files(tmp_path)}
    assert names == {"README.md", "backend/app/main.py", "frontend/src/App.tsx", ".env.example", "data/seed_jobs.jsonl"}


@pytest.mark.parametrize("secret", [
    "sk-ant-api03-" + "A" * 40,
    "sk-" + "b" * 48,
    "AIza" + "C" * 35,
    "ghp_" + "d" * 36,
    "AKIA" + "E" * 16,
    "-----BEGIN " + "RSA PRIVATE KEY-----",  # assembled at runtime so this file doesn't trip the scanner itself
    "ANTHROPIC_API_KEY=" + "f" * 30,
])
def test_real_looking_credentials_are_caught(mz, tmp_path, secret):
    f = write(tmp_path, "backend/notes.md", f"my config:\n{secret}\n")
    findings = mz.scan_for_secrets([f])
    assert findings and "notes.md" in findings[0]


def test_placeholders_and_normal_text_are_not_flagged(mz, tmp_path):
    f = write(tmp_path, ".env.example", "ANTHROPIC_API_KEY=\nOPENAI_API_KEY=\nLLM_MODEL=claude-opus-5-5\n# sk- keys go here\nTASK-1234\n")
    g = write(tmp_path, "docs.md", "Use sk-ant-... style keys; never commit them. AKIA is an AWS prefix.")
    assert mz.scan_for_secrets([f, g]) == []


def test_the_real_project_tree_is_clean_and_complete():
    """Run against the actual repository: no secrets, and every file a recipient needs is present."""
    spec = importlib.util.spec_from_file_location("make_zip_real", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    files = mod.collect_files()
    assert mod.scan_for_secrets(files) == []
    missing = [m for m in mod.missing_required(files) if not m.startswith("docs/screenshots")]  # screenshots are added at packaging time
    assert missing == [], f"required files missing: {missing}"
    assert not any(".venv" in p.parts or "node_modules" in p.parts for p in files)


def test_build_writes_a_zip_with_one_top_folder(mz, tmp_path):
    write(tmp_path, "README.md", "hello")
    write(tmp_path, "backend/app/main.py", "print(1)")
    out = tmp_path.parent / f"{tmp_path.name}-out" / "p.zip"
    digest = mz.build(out, mz.collect_files(tmp_path))
    with zipfile.ZipFile(out) as z:
        assert sorted(z.namelist()) == ["pathfinder/README.md", "pathfinder/backend/app/main.py"]
        assert z.testzip() is None
    assert len(digest) == 64
