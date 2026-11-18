"""Build the deliverable zip: a clean copy of the project with no dependencies, caches, local databases or secrets.

    python scripts/make_zip.py                      # writes ../pathfinder.zip next to the project folder
    python scripts/make_zip.py --out /tmp/p.zip

It refuses to produce a zip if it finds anything that looks like a real credential, and checks that the files a recipient
needs are present. Pure standard library, so it runs anywhere Python does.
"""

import argparse
import hashlib
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP = "pathfinder"  # folder name inside the zip

EXCLUDE_DIRS = {
    ".venv", "venv", ".pgdata", "node_modules", "dist", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
    ".git", ".claude", ".idea", ".vscode", ".vite", "coverage", ".cache",
}
EXCLUDE_FILES = {".env", ".DS_Store", "Thumbs.db", "EVALS.generated.md"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".zip", ".log"}

REQUIRED = [
    "README.md", "DEPLOY.md", "RESUME_BULLETS.md", "EVALS.md", ".env.example", "docker-compose.yml", "Makefile",
    ".github/workflows/ci.yml", "data/seed_jobs.jsonl",
    "backend/requirements.txt", "backend/requirements-dev.txt", "backend/requirements-llm.txt", "backend/Dockerfile",
    "backend/alembic.ini", "backend/pyproject.toml", "backend/scripts/dev_stack.py",
    "frontend/package.json", "frontend/package-lock.json", "frontend/Dockerfile", "frontend/nginx.conf",
    "docs/screenshots/chat.jpg",
]

# Things that look like real credentials. Placeholders in .env.example (empty values) must NOT match.
SECRET_PATTERNS = {
    "Anthropic API key": re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}"),
    "OpenAI-style API key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{32,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_\-]{35}"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "key assigned in an env-style line": re.compile(r"(?m)^\s*(?:ANTHROPIC|OPENAI|GEMINI)_API_KEY\s*=\s*[A-Za-z0-9_\-]{12,}\s*$"),
}
MAX_SCAN_BYTES = 2_000_000


def is_excluded(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    return (
        any(part in EXCLUDE_DIRS for part in rel.parts[:-1])
        or path.name in EXCLUDE_FILES
        or path.suffix in EXCLUDE_SUFFIXES
    )


def collect_files(root: Path = ROOT) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not is_excluded(p))


def scan_for_secrets(files: list[Path]) -> list[str]:
    findings = []
    for p in files:
        if p.stat().st_size > MAX_SCAN_BYTES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary (images, ...)
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{p.relative_to(ROOT).as_posix()}: looks like a {label}")
    return findings


def missing_required(files: list[Path]) -> list[str]:
    have = {p.relative_to(ROOT).as_posix() for p in files}
    return [r for r in REQUIRED if r not in have]


def build(out: Path, files: list[Path]) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in files:
            z.write(p, f"{TOP}/{p.relative_to(ROOT).as_posix()}")
    return hashlib.sha256(out.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "pathfinder.zip"))
    a = ap.parse_args()
    out = Path(a.out).resolve()

    files = collect_files()
    secrets = scan_for_secrets(files)
    if secrets:
        print("REFUSING to build: possible secrets found:\n  " + "\n  ".join(secrets), file=sys.stderr)
        return 1
    missing = missing_required(files)
    if missing:
        print("REFUSING to build: required files are missing:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1

    digest = build(out, files)
    size_mb = out.stat().st_size / 1_000_000
    print(f"wrote {out}\n  {len(files)} files, {size_mb:.2f} MB\n  sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
