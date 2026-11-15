from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "源码"
ARCHIVE = OUT_DIR / "industrial-evolution-school-initial-source.zip"

INCLUDE_DIRS = ("src", "configs", "tests", "tools", "docs")
INCLUDE_FILES = (
    ".dockerignore",
    ".env.example",
    ".gitignore",
    "Dockerfile",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "requirements.lock",
    "requirements-extras.txt",
    "run.sh",
    "运行演示.bat",
)
EXCLUDED_PARTS = {
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "node_modules",
    "runs",
    "datasets",
    "official",
    "submission",
}
ALLOWED_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".html", ".txt", ".toml", ".lock", ".sh", ".bat"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    files: set[Path] = set()
    for relative in INCLUDE_FILES:
        path = ROOT / relative
        if path.is_file() and path.name.lower() != ".env":
            files.add(path)
    for dirname in INCLUDE_DIRS:
        directory = ROOT / dirname
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_SUFFIXES:
                continue
            if any(part in EXCLUDED_PARTS for part in path.relative_to(ROOT).parts):
                continue
            if path.name.lower() == ".env" or "secret" in path.name.lower():
                continue
            files.add(path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(files):
            bundle.write(path, Path("industrial-evolution") / path.relative_to(ROOT))

    print(f"archive={ARCHIVE}")
    print(f"files={len(files)}")
    print(f"size_bytes={ARCHIVE.stat().st_size}")
    print(f"sha256={sha256(ARCHIVE)}")


if __name__ == "__main__":
    main()
