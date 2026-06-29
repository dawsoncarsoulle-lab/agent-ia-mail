from __future__ import annotations

import argparse
import shutil
from pathlib import Path


OUTPUT_PATHS = (
    Path("data/markdown"),
    Path("data/anonymized"),
    Path("data/reports"),
    Path("data/ocr"),
    Path("pages"),
    Path("extracted"),
    Path("reports"),
    Path("validation"),
)

CACHE_PATTERNS = (
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".uv-cache",
    "build",
    "dist",
    "*.egg-info",
)

IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
}


def is_ignored(path: Path) -> bool:
    return any(part in IGNORED_DIR_NAMES for part in path.parts)


def iter_cache_paths(root: Path) -> list[Path]:
    paths: list[Path] = []

    for pattern in CACHE_PATTERNS:
        paths.extend(path for path in root.rglob(pattern) if not is_ignored(path))

    return sorted(set(paths), key=lambda path: str(path))


def remove_path(path: Path, dry_run: bool) -> bool:
    if not path.exists():
        return False

    print(f"{'Would remove' if dry_run else 'Removing'} {path}")

    if dry_run:
        return True

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove generated outputs and local Python build caches."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List paths that would be removed without deleting anything.",
    )

    args = parser.parse_args()
    root = Path.cwd()
    paths = list(OUTPUT_PATHS) + iter_cache_paths(root)

    removed_count = sum(remove_path(path, args.dry_run) for path in paths)

    if removed_count == 0:
        print("Nothing to clean.")


if __name__ == "__main__":
    main()
