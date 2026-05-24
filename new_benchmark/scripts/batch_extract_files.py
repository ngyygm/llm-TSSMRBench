"""Batch extract versioned files from all downloaded repos.

For each repo in data/github_versions/, find source files in each version,
and collect them into a unified versioned_files.jsonl.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "github_versions"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "benchmark_github"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SKIP_DIRS = {
    "test", "tests", "example", "examples", "docs", "__pycache__",
    ".git", ".github", ".tox", ".nox", "node_modules", "venv",
    "egg-info", "dist-info", ".mypy_cache", ".pytest_cache",
    "site-packages", "dist", "build", "scripts", "benchmarks",
}
SKIP_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dll", ".exe", ".png", ".jpg", ".gif",
    ".ico", ".svg", ".woff", ".woff2", ".ttf", ".eot", ".css", ".map",
    ".lock", ".toml.bak",
}
SOURCE_EXTENSIONS = {
    ".py", ".go", ".rs", ".ts", ".js", ".toml", ".cfg", ".ini",
    ".yaml", ".yml", ".json", ".rst", ".md",
}


def should_include(filepath: str) -> bool:
    parts = Path(filepath).parts
    for skip in SKIP_DIRS:
        if skip in parts:
            return False
    ext = Path(filepath).suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return False
    if ext in SOURCE_EXTENSIONS:
        return True
    return False


def extract_repo(repo_slug: str, repo_dir: Path) -> list[dict]:
    """Extract source files from all versions of a repo."""
    # Derive repo name from slug
    repo_name = repo_slug.replace("_", "/", 1)

    # Find version directories (directories that contain actual files)
    version_dirs = []
    for item in sorted(repo_dir.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            # Check if it has actual content (src/ or top-level files)
            has_src = (item / "src").is_dir()
            has_files = any(f.is_file() for f in item.iterdir() if not f.name.startswith("."))
            if has_src or has_files:
                version_dirs.append((item.name, item))

    if not version_dirs:
        print(f"  {repo_slug}: no version directories found")
        return []

    print(f"  {repo_slug} ({repo_name}): {len(version_dirs)} versions")

    all_records = []
    for ver_name, ver_dir in version_dirs:
        # Find source root: prefer src/ subdirectory
        source_root = ver_dir / "src" if (ver_dir / "src").is_dir() else ver_dir

        records = []
        for root, dirs, files in os.walk(source_root):
            dirs[:] = [d for d in sorted(dirs) if d not in SKIP_DIRS and not d.startswith(".")]
            for fname in sorted(files):
                fpath = Path(root) / fname
                try:
                    rel_path = fpath.relative_to(source_root)
                except ValueError:
                    continue
                if not should_include(str(rel_path)):
                    continue
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if len(content.strip()) < 10:  # skip empty/tiny files
                    continue
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
                records.append({
                    "repo": repo_name,
                    "version": ver_name,
                    "file_path": str(rel_path),
                    "content": content,
                    "content_hash": content_hash,
                    "line_count": content.count("\n") + 1,
                    "char_count": len(content),
                })

        print(f"    {ver_name}: {len(records)} files")
        all_records.extend(records)

    return all_records


def find_changed_files(all_records: list[dict]) -> list[dict]:
    """Find files whose content actually changed across versions, grouped by repo."""
    # Group by (repo, file_path)
    by_file: dict[tuple[str, str], dict[str, str]] = {}
    for r in all_records:
        key = (r["repo"], r["file_path"])
        by_file.setdefault(key, {})[r["version"]] = r["content_hash"]

    changed = []
    for (repo, fpath), versions in sorted(by_file.items()):
        hashes = set(versions.values())
        if len(hashes) > 1:
            changed.append({
                "repo": repo,
                "file_path": fpath,
                "versions": sorted(versions.keys()),
                "num_versions": len(versions),
                "num_distinct_hashes": len(hashes),
            })

    return changed


def main():
    # Find all repo directories
    repo_dirs = sorted([
        d for d in DATA_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    print(f"Found {len(repo_dirs)} repos in {DATA_DIR}\n")

    all_records = []
    for repo_dir in repo_dirs:
        records = extract_repo(repo_dir.name, repo_dir)
        all_records.extend(records)
        print()

    print(f"Total records: {len(all_records)}")

    # Save versioned files
    output_path = OUTPUT_DIR / "all_versioned_files.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved to: {output_path}")

    # Find changed files
    changed = find_changed_files(all_records)
    print(f"\nFiles with changes across versions: {len(changed)}")

    # Group by repo
    by_repo = {}
    for c in changed:
        by_repo.setdefault(c["repo"], []).append(c)

    print("\nChanged files per repo:")
    for repo, files in sorted(by_repo.items()):
        # Filter out test/doc files for summary
        interesting = [f for f in files if not any(
            skip in f["file_path"] for skip in ["test", "doc", "example", "benchmark"]
        )]
        print(f"  {repo}: {len(interesting)} changed source files ({len(files)} total)")
        for f in interesting[:5]:
            print(f"    {f['file_path']}: {f['num_versions']} versions, {f['num_distinct_hashes']} distinct")

    # Save change summary
    summary_path = OUTPUT_DIR / "all_change_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(changed, f, indent=2, ensure_ascii=False)
    print(f"\nChange summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
