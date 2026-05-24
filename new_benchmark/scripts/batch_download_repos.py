"""Batch download multiple repos at multiple versions.

For each repo, fetch the list of release tags, pick 4-6 versions,
and download tarballs. Only download if repo size < 300MB.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "github_versions"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Repos sorted by size (small first), skip anything > 300MB
REPOS = [
    # < 50MB - core set
    ("openai/openai-python", "Python"),
    ("psf/requests", "Python"),
    ("pallets/flask", "Python"),       # already done
    ("docker/compose", "Go"),
    ("fastapi/fastapi", "Python"),
    ("qdrant/qdrant", "Rust"),
    ("celery/celery", "Python"),
    ("ollama/ollama", "Go"),
    ("mem0ai/mem0", "Python"),
    ("minio/minio", "Go"),
    ("microsoft/semantic-kernel", "C#"),
    ("Lightning-AI/pytorch-lightning", "Python"),
    ("astral-sh/uv", "Rust"),
    ("prisma/prisma", "TypeScript"),
    ("microsoft/playwright", "TypeScript"),
    # 50-200MB - supplement
    ("wandb/wandb", "Python"),
    ("vllm-project/vllm", "Python"),
    ("PrefectHQ/prefect", "Python"),
    ("scikit-learn/scikit-learn", "Python"),
    ("grafana/grafana", None),          # skip, too big
]

# Actually download these (skip already-done flask and too-big ones)
TARGET_REPOS = [
    ("openai/openai-python", "Python"),
    ("psf/requests", "Python"),
    ("fastapi/fastapi", "Python"),
    ("celery/celery", "Python"),
    ("ollama/ollama", "Go"),
    ("mem0ai/mem0", "Python"),
    ("minio/minio", "Go"),
    ("Lightning-AI/pytorch-lightning", "Python"),
    ("astral-sh/uv", "Rust"),
    ("wandb/wandb", "Python"),
    ("vllm-project/vllm", "Python"),
    ("PrefectHQ/prefect", "Python"),
    ("scikit-learn/scikit-learn", "Python"),
]

MIN_VERSIONS = 4
MAX_VERSIONS = 6


def get_tags(repo: str, count: int = 20) -> list[str]:
    """Fetch recent release tags from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/tags?per_page={count}"
    req = urllib.request.Request(url, headers={"User-Agent": "Python"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            tags = json.loads(resp.read())
            return [t["name"] for t in tags]
    except Exception as e:
        print(f"  Failed to fetch tags for {repo}: {e}")
        return []


def select_versions(tags: list[str]) -> list[str]:
    """Pick 4-6 versions spread across the tag history."""
    if len(tags) <= MAX_VERSIONS:
        return tags[:MAX_VERSIONS]

    # Pick evenly spaced: first, some in middle, last
    selected = [tags[0]]  # newest
    step = max(1, (len(tags) - 1) // (MIN_VERSIONS - 1))
    for i in range(step, len(tags) - 1, step):
        if len(selected) < MAX_VERSIONS - 1:
            selected.append(tags[i])
    selected.append(tags[-1])  # oldest

    # Reverse to chronological order (oldest first)
    selected.reverse()
    return selected


def download_version(repo: str, tag: str, output_dir: Path) -> bool:
    """Download a single version as tarball and extract."""
    repo_slug = repo.replace("/", "_")
    tarball_path = output_dir / f"{tag}.tar.gz"
    extract_dir = output_dir / tag

    if extract_dir.exists() and any(extract_dir.iterdir()):
        print(f"    {tag}: already exists, skipping")
        return True

    # Try GitHub archive API
    url = f"https://api.github.com/repos/{repo}/tarball/refs/tags/{tag}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Python"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
            if len(data) < 1000:
                print(f"    {tag}: response too small ({len(data)} bytes), trying alternate URL")
                # Try without refs/tags/
                url2 = f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"
                req2 = urllib.request.Request(url2, headers={"User-Agent": "Python"})
                with urllib.request.urlopen(req2, timeout=60) as resp2:
                    data = resp2.read()

            tarball_path.write_bytes(data)

        # Extract
        extract_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["tar", "xzf", str(tarball_path), "--strip-components=1", "-C", str(extract_dir)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            print(f"    {tag}: extract failed: {result.stderr[:100]}")
            return False

        size_mb = sum(f.stat().st_size for f in extract_dir.rglob("*") if f.is_file()) / 1024 / 1024
        print(f"    {tag}: {size_mb:.1f} MB extracted")
        return True

    except Exception as e:
        print(f"    {tag}: download failed: {e}")
        return False


def main():
    print(f"Will download {len(TARGET_REPOS)} repos to {DATA_DIR}\n")

    results = []
    for repo, lang in TARGET_REPOS:
        repo_slug = repo.replace("/", "_")
        repo_dir = DATA_DIR / repo_slug
        repo_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  {repo} ({lang})")
        print(f"{'='*60}")

        # Get tags
        tags = get_tags(repo, count=30)
        if not tags:
            print(f"  No tags found, skipping")
            continue

        print(f"  Found {len(tags)} tags, latest: {tags[0]}")

        # Select versions
        versions = select_versions(tags)
        print(f"  Selected versions: {versions}")

        # Download each
        downloaded = []
        for ver in versions:
            ok = download_version(repo, ver, repo_dir)
            if ok:
                downloaded.append(ver)
            time.sleep(0.5)  # rate limit

        results.append({
            "repo": repo,
            "language": lang,
            "versions_downloaded": downloaded,
        })

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['repo']}: {len(r['versions_downloaded'])} versions - {r['versions_downloaded']}")

    # Save manifest
    manifest_path = DATA_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
