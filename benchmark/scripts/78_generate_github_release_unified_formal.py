#!/usr/bin/env python3
"""Generate a large unified GitHub release-note formal dataset with incremental persistence.

Pipeline:
1. discover broad, influential repositories from GitHub search across many domains/languages;
2. fetch recent usable release notes incrementally and persist per repository;
3. for repositories with at least N usable releases, build one unified prototype over the latest window;
4. write every repository result immediately so the run can resume after interruption.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import importlib.util
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "benchmark"))

from src.state_version.github_collection import (  # noqa: E402
    GithubRestClient,
    clip_text,
    iso_day,
    normalize_text,
    parse_datetime,
    should_skip_repo,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "benchmark"
    / "data"
    / "prototypes"
    / "github_release_note_v2"
    / "formal_300repo_unified_v1"
)
DEFAULT_RAW_RELEASE_DIR = ROOT / "benchmark" / "data" / "raw" / "github_release_notes_formal_v1"
DEFAULT_CONFIG = ROOT / "benchmark" / "configs" / "state_version_build_config.yaml"
DEFAULT_CACHE_DIR = ROOT / "benchmark" / "data" / "cache" / "github_release_note_unified_formal_v1"
UNIFIED_SCRIPT = ROOT / "benchmark" / "scripts" / "77_build_github_release_note_unified_prototype.py"


@dataclass(frozen=True)
class DiscoveryQuery:
    label: str
    query: str
    per_query_limit: int = 40


DISCOVERY_QUERIES: list[DiscoveryQuery] = [
    DiscoveryQuery("python", "language:Python stars:>8000 archived:false fork:false"),
    DiscoveryQuery("javascript", "language:JavaScript stars:>15000 archived:false fork:false"),
    DiscoveryQuery("typescript", "language:TypeScript stars:>10000 archived:false fork:false"),
    DiscoveryQuery("go", "language:Go stars:>8000 archived:false fork:false"),
    DiscoveryQuery("rust", "language:Rust stars:>8000 archived:false fork:false"),
    DiscoveryQuery("java", "language:Java stars:>10000 archived:false fork:false"),
    DiscoveryQuery("cpp", "language:C++ stars:>10000 archived:false fork:false"),
    DiscoveryQuery("c", "language:C stars:>8000 archived:false fork:false"),
    DiscoveryQuery("php", "language:PHP stars:>10000 archived:false fork:false"),
    DiscoveryQuery("ruby", "language:Ruby stars:>8000 archived:false fork:false"),
    DiscoveryQuery("kotlin", "language:Kotlin stars:>5000 archived:false fork:false"),
    DiscoveryQuery("swift", "language:Swift stars:>5000 archived:false fork:false"),
    DiscoveryQuery("web-framework", "topic:web-framework stars:>3000 archived:false fork:false"),
    DiscoveryQuery("database", "topic:database stars:>3000 archived:false fork:false"),
    DiscoveryQuery("observability", "topic:observability stars:>2000 archived:false fork:false"),
    DiscoveryQuery("monitoring", "topic:monitoring stars:>2000 archived:false fork:false"),
    DiscoveryQuery("kubernetes", "topic:kubernetes stars:>3000 archived:false fork:false"),
    DiscoveryQuery("devops", "topic:devops stars:>2000 archived:false fork:false"),
    DiscoveryQuery("security", "topic:security stars:>3000 archived:false fork:false"),
    DiscoveryQuery("testing", "topic:testing stars:>2000 archived:false fork:false"),
    DiscoveryQuery("cli", "topic:cli stars:>2000 archived:false fork:false"),
    DiscoveryQuery("api", "topic:api stars:>3000 archived:false fork:false"),
    DiscoveryQuery("browser", "topic:browser stars:>3000 archived:false fork:false"),
    DiscoveryQuery("mobile", "topic:mobile stars:>3000 archived:false fork:false"),
    DiscoveryQuery("devtools", "topic:developer-tools stars:>2000 archived:false fork:false"),
    DiscoveryQuery("static-site", "topic:static-site-generator stars:>2000 archived:false fork:false"),
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify_repo(repo: str) -> str:
    return repo.replace("/", "__").replace("-", "_").replace(".", "_")


def load_unified_module() -> Any:
    spec = importlib.util.spec_from_file_location("release_unified_builder_v77", UNIFIED_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load unified builder module from {UNIFIED_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_release_raw_text(repo_full_name: str, release: dict[str, Any]) -> str:
    return clip_text(
        "\n".join(
            [
                f"Repository: {repo_full_name}",
                f"Release: {release.get('name') or release.get('tag_name')}",
                f"Tag: {release.get('tag_name')}; draft: {release.get('draft')}; prerelease: {release.get('prerelease')}; published_at: {release.get('published_at')}; created_at: {release.get('created_at')}",
                f"Release notes:\n{normalize_text(release.get('body') or '')}",
            ]
        )
    )


def repo_search(client: GithubRestClient, query: str, *, limit: int) -> list[dict[str, Any]]:
    per_page = min(100, limit)
    max_pages = max(1, (limit + per_page - 1) // per_page)
    items: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get_json(
            "/search/repositories",
            params={
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            },
        )
        page_items = list((payload or {}).get("items") or [])
        items.extend(page_items)
        if len(page_items) < per_page or len(items) >= limit:
            break
    return items[:limit]


def discover_candidates(
    client: GithubRestClient,
    *,
    queries: list[DiscoveryQuery],
    discovery_cache_dir: Path,
    max_candidates: int,
) -> list[dict[str, Any]]:
    discovery_cache_dir.mkdir(parents=True, exist_ok=True)
    dedup: dict[str, dict[str, Any]] = {}
    for item in queries:
        cache_name = hashlib.sha256(item.query.encode("utf-8")).hexdigest()[:16]
        cache_path = discovery_cache_dir / f"{cache_name}.json"
        if cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            payload = {
                "label": item.label,
                "query": item.query,
                "items": repo_search(client, item.query, limit=item.per_query_limit),
            }
            write_json(cache_path, payload)
        for repo in payload.get("items") or []:
            full_name = str(repo.get("full_name") or "").strip()
            if not full_name:
                continue
            if should_skip_repo(repo):
                continue
            record = dedup.get(full_name)
            candidate = {
                "repo": full_name,
                "html_url": repo.get("html_url"),
                "description": repo.get("description"),
                "stargazers_count": repo.get("stargazers_count"),
                "language": repo.get("language"),
                "topics": repo.get("topics") or [],
                "source_labels": [item.label],
            }
            if record is None:
                dedup[full_name] = candidate
            else:
                labels = set(record.get("source_labels") or [])
                labels.add(item.label)
                record["source_labels"] = sorted(labels)
                record["stargazers_count"] = max(int(record.get("stargazers_count") or 0), int(repo.get("stargazers_count") or 0))
    candidates = list(dedup.values())
    candidates.sort(
        key=lambda row: (
            -(len(row.get("source_labels") or [])),
            -(int(row.get("stargazers_count") or 0)),
            str(row.get("repo") or ""),
        )
    )
    return candidates[:max_candidates]


def fetch_recent_usable_releases(
    client: GithubRestClient,
    repo: str,
    *,
    min_usable_releases: int,
    max_pages: int,
) -> list[dict[str, Any]]:
    owner, repo_name = repo.split("/", 1)
    usable: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        payload = client.get_json(
            f"/repos/{owner}/{repo_name}/releases",
            params={"per_page": 100, "page": page},
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected releases payload for {repo}: {type(payload).__name__}")
        if not payload:
            break
        for release in payload:
            all_rows.append(release)
            if bool(release.get("draft")):
                continue
            if not normalize_text(release.get("body") or ""):
                continue
            usable.append(release)
        if len(usable) >= min_usable_releases and len(payload) < 100:
            break
        if len(usable) >= min_usable_releases and page >= 1:
            # We only need a recent usable window, not the full release history.
            break
        if len(payload) < 100:
            break
    return all_rows


def convert_local_release_rows(repo: str, releases: list[dict[str, Any]], chunks_per_window: int) -> list[dict[str, Any]]:
    rows = []
    for release in releases:
        body = normalize_text(release.get("body") or "")
        if not body or bool(release.get("draft")):
            continue
        row = {
            "repo": repo,
            "artifact_type": "release_note",
            "artifact_ref": f"release:{release.get('tag_name') or release.get('id')}",
            "tag_name": release.get("tag_name"),
            "title": str(release.get("name") or release.get("tag_name") or "").strip() or None,
            "published_at": release.get("published_at"),
            "created_at": release.get("created_at"),
            "time_hint": iso_day(release.get("published_at") or release.get("created_at")),
            "source_url": release.get("html_url"),
            "raw_text": build_release_raw_text(repo, release),
        }
        rows.append(row)
    rows.sort(key=lambda row: (parse_datetime(row.get("published_at") or row.get("created_at") or row.get("time_hint")), str(row.get("tag_name") or "")))
    return rows[-chunks_per_window:]


def make_spec(repo: str) -> Any:
    module = load_unified_module()
    prototype_id = slugify_repo(repo) + "_release_window"
    return module.ReleaseRepoSpec(repo=repo, prototype_id=prototype_id, window_title=f"{repo} recent release evolution")


def build_one_unified_prototype(
    *,
    module: Any,
    llm: Any,
    repo: str,
    window_rows: list[dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any]:
    spec = module.ReleaseRepoSpec(
        repo=repo,
        prototype_id=slugify_repo(repo) + "_release_window",
        window_title=f"{repo} recent release evolution",
    )
    for row in window_rows:
        if "memory_node_id" not in row:
            row["memory_node_id"] = module.make_memory_node_id(row)
    prompt = module.build_unified_release_window_prompt(spec, window_rows)
    payload = module.llm_cached_json(
        llm,
        prompt,
        cache_dir=cache_dir / "unified_window",
        key_payload={"repo": repo, "window_rows": window_rows},
        kind="unified_release_window",
    )
    return module.normalize_unified_output(payload, spec, window_rows)


def process_repo_candidate(
    *,
    candidate: dict[str, Any],
    config_path: Path,
    raw_release_root: Path,
    out_dir_root: Path,
    cache_dir: Path,
    chunks_per_window: int,
    min_usable_releases: int,
    max_release_pages: int,
    token: str | None,
    api_base_url: str,
    timeout: float,
) -> dict[str, Any]:
    repo = str(candidate["repo"])
    prototype_id = slugify_repo(repo) + "_release_window"
    raw_repo_dir = raw_release_root / repo.replace("/", "__")
    raw_json_path = raw_repo_dir / "releases.json"
    raw_jsonl_path = raw_repo_dir / "releases.jsonl"

    client = GithubRestClient(token=token, api_base_url=api_base_url, timeout=timeout)
    module = load_unified_module()
    llm = module.create_llm(config_path)

    if raw_json_path.exists():
        releases = json.loads(raw_json_path.read_text(encoding="utf-8"))
    else:
        releases = fetch_recent_usable_releases(
            client,
            repo,
            min_usable_releases=min_usable_releases,
            max_pages=max_release_pages,
        )
        raw_repo_dir.mkdir(parents=True, exist_ok=True)
        write_json(raw_json_path, releases)
        write_jsonl(
            raw_jsonl_path,
            [
                {
                    "repo": repo,
                    "id": release.get("id"),
                    "tag_name": release.get("tag_name"),
                    "name": release.get("name"),
                    "draft": release.get("draft"),
                    "prerelease": release.get("prerelease"),
                    "created_at": release.get("created_at"),
                    "published_at": release.get("published_at"),
                    "html_url": release.get("html_url"),
                    "body": release.get("body"),
                }
                for release in releases
            ],
        )

    window_rows = convert_local_release_rows(repo, releases, chunks_per_window)
    if len(window_rows) < min_usable_releases:
        return {
            "status": "skipped",
            "repo": repo,
            "prototype_id": prototype_id,
            "reason": f"only {len(window_rows)} usable releases",
        }

    normalized = build_one_unified_prototype(
        module=module,
        llm=llm,
        repo=repo,
        window_rows=window_rows,
        cache_dir=cache_dir,
    )

    out_dir = out_dir_root / prototype_id
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "prototype.json", normalized)
    (out_dir / "README.md").write_text(
        (
            f"# {normalized['window_title']}\n\n"
            f"- repo: `{normalized['repo']}`\n"
            f"- source_type: `{normalized['source_type']}`\n"
            f"- total_chunks: `{len(normalized['chunks'])}`\n"
            f"- question_count: `{len(normalized['questions'])}`\n"
        ),
        encoding="utf-8",
    )

    return {
        "status": "ok",
        "repo": repo,
        "prototype_id": prototype_id,
        "index_row": {
            "prototype_id": normalized["prototype_id"],
            "repo": normalized["repo"],
            "source_type": normalized["source_type"],
            "window_title": normalized["window_title"],
            "total_chunks": len(normalized["chunks"]),
            "question_count": len(normalized["questions"]),
            "candidate_labels": candidate.get("source_labels") or [],
            "stargazers_count": candidate.get("stargazers_count"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a large unified formal GitHub release-note dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--raw-release-root", type=Path, default=DEFAULT_RAW_RELEASE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--target-repos", type=int, default=300)
    parser.add_argument("--chunks-per-window", type=int, default=30)
    parser.add_argument("--min-usable-releases", type=int, default=30)
    parser.add_argument("--max-discovery-candidates", type=int, default=900)
    parser.add_argument("--max-release-pages", type=int, default=2)
    parser.add_argument("--token", default=None, help="GitHub token. Falls back to GITHUB_TOKEN or GH_TOKEN.")
    parser.add_argument("--api-base-url", default="https://api.github.com")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--repo", default=None, help="Optional single repo full name to build.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    client = GithubRestClient(token=token, api_base_url=args.api_base_url, timeout=args.timeout)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.raw_release_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    status_path = args.out_dir / "build_status.jsonl"
    index_path = args.out_dir / "prototype_index.jsonl"
    candidate_path = args.out_dir / "discovered_candidates.jsonl"

    if args.repo:
        candidates = [{"repo": args.repo, "source_labels": ["manual"], "stargazers_count": None}]
    else:
        candidates = discover_candidates(
            client,
            queries=DISCOVERY_QUERIES,
            discovery_cache_dir=args.cache_dir / "discovery",
            max_candidates=args.max_discovery_candidates,
        )
    write_jsonl(candidate_path, candidates)

    existing_index = {row["prototype_id"]: row for row in read_jsonl(index_path)}
    built_rows: dict[str, dict[str, Any]] = dict(existing_index)
    selected_count = len(built_rows)
    pending_candidates = []
    for candidate in candidates:
        repo = str(candidate["repo"])
        prototype_id = slugify_repo(repo) + "_release_window"
        if not args.force_rebuild and prototype_id in built_rows:
            continue
        pending_candidates.append(candidate)

    candidate_iter = iter(pending_candidates)
    futures: dict[Any, dict[str, Any]] = {}

    def submit_one(executor: ThreadPoolExecutor, candidate: dict[str, Any]) -> None:
        repo = str(candidate["repo"])
        prototype_id = slugify_repo(repo) + "_release_window"
        append_jsonl(
            status_path,
            {
                "time": now_iso(),
                "repo": repo,
                "prototype_id": prototype_id,
                "stage": "start_repo",
                "status": "running",
            },
        )
        future = executor.submit(
            process_repo_candidate,
            candidate=candidate,
            config_path=args.config,
            raw_release_root=args.raw_release_root,
            out_dir_root=args.out_dir,
            cache_dir=args.cache_dir,
            chunks_per_window=args.chunks_per_window,
            min_usable_releases=args.min_usable_releases,
            max_release_pages=args.max_release_pages,
            token=token,
            api_base_url=args.api_base_url,
            timeout=args.timeout,
        )
        futures[future] = candidate

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        while len(futures) < max(1, args.workers):
            if not args.force_rebuild and selected_count >= args.target_repos:
                break
            try:
                submit_one(executor, next(candidate_iter))
            except StopIteration:
                break

        while futures:
            done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                candidate = futures.pop(future)
                repo = str(candidate["repo"])
                prototype_id = slugify_repo(repo) + "_release_window"
                try:
                    result = future.result()
                except Exception as exc:
                    append_jsonl(
                        status_path,
                        {
                            "time": now_iso(),
                            "repo": repo,
                            "prototype_id": prototype_id,
                            "stage": "repo_error",
                            "status": "error",
                            "error": str(exc),
                        },
                    )
                    result = None

                if result:
                    if result["status"] == "ok":
                        built_rows[prototype_id] = result["index_row"]
                        write_jsonl(index_path, list(sorted(built_rows.values(), key=lambda row: row["prototype_id"])))
                        selected_count = len(built_rows)
                        append_jsonl(
                            status_path,
                            {
                                "time": now_iso(),
                                "repo": repo,
                                "prototype_id": prototype_id,
                                "stage": "complete_repo",
                                "status": "ok",
                                "selected_count": selected_count,
                            },
                        )
                    elif result["status"] == "skipped":
                        append_jsonl(
                            status_path,
                            {
                                "time": now_iso(),
                                "repo": repo,
                                "prototype_id": prototype_id,
                                "stage": "skip_repo",
                                "status": "skipped",
                                "reason": result["reason"],
                            },
                        )

                if not args.force_rebuild and selected_count >= args.target_repos:
                    continue
                try:
                    submit_one(executor, next(candidate_iter))
                except StopIteration:
                    pass

    print(
        json.dumps(
            {
                "status": "ok",
                "target_repos": args.target_repos,
                "built_repos": len(built_rows),
                "out_dir": str(args.out_dir),
                "raw_release_root": str(args.raw_release_root),
                "status_file": str(status_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
