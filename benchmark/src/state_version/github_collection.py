"""GitHub collection helpers for state-version source harvesting."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, ProxyHandler, build_opener

from .schemas import RawGithubArtifactRecord


logger = logging.getLogger(__name__)

API_VERSION = "2022-11-28"
DEFAULT_ACCEPT = "application/vnd.github+json"
DEFAULT_SEED_REPOS = [
    "pytorch/pytorch",
    "huggingface/transformers",
    "vllm-project/vllm",
    "ggerganov/llama.cpp",
    "microsoft/vscode",
    "langchain-ai/langchain",
    "milvus-io/milvus",
    "qdrant/qdrant",
    "chroma-core/chroma",
    "facebookresearch/faiss",
]
REPO_NAME_SKIP_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in [
        r"^awesome[-_]",
        r"free[-_]?programming[-_]?books",
        r"public[-_]?apis",
        r"system[-_]?design[-_]?primer",
        r"developer[-_]?roadmap",
        r"coding[-_]?interview",
        r"leetcode",
    ]
]
FOCUS_EVENT_PREFIX_RE = re.compile(r"^\[[^\]]+\]\s*")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+._/-]{3,}")
REFERENCE_NUMBER_RE = re.compile(r"#(\d+)")
STOPWORDS = {
    "about",
    "after",
    "again",
    "allow",
    "build",
    "change",
    "changes",
    "feature",
    "final",
    "first",
    "from",
    "have",
    "into",
    "issue",
    "later",
    "make",
    "plan",
    "proposal",
    "refactor",
    "request",
    "rewrite",
    "state",
    "support",
    "that",
    "their",
    "then",
    "this",
    "update",
    "with",
}


def normalize_text(value: Any) -> str:
    """Collapse arbitrary text-like content into a stable multiline string."""

    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def clip_text(text: str, max_chars: int = 4000) -> str:
    """Clip long text while keeping it audit-friendly."""

    normalized = normalize_text(text)
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def parse_datetime(value: Optional[str]) -> datetime:
    """Parse a GitHub timestamp into a timezone-aware datetime."""

    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_day(value: Optional[str]) -> Optional[str]:
    """Return YYYY-MM-DD when a timestamp is available."""

    if not value:
        return None
    return value[:10]


def normalize_focus_event(title: str) -> str:
    """Turn a GitHub issue title into a cleaner focus event string."""

    cleaned = FOCUS_EVENT_PREFIX_RE.sub("", normalize_text(title))
    cleaned = re.sub(r"^(RFC|Proposal|Feature Request|Tracking)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned or "untitled GitHub evolution event"


def build_bundle_key(repo: str, issue_number: int) -> str:
    """Build a stable grouping key for all artifacts around one issue-centric event."""

    return f"{repo}::issue#{issue_number}"


def should_skip_repo(repo: dict[str, Any]) -> bool:
    """Apply simple list-repo filters to improve source quality."""

    if bool(repo.get("archived")) or bool(repo.get("disabled")) or bool(repo.get("fork")):
        return True
    name = str(repo.get("name") or "")
    description = str(repo.get("description") or "")
    for pattern in REPO_NAME_SKIP_PATTERNS:
        if pattern.search(name) or pattern.search(description):
            return True
    return False


def build_keyword_set(focus_event: str) -> set[str]:
    """Extract conservative lexical anchors from one focus event."""

    keywords = set()
    for token in TOKEN_RE.findall(focus_event.lower()):
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        keywords.add(token)
    return keywords


def release_matches_event(
    release: dict[str, Any],
    issue_number: int,
    linked_pull_numbers: Iterable[int],
    focus_event: str,
) -> bool:
    """Decide whether a release note is probably about the same evolving event."""

    text = " ".join(
        part
        for part in [
            str(release.get("name") or ""),
            str(release.get("tag_name") or ""),
            str(release.get("body") or ""),
        ]
        if part
    ).lower()
    if not text:
        return False

    if f"#{issue_number}" in text:
        return True
    for pull_number in linked_pull_numbers:
        if f"#{pull_number}" in text:
            return True
    return False


def format_labels(labels: Iterable[dict[str, Any]]) -> str:
    """Format issue labels compactly."""

    names = [str(label.get("name")).strip() for label in labels if str(label.get("name") or "").strip()]
    return ", ".join(names)


def build_issue_raw_text(repo: dict[str, Any], issue: dict[str, Any]) -> str:
    """Render one issue into normalized raw text for later summarization."""

    labels = format_labels(issue.get("labels") or [])
    parts = [
        f"Repository: {repo.get('full_name')}",
        f"Issue #{issue.get('number')}: {issue.get('title')}",
        (
            f"State: {issue.get('state')}; state_reason: {issue.get('state_reason')}; "
            f"created: {issue.get('created_at')}; closed: {issue.get('closed_at')}; "
            f"comments: {issue.get('comments')}"
        ),
    ]
    if labels:
        parts.append(f"Labels: {labels}")
    body = normalize_text(issue.get("body"))
    if body:
        parts.append(f"Body:\n{body}")
    return clip_text("\n".join(parts))


def build_pull_raw_text(repo: dict[str, Any], pull: dict[str, Any]) -> str:
    """Render one pull request into normalized raw text."""

    parts = [
        f"Repository: {repo.get('full_name')}",
        f"Pull request #{pull.get('number')}: {pull.get('title')}",
        (
            f"State: {pull.get('state')}; draft: {pull.get('draft')}; merged_at: {pull.get('merged_at')}; "
            f"created: {pull.get('created_at')}; updated: {pull.get('updated_at')}"
        ),
    ]
    base_ref = ((pull.get("base") or {}).get("ref") or "").strip()
    head_ref = ((pull.get("head") or {}).get("ref") or "").strip()
    if base_ref or head_ref:
        parts.append(f"Branch flow: {head_ref or '?'} -> {base_ref or '?'}")
    stats = []
    for key in ["additions", "deletions", "changed_files", "comments", "review_comments", "commits"]:
        if pull.get(key) is not None:
            stats.append(f"{key}={pull.get(key)}")
    if stats:
        parts.append("Stats: " + ", ".join(stats))
    body = normalize_text(pull.get("body"))
    if body:
        parts.append(f"Body:\n{body}")
    return clip_text("\n".join(parts))


def build_release_raw_text(repo: dict[str, Any], release: dict[str, Any]) -> str:
    """Render one release note into normalized raw text."""

    parts = [
        f"Repository: {repo.get('full_name')}",
        f"Release: {release.get('name') or release.get('tag_name')}",
        (
            f"Tag: {release.get('tag_name')}; draft: {release.get('draft')}; prerelease: {release.get('prerelease')}; "
            f"published_at: {release.get('published_at')}; created_at: {release.get('created_at')}"
        ),
    ]
    body = normalize_text(release.get("body"))
    if body:
        parts.append(f"Release notes:\n{body}")
    return clip_text("\n".join(parts))


def describe_timeline_event(event: dict[str, Any], parent_kind: str, parent_number: int) -> str:
    """Build a short synthetic description for non-comment timeline events."""

    event_type = str(event.get("event") or "").strip()
    if not event_type:
        return ""

    prefix = f"{parent_kind.capitalize()} #{parent_number}"
    if event_type == "renamed":
        rename = event.get("rename") or {}
        return f"{prefix} was renamed from '{rename.get('from')}' to '{rename.get('to')}'."
    if event_type in {"labeled", "unlabeled"}:
        label_name = ((event.get("label") or {}).get("name") or "").strip()
        if label_name:
            return f"{prefix} was {event_type} with label '{label_name}'."
    if event_type in {"milestoned", "demilestoned"}:
        milestone_title = ((event.get("milestone") or {}).get("title") or "").strip()
        if milestone_title:
            return f"{prefix} was {event_type} with milestone '{milestone_title}'."
    if event_type in {"closed", "reopened", "merged", "ready_for_review", "converted_to_draft", "review_requested"}:
        return f"{prefix} changed state via '{event_type}'."
    if event_type == "reviewed":
        state = str(event.get("state") or "").strip()
        body = normalize_text(event.get("body"))
        if body:
            return f"{prefix} received review state '{state or 'unknown'}'.\nReview body:\n{body}"
        return f"{prefix} received review state '{state or 'unknown'}'."
    if event_type == "cross-referenced":
        source_issue = ((event.get("source") or {}).get("issue") or {})
        number = source_issue.get("number")
        title = source_issue.get("title")
        kind = "pull request" if source_issue.get("pull_request") else "issue"
        if number:
            return f"{prefix} was cross-referenced by {kind} #{number}: {title}."
    if event_type == "referenced":
        commit_id = str(event.get("commit_id") or "").strip()
        if commit_id:
            return f"{prefix} was referenced from commit {commit_id[:12]}."
    return ""


def extract_linked_pull_numbers(
    timeline_events: Iterable[dict[str, Any]],
    *,
    expected_repo_full_name: Optional[str] = None,
) -> list[int]:
    """Collect same-repository pull request numbers surfaced by issue timeline cross-references."""

    linked: set[int] = set()
    for event in timeline_events:
        source_issue = ((event.get("source") or {}).get("issue") or {})
        source_repo = ((source_issue.get("repository") or {}).get("full_name") or "")
        if expected_repo_full_name and source_repo and source_repo != expected_repo_full_name:
            continue
        if source_issue.get("pull_request") and source_issue.get("number") is not None:
            linked.add(int(source_issue["number"]))
    return sorted(linked)


def select_issue_candidates(
    issues: Iterable[dict[str, Any]],
    issues_per_repo: int,
    min_issue_comments: int,
    min_issue_body_chars: int,
) -> list[dict[str, Any]]:
    """Choose issue-centric candidate events that are likely to contain rich evolution traces."""

    candidates = []
    for issue in issues:
        if issue.get("pull_request"):
            continue
        if int(issue.get("comments") or 0) < min_issue_comments:
            continue
        body = normalize_text(issue.get("body"))
        if len(body) < min_issue_body_chars:
            continue
        candidates.append(issue)

    candidates.sort(
        key=lambda issue: (
            int(issue.get("comments") or 0),
            parse_datetime(issue.get("updated_at")),
            parse_datetime(issue.get("closed_at")),
        ),
        reverse=True,
    )
    return candidates[:issues_per_repo]


def make_raw_artifact_record(
    *,
    repo: dict[str, Any],
    focus_event: str,
    split: str,
    bundle_key: str,
    artifact_type: str,
    artifact_ref: str,
    title: Optional[str],
    time_hint: Optional[str],
    raw_text: str,
    source_url: Optional[str],
    bundle_summary: Optional[str],
    notes: Optional[list[str]] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> RawGithubArtifactRecord:
    """Build one normalized raw artifact record."""

    return RawGithubArtifactRecord(
        repo=str(repo.get("full_name") or ""),
        focus_event=focus_event,
        artifact_type=artifact_type,
        artifact_ref=artifact_ref,
        title=title,
        time_hint=time_hint,
        raw_text=clip_text(raw_text),
        source_url=source_url,
        split=split,
        bundle_key=bundle_key,
        sample_id=None,
        state_chain_id=None,
        bundle_summary=bundle_summary,
        artifact_order=None,
        notes=notes or [],
        metadata=metadata or {},
    )


def collect_issue_bundle(
    *,
    repo: dict[str, Any],
    issue: dict[str, Any],
    issue_timeline: list[dict[str, Any]],
    linked_pulls: dict[int, dict[str, Any]],
    pull_timelines: dict[int, list[dict[str, Any]]],
    releases: list[dict[str, Any]],
    split: str,
    max_issue_comments: int = 6,
    max_issue_events: int = 6,
    max_pull_comments: int = 4,
    max_pull_events: int = 4,
) -> list[RawGithubArtifactRecord]:
    """Compile one issue-centric bundle into normalized raw artifact records."""

    focus_event = normalize_focus_event(str(issue.get("title") or ""))
    issue_number = int(issue["number"])
    bundle_key = build_bundle_key(str(repo.get("full_name") or ""), issue_number)
    bundle_summary = (
        f"Public GitHub artifacts around issue #{issue_number} in {repo.get('full_name')} track the evolving state of "
        f"{focus_event}."
    )
    metadata_base = {
        "repo_html_url": repo.get("html_url"),
        "repo_stars": repo.get("stargazers_count"),
        "repo_topics": repo.get("topics") or [],
        "focus_issue_number": issue_number,
    }
    common_notes = [
        "Collected automatically from the GitHub REST API.",
        "Review the focus_event wording and bundle boundaries before final QA generation.",
    ]

    staged: list[tuple[datetime, RawGithubArtifactRecord]] = []

    staged.append(
        (
            parse_datetime(issue.get("created_at")),
            make_raw_artifact_record(
                repo=repo,
                focus_event=focus_event,
                split=split,
                bundle_key=bundle_key,
                artifact_type="issue",
                artifact_ref=f"#{issue_number}",
                title=str(issue.get("title") or "").strip() or None,
                time_hint=iso_day(issue.get("created_at")),
                raw_text=build_issue_raw_text(repo, issue),
                source_url=issue.get("html_url"),
                bundle_summary=bundle_summary,
                notes=common_notes,
                metadata=dict(metadata_base),
            ),
        )
    )

    issue_comment_count = 0
    issue_event_count = 0
    for event in issue_timeline:
        event_name = str(event.get("event") or "").strip()
        created_at = parse_datetime(event.get("created_at"))
        if event_name == "commented":
            body = normalize_text(event.get("body"))
            if not body or issue_comment_count >= max_issue_comments:
                continue
            issue_comment_count += 1
            staged.append(
                (
                    created_at,
                    make_raw_artifact_record(
                        repo=repo,
                        focus_event=focus_event,
                        split=split,
                        bundle_key=bundle_key,
                        artifact_type="issue_comment",
                        artifact_ref=f"issue#{issue_number}/comment#{event.get('id')}",
                        title=f"Issue comment on #{issue_number}",
                        time_hint=iso_day(event.get("created_at")),
                        raw_text=f"Comment on issue #{issue_number}.\n{body}",
                        source_url=event.get("html_url") or event.get("url"),
                        bundle_summary=None,
                        notes=[],
                        metadata={"parent_issue_number": issue_number},
                    ),
                )
            )
            continue

        if issue_event_count >= max_issue_events:
            continue
        description = describe_timeline_event(event, "issue", issue_number)
        if not description:
            continue
        issue_event_count += 1
        staged.append(
            (
                created_at,
                make_raw_artifact_record(
                    repo=repo,
                    focus_event=focus_event,
                    split=split,
                    bundle_key=bundle_key,
                    artifact_type="issue_event",
                    artifact_ref=f"issue#{issue_number}/event#{event.get('id')}",
                    title=f"Issue event {event_name} on #{issue_number}",
                    time_hint=iso_day(event.get("created_at")),
                    raw_text=description,
                    source_url=event.get("html_url") or event.get("url") or issue.get("html_url"),
                    bundle_summary=None,
                    notes=[],
                    metadata={"parent_issue_number": issue_number, "event": event_name},
                ),
            )
        )

    for pull_number, pull in linked_pulls.items():
        staged.append(
            (
                parse_datetime(pull.get("created_at")),
                make_raw_artifact_record(
                    repo=repo,
                    focus_event=focus_event,
                    split=split,
                    bundle_key=bundle_key,
                    artifact_type="pull_request",
                    artifact_ref=f"PR#{pull_number}",
                    title=str(pull.get("title") or "").strip() or None,
                    time_hint=iso_day(pull.get("created_at")),
                    raw_text=build_pull_raw_text(repo, pull),
                    source_url=pull.get("html_url"),
                    bundle_summary=None,
                    notes=[],
                    metadata={"linked_pull_number": pull_number, "parent_issue_number": issue_number},
                ),
            )
        )

        pull_comment_count = 0
        pull_event_count = 0
        for event in pull_timelines.get(pull_number, []):
            event_name = str(event.get("event") or "").strip()
            created_at = parse_datetime(event.get("created_at"))
            if event_name == "commented":
                body = normalize_text(event.get("body"))
                if not body or pull_comment_count >= max_pull_comments:
                    continue
                pull_comment_count += 1
                staged.append(
                    (
                        created_at,
                        make_raw_artifact_record(
                            repo=repo,
                            focus_event=focus_event,
                            split=split,
                            bundle_key=bundle_key,
                            artifact_type="pull_request_comment",
                            artifact_ref=f"PR#{pull_number}/comment#{event.get('id')}",
                            title=f"Pull request comment on #{pull_number}",
                            time_hint=iso_day(event.get("created_at")),
                            raw_text=f"Comment on pull request #{pull_number}.\n{body}",
                            source_url=event.get("html_url") or event.get("url"),
                            bundle_summary=None,
                            notes=[],
                            metadata={"linked_pull_number": pull_number, "parent_issue_number": issue_number},
                        ),
                    )
                )
                continue

            if pull_event_count >= max_pull_events:
                continue
            description = describe_timeline_event(event, "pull request", pull_number)
            if not description:
                continue
            pull_event_count += 1
            staged.append(
                (
                    created_at,
                    make_raw_artifact_record(
                        repo=repo,
                        focus_event=focus_event,
                        split=split,
                        bundle_key=bundle_key,
                        artifact_type="pull_request_event",
                        artifact_ref=f"PR#{pull_number}/event#{event.get('id')}",
                        title=f"Pull request event {event_name} on #{pull_number}",
                        time_hint=iso_day(event.get("created_at")),
                        raw_text=description,
                        source_url=event.get("html_url") or event.get("url") or pull.get("html_url"),
                        bundle_summary=None,
                        notes=[],
                        metadata={"linked_pull_number": pull_number, "parent_issue_number": issue_number, "event": event_name},
                    ),
                )
            )

    linked_pull_numbers = sorted(linked_pulls)
    matched_releases = [
        release
        for release in releases
        if release_matches_event(release, issue_number, linked_pull_numbers, focus_event)
    ]
    matched_releases.sort(key=lambda release: parse_datetime(release.get("published_at") or release.get("created_at")))
    for release in matched_releases[:2]:
        release_ref = str(release.get("tag_name") or release.get("id"))
        staged.append(
            (
                parse_datetime(release.get("published_at") or release.get("created_at")),
                make_raw_artifact_record(
                    repo=repo,
                    focus_event=focus_event,
                    split=split,
                    bundle_key=bundle_key,
                    artifact_type="release_note",
                    artifact_ref=f"release:{release_ref}",
                    title=str(release.get("name") or release.get("tag_name") or "").strip() or None,
                    time_hint=iso_day(release.get("published_at") or release.get("created_at")),
                    raw_text=build_release_raw_text(repo, release),
                    source_url=release.get("html_url"),
                    bundle_summary=None,
                    notes=[],
                    metadata={"parent_issue_number": issue_number},
                ),
            )
        )

    staged.sort(key=lambda item: (item[0], item[1].artifact_type, item[1].artifact_ref))
    ordered_records: list[RawGithubArtifactRecord] = []
    for index, (_, record) in enumerate(staged, start=1):
        record.artifact_order = index
        ordered_records.append(record)
    return ordered_records


class GithubRestClient:
    """Very small GitHub REST client using only the Python standard library."""

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        api_base_url: str = "https://api.github.com",
        api_version: str = API_VERSION,
        timeout: float = 60.0,
        user_agent: str = "BiTempQA-state-version-collector/0.1",
    ) -> None:
        self.token = token
        self.api_base_url = api_base_url.rstrip("/")
        self.api_version = api_version
        self.timeout = timeout
        self.user_agent = user_agent
        self._opener = build_opener(ProxyHandler({}))

    def _build_url(self, path: str, params: Optional[dict[str, Any]] = None) -> str:
        url = f"{self.api_base_url}{path}"
        if params:
            encoded = urlencode({key: value for key, value in params.items() if value is not None}, doseq=True)
            if encoded:
                url = f"{url}?{encoded}"
        return url

    def get_json(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        accept: str = DEFAULT_ACCEPT,
    ) -> Any:
        """Fetch one JSON payload from GitHub."""

        headers = {
            "Accept": accept,
            "User-Agent": self.user_agent,
            "X-GitHub-Api-Version": self.api_version,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = Request(self._build_url(path, params=params), headers=headers, method="GET")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                return json.loads(body)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"GitHub API request failed: {exc.code} {exc.reason} for {path}\n{detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"GitHub API request failed for {path}: {exc}") from exc

    def paginate(
        self,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        per_page: int = 100,
        max_pages: int = 1,
        accept: str = DEFAULT_ACCEPT,
    ) -> list[Any]:
        """Collect a paginated list endpoint into one list."""

        results: list[Any] = []
        for page in range(1, max_pages + 1):
            page_params = dict(params or {})
            page_params["per_page"] = per_page
            page_params["page"] = page
            payload = self.get_json(path, params=page_params, accept=accept)
            if not isinstance(payload, list):
                raise RuntimeError(f"Expected list payload from {path}, got {type(payload).__name__}")
            results.extend(payload)
            if len(payload) < per_page:
                break
        return results


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write JSONL deterministically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")
