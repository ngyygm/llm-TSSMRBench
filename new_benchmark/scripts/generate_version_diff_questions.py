"""Generate benchmark questions that test version differentiation.

Core idea: Find content unique to one version of a file, then ask questions
about that content WITHOUT mentioning the version number. The retrieval system
must distinguish the correct version from highly similar alternatives.

Output: questions_v2.jsonl with format compatible with evaluate.py
"""

import json
import re
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"

# Files to skip: docs, config, tests, generated code
SKIP_PREFIXES = ("CHANGELOG", "CHANGES", "HISTORY", "README", "NEWS", "AUTHORS", "CONTRIBUTING")
SKIP_SUFFIXES = ("_test.py", "_test.go", "_test.rs", ".md", ".rst", ".txt", ".cfg",
                 ".ini", ".toml", ".yaml", ".yml", ".json", ".lock", ".mod")
CODE_EXTENSIONS = {".py", ".go", ".rs", ".js", ".ts", ".tsx", ".jsx"}

# Functions/methods too generic to make good questions
SKIP_FUNC_NAMES = {
    "__str__", "__init__", "__repr__", "__len__", "__eq__", "__hash__",
    "__call__", "__enter__", "__exit__", "__iter__", "__next__",
    "__getitem__", "__setitem__", "__delitem__", "__contains__",
    "__bool__", "__int__", "__float__", "__str__",
    "main", "setup", "run", "test",
}


def is_source_code(fpath: str) -> bool:
    fname = fpath.split("/")[-1]
    if any(fname.startswith(p) for p in SKIP_PREFIXES):
        return False
    if any(fname.endswith(s) for s in SKIP_SUFFIXES):
        return False
    if "/test" in fpath or "/tests/" in fpath or "/__tests__/" in fpath:
        return False
    if "/docs/" in fpath or "/doc/" in fpath:
        return False
    ext = fpath.rsplit(".", 1)[-1] if "." in fpath else ""
    return f".{ext}" in CODE_EXTENSIONS


def load_file_versions() -> dict[tuple[str, str], dict[str, str]]:
    """Return {(repo, file_path): {version: content}}."""
    result: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    path = DATA_DIR / "benchmark_github" / "all_versioned_files.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[(r["repo"], r["file_path"])][r["version"]] = r["content"]
    return dict(result)


def find_unique_lines_per_version(
    versions: dict[str, str],
) -> dict[str, list[tuple[int, str]]]:
    """For each version, find lines that don't appear in ANY other version."""
    line_to_vers: dict[str, set[str]] = defaultdict(set)
    version_lines: dict[str, list[str]] = {}

    for ver, content in versions.items():
        lines = content.split("\n")
        version_lines[ver] = lines
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            line_to_vers[stripped].add(ver)

    unique: dict[str, list[tuple[int, str]]] = {}
    for ver in versions:
        lines = version_lines[ver]
        unique[ver] = []
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if line_to_vers.get(stripped, set()) == {ver}:
                unique[ver].append((i + 1, stripped))
    return unique


def group_contiguous_lines(
    lines: list[tuple[int, str]], gap: int = 3
) -> list[list[tuple[int, str]]]:
    """Group lines into contiguous blocks (allowing small gaps)."""
    if not lines:
        return []
    groups = [[lines[0]]]
    for line in lines[1:]:
        if line[0] - groups[-1][-1][0] <= gap + 1:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def extract_features(block: list[tuple[int, str]]) -> dict[str, Any]:
    """Extract semantic features from a block of unique lines."""
    features: dict[str, Any] = {
        "functions": [],
        "classes": [],
        "imports": [],
        "assignments": [],
        "other": [],
    }
    for lineno, text in block:
        if text.startswith("def "):
            m = re.match(r"def\s+(\w+)\s*\((.*?)\)", text)
            if m:
                features["functions"].append(
                    {"name": m.group(1), "params": m.group(2), "line": lineno}
                )
        elif text.startswith("class "):
            m = re.match(r"class\s+(\w+)", text)
            if m:
                features["classes"].append({"name": m.group(1), "line": lineno})
        elif text.startswith("from ") or text.startswith("import "):
            features["imports"].append({"text": text, "line": lineno})
        elif "=" in text and not text.startswith(("if ", "for ", "while ")):
            m = re.match(r"(\w+)\s*=", text)
            if m:
                features["assignments"].append(
                    {"name": m.group(1), "value": text, "line": lineno}
                )
        else:
            features["other"].append({"text": text, "line": lineno})
    return features


def make_question_id(repo: str, fpath: str, version: str, block_idx: int) -> str:
    repo_slug = repo.replace("/", "_").replace("-", "_")
    file_slug = fpath.replace("/", "_").replace(".", "_")
    ver_slug = version.replace(".", "_").replace("-", "_")
    return f"q_vd_{repo_slug}_{file_slug}_{ver_slug}_b{block_idx}"


def generate_questions_for_file(
    repo: str, fpath: str, versions: dict[str, str]
) -> list[dict]:
    """Generate version-differentiation questions for one file."""
    hashes = set()
    for content in versions.values():
        hashes.add(hashlib.md5(content.encode()).hexdigest()[:12])
    if len(hashes) < 2:
        return []

    unique = find_unique_lines_per_version(versions)
    ver_list = sorted(versions.keys())
    repo_name = repo.split("/")[-1]
    file_name = fpath.split("/")[-1]

    questions = []
    for ver in ver_list:
        blocks = group_contiguous_lines(unique[ver], gap=3)
        for block_idx, block in enumerate(blocks):
            if len(block) < 3:
                continue  # Skip trivial diffs

            features = extract_features(block)
            q_text = None
            difficulty = "medium"

            # Build file context (without version number)
            file_ctx = f" in {file_name}" if file_name != "__init__.py" else ""

            if features["functions"]:
                fn = features["functions"][0]
                if fn["name"] in SKIP_FUNC_NAMES:
                    # Try next function if available
                    fn = next(
                        (f for f in features["functions"][1:]
                         if f["name"] not in SKIP_FUNC_NAMES),
                        None,
                    )
                if fn:
                    if fn["params"]:
                        q_text = (
                            f"In the {repo_name} library{file_ctx}, what "
                            f"parameters does the `{fn['name']}` function accept?"
                        )
                        difficulty = "low"
                    else:
                        q_text = (
                            f"What does the `{fn['name']}` function do in "
                            f"the {repo_name} library{file_ctx}?"
                        )

            elif features["classes"]:
                cls = features["classes"][0]
                q_text = (
                    f"What base classes or methods does the `{cls['name']}` "
                    f"class define in the {repo_name} library{file_ctx}?"
                )

            elif features["imports"]:
                imp = features["imports"][0]
                q_text = (
                    f"What external dependencies does the {repo_name} "
                    f"library import{file_ctx}?"
                )
                difficulty = "low"

            elif features["assignments"]:
                assign = features["assignments"][0]
                q_text = (
                    f"What is the value of `{assign['name']}` in the "
                    f"{repo_name} library{file_ctx}?"
                )

            else:
                # Find a meaningful keyword from the block
                keywords = []
                for _, text in block:
                    for m in re.finditer(r"`?(\w{4,})`?", text):
                        w = m.group(1)
                        if w.lower() not in {"self", "none", "true", "false", "return"}:
                            keywords.append(w)
                if keywords:
                    keyword = keywords[0]
                    q_text = (
                        f"How is `{keyword}` used in the {repo_name} "
                        f"library{file_ctx}?"
                    )
                else:
                    q_text = (
                        f"What implementation details changed in "
                        f"{file_name} of the {repo_name} library?"
                    )
                difficulty = "high"

            if not q_text:
                continue

            distractor_versions = [v for v in ver_list if v != ver]

            q = {
                "id": make_question_id(repo, fpath, ver, block_idx),
                "type": "single_version",
                "difficulty": difficulty,
                "query_text": q_text,
                "gold_files": [[fpath, ver]],
                "distractor_files": [[fpath, v] for v in distractor_versions],
                "gold_state_summary": " | ".join(
                    text[:100] for _, text in block[:3]
                ),
                "unique_content_preview": [text[:80] for _, text in block[:5]],
                "source_repo": repo,
                "source_file": fpath,
                "gold_version": ver,
                "n_unique_lines": len(block),
                "sc_id": f"sc_vd_{repo.replace('/', '_')}_{hashlib.md5(fpath.encode()).hexdigest()[:8]}",
                "dynamic_top_k": min(len(ver_list), 5),
            }
            questions.append(q)

    return questions


def validate_question(q: dict, file_versions: dict) -> list[str]:
    """Check against the 4 benchmark criteria. Returns violations."""
    violations = []

    # Criterion 3: No version string in query
    ver = q["gold_version"]
    if ver in q["query_text"]:
        violations.append(f"Version string '{ver}' appears in query text")

    for pattern in [r"version\s+\d", r"v\d+\.\d+", r"\d+\.\d+\.\d+"]:
        if re.search(pattern, q["query_text"]):
            violations.append(f"Version-like pattern in query: {pattern}")
            break

    # Criterion 2: Real distractors exist
    if not q["distractor_files"]:
        violations.append("No distractor versions")

    # Criterion 1: Gold has unique content
    repo = q["source_repo"]
    fpath = q["source_file"]
    key = (repo, fpath)
    if key in file_versions:
        versions = file_versions[key]
        gold_content = versions.get(ver, "")
        gold_lines = set(
            l.strip() for l in gold_content.split("\n") if l.strip()
        )
        found_unique = False
        for other_ver, other_content in versions.items():
            if other_ver == ver:
                continue
            other_lines = set(
                l.strip() for l in other_content.split("\n") if l.strip()
            )
            if gold_lines - other_lines:
                found_unique = True
                break
        if not found_unique:
            violations.append("Gold has no unique content")

    return violations


def select_best_questions(questions: list[dict], max_per_repo: int = 15) -> list[dict]:
    """Select best questions: prefer larger unique blocks, deduplicate."""
    by_repo = defaultdict(list)
    for q in questions:
        by_repo[q["source_repo"]].append(q)

    selected = []
    for repo, qs in by_repo.items():
        # Deduplicate by query_text
        seen_queries: set[str] = set()
        unique_qs = []
        for q in qs:
            if q["query_text"] not in seen_queries:
                seen_queries.add(q["query_text"])
                unique_qs.append(q)

        unique_qs.sort(key=lambda x: -x["n_unique_lines"])
        selected.extend(unique_qs[:max_per_repo])
    return selected


def main():
    print("Loading file versions...")
    file_versions = load_file_versions()
    print(f"  Found {len(file_versions)} unique (repo, file) pairs")

    multi_ver = {k: v for k, v in file_versions.items() if len(v) >= 2}
    print(f"  Multi-version files: {len(multi_ver)}")

    # Filter to source code files only
    code_multi_ver = {
        k: v for k, v in multi_ver.items() if is_source_code(k[1])
    }
    print(f"  Multi-version source code files: {len(code_multi_ver)}")

    all_questions = []
    for (repo, fpath), versions in code_multi_ver.items():
        qs = generate_questions_for_file(repo, fpath, versions)
        all_questions.extend(qs)

    print(f"\n  Raw questions generated: {len(all_questions)}")

    selected = select_best_questions(all_questions, max_per_repo=15)
    print(f"  After selection: {len(selected)}")

    valid = []
    invalid_count = 0
    for q in selected:
        violations = validate_question(q, file_versions)
        if not violations:
            valid.append(q)
        else:
            invalid_count += 1

    print(f"  Valid: {len(valid)}, Invalid: {invalid_count}")

    by_repo = defaultdict(int)
    by_diff = defaultdict(int)
    for q in valid:
        by_repo[q["source_repo"]] += 1
        by_diff[q["difficulty"]] += 1

    print("\n  By repo:")
    for repo, count in sorted(by_repo.items(), key=lambda x: -x[1]):
        print(f"    {repo}: {count}")
    print("  By difficulty:")
    for diff, count in sorted(by_diff.items()):
        print(f"    {diff}: {count}")

    out_path = DATA_DIR / "benchmark_github" / "questions_v2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for q in valid:
            output_q = {
                k: v
                for k, v in q.items()
                if k
                not in (
                    "unique_content_preview",
                    "source_repo",
                    "source_file",
                    "gold_version",
                    "n_unique_lines",
                )
            }
            f.write(json.dumps(output_q, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(valid)} questions to {out_path}")


if __name__ == "__main__":
    main()
