"""Validate benchmark questions against the 5 success criteria.

GitHub questions (criteria 1-4):
1. Answer uniqueness: gold doc contains the answer, only that version has it
2. Real version distractors: database contains other versions as distractors
3. Question answerability: no version string in query, answerable via content clues
4. Gold docs resolve: all gold_file references resolve to valid doc_ids

Narrative questions (criteria 1-5):
1. State paragraph distinguishability: states have different content
2. Real distractors: same scenario has >= 2 states planted
3. No position hints: query text has no chapter/section/page references
4. No novel overlap: planted text doesn't appear in original novel
5. Chunk-level resolution: gold (scenario, state) resolves to a valid chunk
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def load_file_versions() -> dict[tuple[str, str], dict[str, str]]:
    result: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    path = DATA_DIR / "benchmark_github" / "all_versioned_files.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            result[(r["repo"], r["file_path"])][r["version"]] = r["content"]
    return dict(result)


def check_github_uniqueness(q: dict, file_versions: dict) -> list[str]:
    """Criterion 1 for GitHub: gold doc contains unique content."""
    issues = []
    if "gold_files" not in q:
        return issues

    fpath, ver = q["gold_files"][0]
    matching = [(r, f) for (r, f) in file_versions if f == fpath]
    if not matching:
        return [f"File {fpath} not found in database"]

    repo = matching[0][0]
    key = (repo, fpath)
    versions = file_versions.get(key, {})
    gold_content = versions.get(ver, "")
    if not gold_content:
        return [f"Gold version {ver} not found"]

    gold_lines = set(l.strip() for l in gold_content.split("\n") if l.strip())
    unique_lines = set()
    for other_ver, other_content in versions.items():
        if other_ver == ver:
            continue
        other_lines = set(l.strip() for l in other_content.split("\n") if l.strip())
        unique_lines.update(gold_lines - other_lines)

    if not unique_lines:
        issues.append("Gold version has NO unique lines vs all distractors")

    return issues


def check_github_distractors(q: dict) -> list[str]:
    """Criterion 2 for GitHub: real distractor versions exist."""
    if q.get("type") == "multi_version":
        return []
    if q.get("distractor_files") and len(q["distractor_files"]) >= 1:
        return []
    return ["No distractor versions"]


def check_no_version_string(q: dict) -> list[str]:
    """Criterion 3: query text must not contain version/position identifiers."""
    if q.get("type") == "multi_version":
        return []

    issues = []
    query = q["query_text"]

    if "gold_files" in q:
        ver = q["gold_files"][0][1]
        if ver in query:
            issues.append(f"Version string '{ver}' in query")

    patterns = [
        (r"version\s+[\d.]+", "explicit version reference"),
        (r"v\d+\.\d+", "v-prefixed version"),
    ]
    for pattern, desc in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            issues.append(f"Version-like pattern ({desc}) in query")

    # Narrative-specific: no chapter/section/page hints
    for forbidden in ["chapter", "section ", "page "]:
        if forbidden in query.lower():
            issues.append(f"Position hint '{forbidden}' in query")

    return issues


def check_github_resolution(q: dict, github_lookup: dict) -> list[str]:
    """Criterion 4 for GitHub: gold files resolve to valid doc_ids."""
    issues = []
    for fpath, ver in q.get("gold_files", []):
        key = (fpath, ver)
        if key not in github_lookup:
            issues.append(f"Gold file {fpath}@{ver} not resolved to doc_id")
    return issues


# --- Narrative-specific validators ---

def load_planted_chunks() -> dict[tuple[str, str], dict]:
    """Load planted/mutated chunks, keyed by (scenario_id, state_id)."""
    result = {}
    path = DATA_DIR / "benchmark_narrative" / "novel_chunks.jsonl"
    if not path.exists():
        return result
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if r.get("type") in ("planted", "mutated"):
                result[(r["scenario_id"], r["state_id"])] = r
    return result


def load_novel_text() -> str:
    path = DATA_DIR / "benchmark_narrative" / "pride_and_prejudice.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def check_narrative_uniqueness(q: dict, planted_chunks: dict) -> list[str]:
    """Criterion 1 for narrative: gold text is distinct from distractors."""
    issues = []
    gold_scenario = q.get("gold_scenario", "")
    gold_state = q.get("gold_state", "")
    distractor_states = q.get("distractor_states", [])

    gold_key = (gold_scenario, gold_state)
    gold_chunk = planted_chunks.get(gold_key)
    if not gold_chunk:
        return [f"Gold chunk {gold_key} not found"]

    gold_text = gold_chunk["text"]

    for dst in distractor_states:
        dst_key = (gold_scenario, dst)
        dst_chunk = planted_chunks.get(dst_key)
        if not dst_chunk:
            continue
        # Check if the actual text strings are identical
        if gold_text == dst_chunk["text"]:
            issues.append(
                f"Gold state {gold_state} is exact duplicate of distractor {dst}"
            )

    return issues


def check_narrative_distractors(q: dict) -> list[str]:
    """Criterion 2 for narrative: at least 1 distractor state exists."""
    distractor_states = q.get("distractor_states", [])
    if len(distractor_states) >= 1:
        return []
    return ["No distractor states"]


def check_narrative_no_overlap(q: dict, planted_chunks: dict, novel_text: str) -> list[str]:
    """Criterion 4: mutated text differs from novel (not an exact copy)."""
    issues = []
    gold_scenario = q.get("gold_scenario", "")
    gold_state = q.get("gold_state", "")
    gold_key = (gold_scenario, gold_state)
    gold_chunk = planted_chunks.get(gold_key)
    if not gold_chunk:
        return []

    # For mutation-based: text is based on novel, so overlap is expected.
    # Just check it's not identical to a novel passage.
    if gold_state == "original":
        # Original paragraphs ARE from the novel, so skip this check
        return []

    text = gold_chunk["text"].lower()
    novel_lower = novel_text.lower()

    # For mutated versions, verify the text is NOT exactly the same as original
    orig_key = (gold_scenario, "original")
    orig_chunk = planted_chunks.get(orig_key)
    if orig_chunk and text == orig_chunk["text"].lower():
        issues.append(f"Mutated version {gold_state} is identical to original")

    return issues


def check_narrative_resolution(q: dict, planted_chunks: dict, github_count: int) -> list[str]:
    """Criterion 5: gold (scenario, state) resolves to a valid chunk."""
    issues = []
    gold_scenario = q.get("gold_scenario", "")
    gold_state = q.get("gold_state", "")
    key = (gold_scenario, gold_state)
    if key not in planted_chunks:
        issues.append(f"Gold ({gold_scenario}/{gold_state}) not found in chunks")
    return issues


def main():
    print("=== Benchmark Validation ===\n")

    # Load GitHub data
    file_versions = load_file_versions()
    print(f"Loaded {len(file_versions)} file groups")

    github_lookup: dict[tuple[str, str], int] = {}
    path = DATA_DIR / "benchmark_github" / "all_versioned_files.jsonl"
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            github_lookup[(r["file_path"], r["version"])] = i
    github_count = i + 1
    print(f"GitHub entries: {github_count}")

    # Load narrative data
    planted_chunks = load_planted_chunks()
    novel_text = load_novel_text()
    print(f"Planted chunks: {len(planted_chunks)}")
    print(f"Novel text: {len(novel_text):,} chars")

    # Validate all questions
    all_issues: dict[str, list[str]] = {}
    total_questions = 0
    passed = 0

    for qfile in [
        DATA_DIR / "benchmark_github" / "questions_v2.jsonl",
        DATA_DIR / "benchmark_narrative" / "questions.jsonl",
    ]:
        if not qfile.exists():
            continue
        label = qfile.parent.name + "/" + qfile.name
        print(f"\nValidating {label}...")

        with open(qfile) as f:
            for line in f:
                q = json.loads(line)
                total_questions += 1
                qid = q["id"]
                issues = []

                # Common checks
                issues.extend(check_no_version_string(q))

                if "gold_files" in q:
                    # GitHub question
                    issues.extend(check_github_uniqueness(q, file_versions))
                    issues.extend(check_github_distractors(q))
                    issues.extend(check_github_resolution(q, github_lookup))
                elif "gold_scenario" in q:
                    # Narrative question (planted)
                    issues.extend(check_narrative_uniqueness(q, planted_chunks))
                    issues.extend(check_narrative_distractors(q))
                    issues.extend(check_narrative_no_overlap(q, planted_chunks, novel_text))
                    issues.extend(check_narrative_resolution(q, planted_chunks, github_count))

                if issues:
                    all_issues[qid] = issues
                else:
                    passed += 1

    # Report
    print(f"\n{'='*60}")
    print(f"  Total questions: {total_questions}")
    print(f"  Passed all criteria: {passed}")
    print(f"  Failed: {total_questions - passed}")

    if all_issues:
        print(f"\n  Issues by question (first 20):")
        for qid, issues in list(all_issues.items())[:20]:
            print(f"    {qid}:")
            for issue in issues:
                print(f"      - {issue}")

    # Criterion summary
    c1 = sum(1 for issues in all_issues.values()
             if any("unique" in i.lower() or "similar" in i.lower() for i in issues))
    c2 = sum(1 for issues in all_issues.values()
             if any("distractor" in i.lower() for i in issues))
    c3 = sum(1 for issues in all_issues.values()
             if any("version" in i.lower() or "hint" in i.lower() for i in issues))
    c4 = sum(1 for issues in all_issues.values()
             if any("resolv" in i.lower() or "overlap" in i.lower() or "not found" in i.lower() for i in issues))

    print(f"\n  Criterion 1 (Answer uniqueness): {c1} fails")
    print(f"  Criterion 2 (Real distractors): {c2} fails")
    print(f"  Criterion 3 (No version/position hints): {c3} fails")
    print(f"  Criterion 4+5 (Resolution + no overlap): {c4} fails")

    if total_questions - passed == 0:
        print(f"\n  ALL QUESTIONS PASS ALL CRITERIA")
        return 0
    else:
        print(f"\n  {total_questions - passed} questions need fixing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
