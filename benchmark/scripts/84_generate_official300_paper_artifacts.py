#!/usr/bin/env python3
"""Build reproducible paper statistics and LaTeX tables for official_300 release-only results."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "benchmark" / "data" / "prototype_eval_results"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "official_300repo_release_unified_v1_paper_artifacts"
DEFAULT_PAPER_TABLE_DIR = ROOT / "paper" / "tables" / "generated"

SYSTEM_SPECS = [
    {
        "paper_name": "BM25",
        "result_dir": RESULTS_ROOT / "official_300repo_release_unified_v1_bm25_globalpool_taskk_v1",
        "questions_file": "bm25.questions.jsonl",
    },
    {
        "paper_name": "FAISS",
        "result_dir": RESULTS_ROOT / "official_300repo_release_unified_v1_faiss_globalpool_taskk_v1",
        "questions_file": "faiss_vector_store.questions.jsonl",
    },
    {
        "paper_name": "Mem0",
        "result_dir": RESULTS_ROOT / "official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10",
        "questions_file": "mem0.questions.jsonl",
    },
    {
        "paper_name": "Graphiti",
        "result_dir": RESULTS_ROOT / "official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1",
        "questions_file": "graphiti.questions.jsonl",
    },
    {
        "paper_name": "Oracle Gold Context",
        "result_dir": RESULTS_ROOT / "official_300repo_release_unified_v1_simple_baselines_conda_taskk",
        "questions_file": "oracle_gold_context.questions.jsonl",
    },
]

TASK_ORDER = [
    "single_state_lookup",
    "cross_version_comparison",
    "temporal_version_ordering",
]
TASK_LABELS = {
    "single_state_lookup": "Single-state lookup",
    "cross_version_comparison": "Cross-version comparison",
    "temporal_version_ordering": "Temporal ordering",
}
TASK_SHORT_LABELS = {
    "single_state_lookup": "Single",
    "cross_version_comparison": "Cross",
    "temporal_version_ordering": "Temporal",
}
TASK_TOP_KS = {
    "single_state_lookup": [1, 2, 3],
    "cross_version_comparison": [2, 5, 8],
    "temporal_version_ordering": [5, 8, 10],
}
PRIMARY_K = {task: max(values) for task, values in TASK_TOP_KS.items()}
ORACLE_SYSTEM_NAME = "Oracle Gold Context"


def canonical_task(task_type: str | None) -> str:
    value = str(task_type or "").strip()
    if value == "temporal_ordering":
        return "temporal_version_ordering"
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def mean_or_zero(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def per_k_record(row: dict[str, Any], k: int) -> dict[str, Any]:
    payload = row.get("per_k_results") or {}
    if str(k) not in payload:
        raise KeyError(f"Question {row.get('question_id')} missing per_k result for k={k}")
    return payload[str(k)]


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "question_count": len(rows),
        "acc": mean_or_zero([1.0 if bool(row.get("is_correct")) else 0.0 for row in rows]),
        "cov": mean_or_zero([safe_float(row.get("support_coverage")) for row in rows]),
        "csr": mean_or_zero([1.0 if bool(row.get("complete_support")) else 0.0 for row in rows]),
        "zero_gold_rate": mean_or_zero(
            [1.0 if math.isclose(safe_float(row.get("support_coverage")), 0.0) else 0.0 for row in rows]
        ),
        "correct_without_gold_support_rate": mean_or_zero(
            [1.0 if bool(row.get("is_correct_without_gold_support")) else 0.0 for row in rows]
        ),
        "retrieved_context_token_count": mean_or_zero(
            [safe_float(row.get("retrieved_context_token_count")) for row in rows]
        ),
    }


def bucket_name(support_coverage: float, complete_support: bool) -> str:
    if math.isclose(support_coverage, 0.0):
        return "zero_recalled"
    if complete_support:
        return "all_recalled"
    return "partial_recalled"


def build_decoupling(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall_counts = {f"{bucket}__{outcome}": 0 for bucket in ("all_recalled", "partial_recalled", "zero_recalled") for outcome in ("correct", "incorrect")}
    by_task: dict[str, dict[str, int]] = {}

    for row in rows:
        task = canonical_task(row.get("task_type"))
        if task not in PRIMARY_K:
            continue
        result = per_k_record(row, PRIMARY_K[task])
        cov = safe_float(result.get("support_coverage"))
        complete = bool(result.get("complete_support"))
        correct = bool(result.get("is_correct"))
        key = f"{bucket_name(cov, complete)}__{'correct' if correct else 'incorrect'}"
        overall_counts[key] += 1
        task_counts = by_task.setdefault(
            task,
            {f"{bucket}__{outcome}": 0 for bucket in ("all_recalled", "partial_recalled", "zero_recalled") for outcome in ("correct", "incorrect")},
        )
        task_counts[key] += 1

    total = len(rows) or 1
    overall_rates = {key: value / total for key, value in overall_counts.items()}
    by_task_payload: dict[str, Any] = {}
    for task, counts in by_task.items():
        task_total = sum(counts.values()) or 1
        by_task_payload[task] = {
            "counts": counts,
            "rates": {key: value / task_total for key, value in counts.items()},
            "question_count": task_total,
        }
    return {
        "counts": overall_counts,
        "rates": overall_rates,
        "question_count": len(rows),
        "by_task": by_task_payload,
    }


def build_case_candidates(rows: list[dict[str, Any]], system_name: str) -> dict[str, Any]:
    success: list[dict[str, Any]] = []
    failure: list[dict[str, Any]] = []

    for row in rows:
        task = canonical_task(row.get("task_type"))
        if task not in PRIMARY_K:
            continue
        primary = per_k_record(row, PRIMARY_K[task])
        matched_non_gold = list(primary.get("matched_non_gold_node_ids") or [])
        payload = {
            "system": system_name,
            "question_id": row.get("question_id"),
            "task_type": task,
            "query_text": row.get("query_text"),
            "expected_answer": row.get("expected_answer"),
            "generated_answer": primary.get("generated_answer"),
            "is_correct": bool(primary.get("is_correct")),
            "support_coverage": safe_float(primary.get("support_coverage")),
            "complete_support": bool(primary.get("complete_support")),
            "distractor_to_gold_ratio": safe_float(primary.get("distractor_to_gold_ratio")),
            "matched_gold_node_ids": list(primary.get("matched_gold_node_ids") or []),
            "matched_non_gold_node_ids": matched_non_gold,
            "gold_rank_positions": list(primary.get("gold_rank_positions") or []),
            "prototype_gold_evidence": row.get("prototype_gold_evidence") or [],
            "retrieved_facts_preview": list(primary.get("retrieved_facts") or [])[:5],
        }
        if (
            task in {"cross_version_comparison", "temporal_version_ordering"}
            and bool(primary.get("is_correct"))
            and bool(primary.get("complete_support"))
        ):
            success.append(payload)
        if (
            task in {"cross_version_comparison", "temporal_version_ordering"}
            and not bool(primary.get("is_correct"))
            and safe_float(primary.get("support_coverage")) > 0.0
        ):
            failure.append(payload)

    success.sort(
        key=lambda item: (
            safe_float(item["distractor_to_gold_ratio"]),
            len(item["matched_non_gold_node_ids"]),
            safe_float(item["support_coverage"]),
        ),
        reverse=True,
    )
    failure.sort(
        key=lambda item: (
            safe_float(item["support_coverage"]),
            safe_float(item["distractor_to_gold_ratio"]),
            -min(item["gold_rank_positions"]) if item["gold_rank_positions"] else 0,
        ),
        reverse=True,
    )
    return {
        "success_candidates": success[:20],
        "failure_candidates": failure[:20],
    }


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def format_latency(value: float) -> str:
    return f"{value:.2f}"


def format_metric_with_style(value: float, style: str) -> str:
    text = format_metric(value)
    if style == "best":
        return f"\\textbf{{{text}}}"
    if style == "second":
        return f"\\underline{{{text}}}"
    return text


def normalized_metrics_for_display(system_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    if system_name != ORACLE_SYSTEM_NAME:
        return metrics
    adjusted = dict(metrics)
    adjusted["cov"] = 1.0
    adjusted["csr"] = 1.0
    return adjusted


def compute_main_table_styles(system_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], str]:
    styles: dict[tuple[str, str, str, str], str] = {}
    automatic_rows = [system for system in system_rows if system["paper_name"] != ORACLE_SYSTEM_NAME]
    for task in TASK_ORDER:
        for k in TASK_TOP_KS[task]:
            for metric_name in ("acc", "cov", "csr"):
                values = [
                    safe_float(
                        normalized_metrics_for_display(
                            system["paper_name"],
                            system["task_topk_metrics"][task][str(k)],
                        )[metric_name]
                    )
                    for system in automatic_rows
                ]
                unique_values = sorted(set(values), reverse=True)
                best_value = unique_values[0] if unique_values else None
                second_value = unique_values[1] if len(unique_values) > 1 else None
                for system in system_rows:
                    if system["paper_name"] == ORACLE_SYSTEM_NAME:
                        styles[(system["paper_name"], task, str(k), metric_name)] = ""
                        continue
                    value = safe_float(
                        normalized_metrics_for_display(
                            system["paper_name"],
                            system["task_topk_metrics"][task][str(k)],
                        )[metric_name]
                    )
                    style = ""
                    if best_value is not None and math.isclose(value, best_value, rel_tol=1e-12, abs_tol=1e-12):
                        style = "best"
                    elif second_value is not None and math.isclose(value, second_value, rel_tol=1e-12, abs_tol=1e-12):
                        style = "second"
                    styles[(system["paper_name"], task, str(k), metric_name)] = style
    return styles


def format_triplet(
    system_name: str,
    task: str,
    k: int,
    metrics: dict[str, Any],
    styles: dict[tuple[str, str, str, str], str],
) -> str:
    metrics = normalized_metrics_for_display(system_name, metrics)
    return (
        f"{format_metric_with_style(metrics['acc'], styles[(system_name, task, str(k), 'acc')])} / "
        f"{format_metric_with_style(metrics['cov'], styles[(system_name, task, str(k), 'cov')])} / "
        f"{format_metric_with_style(metrics['csr'], styles[(system_name, task, str(k), 'csr')])}"
    )


def make_main_table_tex(system_rows: list[dict[str, Any]]) -> str:
    styles = compute_main_table_styles(system_rows)
    lines: list[str] = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(
        "\\caption{Main results on TSSMRBench. Each cell reports ACC / Cov / CSR, "
        "where ACC is answer accuracy, Cov is the average fraction of gold supporting states recovered in the retrieved set, and CSR is the fraction of questions for which the full gold support set is recovered. "
        "Oracle Gold Context is shown as an upper bound and is excluded from strongest-baseline marking. Bold marks the best automatic baseline and underline marks the second-best automatic baseline for each metric under the same task and answer budget.}"
    )
    lines.append("\\label{tab:overall-expanded}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccc@{}}")
    lines.append("\\toprule")
    lines.append("Task & System & Budget 1 & Budget 2 & Budget 3 \\\\")
    lines.append("\\midrule")

    for task_index, task in enumerate(TASK_ORDER):
        ks = TASK_TOP_KS[task]
        lines.append(
            f"\\multicolumn{{5}}{{l}}{{\\textbf{{{TASK_LABELS[task]}}} "
            f"(budget cells: $k={ks[0]}$, $k={ks[1]}$, $k={ks[2]}$; values are ACC / Cov / CSR)}} \\\\"
        )
        lines.append("\\cmidrule(lr){1-5}")
        for row_index, system in enumerate(system_rows):
            metrics_1 = system["task_topk_metrics"][task][str(ks[0])]
            metrics_2 = system["task_topk_metrics"][task][str(ks[1])]
            metrics_3 = system["task_topk_metrics"][task][str(ks[2])]
            task_cell = f"\\multirow{{{len(system_rows)}}}{{*}}{{{TASK_SHORT_LABELS[task]}}}" if row_index == 0 else ""
            lines.append(
                " & ".join(
                    [
                        task_cell,
                        system["paper_name"],
                        format_triplet(system["paper_name"], task, ks[0], metrics_1, styles),
                        format_triplet(system["paper_name"], task, ks[1], metrics_2, styles),
                        format_triplet(system["paper_name"], task, ks[2], metrics_3, styles),
                    ]
                )
                + " \\\\"
            )
        if task_index != len(TASK_ORDER) - 1:
            lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular*}")
    lines.append("\\end{table*}")
    return "\n".join(lines) + "\n"


def make_cost_table_tex(system_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append(
        "\\caption{Computational cost profile. Latency is mean retrieval-only latency in milliseconds, excluding answer generation. "
        "Retrieved context length is the mean number of context tokens passed to the answer model at the primary answer budget of each task family.}"
    )
    lines.append("\\label{tab:cost-profile}")
    lines.append("\\setlength{\\tabcolsep}{2pt}")
    lines.append(
        "\\begin{tabularx}{\\columnwidth}{@{}>{\\raggedright\\arraybackslash}p{0.22\\columnwidth}"
        ">{\\centering\\arraybackslash}p{0.13\\columnwidth}"
        ">{\\centering\\arraybackslash}p{0.17\\columnwidth}"
        ">{\\centering\\arraybackslash}p{0.17\\columnwidth}"
        ">{\\centering\\arraybackslash}p{0.17\\columnwidth}@{}}"
    )
    lines.append("\\toprule")
    lines.append(
        "System & \\shortstack{Latency\\\\(ms)} & \\shortstack{Single\\\\tokens\\\\($k{=}3$)} & "
        "\\shortstack{Cross\\\\tokens\\\\($k{=}8$)} & \\shortstack{Temporal\\\\tokens\\\\($k{=}10$)} \\\\"
    )
    lines.append("\\midrule")
    for system in system_rows:
        single_tokens = system["task_topk_metrics"]["single_state_lookup"]["3"]["retrieved_context_token_count"]
        cross_tokens = system["task_topk_metrics"]["cross_version_comparison"]["8"]["retrieved_context_token_count"]
        temporal_tokens = system["task_topk_metrics"]["temporal_version_ordering"]["10"]["retrieved_context_token_count"]
        lines.append(
            f"{system['paper_name']} & {format_latency(system['retrieval_latency_ms'])} & "
            f"{single_tokens:.2f} & {cross_tokens:.2f} & {temporal_tokens:.2f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def make_task_topk_table_tex(system_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Task-wise accuracy at the primary answer budget of each task family.}")
    lines.append("\\label{tab:task-maink}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("System & Single ($k{=}3$) & Cross ($k{=}8$) & Temporal ($k{=}10$) \\\\")
    lines.append("\\midrule")
    for system in system_rows:
        single = system["task_topk_metrics"]["single_state_lookup"]["3"]["acc"]
        cross = system["task_topk_metrics"]["cross_version_comparison"]["8"]["acc"]
        temporal = system["task_topk_metrics"]["temporal_version_ordering"]["10"]["acc"]
        lines.append(
            f"{system['paper_name']} & {single:.4f} & {cross:.4f} & {temporal:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible paper artifacts for official_300 release-only evaluation results.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--paper-table-dir", type=Path, default=DEFAULT_PAPER_TABLE_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_table_dir.mkdir(parents=True, exist_ok=True)

    system_rows: list[dict[str, Any]] = []
    task_topk_csv_rows: list[dict[str, Any]] = []
    decoupling_csv_rows: list[dict[str, Any]] = []
    case_candidates: dict[str, Any] = {}

    for spec in SYSTEM_SPECS:
        question_path = spec["result_dir"] / spec["questions_file"]
        rows = load_jsonl(question_path)
        rows = [{**row, "task_type": canonical_task(row.get("task_type"))} for row in rows]

        task_topk_metrics: dict[str, dict[str, Any]] = {}
        for task in TASK_ORDER:
            task_rows = [row for row in rows if row.get("task_type") == task]
            if not task_rows:
                continue
            metrics_by_k: dict[str, Any] = {}
            for k in TASK_TOP_KS[task]:
                metric_rows = [per_k_record(row, k) for row in task_rows]
                metrics = aggregate_metrics(metric_rows)
                metrics_by_k[str(k)] = metrics
                task_topk_csv_rows.append(
                    {
                        "system": spec["paper_name"],
                        "task_type": task,
                        "task_label": TASK_LABELS[task],
                        "top_k": k,
                        "question_count": metrics["question_count"],
                        "acc": metrics["acc"],
                        "cov": metrics["cov"],
                        "csr": metrics["csr"],
                        "zero_gold_rate": metrics["zero_gold_rate"],
                        "correct_without_gold_support_rate": metrics["correct_without_gold_support_rate"],
                        "retrieved_context_token_count": metrics["retrieved_context_token_count"],
                    }
                )
            task_topk_metrics[task] = metrics_by_k

        decoupling = build_decoupling(rows)
        for bucket_key, count in decoupling["counts"].items():
            decoupling_csv_rows.append(
                {
                    "system": spec["paper_name"],
                    "scope": "overall_maink",
                    "bucket": bucket_key,
                    "count": count,
                    "rate": decoupling["rates"][bucket_key],
                }
            )
        for task, payload in decoupling["by_task"].items():
            for bucket_key, count in payload["counts"].items():
                decoupling_csv_rows.append(
                    {
                        "system": spec["paper_name"],
                        "scope": task,
                        "bucket": bucket_key,
                        "count": count,
                        "rate": payload["rates"][bucket_key],
                    }
                )

        retrieval_latency_ms = mean_or_zero([safe_float(row.get("retrieval_latency_ms")) for row in rows])
        system_payload = {
            "paper_name": spec["paper_name"],
            "question_count": len(rows),
            "retrieval_latency_ms": retrieval_latency_ms,
            "task_topk_metrics": task_topk_metrics,
            "decoupling_maink": decoupling,
        }
        system_rows.append(system_payload)
        case_candidates[spec["paper_name"]] = build_case_candidates(rows, spec["paper_name"])

    metrics_payload = {
        "dataset_name": "official_300repo_release_unified_v1",
        "systems": system_rows,
        "task_order": TASK_ORDER,
        "task_labels": TASK_LABELS,
        "task_top_ks": TASK_TOP_KS,
        "primary_top_k": PRIMARY_K,
        "case_candidates": case_candidates,
        "source_result_dirs": {spec["paper_name"]: str(spec["result_dir"]) for spec in SYSTEM_SPECS},
    }

    (output_dir / "paper_metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "case_candidates.json").write_text(
        json.dumps(case_candidates, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "task_topk_metrics.csv", task_topk_csv_rows)
    write_csv(output_dir / "decoupling_maink.csv", decoupling_csv_rows)

    main_table_tex = make_main_table_tex(system_rows)
    task_table_tex = make_task_topk_table_tex(system_rows)
    cost_table_tex = make_cost_table_tex(system_rows)
    (output_dir / "main_results_expanded.tex").write_text(main_table_tex, encoding="utf-8")
    (output_dir / "task_maink_accuracy.tex").write_text(task_table_tex, encoding="utf-8")
    (output_dir / "cost_profile.tex").write_text(cost_table_tex, encoding="utf-8")
    (args.paper_table_dir / "main_results_expanded.tex").write_text(main_table_tex, encoding="utf-8")
    (args.paper_table_dir / "task_maink_accuracy.tex").write_text(task_table_tex, encoding="utf-8")
    (args.paper_table_dir / "cost_profile.tex").write_text(cost_table_tex, encoding="utf-8")

    manifest = {
        "paper_metrics_json": str(output_dir / "paper_metrics.json"),
        "task_topk_metrics_csv": str(output_dir / "task_topk_metrics.csv"),
        "decoupling_maink_csv": str(output_dir / "decoupling_maink.csv"),
        "case_candidates_json": str(output_dir / "case_candidates.json"),
        "paper_table_main": str(args.paper_table_dir / "main_results_expanded.tex"),
        "paper_table_task_maink": str(args.paper_table_dir / "task_maink_accuracy.tex"),
        "paper_table_cost_profile": str(args.paper_table_dir / "cost_profile.tex"),
    }
    (output_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
