import json
import math
from pathlib import Path

import pandas as pd
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "benchmark" / "data" / "prototype_eval_results" / "official_300repo_release_unified_v1_paper_artifacts"
PAPER_TABLE_DIR = ROOT / "paper" / "tables" / "generated"

SOURCE_DIRS = {
    "BM25": ROOT / "benchmark" / "data" / "prototype_eval_results" / "official_300repo_release_unified_v1_bm25_globalpool_taskk_v1" / "bm25.questions.jsonl",
    "FAISS": ROOT / "benchmark" / "data" / "prototype_eval_results" / "official_300repo_release_unified_v1_faiss_globalpool_taskk_v1" / "faiss_vector_store.questions.jsonl",
    "Mem0": ROOT / "benchmark" / "data" / "prototype_eval_results" / "official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10" / "mem0.questions.jsonl",
    "Graphiti": ROOT / "benchmark" / "data" / "prototype_eval_results" / "official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1" / "graphiti.questions.jsonl",
}

MAIN_TOP_K = {
    "single_state_lookup": "3",
    "cross_version_comparison": "8",
    "temporal_version_ordering": "10",
}

TASK_LABEL = {
    "single_state_lookup": "Single-state lookup",
    "cross_version_comparison": "Cross-version comparison",
    "temporal_version_ordering": "Temporal ordering",
}


def _open_maybe_gz(path: Path):
    """Open a per-question JSONL, transparently handling .gz.

    Committed raw results are stored as ``*.questions.jsonl.gz``; this lets the
    regression run on the committed repo without manual decompression.
    """
    import gzip

    if path.exists():
        return path.open("r", encoding="utf-8")
    gz = Path(str(path) + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    raise FileNotFoundError(f"Neither {path} nor {gz} exists")


def load_main_k_rows() -> pd.DataFrame:
    rows = []
    for system_name, path in SOURCE_DIRS.items():
        with _open_maybe_gz(path) as f:
            for line in f:
                item = json.loads(line)
                task_type = item["task_type"]
                k = MAIN_TOP_K[task_type]
                per_k = item["per_k_results"][k]
                rows.append(
                    {
                        "system": system_name,
                        "question_id": item["question_id"],
                        "task_type": task_type,
                        "task_label": TASK_LABEL[task_type],
                        "top_k": int(k),
                        "is_correct": int(bool(per_k["is_correct"])),
                    }
                )
    return pd.DataFrame(rows)


def _load_per_k_correctness(system_name: str, task: str, k: str) -> dict[str, int]:
    """Return {question_id: is_correct} for one system/task/k from the raw .jsonl(.gz)."""
    out: dict[str, int] = {}
    with _open_maybe_gz(SOURCE_DIRS[system_name]) as f:
        for line in f:
            item = json.loads(line)
            if item["task_type"] != task:
                continue
            per_k = item["per_k_results"][k]
            out[item["question_id"]] = int(bool(per_k["is_correct"]))
    return out


def paired_two_proportion(a: dict[str, int], b: dict[str, int]) -> dict:
    """Two-proportion z-test plus paired McNemar for two systems on shared questions."""
    common = sorted(set(a) & set(b))
    n = len(common)
    xa = sum(a[q] for q in common)
    xb = sum(b[q] for q in common)
    pa, pb = xa / n, xb / n
    pooled = (xa + xb) / (2 * n)
    se = math.sqrt(2 * pooled * (1 - pooled) / n) if 0 < pooled < 1 else 0.0
    z = (pa - pb) / se if se else 0.0
    # two-sided p from normal CDF approximation
    p_two = math.erfc(abs(z) / math.sqrt(2.0))
    # McNemar (discordant pairs)
    b01 = sum(1 for q in common if a[q] == 0 and b[q] == 1)  # a wrong, b right
    b10 = sum(1 for q in common if a[q] == 1 and b[q] == 0)  # a right, b wrong
    if b01 + b10:
        mcnemar_chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    else:
        mcnemar_chi2 = 0.0
    return {
        "n": n,
        "acc_a": pa,
        "acc_b": pb,
        "acc_diff": pa - pb,
        "z": z,
        "p_two_sided": p_two,
        "mcnemar_b01": b01,
        "mcnemar_b10": b10,
        "mcnemar_chi2": mcnemar_chi2,
    }


def fit_model(df: pd.DataFrame):
    formula = (
        "is_correct ~ "
        "C(task_type, Treatment(reference='single_state_lookup')) + "
        "C(system, Treatment(reference='FAISS'))"
    )
    vc_formulas = {"question_re": "0 + C(question_id)"}
    model = BinomialBayesMixedGLM.from_formula(formula, vc_formulas, df)
    result = model.fit_vb()
    return result


def build_summary(df: pd.DataFrame, result) -> dict:
    names = result.model.exog_names
    coef = dict(zip(names, result.fe_mean))
    sd = dict(zip(names, result.fe_sd))

    task_terms = [
        "C(task_type, Treatment(reference='single_state_lookup'))[T.cross_version_comparison]",
        "C(task_type, Treatment(reference='single_state_lookup'))[T.temporal_version_ordering]",
    ]

    summary_rows = []
    for term in task_terms:
        mean = coef[term]
        se = sd[term]
        lower = mean - 1.96 * se
        upper = mean + 1.96 * se
        summary_rows.append(
            {
                "contrast": "Cross vs Single" if "cross_version" in term else "Temporal vs Single",
                "log_odds": mean,
                "log_odds_sd": se,
                "odds_ratio": math.exp(mean),
                "or_ci_low": math.exp(lower),
                "or_ci_high": math.exp(upper),
            }
        )

    # System fixed effects (odds ratio vs FAISS reference) so that "FAISS is the
    # strongest baseline" and "Graphiti exceeds FAISS" claims carry uncertainty.
    system_terms = [name for name in names if name.startswith("C(system")]
    system_rows = []
    for term in system_terms:
        # term looks like C(system, ...)[T.Graphiti]
        label = term.split("[T.")[1].rstrip("]") if "[T." in term else term
        mean = coef.get(term)
        se = sd.get(term)
        if mean is None or se is None:
            continue
        system_rows.append(
            {
                "system": label,
                "log_odds_vs_faiss": mean,
                "log_odds_sd": se,
                "odds_ratio": math.exp(mean),
                "or_ci_low": math.exp(mean - 1.96 * se),
                "or_ci_high": math.exp(mean + 1.96 * se),
            }
        )

    # Paired test for the specific headline claim "Graphiti exceeds FAISS on
    # cross-version at k=2" (ACC 0.5467 vs 0.5033). Reports whether the 13/300
    # difference is significant once question pairing is accounted for.
    pairwise_claims = []
    try:
        g = _load_per_k_correctness("Graphiti", "cross_version_comparison", "2")
        f_ = _load_per_k_correctness("FAISS", "cross_version_comparison", "2")
        test = paired_two_proportion(f_, g)  # a=FAISS, b=Graphiti
        test["claim"] = "Graphiti vs FAISS, cross-version, k=2"
        pairwise_claims.append(test)
    except Exception as exc:  # pragma: no cover - defensive
        pairwise_claims.append({"claim": "Graphiti vs FAISS, cross-version, k=2", "error": str(exc)})

    return {
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "n_systems": int(df["system"].nunique()),
        "task_rows": summary_rows,
        "system_rows": system_rows,
        "pairwise_claims": pairwise_claims,
    }


def write_outputs(summary: dict):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    json_path = ARTIFACT_DIR / "task_effect_mixed_logit.json"
    tex_path = PAPER_TABLE_DIR / "task_effect_mixed_logit.tex"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Mixed-effects logistic regression on answer correctness at the main task-specific retrieval top-k settings. The model uses fixed effects for task family and system, with a random intercept for question. Odds ratios are reported relative to single-state lookup.}",
        r"\label{tab:task-effect-mixed-logit}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Contrast & Odds ratio & 95\% interval \\",
        r"\midrule",
    ]
    for row in summary["task_rows"]:
        lines.append(
            f"{row['contrast']} & {row['odds_ratio']:.2f} & "
            f"[{row['or_ci_low']:.2f}, {row['or_ci_high']:.2f}] \\\\"
        )
    if summary.get("system_rows"):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{3}{l}{\emph{System effect (odds ratio vs.\ FAISS)}} \\")
        for row in summary["system_rows"]:
            lines.append(
                f"{row['system']} & {row['odds_ratio']:.2f} & "
                f"[{row['or_ci_low']:.2f}, {row['or_ci_high']:.2f}] \\\\"
            )
    pairwise_note = ""
    for claim in summary.get("pairwise_claims", []):
        if "error" in claim:
            continue
        pairwise_note = (
            f"Paired test, Graphiti vs.~FAISS at cross k=2: "
            f"acc.~diff {claim['acc_diff']:+.3f}, z={claim['z']:.2f}, p={claim['p_two_sided']:.3f} "
            f"(McNemar discordant {claim['mcnemar_b01']}/{claim['mcnemar_b10']}, $\\chi^2{{=}}{claim['mcnemar_chi2']:.2f}$)."
        )
    if pairwise_note:
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{3}{l}{\scriptsize " + pairwise_note.strip() + r"} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    df = load_main_k_rows()
    result = fit_model(df)
    summary = build_summary(df, result)
    write_outputs(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
