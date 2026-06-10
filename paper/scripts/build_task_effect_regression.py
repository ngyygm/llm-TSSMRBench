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


def load_main_k_rows() -> pd.DataFrame:
    rows = []
    for system_name, path in SOURCE_DIRS.items():
        with path.open("r", encoding="utf-8") as f:
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

    return {
        "n_rows": int(len(df)),
        "n_questions": int(df["question_id"].nunique()),
        "n_systems": int(df["system"].nunique()),
        "task_rows": summary_rows,
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
