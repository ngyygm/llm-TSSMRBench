#!/usr/bin/env python3
"""Run evaluation for the independent state-version benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.answer_generator import AnswerGenerator
from src.evaluation.judge import LLMJudge
from src.state_version.evaluation import (
    StateVersionEvaluationRunner,
    build_memory_view_dataset,
    load_state_version_dataset,
)
from src.systems.base import MemorySystem
from src.utils.config_env import load_yaml_with_env


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_with_env(path)


def resolve_llm_value(
    section: dict[str, Any],
    key: str,
    generation_cfg: dict[str, Any],
    generation_key: Optional[str] = None,
) -> Any:
    value = section.get(key)
    if value not in (None, ""):
        return value
    llm_cfg = generation_cfg.get("llm", {})
    fallback_key = generation_key or key
    return llm_cfg.get(fallback_key)


def resolve_embedding_value(
    section: dict[str, Any],
    key: str,
    config: dict[str, Any],
    fallback_key: Optional[str] = None,
) -> Any:
    value = section.get(key)
    if value not in (None, ""):
        return value
    embeddings_cfg = config.get("embeddings", {})
    lookup_key = fallback_key or key
    return embeddings_cfg.get(lookup_key)


def create_systems(config: dict[str, Any], generation_cfg: dict[str, Any], only_systems: Optional[list[str]] = None) -> list[MemorySystem]:
    systems: list[MemorySystem] = []
    systems_cfg = config.get("systems", {})

    def normalize(name: str) -> str:
        return name.lower().replace("_", "").replace(" ", "")

    allowed = {normalize(name) for name in only_systems} if only_systems else None

    def enabled(name: str) -> bool:
        if allowed is None:
            return systems_cfg.get(name, {}).get("enabled", False)
        return normalize(name) in allowed

    if enabled("no_context"):
        from src.systems.no_context_baseline import NoContextBaseline

        systems.append(NoContextBaseline())

    if enabled("random_context"):
        from src.systems.random_context_baseline import RandomContextBaseline

        systems.append(RandomContextBaseline())

    if enabled("full_context"):
        from src.systems.full_context_baseline import FullContextBaseline

        systems.append(FullContextBaseline())

    if enabled("bm25"):
        from src.systems.bm25_baseline import BM25Baseline

        system = BM25Baseline()
        system.top_k = systems_cfg["bm25"].get("top_k", 5)
        systems.append(system)

    if enabled("faiss"):
        from src.systems.faiss_baseline import FAISSBaseline

        cfg = systems_cfg["faiss"]
        systems.append(
            FAISSBaseline(
                embedding_model=resolve_embedding_value(cfg, "embedding_model", config, "model"),
                embedding_base_url=resolve_embedding_value(cfg, "embedding_base_url", config, "base_url")
                or resolve_embedding_value(cfg, "embedding_endpoint", config, "endpoint"),
                embedding_api_key=resolve_embedding_value(cfg, "embedding_api_key", config, "api_key"),
                top_k=cfg.get("top_k", 5),
            )
        )

    if enabled("chroma"):
        from src.systems.chroma_baseline import ChromaBaseline

        cfg = systems_cfg["chroma"]
        systems.append(
            ChromaBaseline(
                embedding_model=cfg.get("embedding_model"),
                top_k=cfg.get("top_k", 5),
            )
        )

    if enabled("naive_rag"):
        from src.systems.naive_rag_baseline import NaiveRAGBaseline

        system = NaiveRAGBaseline()
        system.top_k = systems_cfg["naive_rag"].get("top_k", 5)
        systems.append(system)

    if enabled("simple_kg"):
        from src.systems.simple_kg_baseline import SimpleKGBaseline

        cfg = systems_cfg["simple_kg"]
        systems.append(SimpleKGBaseline(top_k=cfg.get("top_k", 5)))

    if enabled("mem0"):
        from src.systems.mem0_baseline import Mem0Baseline

        cfg = systems_cfg["mem0"]
        llm_model = resolve_llm_value(cfg, "llm_model", generation_cfg, generation_key="evaluation_model")
        llm_base_url = resolve_llm_value(cfg, "llm_base_url", generation_cfg, generation_key="base_url")
        llm_api_key = resolve_llm_value(cfg, "llm_api_key", generation_cfg, generation_key="api_key")
        systems.append(
            Mem0Baseline(
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedder_model=cfg.get("embedder_model"),
                embedder_base_url=cfg.get("embedder_base_url"),
                embedder_api_key=cfg.get("embedder_api_key"),
                top_k=cfg.get("top_k", 5),
                use_infer_updates=cfg.get("use_infer_updates", True),
                chain_ingest_workers=cfg.get("chain_ingest_workers", 1),
                internal_fact_k=cfg.get("internal_fact_k"),
                run_id=cfg.get("run_id"),
            )
        )

    if enabled("graphiti"):
        from src.systems.graphiti_baseline import GraphitiBaseline

        cfg = systems_cfg["graphiti"]
        llm_model = resolve_llm_value(cfg, "llm_model", generation_cfg, generation_key="evaluation_model")
        llm_base_url = resolve_llm_value(cfg, "llm_base_url", generation_cfg, generation_key="base_url")
        llm_api_key = resolve_llm_value(cfg, "llm_api_key", generation_cfg, generation_key="api_key")
        systems.append(
            GraphitiBaseline(
                neo4j_uri=cfg.get("neo4j_uri", "bolt://localhost:7687"),
                neo4j_user=cfg.get("neo4j_user", "neo4j"),
                neo4j_password=cfg.get("neo4j_password", "graphiti"),
                llm_model=llm_model,
                llm_base_url=llm_base_url,
                llm_api_key=llm_api_key,
                embedder_model=cfg.get("embedder_model", "Pro/BAAI/bge-m3"),
                embedder_base_url=cfg.get("embedder_base_url"),
                embedder_api_key=cfg.get("embedder_api_key"),
                reranker_model=cfg.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
                reranker_base_url=cfg.get("reranker_base_url"),
                reranker_api_key=cfg.get("reranker_api_key"),
                top_k=cfg.get("top_k", 5),
                max_coroutines=cfg.get("max_coroutines", 10),
                chain_ingest_workers=cfg.get("chain_ingest_workers", 5),
                run_id=cfg.get("run_id"),
            )
        )

    if enabled("tmg"):
        from src.systems.tmg_client import TMGClient

        cfg = systems_cfg["tmg"]
        systems.append(
            TMGClient(
                api_base=cfg.get("api_base", "http://localhost:8732"),
                top_k=cfg.get("top_k", 5),
            )
        )

    return systems


def main() -> None:
    parser = argparse.ArgumentParser(description="Run state-version benchmark evaluation")
    parser.add_argument("--config", default="configs/state_version_experiment_config.yaml")
    parser.add_argument("--systems", nargs="*", default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--question-ids", nargs="*", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--memory-view", choices=["full", "core_only"], default="full")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    repo_root = Path(__file__).parent.parent
    eval_cfg = load_yaml(repo_root / args.config)
    generation_cfg = eval_cfg

    dataset_cfg = eval_cfg["dataset"]
    splits = args.splits or dataset_cfg.get("splits", ["dev", "test"])
    domains = args.domains or dataset_cfg.get("domains", ["github_evolution", "narrative_evolution"])
    dataset = load_state_version_dataset(
        dataset_root=repo_root / dataset_cfg["dataset_root"],
        language=dataset_cfg.get("language", "en"),
        phase=dataset_cfg.get("phase", "formal"),
        domains=domains,
        splits=splits,
    )
    dataset = build_memory_view_dataset(dataset, memory_view=args.memory_view)
    random_pool_nodes: list[tuple[str, str, str]] = []
    systems_cfg = eval_cfg.get("systems", {})
    requested_systems = {name.lower().replace("_", "").replace(" ", "") for name in (args.systems or [])}
    random_enabled = (
        systems_cfg.get("random_context", {}).get("enabled", False)
        if not requested_systems
        else "randomcontext" in requested_systems
    )
    if random_enabled:
        random_pool_splits = systems_cfg.get("random_context", {}).get("pool_splits")
        if not random_pool_splits:
            random_pool_splits = [split for split in ("train", "dev", "test") if split not in splits]
        if random_pool_splits:
            random_pool_dataset = load_state_version_dataset(
                dataset_root=repo_root / dataset_cfg["dataset_root"],
                language=dataset_cfg.get("language", "en"),
                phase=dataset_cfg.get("phase", "formal"),
                domains=domains,
                splits=random_pool_splits,
            )
            random_pool_dataset = build_memory_view_dataset(random_pool_dataset, memory_view=args.memory_view)
            for pool_chain_id in sorted(random_pool_dataset.chains):
                pool_chain = random_pool_dataset.chains[pool_chain_id]
                ordered_nodes = sorted(pool_chain.chain_nodes, key=lambda node: node.surface_order)
                for node in ordered_nodes:
                    random_pool_nodes.append((pool_chain_id, node.node_id, node.text))
            logging.info(
                "Loaded %s random-context pool nodes from splits=%s",
                len(random_pool_nodes),
                random_pool_splits,
            )
    if args.question_ids or args.max_questions:
        allowed = set(args.question_ids or [])
        limited_questions = {}
        seen = 0
        for chain_id, questions in dataset.questions_by_chain.items():
            filtered = []
            for question in questions:
                if allowed and question.question_id not in allowed:
                    continue
                if args.max_questions is not None and seen >= args.max_questions:
                    break
                filtered.append(question)
                seen += 1
            if filtered:
                limited_questions[chain_id] = filtered
            if args.max_questions is not None and seen >= args.max_questions:
                break
        dataset.questions_by_chain = limited_questions
        active_chain_ids = set(limited_questions)
        dataset.chains = {chain_id: chain for chain_id, chain in dataset.chains.items() if chain_id in active_chain_ids}
        dataset.chain_split = {chain_id: split for chain_id, split in dataset.chain_split.items() if chain_id in active_chain_ids}
        dataset.chain_domain = {chain_id: domain for chain_id, domain in dataset.chain_domain.items() if chain_id in active_chain_ids}
        logging.info(
            "Filtered evaluation set to %s chains and %s questions",
            len(dataset.chains),
            sum(len(items) for items in dataset.questions_by_chain.values()),
        )

    systems = create_systems(eval_cfg, generation_cfg, args.systems)
    logging.info("Systems to evaluate: %s", [system.name for system in systems])

    default_output_dir = eval_cfg.get("reporting", {}).get("output_dir", "data/state_version_eval_results")
    if args.output_dir:
        output_dir_value = args.output_dir
    elif args.memory_view == "core_only":
        output_dir_value = f"{default_output_dir}_core_only"
    else:
        output_dir_value = default_output_dir
    output_root = repo_root / output_dir_value
    output_root.mkdir(parents=True, exist_ok=True)

    answer_cfg = eval_cfg.get("answer_generator", {})
    answer_generator = AnswerGenerator(
        base_url=answer_cfg.get("base_url"),
        api_key=answer_cfg.get("api_key"),
        model=answer_cfg.get("model"),
        temperature=answer_cfg.get("temperature", eval_cfg.get("llm", {}).get("generation_temperature", 0.0)),
        extra_body=answer_cfg.get("extra_body") or eval_cfg.get("llm", {}).get("extra_body") or {},
    )

    judge_cfg = eval_cfg.get("llm_judge", {})
    judge = None
    if judge_cfg.get("enabled", True) and not args.no_judge:
        judge = LLMJudge(
            base_url=judge_cfg.get("base_url"),
            api_key=judge_cfg.get("api_key"),
            model=judge_cfg.get("model"),
            temperature=judge_cfg.get("temperature", eval_cfg.get("llm", {}).get("generation_temperature", 0.0)),
            max_workers=judge_cfg.get("max_workers", 3),
            timeout=judge_cfg.get("timeout", 60),
            cache_path=output_root / "judge_cache.json",
            mode="answer_judge",
            extra_body=judge_cfg.get("extra_body") or eval_cfg.get("llm", {}).get("extra_body") or {},
        )
    runtime_cfg = eval_cfg.get("runtime", {})
    max_workers = args.max_workers if args.max_workers is not None else runtime_cfg.get("max_workers", 10)
    save_every = args.save_every if args.save_every is not None else runtime_cfg.get("save_every", 10)
    resume = not args.no_resume if args.no_resume else runtime_cfg.get("resume", True)
    runs = []
    for system in systems:
        runner = StateVersionEvaluationRunner(
            dataset=dataset,
            systems=[system],
            answer_generator=answer_generator,
            judge=judge,
            random_context_pool_nodes=random_pool_nodes,
        )
        run = runner.run_system(
            system,
            output_dir=output_root,
            resume=resume,
            max_workers=max_workers,
            save_every=save_every,
        )
        runs.append(run)
        logging.info("Saved results for %s to %s", run["system_name"], output_root)

    print("\n" + "=" * 128)
    print(
        f"{'System':<24} {'QA Acc':>8} {'Ans Acc':>8} {'GoldCov':>9} "
        f"{'CompSupp':>9} {'D/G Ratio':>10} {'Latency':>10} {'CtxSize':>10}"
    )
    print("-" * 128)
    for run in runs:
        summary = run["summary"]["overall"]
        print(
            f"{run['system_name']:<24} "
            f"{(summary['qa_accuracy'] or 0.0):>8.3f} "
            f"{(summary['answerable_accuracy'] or 0.0):>8.3f} "
            f"{(summary['gold_coverage_at_k'] or 0.0):>9.3f} "
            f"{(summary['complete_support_rate'] or 0.0):>9.3f} "
            f"{(summary['distractor_to_gold_ratio'] or 0.0):>10.3f} "
            f"{(summary['retrieval_latency_ms'] or 0.0):>10.1f} "
            f"{(summary['retrieved_context_token_count'] or 0.0):>10.1f}"
        )
    print("=" * 128)


if __name__ == "__main__":
    main()
