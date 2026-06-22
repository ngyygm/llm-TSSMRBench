---
language:
- en
pretty_name: TSSMRBench Official 300 Public Release
license: other
license_name: Multi-source benchmark release with author-created annotations and rewritten memory units
task_categories:
- question-answering
tags:
- benchmark
- llm-agents
- memory-systems
- retrieval
- temporal-reasoning
- version-aware-retrieval
- text
---

# TSSMRBench Official 300 Public Release

## Dataset Summary

This repository contains the public release package for **TSSMRBench**, a benchmark for evaluating whether an agent memory system can retrieve the **stage-correct temporal semantic state** from an evolving memory trajectory.

The source material for this release is derived from public release histories of 300 influential GitHub repositories. For public redistribution, the package **removes the original `raw_text` release-note content** and retains only the normalized `memory_unit_text` representations, benchmark structure, question annotations, and source links.

The release is distributed as a single merged public file:

- `data/official_300_merged_public.json`: a single-file public merged release with the same scenario structure as the internal merged dataset, but with `raw_text` removed.

## What TSSMRBench Evaluates

TSSMRBench focuses on **temporal semantic state memory retrieval**. Each scenario is an evolving memory trajectory composed of versioned semantic states. The benchmark asks whether a memory system can retrieve the state evidence required by a query when several nearby versions are semantically related but temporally different.

The benchmark contains three task families:

- `single_state_lookup`: retrieve one stage-correct state.
- `cross_version_comparison`: jointly retrieve two states and compare them.
- `temporal_version_ordering`: recover several states and place them in the correct temporal progression.

## Public Release Design

This Hugging Face release is intended to be easier to redistribute and audit than the internal experiment package:

- Original `raw_text` fields are removed.
- Local absolute paths are removed.
- `source_url` links are preserved for traceability.
- `memory_unit_text` fields retain the normalized textual state descriptions used in the benchmark.

## Data Fields

The merged file `official_300_merged_public.json` stores one scenario record per repository. Each scenario contains:

- `prototype_id`: scenario identifier.
- `repo`: source repository.
- `window_title`: scenario title.
- `window_summary`: summary of the scenario's version history.
- `chunks`: the 30 temporal semantic state nodes, each with fields such as `memory_node_id`, `artifact_ref`, `time_hint`, `published_at`, `source_url`, and `memory_unit_text`.
- `questions`: the benchmark questions associated with the scenario, including `question_id`, `task_type`, `query_text`, `options`, `correct_option_id`, `expected_answer`, `source_chunk_ids`, and `answer_support`.

## Intended Use

This release is intended for:

- benchmarking agent memory systems;
- retrieval and reranking analysis on evolving memories;
- error analysis on version-sensitive memory retrieval;
- research on temporal reasoning grounded in retrieved memory evidence.

## Limitations

- The package is derived from public repository release histories and therefore reflects the style, granularity, and coverage of those sources.
- `memory_unit_text` is a normalized rewrite of source material rather than the raw release note text.
- This public package is designed for benchmark use and traceable redistribution, not as a substitute for the original upstream release documentation.

## Licensing Note

This dataset package uses `license: other` because it combines:

- benchmark structure, task design, and annotations created by the authors; and
- derived state descriptions and source links originating from multiple upstream repositories with non-uniform licenses.

Please read the `LICENSE` file in this repository before redistribution or reuse.

## Citation

If you use this dataset, please cite the TSSMRBench paper and repository once the final bibliographic entry is available. Until then, please reference the repository URL directly.
