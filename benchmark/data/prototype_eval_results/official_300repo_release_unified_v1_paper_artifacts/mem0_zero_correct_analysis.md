# Mem0 `zero+correct` Analysis

This note explains why `Mem0` shows an unexpectedly large `zero_recalled__correct` bucket in the node-level decoupling summary.

## Summary

- Source run:
  - `benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/mem0.questions.jsonl`
- Node-level decoupling summary:
  - `benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_paper_artifacts/decoupling_maink.csv`
- Raw node-level count:
  - `zero_recalled__correct = 89 / 900 = 9.89%`
- After recounting with Mem0's grouped source-node identifiers:
  - `2 / 900`

The main conclusion is that this bucket is mostly an evaluation-granularity artifact, not genuine "correct without retrieval".

## Why this happens

Mem0 retrieves fine-grained memory facts and groups them by source state node before returning them.

- In `benchmark/src/systems/mem0_baseline.py`, `_group_raw_results_by_source_node()` builds grouped results with explicit `source_node_id`.
- In `benchmark/src/systems/mem0_baseline.py`, `query()` returns these grouped results in `metadata["grouped_results"]`.
- In `benchmark/src/state_version/evaluation.py`, `_match_retrieved_nodes()` only consumes explicit node ids from `metadata["retrieved_source_node_ids"]`.
- When that field is absent, it falls back to fuzzy text-overlap matching over the bundled fact text.

So the returned facts can already come from the correct source state node, but node-level coverage is still recorded as zero because the matcher never directly uses `grouped_results[].source_node_id`.

## Representative cases

### 1. Single-state lookup

- Question ID:
  - `Alamofire__Alamofire_release_window__q1_single_state_lookup`
- Query:
  - `In the Alamofire/Alamofire releases, which release adds Combine support with DataResponsePublisher, DownloadResponsePublisher, and DataStreamPublisher?`
- Gold state:
  - `Alamofire__Alamofire__release__5_2_0`
- Recorded node-level result:
  - `support_coverage = 0.0`
  - `matched_node_ids = ['Alamofire__Alamofire__release__5_6_3', 'Alamofire__Alamofire__release__5_1_0']`
- Retrieved grouped source states:
  - `Alamofire__Alamofire__release__5_2_0`
  - `Alamofire__Alamofire__release__5_6_3`
  - `Alamofire__Alamofire__release__5_1_0`
- Retrieved memory fact from the top grouped result:
  - `In Alamofire/Alamofire release 5.2.0, the release adds Combine support with DataResponsePublisher, DownloadResponsePublisher, and DataStreamPublisher.`

Diagnosis:
- The correct source node is the top grouped result.
- The answer is correct because the gold fact was retrieved.
- Node-level coverage becomes zero only because the bundled fact text is not aligned back to the gold node id.

### 2. Cross-version comparison

- Question ID:
  - `FuelLabs__sway_release_window__q2_cross_version_comparison`
- Query:
  - `Across the FuelLabs/sway releases, how does the handling of panic differ between v0.68.2 and v0.69.0?`
- Gold states:
  - `FuelLabs__sway__release__v0_68_2`
  - `FuelLabs__sway__release__v0_69_0`
- Recorded node-level result:
  - `support_coverage = 0.0`
- Retrieved grouped source states:
  - `FuelLabs__sway__release__v0_69_0`
  - `FuelLabs__sway__release__v0_68_2`
  - `FuelLabs__sway__release__v0_68_6`
  - `FuelLabs__sway__release__v0_68_9`
- Retrieved memory facts:
  - `Breaking changes in FuelLabs/sway release v0.69.0 include panic becoming a keyword.`
  - `In the FuelLabs/sway release v0.68.2, the release note includes implementing panic expression.`

Diagnosis:
- Both gold states are already present in the grouped retrieval output.
- The answer is supported by the retrieved memory facts.
- The zero-coverage label again comes from node-level alignment failure, not from retrieval failure.

### 3. Temporal ordering

- Question ID:
  - `ChatGPTNextWeb__NextChat_release_window__q3_temporal_version_ordering`
- Query:
  - `In the ChatGPTNextWeb/NextChat release window, order the following release-content states from earliest to latest: (1) a release that adds support for the Google Gemini Pro model, (2) a release that adds support for the OpenAI o1 model, (3) a release that adds support for the OpenAI Realtime API, (4) a release that adds DeepSeek as a new model provider.`
- Gold states:
  - `ChatGPTNextWeb__NextChat__release__v2_10_1`
  - `ChatGPTNextWeb__NextChat__release__v2_15_2`
  - `ChatGPTNextWeb__NextChat__release__v2_15_8`
  - `ChatGPTNextWeb__NextChat__release__v2_16_0`
- Recorded node-level result:
  - `support_coverage = 0.0`
- Retrieved grouped source states include:
  - `ChatGPTNextWeb__NextChat__release__v2_10_1`
  - `ChatGPTNextWeb__NextChat__release__v2_15_2`
  - `ChatGPTNextWeb__NextChat__release__v2_15_8`
  - `ChatGPTNextWeb__NextChat__release__v2_16_0`
- Retrieved memory facts include:
  - `In the ChatGPTNextWeb/NextChat release v2.10.1, the update adds support for the Google Gemini Pro model.`
  - `In the ChatGPTNextWeb/NextChat release v2.15.2, the update adds support for the OpenAI o1 model.`
  - `In the ChatGPTNextWeb/NextChat release v2.15.8, the update adds support for the OpenAI Realtime API.`

Diagnosis:
- This case is especially important because the question needs multiple states.
- Mem0 retrieves all required grouped source states, but node-level coverage is still recorded as zero.
- The anomaly therefore persists even when the system has enough evidence to solve a multi-state ordering question.

## Residual cases after grouped-id recount

Only two `zero+correct` cases remain after checking grouped source-node ids directly:

1. `YunaiV__ruoyi_vue_pro_release_window__q2_cross_version_comparison`
2. `microsoft__monaco_editor_release_window__q1_single_state_lookup`

These are identifier-mismatch cases rather than clear retrieval-free successes.

- `YunaiV/ruoyi-vue-pro`:
  - gold uses `v2.4.0(jdk8/11)` and `v2.4.1(jdk17/21)`
  - retrieved grouped ids point to the opposite JDK variants
- `microsoft/monaco-editor`:
  - gold uses `v0.53.0-dev-20250830`
  - retrieved grouped id is `v0.53.0`

So even the residual bucket is better interpreted as an identifier-normalization issue than as genuine unsupported correctness.

## Takeaway for the paper

The large raw `zero+correct` bucket for Mem0 should be described as:

- a fact-level vs. node-level alignment gap
- caused by Mem0 returning grouped fact evidence with source-node metadata
- while the current node matcher only consumes `retrieved_source_node_ids` and otherwise falls back to fuzzy overlap matching

It should not be described as evidence that Mem0 often answers correctly without retrieving relevant memory.
