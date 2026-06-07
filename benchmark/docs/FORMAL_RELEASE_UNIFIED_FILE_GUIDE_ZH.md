# 正式数据集与评测结果文件说明

本文档用于说明当前**正式 GitHub Release Unified 数据集**及其**正式评测结果**所在位置，方便后续论文写作、结果核对和补充统计。

## 1. 当前正式数据集

当前正式数据集的根目录是：

- [formal_300repo_unified_v1](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1:1)

这是当前应优先使用的正式版本。该目录下最重要的文件有：

- [official_300_merged.json](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_merged.json:1)
  - 正式实验主入口文件。
  - 合并后的 `300` 个项目统一数据文件。
  - 全量统一混池实验应直接使用此文件。

- [prototype_index_official_300.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/prototype_index_official_300.jsonl:1)
  - 正式 `300` 个项目样例的索引。
  - 每一行对应一个项目窗口样例。

- [prototype_index_reserve.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/prototype_index_reserve.jsonl:1)
  - 备用样例索引。
  - 不属于正式 `300`，仅在正式样例替换时参考。

- [official_300_replacement_report.json](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_replacement_report.json:1)
  - 正式 `300` 中发生替换的项目记录。

- [successful_repos.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/successful_repos.jsonl:1)
  - 成功构建出的全部可实验项目列表。

- [successful_repos.txt](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/successful_repos.txt:1)
  - 上述列表的纯文本版。

## 2. 单项目样例文件格式

正式集中的每个项目样例都有一个独立目录，例如：

- [fastapi__fastapi_release_window](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/fastapi__fastapi_release_window:1)

每个项目目录下最重要的文件是：

- `prototype.json`
  - 单个项目的完整样例文件。
  - 内含：
    - `chunks`
    - `questions`
  - `chunks` 对应该项目窗口内的 `30` 个 release-note memory chunks。
  - `questions` 对应该项目的 `3` 道题。

当前 unified 正式格式中：

- 不再单独区分“主链文件”和“QA 文件”
- 一个 `prototype.json` 同时包含：
  - chunk 数据
  - QA 数据
- 每道题通过 `source_chunk_ids` 指明所依赖的 gold chunks

## 3. 原始 release 源数据

正式数据集对应的原始 GitHub release notes 下载目录是：

- [github_release_notes_formal_v1](/d:/github_project/BiTempQA/benchmark/data/raw/github_release_notes_formal_v1:1)

每个仓库子目录中通常有：

- `releases.json`
- `releases.jsonl`

这些文件是正式数据集的原始来源，可用于：

- 核对某个 `memory_unit_text` 是否忠实于原始发行说明
- 追查 QA 题目证据
- 后续补做人工审查

## 4. 当前应使用的正式评测结果

### 4.1 Simple Baselines，全局统一混池

这组结果对应：

- `300` 项目所有 chunk 先统一存入一个混合记忆池
- 再进行检索与评测

#### BM25

- [official_300repo_release_unified_v1_bm25_globalpool_taskk_v1](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1:1)

重要文件：

- [aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/aggregate_summary.json:1)
- [bm25.summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/bm25.summary.json:1)
- [bm25.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/bm25.questions.jsonl:1)

#### FAISS

- [official_300repo_release_unified_v1_faiss_globalpool_taskk_v1](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1:1)

重要文件：

- [aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/aggregate_summary.json:1)
- [faiss_vector_store.summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/faiss_vector_store.summary.json:1)
- [faiss_vector_store.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/faiss_vector_store.questions.jsonl:1)

### 4.2 Mem0，全局统一混池

#### Mem0 主结果，`internal_fact_k = 60`

- [official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2:1)

重要文件：

- [aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/aggregate_summary.json:1)
- [mem0.summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0.summary.json:1)
- [mem0.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0.questions.jsonl:1)

#### Mem0 对照结果，`internal_fact_k = 10`

- [official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10:1)

重要文件：

- [aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/aggregate_summary.json:1)
- [mem0.summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/mem0.summary.json:1)
- [mem0.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/mem0.questions.jsonl:1)

### 4.3 Graphiti，全局统一混池

- [official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1:1)

重要文件：

- [aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/aggregate_summary.json:1)
- [graphiti.summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti.summary.json:1)
- [graphiti.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti.questions.jsonl:1)

## 5. 中间结果文件说明

下面这些文件很重要，后续做论文统计时会频繁用到。

### 5.1 `aggregate_summary.json`

作用：

- 汇总某个系统在整个正式集上的总体指标。
- 适合直接提取：
  - `ACC`
  - `Cov`
  - `CSR`
  - `Latency`

### 5.2 `*.summary.json`

作用：

- 给出更完整的 summary。
- 一般包含：
  - `overall`
  - `breakdowns`
- 常用于按题型查看：
  - `single_state_lookup`
  - `cross_version_comparison`
  - `temporal_version_ordering`

### 5.3 `*.questions.jsonl`

作用：

- 最重要的逐题结果文件。
- 后续统计表格时，最细粒度的数据来自这里。

每一行通常包含：

- 题目元信息
- gold chunks
- 检回结果
- 答案生成结果
- 支持度信息
- 多个 `k` 的复用评测结果

特别重要的字段包括：

- `task_type`
- `per_k_results`
- `support_coverage`
- `complete_support`
- `distractor_to_gold_ratio`
- `retrieved_context_token_count`
- `is_correct_without_gold_support`
- `first_gold_rank`
- `gold_rank_positions`
- `any_gold_within`
- `all_gold_within`

这些字段支持后续统计：

- 按题型的 `ACC/Cov/CSR`
- 按 `top-k` 的 `ACC/Cov/CSR`
- `Zero-Gold`
- `D/G`
- `Context Tokens`
- “答对但没有检回 gold”的比例
- `Top-1 vs Top-k stability`

## 6. 统一混池实验的存入进度文件

对于需要先全量存记忆的系统，中间通常会有 ingest 进度文件。

### Mem0

主结果目录中：

- [mem0.ingest_checkpoint.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0.ingest_checkpoint.json:1)
- [mem0.ingest_progress.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0.ingest_progress.jsonl:1)
- [mem0_memory_run.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0_memory_run.json:1)

作用：

- `mem0.ingest_checkpoint.json`
  - 记录节点级断点续传信息。
- `mem0.ingest_progress.jsonl`
  - 记录节点级增量落盘的 ingest 进度。
- `mem0_memory_run.json`
  - 记录当前 run 对应的记忆库运行标识，可用于复用旧记忆库重跑检索与答案生成。

### Graphiti

主结果目录中：

- [graphiti.ingest_checkpoint.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti.ingest_checkpoint.json:1)
- [graphiti.ingest_progress.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti.ingest_progress.jsonl:1)
- [graphiti_memory_run.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti_memory_run.json:1)

作用与 Mem0 对应。

## 7. 当前论文写作时建议优先引用的文件

如果后续只是写论文并查结果，建议优先使用：

### 数据集

- [official_300_merged.json](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/official_300_merged.json:1)
- [prototype_index_official_300.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototypes/github_release_note_v2/formal_300repo_unified_v1/prototype_index_official_300.jsonl:1)

### Baselines

- [official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/aggregate_summary.json:1)
- [official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/aggregate_summary.json:1)
- [official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/aggregate_summary.json:1)
- [official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/aggregate_summary.json:1)
- [official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/aggregate_summary.json](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/aggregate_summary.json:1)

### 逐题统计

- [official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/bm25.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_bm25_globalpool_taskk_v1/bm25.questions.jsonl:1)
- [official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/faiss_vector_store.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_faiss_globalpool_taskk_v1/faiss_vector_store.questions.jsonl:1)
- [official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_resume50_v2/mem0.questions.jsonl:1)
- [official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/mem0.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_mem0_deepseekflash_globalpool_taskk_internal10/mem0.questions.jsonl:1)
- [official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti.questions.jsonl](/d:/github_project/BiTempQA/benchmark/data/prototype_eval_results/official_300repo_release_unified_v1_graphiti_deepseekflash_globalpool_taskk_resume50_v1/graphiti.questions.jsonl:1)

## 8. 不建议用于论文主结果的旧目录

`benchmark/data/prototype_eval_results` 下仍然存在很多早期打通链路时留下的实验目录，例如：

- `multirepo_10chain_*`
- `release_note_*`
- `simple_parallel_*`
- `_smoketest_*`
- `official_*limit10`

这些目录更多用于：

- 早期原型验证
- 单样例打通
- 局部 smoke test

不建议作为论文主结果引用。论文主结果应优先使用本文档第 4 节列出的正式目录。

