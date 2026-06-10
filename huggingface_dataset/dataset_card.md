---
language:
- zh
license: mit
task_categories:
- question-answering
tags:
- temporal-reasoning
- bitemporal
- memory-systems
- llm-agents
- benchmark
- chinese
size_categories:
- n<1K
---

# BiTempQA: A Diagnostic Benchmark for Bitemporal Reasoning in LLM Agent Memory Systems

## Dataset Description

BiTempQA is the first diagnostic benchmark explicitly designed to evaluate bitemporal reasoning — reasoning about **when events occurred** (`event_time`) vs. **when the system learned about them** (`record_time`) — in LLM agent memory systems.

### Dataset Summary

- **308 Chinese QA pairs** across **10 scenario types** and **9 question types** at **3 difficulty levels**
- Every memory entry carries explicit `event_time` and `record_time` annotations
- **56.5% of questions require reasoning about both timestamps simultaneously**
- Three answer formats: multiple choice (43.8%), abstractive (53.0%), boolean (3.2%)

### Supported Tasks

- Bitemporal reasoning question answering
- Memory system evaluation
- Temporal reasoning diagnosis

### Languages

Chinese (zh)

## Dataset Structure

### Data Splits

| Split | Files | QA Pairs |
|-------|-------|----------|
| train | `train.json` | ~240 |
| dev | `dev.json` | ~30 |
| test | `test.json` | ~38 |

### Data Fields

Each QA pair contains:

- `scenario_id`: Scenario identifier
- `question_id`: Unique question identifier
- `question`: Question text (Chinese)
- `answer`: Gold answer
- `answer_type`: "mc" (multiple choice), "abstractive", or "boolean"
- `options`: Multiple choice options (if applicable)
- `difficulty`: "L1" (easy), "L2" (medium), or "L3" (hard)
- `question_type`: One of 9 types (point_in_time, temporal_order, first_recorded, period_query, change_detection, multi_hop_temporal, counterfactual, complex_temporal, version_conflict)
- `requires_event_time`: Whether the question requires event-time reasoning
- `requires_record_time`: Whether the question requires record-time reasoning
- `requires_version_tracking`: Whether version tracking is needed
- `requires_knowledge_retraction`: Whether knowledge retraction is involved

Each scenario contains:

- `memory_writes`: List of memory entries, each with `text`, `event_time`, `record_time`
- `scenario_type`: One of 10 types (entity_attribute_evolution, relationship_evolution, contradictory_information, late_arriving_facts, future_dated_information, entity_identity_resolution, knowledge_retraction, multi_source_information, gradual_accumulation, temporal_ambiguity)

## Additional Resources

- **Code & Paper**: https://github.com/ngyygm/llm-TSSMRBench
- **Scenario Templates**: `scenario_templates/` directory

## Citation

```bibtex
@inproceedings{bitempqa2026,
  title={BiTempQA: A Diagnostic Benchmark for Bitemporal Reasoning in LLM Agent Memory Systems},
  author={Anonymous},
  booktitle={Proceedings of ACL 2026},
  year={2026}
}
```

## License

MIT License
