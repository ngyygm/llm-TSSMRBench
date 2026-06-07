"""Schemas for the independent state-version benchmark pipeline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

Domain = Literal["github_evolution", "narrative_evolution"]
LanguageCode = Literal["en", "zh"]
ProgressLabel = Literal["planned", "active", "resolved", "invalidated"]
PerspectiveLabel = Literal["contemporaneous", "retrospective"]
RelationLabel = Literal["introduces", "continues", "revises", "cancels", "explains"]
SalienceLabel = Literal["core", "distractor"]
CompetitionStrength = Literal["low", "medium", "high"]
LexicalOverlapBand = Literal["low", "medium", "high"]
DifficultyLevel = Literal["low", "high"]
QuestionFamily = Literal["single_version", "multi_version"]
Answerability = Literal["answerable"]
AnswerFormat = Literal["multiple_choice", "boolean", "abstractive"]
StatusValue = Literal["pending", "ready", "generated", "reviewed"]


class SourcePointer(BaseModel):
    """Pointer back to the raw source used for audit."""

    artifact_type: str
    artifact_ref: str
    span_hint: str


class SourceBundleItem(BaseModel):
    """One summarized raw artifact that feeds state-chain generation."""

    artifact_type: str
    artifact_ref: str
    title: Optional[str] = None
    time_hint: Optional[str] = None
    summary: str

    @field_validator("artifact_type", "artifact_ref", "summary")
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source bundle fields must not be blank")
        return value


class SourceBundleRecord(BaseModel):
    """One source-bundle record aligned to one pending chain-construction task."""

    sample_id: str
    state_chain_id: str
    domain: Domain
    language: LanguageCode
    focus_event: str
    source_title: str
    bundle_summary: Optional[str] = None
    source_bundle_items: List[SourceBundleItem] = Field(min_length=1)
    notes: List[str] = Field(default_factory=list)
    source_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "state_chain_id", "focus_event", "source_title")
    @classmethod
    def id_like_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required source-bundle fields must not be blank")
        return value


class RawGithubArtifactRecord(BaseModel):
    """One normalized raw GitHub artifact before source-bundle assembly."""

    repo: str
    focus_event: str
    artifact_type: str
    artifact_ref: str
    title: Optional[str] = None
    time_hint: Optional[str] = None
    summary: Optional[str] = None
    raw_text: Optional[str] = None
    source_url: Optional[str] = None
    split: Optional[str] = None
    bundle_key: Optional[str] = None
    sample_id: Optional[str] = None
    state_chain_id: Optional[str] = None
    bundle_summary: Optional[str] = None
    artifact_order: Optional[int] = None
    notes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("repo", "focus_event", "artifact_type", "artifact_ref")
    @classmethod
    def artifact_required_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("raw artifact required fields must not be blank")
        return value

    @model_validator(mode="after")
    def validate_artifact_content(self) -> "RawGithubArtifactRecord":
        summary = (self.summary or "").strip()
        raw_text = (self.raw_text or "").strip()
        if not summary and not raw_text:
            raise ValueError("raw github artifact must provide at least one of summary or raw_text")
        if self.artifact_order is not None and self.artifact_order < 0:
            raise ValueError("artifact_order cannot be negative")
        return self


class ChainProfile(BaseModel):
    """Sample-level chain profile used for audit and distribution tracking."""

    node_count: int = Field(gt=0)
    competition_strength: CompetitionStrength
    lexical_overlap_band: LexicalOverlapBand


class StateChainNode(BaseModel):
    """One state node in a state-version chain."""

    node_id: str
    surface_order: int = Field(ge=1)
    text: str
    progress_label: ProgressLabel
    perspective_label: PerspectiveLabel
    relation_label: RelationLabel
    salience_label: SalienceLabel
    supersedes: List[str] = Field(default_factory=list)
    depends_on: List[str] = Field(default_factory=list)
    source_pointer: SourcePointer

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class StateChainSample(BaseModel):
    """A full state-version chain sample."""

    sample_id: str
    state_chain_id: str
    domain: Domain
    language: LanguageCode
    focus_event: str
    chain_summary: str
    source_kind: Domain
    source_title: str
    chain_profile: ChainProfile
    chain_nodes: List[StateChainNode] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_sample(self) -> "StateChainSample":
        if self.source_kind != self.domain:
            raise ValueError("source_kind must match domain")
        if self.chain_profile.node_count != len(self.chain_nodes):
            raise ValueError("chain_profile.node_count must equal len(chain_nodes)")
        return self


class ChoiceOption(BaseModel):
    """One structured answer option."""

    option_id: str
    text: str

    @field_validator("text")
    @classmethod
    def option_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("option text must not be blank")
        return value


class StateQuestion(BaseModel):
    """One benchmark question anchored to a frozen state chain."""

    question_id: str
    state_chain_id: str
    difficulty_level: DifficultyLevel
    question_family: Optional[QuestionFamily] = None
    answerability: Answerability
    answer_format: AnswerFormat
    query_text: str
    options: Optional[List[ChoiceOption]] = None
    correct_option_id: Optional[str] = None
    expected_answer: str
    gold_node_ids: List[str] = Field(default_factory=list)
    adversarial_node_ids: List[str] = Field(default_factory=list)
    oracle_context_node_ids: List[str] = Field(default_factory=list)
    dynamic_top_k: Optional[int] = None
    reasoning_chain: str | List[str] = Field(default_factory=list)

    @field_validator("query_text", "expected_answer")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text fields must not be blank")
        return value

    @model_validator(mode="after")
    def validate_question(self) -> "StateQuestion":
        if isinstance(self.reasoning_chain, str):
            self.reasoning_chain = [self.reasoning_chain.strip()] if self.reasoning_chain.strip() else []

        if self.reasoning_chain and not 2 <= len(self.reasoning_chain) <= 4:
            raise ValueError("answerable questions must use 2 to 4 short reasoning_chain steps")

        if self.answer_format == "abstractive":
            if self.options is not None or self.correct_option_id is not None:
                raise ValueError("abstractive questions must set options and correct_option_id to null")
        else:
            if not self.options:
                raise ValueError("structured questions must provide options")
            if self.correct_option_id is None:
                raise ValueError("structured questions must provide correct_option_id")

        if self.question_family is None:
            self.question_family = "multi_version" if len(self.gold_node_ids) > 1 else "single_version"

        if self.dynamic_top_k is None:
            self.dynamic_top_k = math.ceil(1.5 * len(self.gold_node_ids))

        return self


class CanonicalAbstentions(BaseModel):
    """Canonical insufficiency outputs used for exact validation."""

    multiple_choice: str
    boolean: str
    abstractive: str


class QuestionCountConfig(BaseModel):
    """Question-count targets shared by all phases."""

    min_per_chain: int = Field(ge=1)
    max_per_chain: int = Field(ge=1)
    target_per_chain: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "QuestionCountConfig":
        if self.min_per_chain > self.max_per_chain:
            raise ValueError("min_per_chain cannot be larger than max_per_chain")
        if not self.min_per_chain <= self.target_per_chain <= self.max_per_chain:
            raise ValueError("target_per_chain must fall within [min_per_chain, max_per_chain]")
        return self


class HardLimitConfig(BaseModel):
    """Hard filtering limits used by the validator."""

    min_nodes_by_domain: Dict[Domain, int]
    max_nodes_by_domain: Dict[Domain, int]
    min_text_units_by_language: Dict[LanguageCode, int]
    distribution_tolerance: float = Field(default=0.05, ge=0.0, le=1.0)


class DistributionTargets(BaseModel):
    """Dataset-level distribution targets."""

    difficulty: Dict[str, float]
    answerability: Dict[str, float]
    answer_format: Dict[str, float]
    question_family: Dict[str, float]

    @staticmethod
    def _validate_distribution(name: str, payload: Dict[str, float]) -> None:
        if not payload:
            raise ValueError(f"{name} must not be empty")
        total = sum(payload.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{name} must sum to 1.0, got {total}")

    @model_validator(mode="after")
    def validate_distributions(self) -> "DistributionTargets":
        self._validate_distribution("difficulty", self.difficulty)
        self._validate_distribution("answerability", self.answerability)
        self._validate_distribution("answer_format", self.answer_format)
        self._validate_distribution("question_family", self.question_family)
        return self


class PromptBundle(BaseModel):
    """Prompt paths used by the construction pipeline."""

    chain_generation: Dict[Domain, str]
    chain_review: Dict[Domain, str]
    qa_generation: Dict[Domain, str]
    qa_review: str


class PhaseSpec(BaseModel):
    """One construction phase, such as smoke or formal."""

    total_chains: int = Field(gt=0)
    domain_counts: Dict[Domain, int]
    splits: Dict[str, Dict[Domain, int]]
    target_questions_per_chain: Optional[int] = Field(default=None, ge=1)
    recommended_question_plan: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_phase(self) -> "PhaseSpec":
        if sum(self.domain_counts.values()) != self.total_chains:
            raise ValueError("sum(domain_counts.values()) must equal total_chains")

        split_totals_by_domain = {domain: 0 for domain in self.domain_counts}
        split_total = 0
        for split_counts in self.splits.values():
            split_total += sum(split_counts.values())
            for domain, count in split_counts.items():
                split_totals_by_domain[domain] = split_totals_by_domain.get(domain, 0) + count

        if split_total != self.total_chains:
            raise ValueError("sum of split counts must equal total_chains")
        if split_totals_by_domain != self.domain_counts:
            raise ValueError("split domain counts must sum back to domain_counts")

        if self.recommended_question_plan:
            plan_total = sum(self.recommended_question_plan.values())
            if self.target_questions_per_chain is not None and plan_total != self.target_questions_per_chain:
                raise ValueError("recommended_question_plan must sum to target_questions_per_chain")

        return self


class DatasetBuildConfig(BaseModel):
    """Top-level build configuration for one dataset family."""

    dataset_name: str
    language: LanguageCode
    dataset_root: str
    question_count: QuestionCountConfig
    hard_limits: HardLimitConfig
    distribution_targets: DistributionTargets
    canonical_abstentions: CanonicalAbstentions
    prompts: PromptBundle
    phases: Dict[str, PhaseSpec]

    def resolve_dataset_root(self, repo_root: Path) -> Path:
        return (repo_root / self.dataset_root).resolve()


class BuildStatus(BaseModel):
    """Simple progress tracking for a manifest entry."""

    source_bundle: StatusValue = "pending"
    state_chain: StatusValue = "pending"
    questions: StatusValue = "pending"
    review: StatusValue = "pending"


class BuildTask(BaseModel):
    """One pending chain-construction task."""

    sample_id: str
    state_chain_id: str
    phase: str
    split: str
    domain: Domain
    language: LanguageCode
    target_question_count_range: Dict[str, int]
    recommended_question_count: int = Field(ge=1)
    recommended_question_plan: Dict[str, int] = Field(default_factory=dict)
    prompt_bundle: Dict[str, str]
    status: BuildStatus = Field(default_factory=BuildStatus)
    notes: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_build_task(self) -> "BuildTask":
        expected_keys = {"min", "max"}
        if set(self.target_question_count_range) != expected_keys:
            raise ValueError("target_question_count_range must contain exactly {'min', 'max'}")
        if self.target_question_count_range["min"] > self.target_question_count_range["max"]:
            raise ValueError("target_question_count_range has invalid bounds")
        return self
