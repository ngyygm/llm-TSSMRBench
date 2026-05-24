"""Simple KG retriever: rule-based entity/relation extraction with versioned triples.

No external dependencies or LLM calls. Uses regex patterns for Chinese and English
text to extract (subject, predicate, object) triples, then matches queries by term overlap.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .base import DatabaseEntry, Retriever


class SimpleKGRetriever(Retriever):
    """Rule-based knowledge graph retriever.

    Extracts triples via regex, stores versioned entity attributes and relations,
    and retrieves by term overlap scoring.
    """

    def __init__(self):
        self._doc_ids: list[int] = []
        self._doc_texts: list[str] = []
        self._doc_terms: list[set[str]] = []
        self._doc_lower: list[str] = []
        self._triples_by_doc: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._triple_terms: dict[int, set[str]] = {}  # triple_id -> terms
        self._entities: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
        self._relations: dict[tuple, list[dict]] = defaultdict(list)
        self._triple_counter = 0
        self._term_to_docs: dict[str, set[int]] = defaultdict(set)  # inverted index

    def add_documents(self, docs: list[DatabaseEntry]) -> None:
        self._doc_ids = [doc.doc_id for doc in docs]
        self._doc_texts = [doc.text for doc in docs]
        self._doc_terms = [self._terms(doc.text) for doc in docs]
        self._doc_lower = [doc.text.lower() for doc in docs]

        # Build inverted index for fast candidate lookup
        for idx in range(len(docs)):
            for term in self._doc_terms[idx]:
                self._term_to_docs[term].add(idx)

        # Only extract triples from narrative docs (GitHub code files won't match)
        for idx, doc in enumerate(docs):
            if doc.source == "narrative":
                self._extract_and_update(doc.text, idx)
        print(f"  Simple KG: {len(docs)} docs, {self._triple_counter} triples extracted")

    def find(self, query: str, top_k: int = 10) -> list[int]:
        q_terms = self._terms(query)
        if not q_terms:
            return []

        query_lower = query.lower()

        # Use inverted index to find candidate docs (those with any term overlap)
        candidate_docs: set[int] = set()
        for term in q_terms:
            candidate_docs.update(self._term_to_docs.get(term, set()))

        scored: list[tuple[float, int]] = []
        for idx in candidate_docs:
            score = 0.0

            # Triple-based scoring (pre-computed terms)
            for triple in self._triples_by_doc[idx]:
                tid = triple["_tid"]
                t_terms = self._triple_terms[tid]
                overlap = len(q_terms & t_terms)
                subject = triple["subject"]
                obj = triple["object"]
                exact_boost = 0
                if subject and subject.lower() in query_lower:
                    exact_boost += 5
                if obj and obj.lower() in query_lower:
                    exact_boost += 3
                score = max(score, overlap + exact_boost)

            # Raw text overlap fallback
            text_overlap = len(q_terms & self._doc_terms[idx])
            score = max(score, text_overlap * 0.5)

            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [self._doc_ids[idx] for _, idx in scored[:top_k]]

    # ─── Internal helpers ───

    @property
    def _doc_triples(self) -> dict[int, list[dict[str, Any]]]:
        return self._triples_by_doc

    @staticmethod
    def _terms(text: str) -> set[str]:
        text = text or ""
        lowered = text.lower()
        stopwords = {
            "the", "a", "an", "of", "to", "in", "on", "from", "until", "and",
            "was", "were", "is", "are", "be", "been", "as", "by", "for", "his",
            "her", "their", "official", "name", "changed", "recorded", "confirmed",
            "when", "what", "which", "did", "does", "with", "this", "that",
        }
        terms = {
            token
            for token in re.findall(r"[a-zA-Z][a-zA-Z0-9'.-]*|\d{3,4}", lowered)
            if token not in stopwords and len(token) > 1
        }
        for chunk in re.findall(r"[一-鿿]{2,}", text):
            terms.add(chunk)
            terms.update(chunk[i:i + 2] for i in range(max(len(chunk) - 1, 0)))
        return terms

    @staticmethod
    def _clean_span(value: str) -> str:
        value = re.sub(r"\s+", " ", value or "").strip()
        value = re.sub(r"^On\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^From\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s+until\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(?:On|From)\s+[^,]+,\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^\d{4},\s*", "", value)
        value = re.sub(r"^(?:changed|set|confirmed|recorded|renamed)\s+to\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^\d{4}年\d{1,2}月\d{1,2}日[，,]\s*", "", value)
        value = re.sub(r"^(?:据|根据|官方确认|一份历史文件显示|另一来源称)[^，,。]*[，,]\s*", "", value)
        value = value.strip(" \t\r\n，。;:；：""''\"'()（）")
        if "，" in value and len(value) > 30:
            value = value.split("，")[-1].strip()
        if "," in value and len(value) > 40:
            value = value.split(",")[-1].strip()
        return value

    def _extract_and_update(self, text: str, doc_idx: int) -> None:
        normalized = re.sub(r"\s+", " ", text).strip()

        patterns = [
            (r"(?:On\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s+)?the official name of (?P<s>.+?) was (?:changed|set|confirmed|recorded) to (?P<o>.+?)(?:,\s+as\b|\.|$)", "official_name"),
            (r"(?:On\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s+)?the official name of (?P<s>.+?) was (?!(?:changed|set|confirmed|recorded|renamed)\b)(?P<o>.+?)(?:,\s+as\b|\.|$)", "official_name"),
            (r"(?:From\s+[^,]+,\s+until\s+[^,]+,\s+)?(?P<s>.+?)'s official name was (?P<o>.+?)(?:\.|$)", "official_name"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) was officially named (?P<o>.+?)(?:\.|$)", "official_name"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) was renamed to (?P<o>.+?)(?:\.|$)", "official_name"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) became a member of (?P<o>.+?)(?: on [^.]+)?(?:\.|$)", "member_of"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) (?:joined|rejoined) (?P<o>.+?)(?: on [^.]+)?(?:\.|$)", "member_of"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) (?:left|departed from) (?P<o>.+?)(?: on [^.]+)?(?:\.|$)", "former_member_of"),
            (r"(?P<s>.+?) began working in (?P<o>.+?)(?: in \d{4})?(?:\.|$)", "work_location"),
            (r"(?P<s>.+?) continued (?:his |her |their )?work in (?P<o>.+?)(?: in \d{4})?(?:\.|$)", "work_location"),
            (r"(?P<s>.+?)'s work in (?P<o>.+?) ended", "former_work_location"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) was employed by (?P<o>.+?)(?:\.|$)", "employer"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) became an employee of (?P<o>.+?)(?:\.|$)", "employer"),
            (r"(?:On\s+[^,]+,\s+)?(?P<s>.+?) (?:became|was appointed as|was elected as) (?P<o>.+?)(?:\.|$)", "position_or_role"),
            (r"(?P<s>[^，。；]{2,60}?)的官方名称(?:为|是|被改为|被确认为|变更为|更名为|改名为|记录为|确定为)\s*(?P<o>[^，。；]+)", "official_name"),
            (r"(?P<s>[^，。；]{2,60}?)的名称(?:为|是|被改为|被确认为|变更为|更名为|改名为|记录为|确定为)\s*(?P<o>[^，。；]+)", "official_name"),
            (r"(?P<s>[^，。；]{2,60}?)(?:于|在)?\d{4}年[^，。；]*?(?:开始为|加入|重新加入|再次成为)(?P<o>[^，。；]+?)(?:效力|成员|的成员|。|，|$)", "member_of"),
            (r"(?P<s>[^，。；]{2,60}?)(?:结束了|结束|离开|退出)[^，。；]*?(?:在|为)?(?P<o>[^，。；]+?)(?:的效力|的成员身份|效力|成员身份|。|，|$)", "former_member_of"),
            (r"(?P<s>[^，。；]{2,60}?)(?:转投|转会至|转至)(?P<o>[^，。；]+)", "member_of"),
            (r"(?P<s>[^，。；]{2,60}?)(?:担任|就职于|任职于|受雇于)(?P<o>[^，。；]+)", "position_or_employer"),
            (r"(?P<s>[^，。；]{2,60}?)(?:开始在|在)(?P<o>[^，。；]{2,60}?)(?:工作|任职)", "work_location"),
            (r"(?P<s>[^，。；]{2,60}?)(?:被任命为|当选为|成为)(?P<o>[^，。；]+)", "position_or_role"),
        ]

        for pattern, predicate in patterns:
            for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
                subject = self._clean_span(match.group("s"))
                obj = self._clean_span(match.group("o"))
                if not subject or not obj or subject == obj:
                    continue
                if len(subject) > 100 or len(obj) > 160:
                    continue

                triple_text = f"{subject} {predicate} {obj} {text}"
                tid = self._triple_counter
                self._triple_counter += 1

                self._triple_terms[tid] = self._terms(triple_text)
                self._triples_by_doc[doc_idx].append({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "_tid": tid,
                })
                self._entities[subject][predicate].append({"value": obj, "source": text})
                self._relations[(subject, obj)].append({"relation": predicate, "source": text})
