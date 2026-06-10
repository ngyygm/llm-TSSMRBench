"""Simple KG baseline：基于规则抽取的轻量知识图谱记忆后端。"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .base import MemorySystem, QueryResult


class SimpleKGBaseline(MemorySystem):
    """简单知识图谱基线。

    该后端不调用 LLM、不使用向量检索，只用中英文规则从 memory_write 文本中抽取
    (subject, predicate, object) 三元组，并在查询时按问题词项匹配三元组和原始证据。
    """

    def __init__(
        self,
        name: str = "Simple KG",
        top_k: int = 5,
        expose_timestamps_in_context: bool = False,
    ):
        super().__init__(name)
        self.top_k = top_k
        # 中文注释：主实验默认不把 event_time / record_time 显式拼进 KG 行。
        # 只有补充实验显式开启 expose_timestamps_in_context 时，才返回带时间标签的 KG 事实文本。
        self.expose_timestamps_in_context = expose_timestamps_in_context
        # 中文注释：实体属性版本表，保留 event_time 和 record_time 便于输出证据。
        self.entities: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
        # 中文注释：关系版本表，主要用于诊断 KG 是否真正抽取到边。
        self.relations: Dict[tuple, List[Dict]] = defaultdict(list)
        # 中文注释：三元组是 Simple KG 的核心符号记忆单元。
        self.triples: List[Dict] = []
        # 中文注释：原始证据只在与问题有词项重叠时参与检索，不能作为无条件 fallback。
        self.texts: List[str] = []
        self.text_metadata: List[Dict] = []
        self.text_entities: List[Set[str]] = []
        self.text_terms: List[Set[str]] = []
        self._write_counter: int = 0

    def remember(
        self,
        text: str,
    ) -> str:
        self._write_counter += 1
        write_id = f"kg_{self._write_counter}"

        self.texts.append(text)
        self.text_metadata.append({})
        self.text_terms.append(self._terms(text))
        self._extract_and_update(text, "", "")
        return write_id

    @staticmethod
    def _clean_span(value: str) -> str:
        """清理规则抽取出的实体或属性值边界。"""
        value = re.sub(r"\s+", " ", value or "").strip()
        value = re.sub(r"^On\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^From\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s+until\s+[A-Za-z]+\s+\d{1,2},\s+\d{4},\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^(?:On|From)\s+[^,]+,\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^\d{4},\s*", "", value)
        value = re.sub(r"^(?:changed|set|confirmed|recorded|renamed)\s+to\s+", "", value, flags=re.IGNORECASE)
        value = re.sub(r"^\d{4}年\d{1,2}月\d{1,2}日[，,]\s*", "", value)
        value = re.sub(r"^(?:据|根据|官方确认|一份历史文件显示|另一来源称)[^，,。]*[，,]\s*", "", value)
        value = value.strip(" \t\r\n，。;:；：“”‘’\"'()（）")

        # 中文注释：少数摘要会把来源提示放在实体前，过长时取最后一个分句主体。
        if "，" in value and len(value) > 30:
            value = value.split("，")[-1].strip()
        if "," in value and len(value) > 40:
            value = value.split(",")[-1].strip()
        return value

    @staticmethod
    def _terms(text: str) -> Set[str]:
        """生成中英文混合文本的轻量检索词项。"""
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

        # 中文注释：中文没有空格，保留连续中文片段，并加入 bigram 以提高短问题命中率。
        for chunk in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            terms.add(chunk)
            terms.update(chunk[i:i + 2] for i in range(max(len(chunk) - 1, 0)))
        return terms

    def _add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        event_time: str,
        record_time: str,
        source_text: str,
        mentioned: Set[str],
    ) -> None:
        """保存一个版本化三元组，同时维护实体属性表和关系表。"""
        subject = self._clean_span(subject)
        obj = self._clean_span(obj)
        if not subject or not obj or subject == obj:
            return
        if len(subject) > 100 or len(obj) > 160:
            return

        triple = {
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "event_time": event_time,
            "record_time": record_time,
            "source": source_text,
        }
        self.triples.append(triple)
        mentioned.update({subject, obj})

        self.entities[subject][predicate].append(
            {
                "value": obj,
                "event_time": event_time,
                "record_time": record_time,
                "source": source_text,
            }
        )
        self.relations[(subject, obj)].append(
            {
                "relation": predicate,
                "event_time": event_time,
                "record_time": record_time,
                "source": source_text,
            }
        )

    def _format_kg_line(self, triple: Dict) -> str:
        """构造返回给答案生成器的 KG 事实文本。"""
        base_line = f"KG: {triple['subject']} | {triple['predicate']} | {triple['object']}"
        if not self.expose_timestamps_in_context:
            return base_line
        return (
            f"{base_line} | event_time={triple['event_time']} "
            f"| record_time={triple['record_time']}"
        )

    def _extract_and_update(self, text: str, event_time: str, record_time: str) -> None:
        """基于中英文模板抽取三元组，并保留双时间戳。"""
        mentioned: Set[str] = set()
        normalized = re.sub(r"\s+", " ", text).strip()

        # 中文注释：模板覆盖当前 Wikidata 摘要中最常见的名称、成员、任职、地点和状态变化。
        patterns: List[Tuple[str, str]] = [
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
                self._add_triple(
                    match.group("s"),
                    predicate,
                    match.group("o"),
                    event_time,
                    record_time,
                    text,
                    mentioned,
                )

        self.text_entities.append(mentioned)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        start = time.time()

        effective_top_k = top_k if top_k is not None else self.top_k
        q_terms = self._terms(question)
        scored: List[Tuple[float, int, str, str]] = []

        for idx, triple in enumerate(self.triples):
            subject = triple["subject"]
            obj = triple["object"]
            triple_text = f"{subject} {triple['predicate']} {obj} {triple['source']}"
            overlap = len(q_terms & self._terms(triple_text))
            exact_boost = 0
            if subject and subject.lower() in question.lower():
                exact_boost += 5
            if obj and obj.lower() in question.lower():
                exact_boost += 3
            score = overlap + exact_boost
            if score > 0:
                kg_line = self._format_kg_line(triple)
                scored.append((float(score), idx, triple["source"], kg_line))

        # 中文注释：规则未覆盖时，允许原始证据按词项命中参与排序；无命中则返回空上下文。
        for idx, text in enumerate(self.texts):
            overlap = len(q_terms & self.text_terms[idx])
            if overlap > 0:
                scored.append((float(overlap) * 0.5, len(self.triples) + idx, text, ""))

        scored.sort(key=lambda item: (-item[0], item[1]))

        context_parts: List[str] = []
        relevant_texts: List[str] = []
        seen: Set[str] = set()
        for _score, _idx, source_text, kg_line in scored:
            if len(relevant_texts) >= effective_top_k:
                break
            if source_text in seen:
                continue
            seen.add(source_text)
            relevant_texts.append(source_text)
            context_parts.append(source_text)
            if kg_line:
                context_parts.append(kg_line)

        context = "\n".join(context_parts)
        confidence = min(scored[0][0] / 10.0, 1.0) if scored else 0.0
        return QueryResult(
            answer=context,
            retrieved_context=context,
            retrieved_facts=relevant_texts,
            confidence=confidence,
            latency_ms=(time.time() - start) * 1000,
            metadata={
                "num_triples": len(self.triples),
                "num_results": len(relevant_texts),
                "timestamps_exposed": self.expose_timestamps_in_context,
            },
        )

    def reset(self) -> None:
        self.entities.clear()
        self.relations.clear()
        self.triples.clear()
        self.texts.clear()
        self.text_metadata.clear()
        self.text_entities.clear()
        self.text_terms.clear()
        self._write_counter = 0
