from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from medsim.clinical.schema import EpisodeResult, FinalClinicalDecision, ManagerAssessment, VisitRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _safe_jsonl_read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    decoder = json.JSONDecoder()
    # Use utf-8-sig and explicit BOM strip to tolerate BOM-prefixed JSONL files.
    with open(path, "r", encoding="utf-8-sig") as file:
        for line_no, line in enumerate(file, start=1):
            row = str(line or "").strip().lstrip("\ufeff")
            if not row:
                continue
            cursor = 0
            while cursor < len(row):
                while cursor < len(row) and row[cursor].isspace():
                    cursor += 1
                if cursor >= len(row):
                    break
                try:
                    payload, end = decoder.raw_decode(row, cursor)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSONL row ignored path=%s line=%s", path, line_no)
                    break
                if isinstance(payload, dict):
                    items.append(payload)
                cursor = end
    return items

def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _rank_entries(entries: List[Dict[str, Any]], query_text: str, top_k: int) -> List[Dict[str, Any]]:
    if not entries:
        return []
    corpus = [str(entry.get("search_text", "")).strip() for entry in entries]
    if not any(corpus):
        return entries[:top_k]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(corpus + [query_text or ""])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]
    return [entries[int(index)] for index in ranked_indices]


def _normalize_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in entries:
        experience = str(item.get("experience", "")).strip()
        if not experience:
            continue
        row = dict(item)
        row["experience"] = experience
        if not str(row.get("search_text", "")).strip():
            row["search_text"] = " ".join([experience, str(row.get("case_id", ""))]).strip()
        normalized.append(row)
    return normalized


def _normalize_rule_card_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    allowed_types = {"success", "avoid", "decision"}
    for item in entries:
        card_type = str(item.get("card_type", "")).strip().lower()
        rule_text = str(item.get("rule_text", "")).strip()
        if card_type not in allowed_types or not rule_text:
            continue
        row = dict(item)
        row["card_type"] = card_type
        row["rule_text"] = rule_text
        if not str(row.get("search_text", "")).strip():
            row["search_text"] = " ".join(
                [
                    card_type,
                    rule_text,
                    str(row.get("case_id", "")),
                    str(row.get("outcome", "")),
                ]
            ).strip()
        normalized.append(row)
    return normalized


class ClinicalMemoryManager:
    """Manage success/correction KB entries with LLM-generated experience text."""

    def __init__(self, namespace: str, kb_root_dir: Optional[str] = None) -> None:
        self.namespace = str(namespace or "default").strip() or "default"
        if str(kb_root_dir or "").strip():
            root = Path(str(kb_root_dir))
            if not root.is_absolute():
                root = (PROJECT_ROOT / root).resolve()
            self.root = root
        else:
            self.root = PROJECT_ROOT / "output" / "knowledge" / self.namespace
        self.success_kb_path = self.root / "success_kb.jsonl"
        self.correction_kb_path = self.root / "correction_kb.jsonl"
        self.rule_cards_kb_path = self.root / "rule_cards_kb.jsonl"

    def _load_success_entries(self) -> List[Dict[str, Any]]:
        return _normalize_entries(_safe_jsonl_read(self.success_kb_path))

    def _load_correction_entries(self) -> List[Dict[str, Any]]:
        return _normalize_entries(_safe_jsonl_read(self.correction_kb_path))

    def _load_rule_card_entries(self) -> List[Dict[str, Any]]:
        return _normalize_rule_card_entries(_safe_jsonl_read(self.rule_cards_kb_path))

    def retrieve_success_examples(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        return _rank_entries(self._load_success_entries(), query_text, top_k)

    def retrieve_correction_examples(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        return _rank_entries(self._load_correction_entries(), query_text, top_k)

    def retrieve_examples(self, query_text: str, top_k: int, include_failure_reflection: bool) -> List[Dict[str, Any]]:
        entries = self.retrieve_success_examples(query_text, top_k)
        if include_failure_reflection:
            entries.extend(self.retrieve_correction_examples(query_text, top_k))
        return entries[:top_k]

    def retrieve_rule_cards(
        self,
        query_text: str,
        top_k: int,
        allowed_types: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        normalized_top_k = max(0, min(int(top_k), 3))
        if normalized_top_k == 0:
            return []
        entries = self._load_rule_card_entries()
        if allowed_types:
            allowed = {str(item).strip().lower() for item in allowed_types if str(item).strip()}
            entries = [row for row in entries if str(row.get("card_type", "")).strip().lower() in allowed]
        return _rank_entries(entries, query_text, normalized_top_k)[:normalized_top_k]

    def write_success_entry(
        self,
        episode: EpisodeResult,
        visit: VisitRecord,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
        experience: str,
    ) -> Dict[str, Any]:
        del assessment
        normalized_exp = str(experience or "").strip()
        payload = {
            "case_id": episode.case.case_id,
            "visit_index": visit.visit_index,
            "experience": normalized_exp,
            "search_text": " ".join(
                [
                    normalized_exp,
                    episode.case.gold_diagnosis,
                    " ".join(decision.diagnosis),
                    " ".join(decision.evidence[:3]),
                ]
            ).strip(),
        }
        _append_jsonl(self.success_kb_path, payload)
        return payload

    def write_correction_entry(
        self,
        episode: EpisodeResult,
        visit: VisitRecord,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
        experience: str,
    ) -> Dict[str, Any]:
        normalized_exp = str(experience or "").strip()
        payload = {
            "case_id": episode.case.case_id,
            "visit_index": visit.visit_index,
            "experience": normalized_exp,
            "search_text": " ".join(
                [
                    normalized_exp,
                    episode.case.gold_diagnosis,
                    " ".join(decision.diagnosis),
                    " ".join(assessment.corrected_diagnosis),
                    " ".join(decision.evidence[:3]),
                ]
            ).strip(),
        }
        _append_jsonl(self.correction_kb_path, payload)
        return payload

    def write_rule_card_entries(
        self,
        episode: EpisodeResult,
        visit: VisitRecord,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
        rule_cards: List[Dict[str, str]],
        *,
        outcome: str,
    ) -> List[Dict[str, Any]]:
        del assessment
        normalized_cards = _normalize_rule_card_entries(rule_cards)
        writes: List[Dict[str, Any]] = []
        for card in normalized_cards:
            payload = {
                "case_id": episode.case.case_id,
                "visit_index": visit.visit_index,
                "card_type": card["card_type"],
                "rule_text": card["rule_text"],
                "outcome": str(outcome or "").strip().lower() or "unknown",
                "search_text": " ".join(
                    [
                        card["card_type"],
                        card["rule_text"],
                        episode.case.gold_diagnosis,
                        " ".join(decision.diagnosis),
                        " ".join(decision.evidence[:3]),
                        str(outcome or ""),
                    ]
                ).strip(),
            }
            _append_jsonl(self.rule_cards_kb_path, payload)
            writes.append(payload)
        return writes

    def write_learning_trace(self, episode: EpisodeResult) -> Dict[str, Any]:
        rule_card_writes = sum(1 for item in episode.kb_writes if str(item.type).startswith("rule_card"))
        return {
            "namespace": self.namespace,
            "success_kb_path": str(self.success_kb_path),
            "correction_kb_path": str(self.correction_kb_path),
            "rule_cards_kb_path": str(self.rule_cards_kb_path),
            "kb_writes": len(episode.kb_writes),
            "rule_card_writes": rule_card_writes,
        }




