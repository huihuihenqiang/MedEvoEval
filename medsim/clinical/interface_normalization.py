"""Semantics-preserving normalization for the doctor's action interface.

This module deliberately does not invent clinical content.  It may only:

* move an existing value from a known field alias to its canonical field;
* canonicalize an action alias;
* map a specific test name to one unambiguous category in the benchmark's
  orderable-test catalog.

Question text, utterances, diagnoses, evidence, plans, and rationales are
fingerprinted before and after normalization.  A repair is rejected if any of
those values change.  Ambiguous test mappings remain errors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


STRICT_ACTION_ALIASES = {
    "QUESTION": "ASK",
    "继续问": "ASK",
    "TEST": "REQUEST_TEST",
    "去检查": "REQUEST_TEST",
    "DECIDE": "FINALIZE",
    "做出判断": "FINALIZE",
}

NORMALIZED_ACTION_ALIASES = {
    **STRICT_ACTION_ALIASES,
    "ASK_QUESTION": "ASK",
    "ASK QUESTION": "ASK",
    "INQUIRE": "ASK",
    "询问": "ASK",
    "提问": "ASK",
    "ORDER_TEST": "REQUEST_TEST",
    "ORDER TEST": "REQUEST_TEST",
    "REQUEST_EXAM": "REQUEST_TEST",
    "REQUEST EXAM": "REQUEST_TEST",
    "EXAM": "REQUEST_TEST",
    "CHECK": "REQUEST_TEST",
    "检查": "REQUEST_TEST",
    "申请检查": "REQUEST_TEST",
    "DIAGNOSE": "FINALIZE",
    "FINAL": "FINALIZE",
    "SUBMIT": "FINALIZE",
    "终诊": "FINALIZE",
    "CALL_CONSULTANT": "CALL_MDT",
    "CALL CONSULTANT": "CALL_MDT",
    "CONSULT": "CALL_MDT",
    "会诊": "CALL_MDT",
}

FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "action": ("action_type", "actionType", "type", "intent", "动作"),
    "test_name": ("test", "testName", "exam_name", "examination", "检查项目"),
    "utterance": ("response", "message", "text", "reply", "回复"),
    "question": ("question_text", "query", "ask", "问题"),
}

# These fields carry clinical or conversational substance.  Their canonical
# values must be byte-for-byte identical across an interface repair.
PROTECTED_FIELDS = (
    "utterance",
    "question",
    "requested_history",
    "rationale",
    "working_diagnoses",
    "needs_mdt",
    "diagnosis",
    "evidence",
    "plan",
    "followup",
)


def normalize_token(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[\s\-_()/,:;，。；：]+", "", text)


def _canonical_value(payload: Mapping[str, Any], field_name: str) -> Any:
    if field_name in payload:
        return payload[field_name]
    for alias in FIELD_ALIASES.get(field_name, ()):
        if alias in payload:
            return payload[alias]
    return None


def protected_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = {field: _canonical_value(payload, field) for field in PROTECTED_FIELDS}
    rendered = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def strict_action(value: Any) -> str:
    raw = str(value or "").strip()
    upper = raw.upper()
    return STRICT_ACTION_ALIASES.get(upper, STRICT_ACTION_ALIASES.get(raw, upper))


def normalized_action(value: Any) -> str:
    raw = str(value or "").strip()
    upper = raw.upper()
    return NORMALIZED_ACTION_ALIASES.get(upper, NORMALIZED_ACTION_ALIASES.get(raw, upper))


def _catalog_lookup(catalog: Iterable[str]) -> Dict[str, str]:
    return {
        normalize_token(item): str(item).strip()
        for item in catalog
        if str(item or "").strip()
    }


# Fallback rules cover valid medical test phrases that do not appear verbatim in
# the dataset-wide alias registry.  Each phrase must hit exactly one category.
_CATEGORY_PATTERNS: Dict[str, Tuple[str, ...]] = {
    "体格检查类": (
        r"体格检查",
        r"查体",
        r"触诊",
        r"叩诊",
        r"听诊",
        r"指检",
        r"眼底检查",
        r"妇科检查",
        r"神经系统检查",
        r"生命体征",
        r"活动度",
        r"压痛",
        r"视诊",
        r"血压测",
        r"血氧饱和度",
        r"皮肤检查",
    ),
    "实验室检查类": (
        r"血常规",
        r"尿常规",
        r"粪便",
        r"便常规",
        r"隐血",
        r"肝功能",
        r"肾功能",
        r"甲状腺功能",
        r"电解质",
        r"血糖",
        r"血脂",
        r"尿酸",
        r"生化",
        r"凝血",
        r"妊娠试验",
        r"激素",
        r"二聚体",
        r"(真菌|显微|涂片)镜检",
        r"培养",
        r"抗体",
        r"抗原",
        r"精液",
        r"实验室",
        r"\blab(?:oratory)?\b",
    ),
    "影像学检查类": (
        r"影像",
        r"超声",
        r"彩超",
        r"磁共振",
        r"核磁",
        r"(?<![a-z])mri(?![a-z])",
        r"(?<![a-z])ct(?![a-z])",
        r"(?<![a-z])pet(?![a-z])",
        r"x[\s-]?线",
        r"x光",
        r"b超",
        r"x[\s-]?ray",
        r"平片",
        r"造影",
    ),
    "功能检查类": (
        r"心电",
        r"脑电",
        r"肌电",
        r"肺功能",
        r"睡眠监测",
        r"动态血压",
        r"holter",
        r"功能检查",
    ),
    "内镜检查类": (
        r"内镜",
        r"胃镜",
        r"肠镜",
        r"支气管镜",
        r"膀胱镜",
        r"宫腔镜",
        r"腹腔镜",
    ),
    "病理/活检类": (
        r"病理",
        r"活检",
        r"组织学",
        r"细胞学",
        r"\bbiopsy\b",
        r"\bpatholog",
    ),
    "分子/基因检测类": (
        r"基因",
        r"分子",
        r"测序",
        r"核酸",
        r"\bpcr\b",
    ),
    "治疗/处置类": (
        r"治疗",
        r"处置",
        r"给药",
        r"输液",
        r"手术",
        r"复位",
        r"固定",
    ),
    "会诊/转诊类": (
        r"会诊",
        r"转诊",
        r"\bconsult",
        r"\breferr",
    ),
}


def map_test_to_category(
    test_name: Any,
    allowed_catalog: Sequence[str],
    alias_to_category: Optional[Mapping[str, str]] = None,
) -> Tuple[Optional[str], List[str], str]:
    """Return (canonical category, candidates, method).

    A result is returned only when the mapping is unique.  The dataset registry
    is checked before keyword rules; exact allowed-category spelling is always
    accepted without counting as a repair.
    """

    text = str(test_name or "").strip()
    key = normalize_token(text)
    allowed = _catalog_lookup(allowed_catalog)
    if key in allowed:
        return allowed[key], [allowed[key]], "catalog_exact"

    registry = alias_to_category or {}
    registry_lookup = {normalize_token(alias): str(category).strip() for alias, category in registry.items()}
    category = registry_lookup.get(key)
    if category and normalize_token(category) in allowed:
        canonical = allowed[normalize_token(category)]
        return canonical, [canonical], "dataset_alias"

    candidates: List[str] = []
    lowered = text.lower()
    for category_name, patterns in _CATEGORY_PATTERNS.items():
        canonical = allowed.get(normalize_token(category_name))
        if not canonical:
            continue
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            candidates.append(canonical)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) == 1:
        return candidates[0], candidates, "taxonomy_keyword"
    return None, candidates, "ambiguous" if candidates else "unmapped"


@dataclass
class InterfaceNormalizationAudit:
    raw_error_type: str = ""
    repair_applied: bool = False
    repair_methods: List[str] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)
    original_test_name: str = ""
    normalized_test_name: str = ""
    candidate_categories: List[str] = field(default_factory=list)
    semantic_preserved: bool = True
    protected_before_sha256: str = ""
    protected_after_sha256: str = ""
    unresolved: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "raw_error_type": self.raw_error_type,
            "repair_applied": self.repair_applied,
            "repair_methods": list(self.repair_methods),
            "changes": list(self.changes),
            "original_test_name": self.original_test_name,
            "normalized_test_name": self.normalized_test_name,
            "candidate_categories": list(self.candidate_categories),
            "semantic_preserved": self.semantic_preserved,
            "protected_before_sha256": self.protected_before_sha256,
            "protected_after_sha256": self.protected_after_sha256,
            "unresolved": self.unresolved,
        }


def _raw_validation_error(payload: Mapping[str, Any], allowed_catalog: Sequence[str]) -> str:
    action_value = payload.get("action")
    if not str(action_value or "").strip():
        return "missing_action"
    if not str(payload.get("utterance", "") or "").strip() and not str(payload.get("question", "") or "").strip():
        return "missing_utterance"
    action = strict_action(action_value)
    if action not in {"ASK", "REQUEST_TEST", "FINALIZE", "CALL_MDT"}:
        return "invalid_action"
    if action == "REQUEST_TEST":
        test_name = str(payload.get("test_name", "") or "").strip()
        if not test_name:
            return "missing_test_name"
        if normalize_token(test_name) not in _catalog_lookup(allowed_catalog):
            return "invalid_test_name"
    if action == "FINALIZE" and not all(payload.get(field) for field in ("diagnosis", "evidence", "plan")):
        return "missing_finalize_fields"
    return ""


def normalize_doctor_payload(
    payload: Mapping[str, Any],
    *,
    allowed_catalog: Sequence[str],
    alias_to_category: Optional[Mapping[str, str]] = None,
) -> Tuple[Dict[str, Any], InterfaceNormalizationAudit]:
    """Normalize interface-only fields and return a complete audit record."""

    original = copy.deepcopy(dict(payload))
    normalized = copy.deepcopy(original)
    audit = InterfaceNormalizationAudit(
        raw_error_type=_raw_validation_error(original, allowed_catalog),
        protected_before_sha256=protected_fingerprint(original),
    )

    # Move values, never synthesize them.
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical in normalized:
            continue
        for alias in aliases:
            if alias in normalized:
                normalized[canonical] = normalized[alias]
                audit.changes.append(f"field:{alias}->{canonical}")
                audit.repair_methods.append("field_alias")
                break

    raw_action = normalized.get("action")
    action = normalized_action(raw_action)
    if action and action != str(raw_action or "").strip().upper():
        # Existing strict aliases are baseline behavior, not an added repair.
        if strict_action(raw_action) != action:
            audit.changes.append(f"action:{raw_action}->{action}")
            audit.repair_methods.append("action_alias")
        normalized["action"] = action
    elif action:
        normalized["action"] = action

    if action == "REQUEST_TEST":
        original_test = str(normalized.get("test_name", "") or "").strip()
        audit.original_test_name = original_test
        mapped, candidates, method = map_test_to_category(
            original_test,
            allowed_catalog,
            alias_to_category,
        )
        audit.candidate_categories = candidates
        if mapped is not None:
            audit.normalized_test_name = mapped
            if normalize_token(original_test) != normalize_token(mapped):
                normalized["test_name"] = mapped
                audit.changes.append(f"test_name:{original_test}->{mapped}")
                audit.repair_methods.append(method)
        elif audit.raw_error_type in {"invalid_test_name", "missing_test_name"}:
            audit.unresolved = True

    audit.repair_methods = list(dict.fromkeys(audit.repair_methods))
    audit.protected_after_sha256 = protected_fingerprint(normalized)
    audit.semantic_preserved = audit.protected_before_sha256 == audit.protected_after_sha256
    if not audit.semantic_preserved:
        audit.unresolved = True
        audit.repair_applied = False
        return original, audit

    if audit.raw_error_type and _raw_validation_error(normalized, allowed_catalog):
        audit.unresolved = True
    audit.repair_applied = bool(audit.changes) and bool(audit.raw_error_type) and not audit.unresolved
    return normalized, audit
