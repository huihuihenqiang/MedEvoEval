from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from medsim.clinical.interface_normalization import (
    STRICT_ACTION_ALIASES,
    normalize_doctor_payload,
)
from medsim.clinical.llm_io import COMPLETION_AUDIT_KEY, LLMJSONParseError, StructuredLLMClient
from medsim.clinical.prompts import (
    build_consultant_prompt,
    build_consultant_user_prompt,
    build_doctor_prompt,
    build_doctor_user_prompt,
    build_manager_extraction_prompt,
    build_manager_extraction_user_prompt,
    build_manager_learning_prompt,
    build_manager_learning_user_prompt,
    build_manager_rule_cards_prompt,
    build_manager_rule_cards_user_prompt,
    build_mdt_expert_prompt,
    build_mdt_user_prompt,
    build_patient_prompt,
    build_patient_user_prompt,
    should_allow_call_mdt,
    should_strongly_suggest_mdt,
)
from medsim.clinical.schema import (
    DoctorAction,
    DoctorCaseView,
    EvidenceScoreStats,
    ExamView,
    ExpertOpinion,
    FinalClinicalDecision,
    ManagementScoreStats,
    ManagerAssessment,
    ManagerView,
    PatientTurn,
    PatientView,
    TestServiceResponse,
)

NO_THINKING_EXTRA_BODY = {"enable_thinking": False}
THINKING_EXTRA_BODY = {"enable_thinking": True}


class DoctorActionError(RuntimeError):
    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.error_type = str(error_type or "doctor_action_error")
        self.cause = cause
        self.details = dict(details or {})
        super().__init__(message)


def _normalize_compare_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    return re.sub(r"[\s\-_()/,:;，。；：]+", "", lowered)


def _contains_text(options: List[str], targets: List[str]) -> bool:
    normalized_targets = [_normalize_compare_text(item) for item in targets if _normalize_compare_text(item)]
    if not normalized_targets:
        return False
    for option in options:
        normalized_option = _normalize_compare_text(option)
        if not normalized_option:
            continue
        for target in normalized_targets:
            if target in normalized_option or normalized_option in target:
                return True
    return False


def _contains_exact_text(options: List[str], targets: List[str]) -> bool:
    normalized_targets = {_normalize_compare_text(item) for item in targets if _normalize_compare_text(item)}
    if not normalized_targets:
        return False
    for option in options:
        normalized_option = _normalize_compare_text(option)
        if normalized_option and normalized_option in normalized_targets:
            return True
    return False


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def _prefer_chinese_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text if _contains_chinese(text) else fallback


def _coerce_list(value: Any, fallback: List[str]) -> List[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return list(fallback)


def _prefer_chinese_list(value: Any, fallback: List[str]) -> List[str]:
    items = _coerce_list(value, fallback)
    if not items:
        return list(fallback)
    if any(_contains_chinese(item) for item in items):
        return items
    return list(fallback)


def _coerce_confidence(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, (int, float)):
        return float(value)
    normalized = str(value).strip().lower()
    if not normalized:
        return float(fallback)
    label_map = {
        "high": 0.9,
        "medium": 0.6,
        "low": 0.3,
        "very high": 0.95,
        "very low": 0.15,
    }
    if normalized in label_map:
        return label_map[normalized]
    try:
        return float(normalized)
    except ValueError:
        return float(fallback)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _deduplicate(items: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_diagnosis_phrase(text: str) -> str:
    compact = " ".join(str(text or "").strip().split())
    if not compact:
        return ""
    compact = re.sub(r"^(诊断(为|是)?[:：]\s*)", "", compact)
    compact = re.sub(r"^[\-*•\d\.\)\(]+\s*", "", compact)
    compact = re.split(r"[；;、/|｜，,]", compact, maxsplit=1)[0].strip()
    compact = re.sub(r"[。；;，,、]+$", "", compact).strip()
    return compact


def _coerce_single_diagnosis(items: List[str], fallback: str = "当前诊断仍需进一步明确") -> List[str]:
    for item in items:
        phrase = _normalize_diagnosis_phrase(item)
        if phrase:
            return [phrase]
    fallback_text = _normalize_diagnosis_phrase(fallback) or "当前诊断仍需进一步明确"
    return [fallback_text]


def _index_points(prefix: str, points: List[str]) -> Dict[str, str]:
    return {f"{prefix}{idx}": point for idx, point in enumerate(points, start=1)}


def _filter_allowed_ids(raw_items: Any, allowed: Dict[str, str]) -> List[str]:
    if not allowed:
        return []
    allowed_set = set(allowed.keys())
    normalized = []
    for item in _coerce_list(raw_items, []):
        token = str(item).strip().upper()
        if token in allowed_set:
            normalized.append(token)
    return _deduplicate(normalized)


def _match_ids_by_text(text: str, indexed_points: Dict[str, str]) -> List[str]:
    normalized_text = _normalize_compare_text(text)
    if not normalized_text:
        return []
    matched: List[str] = []
    for token, point in indexed_points.items():
        normalized_point = _normalize_compare_text(point)
        if normalized_point and normalized_point in normalized_text:
            matched.append(token)
    return _deduplicate(matched)


def _normalize_test_name(value: str) -> str:
    # Strict matching policy: trim + case-insensitive only.
    return str(value or "").strip().lower()


def _complete_json_strict(
    llm_client: Optional[StructuredLLMClient],
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    fallback_payload: Dict[str, Any],
    trace_label: str,
    request_extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if llm_client is None:
        raise RuntimeError("LLM client is not initialized.")
    return llm_client.complete_json(
        model_name,
        system_prompt,
        user_prompt,
        fallback_payload,
        trace_label=trace_label,
        request_extra_body=request_extra_body,
    )


class PatientAgentNPC:
    def __init__(self, patient_view: PatientView, config, llm_client: Optional[StructuredLLMClient]) -> None:
        self.patient_view = patient_view
        self.config = config
        self.llm_client = llm_client

    def respond(self, question: str, disclosed_facts: List[str]) -> PatientTurn:
        hidden_facts = [fact for fact in self.patient_view.history_facts if fact not in disclosed_facts]
        revealed = hidden_facts[:1] or [self.patient_view.chief_complaint]
        fallback = {
            "utterance": "我现在最明显的不舒服就是这次就诊想说的这些症状。",
        }

        payload = fallback
        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if can_use_llm:
            try:
                payload = _complete_json_strict(
                    self.llm_client,
                    self.config.models.patient,
                    build_patient_prompt(self.patient_view, disclosed_facts),
                    build_patient_user_prompt(question),
                    fallback,
                    trace_label="patient",
                    request_extra_body=NO_THINKING_EXTRA_BODY,
                )
            except Exception:
                payload = fallback

        return PatientTurn(
            utterance=_prefer_chinese_text(payload.get("utterance"), fallback["utterance"]),
            revealed_facts=revealed,
            emotion="平静",
            cooperation="配合",
        )


class TestService:
    def __init__(self, exam_view: ExamView, config, llm_client: Optional[StructuredLLMClient]) -> None:
        del llm_client
        self.exam_view = exam_view
        allowed_catalog = getattr(getattr(config, "clinical_tests", None), "allowed_catalog", None)
        if allowed_catalog:
            catalog_source = list(allowed_catalog)
        else:
            # Backward-compatible fallback for tests that initialize TestService without runtime config.
            catalog_source = list(exam_view.test_catalog or exam_view.tests.keys())
        self.allowed_lookup: Dict[str, str] = {
            _normalize_test_name(item): str(item).strip() for item in catalog_source if str(item).strip()
        }
        self.case_lookup: Dict[str, Tuple[str, str]] = {
            _normalize_test_name(name): (name, str(result))
            for name, result in exam_view.tests.items()
            if str(name or "").strip()
        }

    @staticmethod
    def normalize_order_name(test_name: str) -> str:
        return _normalize_test_name(test_name)

    def _lookup_result(self, test_name: str) -> TestServiceResponse:
        requested_text = str(test_name or "").strip() or "检查项目"
        requested_key = _normalize_test_name(requested_text)
        allowed_name = self.allowed_lookup.get(requested_key)

        if not allowed_name:
            return TestServiceResponse(
                test_name=requested_text,
                returned=False,
                result_text="检查不在允许目录，判定为无效检查请求。",
                evidence_tags=["not_allowed"],
                match_type="not_allowed",
            )

        case_hit = self.case_lookup.get(requested_key)
        if case_hit is None:
            return TestServiceResponse(
                test_name=allowed_name,
                returned=False,
                result_text="当前病例无此检查结果，判定为无效检查。",
                evidence_tags=[allowed_name, "not_in_case"],
                match_type="not_in_case",
            )

        matched_name, matched_result = case_hit
        return TestServiceResponse(
            test_name=matched_name,
            returned=True,
            result_text=matched_result,
            evidence_tags=[matched_name, "released_after_order", "match:exact"],
            match_type="exact",
        )

    def request_test(self, test_name: str) -> TestServiceResponse:
        normalized_name = (test_name or "").strip() or "检查项目"
        return self._lookup_result(normalized_name)


class MDTExpertAgent:
    def __init__(
        self,
        specialty: str,
        doctor_view: DoctorCaseView,
        config,
        llm_client: Optional[StructuredLLMClient],
    ) -> None:
        self.specialty = specialty
        self.doctor_view = doctor_view
        self.config = config
        self.llm_client = llm_client

    def review_case(self, evidence: List[str]) -> ExpertOpinion:
        fallback = {
            "specialty": self.specialty,
            "assessment": f"{self.specialty} 专科建议先补齐最关键的信息，再做最终判断。",
            "supporting_evidence": evidence[:3],
            "recommended_next_step": "优先补齐最可能改变诊断的关键信息。",
            "confidence": 0.72,
        }

        payload = fallback
        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if can_use_llm:
            try:
                payload = _complete_json_strict(
                    self.llm_client,
                    self.config.models.mdt_expert,
                    build_mdt_expert_prompt("当前门诊病例需要进一步鉴别。", self.specialty, evidence),
                    build_mdt_user_prompt(),
                    fallback,
                    trace_label=f"mdt:{self.specialty}",
                    request_extra_body=NO_THINKING_EXTRA_BODY,
                )
            except Exception:
                payload = fallback

        return ExpertOpinion(
            specialty=_prefer_chinese_text(payload.get("specialty"), self.specialty),
            assessment=_prefer_chinese_text(payload.get("assessment"), fallback["assessment"]),
            supporting_evidence=_prefer_chinese_list(payload.get("supporting_evidence"), fallback["supporting_evidence"]),
            recommended_next_step=_prefer_chinese_text(
                payload.get("recommended_next_step"),
                fallback["recommended_next_step"],
            ),
            confidence=_coerce_confidence(payload.get("confidence"), fallback["confidence"]),
        )


class ConsultantAgent:
    def __init__(
        self,
        doctor_view: DoctorCaseView,
        config,
        llm_client: Optional[StructuredLLMClient],
    ) -> None:
        self.doctor_view = doctor_view
        self.config = config
        self.llm_client = llm_client

    def review_case(
        self,
        *,
        turn_index: int,
        question_count: int,
        test_count: int,
        released_results: List[str],
        tests_requested: List[str],
        context_prompt: str,
    ) -> Dict[str, Any]:
        fallback = {
            "advice_action": "ASK",
            "utterance": "建议先补充关键信息后再判断是否终诊。",
            "consult_opinion": "当前证据仍不足以稳健终诊，建议继续针对性追问。",
            "working_diagnoses": ["当前仍需进一步鉴别的诊断方向"],
            "key_evidence_for": released_results[:3],
            "key_evidence_against": [],
            "missing_points": ["症状演变时间线", "关键阴性体征"],
            "recommended_question": ["请补充症状的起病方式与变化规律。"],
            "recommended_diagnosis": [],
            "confidence": 0.65,
            "rationale": ["信息不完整时应优先补齐高价值病史。"],
        }

        payload = fallback
        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if can_use_llm:
            try:
                payload = _complete_json_strict(
                    self.llm_client,
                    self.config.models.mdt_expert,
                    build_consultant_prompt(context_prompt),
                    build_consultant_user_prompt(
                        turn_index=turn_index,
                        question_count=question_count,
                        test_count=test_count,
                        released_results=released_results,
                        tests_requested=tests_requested,
                    ),
                    fallback,
                    trace_label="consultant",
                    request_extra_body=NO_THINKING_EXTRA_BODY,
                )
            except Exception:
                payload = fallback
        return self._normalize_consultant_payload(payload, fallback)

    def _normalize_consultant_payload(self, payload: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
        raw_action = str(payload.get("advice_action", "")).strip().upper()
        advice_action = "FINALIZE" if raw_action == "FINALIZE" else "ASK"
        utterance = _prefer_chinese_text(payload.get("utterance"), fallback["utterance"])
        consult_opinion = _prefer_chinese_text(payload.get("consult_opinion"), fallback["consult_opinion"])
        working_diagnoses = _prefer_chinese_list(payload.get("working_diagnoses"), fallback["working_diagnoses"])
        key_evidence_for = _prefer_chinese_list(payload.get("key_evidence_for"), fallback["key_evidence_for"])
        key_evidence_against = _prefer_chinese_list(payload.get("key_evidence_against"), fallback["key_evidence_against"])
        missing_points = _prefer_chinese_list(payload.get("missing_points"), fallback["missing_points"])
        rationale = _prefer_chinese_list(payload.get("rationale"), fallback["rationale"])
        confidence = _clamp(_coerce_confidence(payload.get("confidence"), fallback["confidence"]), 0.0, 1.0)

        if advice_action == "ASK":
            recommended_question = _prefer_chinese_list(payload.get("recommended_question"), fallback["recommended_question"])
            recommended_question = recommended_question[:2]
            if not recommended_question:
                recommended_question = list(fallback["recommended_question"][:2])
            recommended_diagnosis: List[str] = []
        else:
            raw_diag = _prefer_chinese_list(payload.get("recommended_diagnosis"), fallback["recommended_diagnosis"])
            recommended_diagnosis = _coerce_single_diagnosis(raw_diag or ["当前建议先不终诊"])
            recommended_question = []

        return {
            "advice_action": advice_action,
            "utterance": utterance,
            "consult_opinion": consult_opinion,
            "working_diagnoses": working_diagnoses,
            "key_evidence_for": key_evidence_for,
            "key_evidence_against": key_evidence_against,
            "missing_points": missing_points,
            "recommended_question": recommended_question,
            "recommended_diagnosis": recommended_diagnosis,
            "confidence": confidence,
            "rationale": rationale,
        }


class ManagerAgent:
    def __init__(self, manager_view: ManagerView, config, llm_client: Optional[StructuredLLMClient]) -> None:
        self.manager_view = manager_view
        self.config = config
        self.llm_client = llm_client

    def assess_case(self, visit_summary: str, decision: FinalClinicalDecision) -> ManagerAssessment:
        evidence_key_map = _index_points("E", self.manager_view.ground_truth.evidence_rubric.key_points)
        evidence_error_map = _index_points("EE", self.manager_view.ground_truth.evidence_rubric.error_points_examples)
        management_key_map = _index_points("M", self.manager_view.ground_truth.management_rubric.key_points)
        management_error_map = _index_points("MM", self.manager_view.ground_truth.management_rubric.danger_points_examples)

        extracted = self._extract_point_ids(
            visit_summary=visit_summary,
            decision=decision,
            evidence_key_map=evidence_key_map,
            evidence_error_map=evidence_error_map,
            management_key_map=management_key_map,
            management_error_map=management_error_map,
        )

        matched_evidence_ids = extracted["matched_evidence_points"]
        incorrect_evidence_ids = extracted["incorrect_evidence_points"]
        matched_management_ids = extracted["matched_management_points"]
        incorrect_management_ids = extracted["incorrect_management_points"]

        matched_evidence_points = [evidence_key_map[item] for item in matched_evidence_ids if item in evidence_key_map]
        incorrect_evidence_points = [evidence_error_map[item] for item in incorrect_evidence_ids if item in evidence_error_map]
        matched_management_points = [management_key_map[item] for item in matched_management_ids if item in management_key_map]
        incorrect_management_points = [
            management_error_map[item] for item in incorrect_management_ids if item in management_error_map
        ]

        scoring = self.config.scoring
        me = len(matched_evidence_points)
        he = len(incorrect_evidence_points)
        mt = len(matched_management_points)
        ht = len(incorrect_management_points)
        ke = len(evidence_key_map)
        kt = len(management_key_map)

        evidence_score = max(0.0, ((me - scoring.lambda_evidence * he) / ke) if ke else 0.0)
        plan_score = max(0.0, ((mt - scoring.mu_management * ht) / kt) if kt else 0.0)
        diagnosis_grade, diagnosis_score = self._score_diagnosis_with_context(
            candidate_diagnosis=decision.diagnosis,
            incorrect_management_points=incorrect_management_points,
            matched_management_points=matched_management_points,
            plan_score=plan_score,
        )

        total_score = _clamp(
            (scoring.weight_diag * diagnosis_score)
            + (scoring.weight_evidence * evidence_score)
            + (scoring.weight_plan * plan_score),
            0.0,
            1.0,
        )

        correct_diagnosis = diagnosis_score == 1.0
        effective_treatment = plan_score >= scoring.effective_treatment_threshold
        cured = total_score >= scoring.pass_threshold
        missing_points = [
            point for point in self.manager_view.ground_truth.evidence_rubric.key_points if point not in matched_evidence_points
        ][:3]
        corrected_plan = (
            list(self.manager_view.ground_truth.management_rubric.key_points) or ["根据标准诊断给出规范化后续处理。"]
        )
        feedback_text = self._build_feedback_text(
            diagnosis_grade=diagnosis_grade,
            diagnosis_score=diagnosis_score,
            evidence_score=evidence_score,
            plan_score=plan_score,
            total_score=total_score,
            cured=cured,
        )
        confidence = _clamp(0.4 + 0.6 * total_score, 0.0, 1.0)

        return ManagerAssessment(
            correct_diagnosis=correct_diagnosis,
            effective_treatment=effective_treatment,
            cured=cured,
            feedback_text=feedback_text,
            missing_points=missing_points,
            suggested_questions=[],
            suggested_tests=[],
            corrected_diagnosis=[self.manager_view.ground_truth.diagnosis.display_name],
            corrected_plan=corrected_plan,
            needs_revisit=False,
            confidence=confidence,
            diagnosis_grade=diagnosis_grade,
            diagnosis_score=diagnosis_score,
            evidence_score=evidence_score,
            plan_score=plan_score,
            total_score=total_score,
            evidence_stats=EvidenceScoreStats(matched=me, incorrect=he, total_key_points=ke),
            management_stats=ManagementScoreStats(matched=mt, incorrect=ht, total_key_points=kt),
            matched_evidence_points=matched_evidence_points,
            incorrect_evidence_points=incorrect_evidence_points,
            matched_management_points=matched_management_points,
            incorrect_management_points=incorrect_management_points,
        )

    def _diagnosis_direction(self, candidate_diagnosis: List[str]) -> str:
        if _contains_exact_text(candidate_diagnosis, self.manager_view.diagnosis_aliases):
            return "exact"
        if _contains_text(candidate_diagnosis, self.manager_view.diagnosis_aliases):
            return "mostly_correct"
        if _contains_text(candidate_diagnosis, self.manager_view.diagnosis_partial_aliases):
            return "partially_correct"
        return "wrong_direction"

    def _score_diagnosis_with_context(
        self,
        *,
        candidate_diagnosis: List[str],
        incorrect_management_points: List[str],
        matched_management_points: List[str],
        plan_score: float,
    ) -> Tuple[str, float]:
        direction_grade = self._diagnosis_direction(candidate_diagnosis)
        high_risk_signal = bool(incorrect_management_points)
        major_strategy_deviation = (
            plan_score < float(self.config.scoring.effective_treatment_threshold)
            or (not matched_management_points and high_risk_signal)
        )

        if high_risk_signal and (direction_grade in {"wrong_direction", "partially_correct"} or major_strategy_deviation):
            return "dangerous", 0.0

        if direction_grade == "exact":
            if major_strategy_deviation:
                return "partially_correct", 0.5
            return "exact", 1.0

        if direction_grade == "mostly_correct":
            if major_strategy_deviation:
                return "partially_correct", 0.5
            return "mostly_correct", 0.75

        if direction_grade == "partially_correct":
            if major_strategy_deviation:
                return "wrong_low_risk", 0.25
            return "partially_correct", 0.5

        if high_risk_signal:
            return "dangerous", 0.0
        return "wrong_low_risk", 0.25

    def _extract_point_ids(
        self,
        visit_summary: str,
        decision: FinalClinicalDecision,
        evidence_key_map: Dict[str, str],
        evidence_error_map: Dict[str, str],
        management_key_map: Dict[str, str],
        management_error_map: Dict[str, str],
    ) -> Dict[str, List[str]]:
        fallback_payload = {
            "matched_evidence_points": [],
            "incorrect_evidence_points": [],
            "matched_management_points": [],
            "incorrect_management_points": [],
        }

        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if can_use_llm:
            try:
                payload = _complete_json_strict(
                    self.llm_client,
                    self.config.models.manager,
                    build_manager_extraction_prompt(
                        self.manager_view,
                        decision.diagnosis,
                        decision.evidence,
                        decision.plan,
                        decision.followup,
                        evidence_key_map,
                        evidence_error_map,
                        management_key_map,
                        management_error_map,
                    ),
                    build_manager_extraction_user_prompt(),
                    fallback_payload,
                    trace_label="manager_extraction",
                    request_extra_body=THINKING_EXTRA_BODY,
                )
                return {
                    "matched_evidence_points": _filter_allowed_ids(payload.get("matched_evidence_points"), evidence_key_map),
                    "incorrect_evidence_points": _filter_allowed_ids(
                        payload.get("incorrect_evidence_points"),
                        evidence_error_map,
                    ),
                    "matched_management_points": _filter_allowed_ids(
                        payload.get("matched_management_points"),
                        management_key_map,
                    ),
                    "incorrect_management_points": _filter_allowed_ids(
                        payload.get("incorrect_management_points"),
                        management_error_map,
                    ),
                }
            except Exception:
                pass

        evidence_text = " ".join(decision.evidence + [visit_summary])
        management_text = " ".join(decision.plan + decision.followup + [visit_summary])
        return {
            "matched_evidence_points": _match_ids_by_text(evidence_text, evidence_key_map),
            "incorrect_evidence_points": _match_ids_by_text(evidence_text, evidence_error_map),
            "matched_management_points": _match_ids_by_text(management_text, management_key_map),
            "incorrect_management_points": _match_ids_by_text(management_text, management_error_map),
        }

    def generate_learning_experience(
        self,
        visit_summary: str,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
    ) -> str:
        is_success = bool(assessment.cured)
        fallback = self._build_learning_fallback(visit_summary, decision, assessment, is_success=is_success)
        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if not can_use_llm:
            return fallback
        try:
            payload = _complete_json_strict(
                self.llm_client,
                self.config.models.manager,
                build_manager_learning_prompt(
                    manager_view=self.manager_view,
                    visit_summary=visit_summary,
                    final_diagnosis=decision.diagnosis,
                    final_evidence=decision.evidence,
                    final_plan=decision.plan,
                    final_followup=decision.followup,
                    matched_evidence_points=assessment.matched_evidence_points,
                    matched_management_points=assessment.matched_management_points,
                    incorrect_evidence_points=assessment.incorrect_evidence_points,
                    incorrect_management_points=assessment.incorrect_management_points,
                    diagnosis_grade=assessment.diagnosis_grade,
                    diagnosis_score=assessment.diagnosis_score,
                    evidence_score=assessment.evidence_score,
                    plan_score=assessment.plan_score,
                    total_score=assessment.total_score,
                    is_success=is_success,
                ),
                build_manager_learning_user_prompt(is_success=is_success),
                {"experience": fallback},
                trace_label="manager_learning",
                request_extra_body=THINKING_EXTRA_BODY,
            )
            return self._coerce_learning_experience(payload.get("experience"), fallback)
        except Exception:
            return fallback

    def generate_rule_cards(
        self,
        visit_summary: str,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
    ) -> List[Dict[str, str]]:
        is_success = bool(assessment.cured)
        fallback = self._build_rule_card_fallback(visit_summary, decision, assessment, is_success=is_success)
        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if not can_use_llm:
            return fallback
        try:
            payload = _complete_json_strict(
                self.llm_client,
                self.config.models.manager,
                build_manager_rule_cards_prompt(
                    manager_view=self.manager_view,
                    visit_summary=visit_summary,
                    final_diagnosis=decision.diagnosis,
                    final_evidence=decision.evidence,
                    final_plan=decision.plan,
                    final_followup=decision.followup,
                    matched_evidence_points=assessment.matched_evidence_points,
                    matched_management_points=assessment.matched_management_points,
                    incorrect_evidence_points=assessment.incorrect_evidence_points,
                    incorrect_management_points=assessment.incorrect_management_points,
                    diagnosis_grade=assessment.diagnosis_grade,
                    diagnosis_score=assessment.diagnosis_score,
                    evidence_score=assessment.evidence_score,
                    plan_score=assessment.plan_score,
                    total_score=assessment.total_score,
                    is_success=is_success,
                ),
                build_manager_rule_cards_user_prompt(is_success=is_success),
                {"rule_cards": fallback},
                trace_label="manager_rule_cards",
                request_extra_body=THINKING_EXTRA_BODY,
            )
            return self._coerce_rule_cards(payload.get("rule_cards"), fallback, is_success=is_success)
        except Exception:
            return fallback

    def _coerce_rule_cards(
        self,
        value: Any,
        fallback: List[Dict[str, str]],
        *,
        is_success: bool,
    ) -> List[Dict[str, str]]:
        if not isinstance(value, list):
            return list(fallback)

        parsed: List[Dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            card_type = str(item.get("card_type", "")).strip().lower()
            rule_text = " ".join(str(item.get("rule_text", "")).strip().split())
            if card_type not in {"success", "avoid", "decision"}:
                continue
            if not rule_text or not self._is_valid_rule_text(card_type, rule_text):
                continue
            parsed.append({"card_type": card_type, "rule_text": rule_text})

        expected_order = ["success", "decision"] if is_success else ["avoid", "success"]
        selected: List[Dict[str, str]] = []
        for expected_type in expected_order:
            hit = next((row for row in parsed if row["card_type"] == expected_type), None)
            if hit is None:
                return list(fallback)
            selected.append(hit)
        return selected

    def _is_valid_rule_text(self, card_type: str, rule_text: str) -> bool:
        normalized = str(rule_text or "").upper()
        if card_type == "success":
            return ("IF" in normalized) and ("THEN" in normalized)
        if card_type == "avoid":
            return ("IF" in normalized) and ("AVOID" in normalized)
        if card_type == "decision":
            return ("IF" in normalized) and bool(re.search(r"ACTION\\s*=\\s*(ASK|FINALIZE|CALL_MDT)", normalized))
        return False

    def _build_rule_card_fallback(
        self,
        visit_summary: str,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
        *,
        is_success: bool,
    ) -> List[Dict[str, str]]:
        diagnosis_name = (
            self.manager_view.ground_truth.diagnosis.display_name
            or (decision.diagnosis[0] if decision.diagnosis else "目标诊断")
        )
        evidence_anchor = "、".join((assessment.matched_evidence_points or decision.evidence or [visit_summary or "关键证据"])[:2])
        correction_anchor = "、".join((assessment.incorrect_evidence_points + assessment.incorrect_management_points)[:2])
        if not correction_anchor:
            correction_anchor = "偏离关键证据与主处理策略"

        if is_success:
            return [
                {
                    "card_type": "success",
                    "rule_text": f"IF 出现 {evidence_anchor} THEN 优先收敛到 {diagnosis_name} 并按主处理路径推进。",
                },
                {
                    "card_type": "decision",
                    "rule_text": "IF 关键证据已闭环且高风险征象被排除 ACTION=FINALIZE",
                },
            ]

        corrected_target = "、".join((assessment.corrected_diagnosis or [diagnosis_name])[:1])
        return [
            {
                "card_type": "avoid",
                "rule_text": f"IF 证据仍不完整或存在 {correction_anchor} AVOID 在未排除高风险疾病前过早终诊。",
            },
            {
                "card_type": "success",
                "rule_text": f"IF 抓住 {evidence_anchor} THEN 先补齐关键证据再收敛到 {corrected_target}。",
            },
        ]

    def _coerce_learning_experience(self, value: Any, fallback: str) -> str:
        text = str(value or "").strip()
        if not text:
            return fallback
        required_tokens = ("病例签名：", "决策摘要：", "反思规则：")
        if not all(token in text for token in required_tokens):
            return fallback
        if not _contains_chinese(text):
            return fallback
        compact = " ".join(text.split())
        return compact if compact else fallback

    def _build_learning_fallback(
        self,
        visit_summary: str,
        decision: FinalClinicalDecision,
        assessment: ManagerAssessment,
        *,
        is_success: bool,
    ) -> str:
        diagnosis_name = (
            self.manager_view.ground_truth.diagnosis.display_name
            or (decision.diagnosis[0] if decision.diagnosis else "待明确病型")
        )
        symptom_signature = "、".join((decision.evidence or [visit_summary or "核心症状未充分披露"])[:2])
        key_evidence = "、".join((assessment.matched_evidence_points or decision.evidence or ["关键阳性证据"])[:2])
        efficient_path = "、".join((assessment.matched_management_points or decision.plan or ["先抓关键证据再快速收敛"])[:2])
        convergence_logic = (
            f"围绕{diagnosis_name}先校验证据再单诊断收敛"
            if diagnosis_name
            else "先校验证据再单诊断收敛"
        )
        if is_success:
            return (
                f"病例签名：{diagnosis_name}，核心症状/场景为{symptom_signature}；"
                f"决策摘要：关键证据对为{key_evidence}，最高效路径为{efficient_path}，诊断收敛动作为{convergence_logic}；"
                "反思规则：优先锁定最关键证据并沿最高效路径推进，以单一诊断收敛逻辑完成可复用决策。"
            )

        wrong_points = "、".join((assessment.incorrect_evidence_points + assessment.incorrect_management_points)[:2])
        if not wrong_points:
            wrong_points = "偏离关键证据与处理主线"
        corrected_target = "、".join((assessment.corrected_diagnosis or [diagnosis_name])[:1])
        warning_signals = "、".join((assessment.missing_points or decision.evidence or ["关键警示征象"])[:2])
        return (
            f"病例签名：{diagnosis_name}，核心症状/场景为{symptom_signature}；"
            f"决策摘要：关键证据对为{key_evidence}，最高效路径应为{efficient_path}，诊断收敛动作应围绕{corrected_target}展开；"
            f"反思规则：错在{wrong_points}且原因是未优先抓住决定性证据，正确改法是先按{efficient_path}补齐关键证据再收敛到{corrected_target}，下次遇到{warning_signals}等征象要立即警惕并提前纠偏。"
        )

    def _build_feedback_text(
        self,
        diagnosis_grade: str,
        diagnosis_score: float,
        evidence_score: float,
        plan_score: float,
        total_score: float,
        cured: bool,
    ) -> str:
        diagnosis_name = self.manager_view.ground_truth.diagnosis.display_name or "标准诊断"
        diagnosis_reflections = {
            "exact": "诊断与标准完全一致。",
            "mostly_correct": "诊断方向基本正确，存在轻微偏差。",
            "partially_correct": "诊断部分正确，仍有中等偏差。",
            "wrong_low_risk": "诊断方向明显偏差，但当前风险相对有限。",
            "dangerous": "诊断或处置存在高风险偏差，需立即纠偏。",
        }
        diagnosis_reflection = diagnosis_reflections.get(diagnosis_grade, "诊断等级未识别，建议复核。")
        if cured:
            return (
                "本轮评估通过。"
                f"诊断评级={diagnosis_grade}，诊断分={diagnosis_score:.2f}，证据分={evidence_score:.2f}，"
                f"处理分={plan_score:.2f}，总分={total_score:.2f}。"
                f" {diagnosis_reflection}"
            )
        return (
            "本轮评估未通过。"
            f"标准诊断应收敛到 {diagnosis_name}。"
            f"诊断评级={diagnosis_grade}，诊断分={diagnosis_score:.2f}，证据分={evidence_score:.2f}，"
            f"处理分={plan_score:.2f}，总分={total_score:.2f}。"
            f" {diagnosis_reflection}"
        )


class AttendingDoctorAgent:
    OPENING_QUESTION = "您好，请问您今天因为什么来就诊？现在最不舒服的地方是什么？"

    def __init__(self, doctor_view: DoctorCaseView, config, llm_client: Optional[StructuredLLMClient]) -> None:
        self.doctor_view = doctor_view
        self.config = config
        self.llm_client = llm_client

    def next_action(
        self,
        turn_index: int,
        context_packet: Dict[str, Any],
        released_results: List[str],
        tests_requested: List[str],
        mdt_already_used: bool,
        question_count: int,
        test_count: int,
    ) -> DoctorAction:
        if self._is_opening_turn(context_packet):
            return self._opening_action()

        can_use_llm = self.llm_client is not None and self.llm_client.is_available()
        if not can_use_llm:
            raise DoctorActionError(
                "llm_unavailable",
                "Doctor decision requires LLM but client is unavailable.",
            )

        try:
            allow_request_test = test_count < int(self.config.workflow.max_tests_per_visit)
            allow_call_mdt = bool(getattr(self.config.mdt, "enabled", False))
            allow_call_mdt_now = (
                should_allow_call_mdt(question_count, test_count, mdt_already_used) if allow_call_mdt else False
            )
            strongly_suggest_mdt = (
                should_strongly_suggest_mdt(question_count, test_count, mdt_already_used) if allow_call_mdt else False
            )
            payload = _complete_json_strict(
                self.llm_client,
                self.config.models.attending,
                build_doctor_prompt(
                    self.doctor_view,
                    context_packet["rendered_prompt"],
                    allow_request_test=allow_request_test,
                    allow_call_mdt=allow_call_mdt,
                ),
                build_doctor_user_prompt(
                    turn_index=turn_index,
                    question_count=question_count,
                    test_count=test_count,
                    released_results=released_results,
                    tests_requested=tests_requested,
                    allow_request_test=allow_request_test,
                    allow_call_mdt=allow_call_mdt,
                    allow_call_mdt_now=allow_call_mdt_now,
                    strongly_suggest_mdt=strongly_suggest_mdt,
                    mdt_called=mdt_already_used,
                ),
                {},
                trace_label="doctor",
                request_extra_body=NO_THINKING_EXTRA_BODY,
            )
        except LLMJSONParseError as exc:
            raise DoctorActionError(
                "json_parse_error",
                str(exc),
                cause=exc,
                details={"normalization_enabled": True, "repair_applied": False},
            ) from exc
        except Exception as exc:
            raise DoctorActionError(
                "llm_inference_error",
                f"Doctor action generation failed: {exc}",
                cause=exc,
            ) from exc

        normalization_enabled = bool(
            getattr(getattr(self.config, "interface_normalization", None), "enabled", False)
        )
        completion_audit = payload.pop(COMPLETION_AUDIT_KEY, {}) if normalization_enabled else {}
        interface_audit: Dict[str, Any] = {}
        if normalization_enabled:
            payload, normalization_audit = normalize_doctor_payload(
                payload,
                allowed_catalog=self.doctor_view.available_test_catalog,
                alias_to_category=self.doctor_view.test_alias_to_category,
            )
            interface_audit = normalization_audit.as_dict()
            interface_audit["completion"] = completion_audit
            if normalization_audit.unresolved:
                error_type = normalization_audit.raw_error_type or "interface_normalization_failed"
                raise DoctorActionError(
                    error_type,
                    "Interface error could not be repaired unambiguously without changing protected content.",
                    details=interface_audit,
                )

        action_text = str(payload.get("action", "")).strip()
        if not action_text:
            raise DoctorActionError("missing_action", "Doctor payload missing required field: action.")

        utterance = str(payload.get("utterance", "") or "").strip()
        question = str(payload.get("question", "") or "").strip()
        if not utterance and not question:
            raise DoctorActionError("missing_utterance", "Doctor payload missing utterance/question text.")

        action = DoctorAction(
            action=action_text,
            utterance=_prefer_chinese_text(utterance, utterance or question),
            question=_prefer_chinese_text(question, question or utterance),
            requested_history=_coerce_list(payload.get("requested_history"), []),
            test_name=str(payload.get("test_name", "") or "").strip(),
            rationale=_prefer_chinese_list(payload.get("rationale"), _coerce_list(payload.get("rationale"), [])),
            working_diagnoses=_coerce_list(payload.get("working_diagnoses"), []),
            needs_mdt=bool(payload.get("needs_mdt", False)),
            diagnosis=_coerce_list(payload.get("diagnosis"), []),
            evidence=_coerce_list(payload.get("evidence"), []),
            plan=_prefer_chinese_list(payload.get("plan"), _coerce_list(payload.get("plan"), [])),
            followup=_prefer_chinese_list(payload.get("followup"), _coerce_list(payload.get("followup"), [])),
            interface_normalization_audit=interface_audit,
        )

        normalized_action = str(action.action or "").strip().upper()
        action.action = STRICT_ACTION_ALIASES.get(
            normalized_action,
            STRICT_ACTION_ALIASES.get(str(action.action or "").strip(), normalized_action),
        )

        allowed_actions = {"ASK", "FINALIZE"}
        if allow_request_test:
            allowed_actions.add("REQUEST_TEST")
        if allow_call_mdt:
            allowed_actions.add("CALL_MDT")
        if action.action not in allowed_actions:
            raise DoctorActionError("invalid_action", f"Unsupported doctor action: {action.action}")
        if action.action == "REQUEST_TEST" and not action.test_name:
            raise DoctorActionError("missing_test_name", "REQUEST_TEST action requires non-empty test_name.")
        if action.action == "REQUEST_TEST":
            allowed_lookup = {
                _normalize_test_name(item): str(item).strip()
                for item in self.doctor_view.available_test_catalog
                if str(item or "").strip()
            }
            requested_key = _normalize_test_name(action.test_name)
            if requested_key not in allowed_lookup:
                raise DoctorActionError(
                    "invalid_test_name",
                    f"REQUEST_TEST test_name must be from allowed catalog, got: {action.test_name}",
                    details=interface_audit,
                )
            # Canonicalize to configured catalog spelling to keep logs/results consistent.
            action.test_name = allowed_lookup[requested_key]
        if action.action == "FINALIZE":
            action.diagnosis = _coerce_single_diagnosis(action.diagnosis)
            if not action.diagnosis or not action.evidence or not action.plan:
                raise DoctorActionError(
                    "missing_finalize_fields",
                    "FINALIZE action requires diagnosis/evidence/plan fields.",
                )
            if not action.followup:
                action.followup = ["建议近期复诊并持续观察病情变化。"]
        if action.action == "ASK" and not action.question:
            action.question = action.utterance
        if action.action != "ASK":
            action.question = ""

        if not action.utterance:
            if action.action == "ASK":
                action.utterance = action.question
            elif action.action == "REQUEST_TEST":
                action.utterance = f"我建议先做 {action.test_name} 以进一步明确诊断。"
            elif action.action == "CALL_MDT":
                action.utterance = "当前信息存在不确定性，我申请顾问医生复核。"
            else:
                action.utterance = "我现在基于现有证据给出最终判断。"
        if not action.rationale:
            action.rationale = ["模型返回未提供 rationale。"]
        return action

    def build_forced_final_decision(
        self,
        dialogue_evidence: List[str],
        released_results: List[str],
        manager_feedback: Optional[ManagerAssessment],
    ) -> FinalClinicalDecision:
        if manager_feedback and manager_feedback.corrected_diagnosis:
            diagnosis = _coerce_single_diagnosis(list(manager_feedback.corrected_diagnosis))
            plan = list(manager_feedback.corrected_plan or ["根据修正后的诊断给出针对性的保守处理。"])
        else:
            diagnosis = _coerce_single_diagnosis(["当前诊断仍需进一步明确"])
            plan = ["先给予对症处理。", "交代危险信号，如症状加重及时复诊或急诊就医。"]
        evidence = (dialogue_evidence + released_results)[:6] or ["当前已披露信息有限"]
        return FinalClinicalDecision(
            diagnosis=diagnosis,
            evidence=evidence,
            plan=plan,
            followup=["如症状加重应立即就医。", "按医嘱完成复查或复诊。"],
            patient_instruction="请按医嘱处理，如出现危险信号及时到急诊就诊。",
        )

    def _suggest_test(self, tests_requested: List[str]) -> str:
        normalized_requested = {_normalize_test_name(item) for item in tests_requested}
        preferred_patterns = [
            "生命体征",
            "神经系统查体",
            "脑神经查体",
            "肌力查体",
            "肌电图",
            "心电图",
            "血常规",
            "尿常规",
            "ct",
            "mri",
            "xr",
            "超声",
        ]
        for pattern in preferred_patterns:
            for candidate in self.doctor_view.available_test_catalog:
                normalized = _normalize_test_name(candidate)
                if normalized in normalized_requested:
                    continue
                if _normalize_test_name(pattern) in normalized:
                    return candidate
        for candidate in self.doctor_view.available_test_catalog:
            normalized = _normalize_test_name(candidate)
            if normalized not in normalized_requested:
                return candidate
        return ""

    def _is_opening_turn(self, context_packet: Dict[str, Any]) -> bool:
        recent_lines = context_packet.get("recent_dialogue_window", [])
        return not any(getattr(line, "phase", "") == "consultation" for line in recent_lines)

    def _opening_action(self) -> DoctorAction:
        return DoctorAction(
            action="ASK",
            utterance=self.OPENING_QUESTION,
            question=self.OPENING_QUESTION,
            requested_history=["主诉", "当前最不舒服的症状"],
            rationale=["患者刚进入门诊，必须先询问主诉，不能预设具体病情。"],
        )






