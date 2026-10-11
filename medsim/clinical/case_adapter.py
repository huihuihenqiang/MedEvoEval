from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from medsim.clinical.schema import (
    CaseSource,
    ClinicalCase,
    DiagnosisGroundTruth,
    DoctorCaseView,
    EvidenceRubric,
    ExamView,
    GroundTruth,
    HistoryOfPresentIllness,
    ManagementRubric,
    ManagerView,
    NeurologicalExam,
    PatientDemographics,
    PatientView,
    PhysicalExam,
    ReviewOfSystems,
    SocialHistory,
    TestResults,
    VitalSigns,
)
from medsim.clinical.experiment4 import extract_dataset_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE = PROJECT_ROOT / "datasets" / "_medqa.jsonl"
EXAM_CATEGORY_NAMES = [
    "体格检查类",
    "实验室检查类",
    "影像学检查类",
    "功能检查类",
    "内镜检查类",
    "病理/活检类",
    "分子/基因检测类",
    "治疗/处置类",
    "会诊/转诊类",
]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _clean_label(value: str) -> str:
    return _clean_text(value).replace("_", " ")


def _normalize_catalog_name(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_category_exam_payload(payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    keys = set(payload.keys())
    if "physical_exam" in keys or "test_results" in keys:
        return False
    return any(key in keys for key in EXAM_CATEGORY_NAMES)


def _render_exam_value(value: Any) -> str:
    if isinstance(value, dict):
        parts: List[str] = []
        for key, child in value.items():
            rendered = _render_exam_value(child)
            if rendered:
                parts.append(f"{_clean_label(str(key))}：{rendered}")
        return "；".join(parts)
    if isinstance(value, list):
        items = [_clean_text(item) for item in value if _clean_text(item)]
        return "；".join(items)
    return _clean_text(value)


def _summary_text_from_map(values: Dict[str, str]) -> str:
    items = [f"{key}：{val}" for key, val in values.items() if _clean_text(key) and _clean_text(val)]
    return "；".join(items)


def _flatten_category_exam_payload(payload: Dict[str, Any]) -> Dict[str, str]:
    results: Dict[str, str] = {}
    for category in EXAM_CATEGORY_NAMES:
        bucket = payload.get(category, {}) or {}
        if not isinstance(bucket, dict):
            continue
        per_category: Dict[str, str] = {}
        for exam_name, exam_value in bucket.items():
            key = _clean_text(exam_name)
            rendered = _render_exam_value(exam_value)
            if not key or not rendered:
                continue
            per_category[key] = rendered
            results[key] = rendered
        summary = _summary_text_from_map(per_category)
        if summary:
            results[category] = summary
    return results


def _build_global_test_alias_to_category(
    raw_cases: List[Dict[str, Any]],
    allowed_lookup: Dict[str, str],
) -> Dict[str, str]:
    """Build a dataset-wide, case-independent specific-test registry.

    The interface adapter receives the same registry for every scenario, so a
    successful mapping cannot reveal whether the current hidden case happens to
    contain that test or result.  Aliases observed under multiple categories are
    intentionally excluded as ambiguous.
    """

    candidates: Dict[str, Dict[str, Any]] = {}
    for raw_case in raw_cases:
        exam_payload = raw_case.get("exam_view", {}) or {}
        if not _is_category_exam_payload(exam_payload):
            continue
        for raw_category, bucket in exam_payload.items():
            category = allowed_lookup.get(_normalize_catalog_name(raw_category))
            if not category or not isinstance(bucket, dict):
                continue
            for raw_alias in bucket:
                alias = _clean_text(raw_alias)
                key = _normalize_catalog_name(alias)
                if not key:
                    continue
                entry = candidates.setdefault(key, {"alias": alias, "categories": set()})
                entry["categories"].add(category)
    return {
        str(entry["alias"]): next(iter(entry["categories"]))
        for entry in candidates.values()
        if len(entry["categories"]) == 1
    }


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else None


def _unique_items(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        text = _clean_text(item)
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _flatten_patient_facts(patient_view: PatientView) -> List[str]:
    facts: List[str] = []
    if patient_view.demographics.age is not None:
        facts.append(f"年龄：{patient_view.demographics.age}岁")
    if patient_view.demographics.sex:
        facts.append(f"性别：{patient_view.demographics.sex}")
    if patient_view.demographics.occupation:
        facts.append(f"职业：{patient_view.demographics.occupation}")
    if patient_view.chief_complaint:
        facts.append(f"主诉：{patient_view.chief_complaint}")
    if patient_view.history_of_present_illness.duration:
        facts.append(f"病程：{patient_view.history_of_present_illness.duration}")
    facts.extend(f"症状：{item}" for item in patient_view.history_of_present_illness.symptoms)
    facts.extend(f"病情特点：{item}" for item in patient_view.history_of_present_illness.progression_features)
    if patient_view.past_medical_history:
        facts.append(f"既往史：{patient_view.past_medical_history}")
    if patient_view.social_history.smoking:
        facts.append(f"吸烟史：{patient_view.social_history.smoking}")
    if patient_view.social_history.alcohol:
        facts.append(f"饮酒史：{patient_view.social_history.alcohol}")
    facts.extend(f"否认：{item}" for item in patient_view.review_of_systems.negative_findings)
    return _unique_items(facts)


def _flatten_result_bucket(section: Any, prefix: str = "") -> Dict[str, str]:
    results: Dict[str, str] = {}
    if isinstance(section, dict):
        scalar_parts: List[str] = []
        for key, value in section.items():
            label = _clean_label(key)
            if isinstance(value, dict):
                child_prefix = label if not prefix else f"{prefix} / {label}"
                results.update(_flatten_result_bucket(value, child_prefix))
                continue
            if isinstance(value, list):
                joined = "；".join(_clean_text(item) for item in value if _clean_text(item))
                if joined:
                    child_prefix = label if not prefix else f"{prefix} / {label}"
                    results[child_prefix] = joined
                continue
            text = _clean_text(value)
            if text:
                scalar_parts.append(f"{label}：{text}")
        if prefix and scalar_parts:
            results.setdefault(prefix, "；".join(scalar_parts))
        elif not prefix:
            for item in scalar_parts:
                label, text = item.split("：", 1)
                results[label] = text
        return results

    text = _clean_text(section)
    if text and prefix:
        results[prefix] = text
    return results


def _build_physical_exam_map(physical_exam: PhysicalExam) -> Dict[str, str]:
    results: Dict[str, str] = {}
    vital_parts: List[str] = []
    if physical_exam.vital_signs.temperature:
        results["体温"] = physical_exam.vital_signs.temperature
        vital_parts.append(f"体温：{physical_exam.vital_signs.temperature}")
    if physical_exam.vital_signs.blood_pressure:
        results["血压"] = physical_exam.vital_signs.blood_pressure
        vital_parts.append(f"血压：{physical_exam.vital_signs.blood_pressure}")
    if physical_exam.vital_signs.heart_rate:
        results["心率"] = physical_exam.vital_signs.heart_rate
        vital_parts.append(f"心率：{physical_exam.vital_signs.heart_rate}")
    if physical_exam.vital_signs.respiratory_rate:
        results["呼吸频率"] = physical_exam.vital_signs.respiratory_rate
        vital_parts.append(f"呼吸频率：{physical_exam.vital_signs.respiratory_rate}")
    if vital_parts:
        results["生命体征"] = "；".join(vital_parts)

    neuro_parts: List[str] = []
    if physical_exam.neurological_exam.cranial_nerves:
        text = "；".join(physical_exam.neurological_exam.cranial_nerves)
        results["脑神经查体"] = text
        neuro_parts.append(f"脑神经：{text}")
    if physical_exam.neurological_exam.motor_strength:
        text = "；".join(physical_exam.neurological_exam.motor_strength)
        results["肌力查体"] = text
        neuro_parts.append(f"肌力：{text}")
    if physical_exam.neurological_exam.reflexes:
        results["反射"] = physical_exam.neurological_exam.reflexes
        neuro_parts.append(f"反射：{physical_exam.neurological_exam.reflexes}")
    if physical_exam.neurological_exam.sensation:
        results["感觉"] = physical_exam.neurological_exam.sensation
        neuro_parts.append(f"感觉及其他查体：{physical_exam.neurological_exam.sensation}")
    if neuro_parts:
        results["神经系统查体"] = "；".join(neuro_parts)

    return results


def _flatten_exam_view(exam_view: ExamView) -> Dict[str, str]:
    physical_map = _build_physical_exam_map(exam_view.physical_exam)
    blood_map = _flatten_result_bucket(exam_view.test_results.blood_tests)
    emg_map = _flatten_result_bucket(exam_view.test_results.electromyography)
    imaging_map = _flatten_result_bucket(exam_view.test_results.imaging)

    results: Dict[str, str] = {}
    results.update(physical_map)
    results.update(blood_map)
    results.update(emg_map)
    results.update(imaging_map)

    physical_summary = _summary_text_from_map(physical_map)
    if physical_summary:
        results["体格检查类"] = physical_summary
    lab_summary = _summary_text_from_map(blood_map)
    if lab_summary:
        results["实验室检查类"] = lab_summary
    function_summary = _summary_text_from_map(emg_map)
    if function_summary:
        results["功能检查类"] = function_summary
    imaging_summary = _summary_text_from_map(imaging_map)
    if imaging_summary:
        results["影像学检查类"] = imaging_summary

    return {key: value for key, value in results.items() if _clean_text(value)}


def _infer_triage_acuity(problem_text: str, history_facts: Iterable[str]) -> str:
    combined = " ".join([problem_text, *history_facts]).lower()
    urgent_terms = [
        "胸痛",
        "呼吸困难",
        "意识丧失",
        "突发",
        "卒中",
        "晕厥",
        "高热",
        "抽搐",
        "weakness",
        "loss of consciousness",
        "syncope",
    ]
    return "urgent" if any(term in combined for term in urgent_terms) else "routine"


def _build_onset_context(patient_view: PatientView) -> str:
    parts: List[str] = []
    if patient_view.history_of_present_illness.duration:
        parts.append(f"病程{patient_view.history_of_present_illness.duration}")
    if patient_view.history_of_present_illness.symptoms:
        parts.append("，".join(patient_view.history_of_present_illness.symptoms[:2]))
    if patient_view.history_of_present_illness.progression_features:
        parts.append("，".join(patient_view.history_of_present_illness.progression_features[:1]))
    return "；".join(part for part in parts if part) or patient_view.chief_complaint


def _build_patient_view(payload: Dict[str, Any]) -> PatientView:
    demographics_payload = payload.get("demographics", {}) or {}
    hpi_payload = payload.get("history_of_present_illness", {}) or {}
    social_payload = payload.get("social_history", {}) or {}
    ros_payload = payload.get("review_of_systems", {}) or {}
    patient_view = PatientView(
        demographics=PatientDemographics(
            age=_coerce_int(demographics_payload.get("age")),
            sex=_clean_text(demographics_payload.get("sex")),
            occupation=_clean_text(demographics_payload.get("occupation")),
        ),
        chief_complaint=_clean_text(payload.get("chief_complaint")),
        history_of_present_illness=HistoryOfPresentIllness(
            duration=_clean_text(hpi_payload.get("duration")),
            symptoms=_unique_items(hpi_payload.get("symptoms", []) or []),
            progression_features=_unique_items(hpi_payload.get("progression_features", []) or []),
        ),
        past_medical_history=_clean_text(payload.get("past_medical_history")),
        social_history=SocialHistory(
            smoking=_clean_text(social_payload.get("smoking")),
            alcohol=_clean_text(social_payload.get("alcohol")),
        ),
        review_of_systems=ReviewOfSystems(
            negative_findings=_unique_items(ros_payload.get("negative_findings", []) or [])
        ),
    )
    patient_view.history_facts = _flatten_patient_facts(patient_view)
    return patient_view


def _build_exam_view(payload: Dict[str, Any]) -> ExamView:
    if _is_category_exam_payload(payload):
        exam_view = ExamView()
        exam_view.tests = _flatten_category_exam_payload(payload)
        exam_view.test_catalog = sorted(exam_view.tests.keys())
        return exam_view

    physical_payload = payload.get("physical_exam", {}) or {}
    vital_payload = physical_payload.get("vital_signs", {}) or {}
    neuro_payload = physical_payload.get("neurological_exam", {}) or {}
    test_payload = payload.get("test_results", {}) or {}

    exam_view = ExamView(
        physical_exam=PhysicalExam(
            vital_signs=VitalSigns(
                temperature=_clean_text(vital_payload.get("temperature")),
                blood_pressure=_clean_text(vital_payload.get("blood_pressure")),
                heart_rate=_clean_text(vital_payload.get("heart_rate")),
                respiratory_rate=_clean_text(vital_payload.get("respiratory_rate")),
            ),
            neurological_exam=NeurologicalExam(
                cranial_nerves=_unique_items(neuro_payload.get("cranial_nerves", []) or []),
                motor_strength=_unique_items(neuro_payload.get("motor_strength", []) or []),
                reflexes=_clean_text(neuro_payload.get("reflexes")),
                sensation=_clean_text(neuro_payload.get("sensation")),
            ),
        ),
        test_results=TestResults(
            blood_tests=test_payload.get("blood_tests", {}) or {},
            electromyography=test_payload.get("electromyography", {}) or {},
            imaging=test_payload.get("imaging", {}) or {},
        ),
    )
    exam_view.tests = _flatten_exam_view(exam_view)
    exam_view.test_catalog = sorted(exam_view.tests.keys())
    return exam_view


def _build_manager_view(payload: Dict[str, Any]) -> ManagerView:
    ground_truth_payload = payload.get("ground_truth", {}) or {}
    diagnosis_payload = ground_truth_payload.get("diagnosis", {}) or {}
    evidence_payload = ground_truth_payload.get("evidence_rubric", {}) or {}
    management_payload = ground_truth_payload.get("management_rubric", {}) or {}
    return ManagerView(
        ground_truth=GroundTruth(
            diagnosis=DiagnosisGroundTruth(
                primary_label=_clean_text(diagnosis_payload.get("primary_label")),
                display_name=_clean_text(diagnosis_payload.get("display_name")),
                acceptable_names=_unique_items(diagnosis_payload.get("acceptable_names", []) or []),
                partial_credit_names=_unique_items(diagnosis_payload.get("partial_credit_names", []) or []),
            ),
            evidence_rubric=EvidenceRubric(
                key_points=_unique_items(evidence_payload.get("key_points", []) or []),
                error_points_examples=_unique_items(evidence_payload.get("error_points_examples", []) or []),
            ),
            management_rubric=ManagementRubric(
                key_points=_unique_items(management_payload.get("key_points", []) or []),
                danger_points_examples=_unique_items(management_payload.get("danger_points_examples", []) or []),
            ),
        )
    )


def build_doctor_case_view(case: ClinicalCase) -> DoctorCaseView:
    return DoctorCaseView(
        case_id=case.case_id,
        arrival_mode="outpatient_initial_visit",
        opening_instruction="患者刚进入门诊，你尚不知道主诉，必须先询问本次就诊原因与最不舒服的问题。",
        available_test_catalog=list(case.available_test_catalog),
        difficulty_tags=list(case.difficulty_tags),
        test_alias_to_category=dict(case.test_alias_to_category),
    )


def medqa_to_clinical_case(
    raw_case: Dict[str, Any],
    scenario_id: int,
    allowed_test_catalog: List[str],
    test_alias_to_category: Optional[Dict[str, str]] = None,
) -> ClinicalCase:
    patient_view = _build_patient_view(raw_case.get("patient_view", {}) or {})
    exam_view = _build_exam_view(raw_case.get("exam_view", {}) or {})
    manager_view = _build_manager_view(raw_case.get("manager_view", {}) or {})
    case_id = _clean_text(raw_case.get("case_id")) or f"medqa_case_{scenario_id + 1:03d}"
    task = _clean_text(raw_case.get("task")) or patient_view.chief_complaint
    triage_acuity = _infer_triage_acuity(task, patient_view.history_facts)

    return ClinicalCase(
        case_id=case_id,
        source=CaseSource(type="MedQA", raw_case_id=case_id),
        patient_view=patient_view,
        exam_view=exam_view,
        manager_view=manager_view,
        onset_context=_build_onset_context(patient_view),
        triage_payload={
            "acuity": triage_acuity,
            "reason": task,
            "chief_complaint": patient_view.chief_complaint,
        },
        registration_payload={
            "visit_type": "outpatient_initial_visit",
            "registration_status": "documents_verified",
            "payment_channel": "self_or_insurance_pending",
        },
        presenting_problem=task,
        difficulty_tags=sorted(
            {
                "medqa",
                triage_acuity,
                "exam_required" if exam_view.tests else "history_only",
            }
        ),
        available_test_catalog=list(allowed_test_catalog),
        test_alias_to_category=dict(test_alias_to_category or {}),
        dataset_metadata=extract_dataset_metadata(raw_case),
    )


class ScenarioLoaderMedQA:
    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        allowed_catalog: Optional[List[str]] = None,
    ) -> None:
        data_path = dataset_path or DATASET_FILE
        if not data_path.exists():
            raise FileNotFoundError(f"MedQA dataset not found: {data_path}")

        with open(data_path, "r", encoding="utf-8") as file:
            raw_cases = [json.loads(line) for line in file if line.strip()]

        if allowed_catalog is None:
            from medsim.clinical.config import load_clinical_config

            allowed_catalog = list(load_clinical_config().clinical_tests.allowed_catalog)

        cleaned_catalog: List[str] = []
        seen_catalog: set[str] = set()
        for item in allowed_catalog or []:
            text = _clean_text(item)
            key = _normalize_catalog_name(text)
            if not text or key in seen_catalog:
                continue
            seen_catalog.add(key)
            cleaned_catalog.append(text)
        if not cleaned_catalog:
            raise ValueError("Allowed test catalog is empty. Please configure clinical_tests.allowed_catalog.")

        allowed_lookup = {_normalize_catalog_name(item): item for item in cleaned_catalog}
        self.raw_cases = raw_cases
        self.allowed_test_catalog = list(cleaned_catalog)
        self.global_test_catalog = list(cleaned_catalog)

        for idx, item in enumerate(raw_cases):
            exam_payload = item.get("exam_view", {}) or {}
            case_id = _clean_text(item.get("case_id")) or f"medqa_case_{idx + 1:03d}"
            if _is_category_exam_payload(exam_payload):
                normalized_seen = {_normalize_catalog_name(key): key for key in exam_payload.keys()}
                for field_name in exam_payload.keys():
                    normalized_field = _normalize_catalog_name(field_name)
                    if normalized_field not in allowed_lookup:
                        raise ValueError(
                            "Found test field not in configured catalog: "
                            f"scenario={idx} case_id={case_id} field={field_name}"
                        )

                missing_categories = [
                    category
                    for category in cleaned_catalog
                    if _normalize_catalog_name(category) not in normalized_seen
                ]
                if missing_categories:
                    raise ValueError(
                        "Missing required exam categories: "
                        f"scenario={idx} case_id={case_id} missing={missing_categories}"
                    )

        self.test_alias_to_category = _build_global_test_alias_to_category(raw_cases, allowed_lookup)
        self.scenarios = [
            medqa_to_clinical_case(
                item,
                idx,
                self.allowed_test_catalog,
                self.test_alias_to_category,
            )
            for idx, item in enumerate(raw_cases)
        ]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self) -> ClinicalCase:
        return self.scenarios[random.randint(0, len(self.scenarios) - 1)]

    def get_scenario(self, scenario_id: Optional[int]) -> ClinicalCase:
        if scenario_id is None:
            return self.sample_scenario()
        return self.scenarios[scenario_id]
