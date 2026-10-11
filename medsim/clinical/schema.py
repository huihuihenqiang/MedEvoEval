from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional


def dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {key: dataclass_to_dict(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {key: dataclass_to_dict(val) for key, val in value.items()}
    if isinstance(value, list):
        return [dataclass_to_dict(item) for item in value]
    return value


@dataclass
class RuntimeOptions:
    scenario_source: str = "medqa"
    dataset_path: Optional[str] = None
    run_n_cases: Optional[int] = 1
    write_legacy_outputs: bool = True
    use_simulacra_bridge: bool = True
    enable_llm: bool = True
    seed: int = 7


@dataclass
class ModelOptions:
    attending: str = "qwen-plus"
    patient: str = "qwen-plus"
    manager: str = "qwen-plus"
    mdt_expert: str = "qwen-plus"


@dataclass
class WorkflowOptions:
    max_total_turns: int = 10
    max_tests_per_visit: int = 3


@dataclass
class MDTOptions:
    enabled: bool = False
    team_size: int = 0
    specialties: List[str] = field(
        default_factory=lambda: ["cardiology", "pulmonology", "gastroenterology", "imaging"]
    )
    collaboration: str = "consensus"


@dataclass
class MemoryOptions:
    mode: str = "retrieve_replay"
    enable_retrieval: bool = True
    read_only: bool = False
    include_failure_correction: bool = True
    namespace: str = "default"
    kb_root_dir: Optional[str] = None
    retrieval_topk: int = 3
    conversation_window: int = 6
    kb_snippet_token_limit: int = 160


@dataclass
class SimulacraOptions:
    visible_roles: str = "minimal_closed_loop"
    examiner_station: str = "the Ville:Hospital:Radiology Department:desk"
    pharmacy_station: str = "the Ville:Hospital:Pharmacy:desk"


@dataclass
class APIOptions:
    dashscope: str = ""


@dataclass
class APIBaseOptions:
    dashscope: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


@dataclass
class NetworkOptions:
    use_system_proxy: bool = False


@dataclass
class ScoringOptions:
    lambda_evidence: float = 0.5
    mu_management: float = 1.0
    weight_diag: float = 0.5
    weight_evidence: float = 0.3
    weight_plan: float = 0.2
    pass_threshold: float = 0.7
    effective_treatment_threshold: float = 0.5


@dataclass
class ClinicalTestsOptions:
    # Authoritative list of orderable tests. Doctor REQUEST_TEST must select from this catalog.
    allowed_catalog: List[str] = field(default_factory=list)


@dataclass
class InterfaceNormalizationOptions:
    # Disabled by default for the raw-interface condition.
    enabled: bool = False
    deterministic_json_repair: bool = True
    # The primary rebuttal experiment keeps this false: an LLM re-prompt cannot
    # prove that it preserved unparseable medical content.
    allow_format_only_reprompt: bool = False


@dataclass
class ClinicalRuntimeConfig:
    runtime: RuntimeOptions = field(default_factory=RuntimeOptions)
    models: ModelOptions = field(default_factory=ModelOptions)
    workflow: WorkflowOptions = field(default_factory=WorkflowOptions)
    clinical_tests: ClinicalTestsOptions = field(default_factory=ClinicalTestsOptions)
    interface_normalization: InterfaceNormalizationOptions = field(
        default_factory=InterfaceNormalizationOptions
    )
    mdt: MDTOptions = field(default_factory=MDTOptions)
    memory: MemoryOptions = field(default_factory=MemoryOptions)
    simulacra: SimulacraOptions = field(default_factory=SimulacraOptions)
    api_keys: APIOptions = field(default_factory=APIOptions)
    api_base: APIBaseOptions = field(default_factory=APIBaseOptions)
    network: NetworkOptions = field(default_factory=NetworkOptions)
    scoring: ScoringOptions = field(default_factory=ScoringOptions)


@dataclass
class CaseSource:
    type: str
    raw_case_id: str


@dataclass
class PatientDemographics:
    age: Optional[int] = None
    sex: str = ""
    occupation: str = ""


@dataclass
class HistoryOfPresentIllness:
    duration: str = ""
    symptoms: List[str] = field(default_factory=list)
    progression_features: List[str] = field(default_factory=list)


@dataclass
class SocialHistory:
    smoking: str = ""
    alcohol: str = ""


@dataclass
class ReviewOfSystems:
    negative_findings: List[str] = field(default_factory=list)


@dataclass
class PatientView:
    demographics: PatientDemographics = field(default_factory=PatientDemographics)
    chief_complaint: str = ""
    history_of_present_illness: HistoryOfPresentIllness = field(default_factory=HistoryOfPresentIllness)
    past_medical_history: str = ""
    social_history: SocialHistory = field(default_factory=SocialHistory)
    review_of_systems: ReviewOfSystems = field(default_factory=ReviewOfSystems)
    history_facts: List[str] = field(default_factory=list)

    @property
    def patient_profile(self) -> Dict[str, Any]:
        return {
            "demographics": dataclass_to_dict(self.demographics),
            "chief_complaint": self.chief_complaint,
            "history_of_present_illness": dataclass_to_dict(self.history_of_present_illness),
            "past_medical_history": self.past_medical_history,
            "social_history": dataclass_to_dict(self.social_history),
            "review_of_systems": dataclass_to_dict(self.review_of_systems),
        }


@dataclass
class VitalSigns:
    temperature: str = ""
    blood_pressure: str = ""
    heart_rate: str = ""
    respiratory_rate: str = ""


@dataclass
class NeurologicalExam:
    cranial_nerves: List[str] = field(default_factory=list)
    motor_strength: List[str] = field(default_factory=list)
    reflexes: str = ""
    sensation: str = ""


@dataclass
class PhysicalExam:
    vital_signs: VitalSigns = field(default_factory=VitalSigns)
    neurological_exam: NeurologicalExam = field(default_factory=NeurologicalExam)


@dataclass
class TestResults:
    blood_tests: Dict[str, Any] = field(default_factory=dict)
    electromyography: Dict[str, Any] = field(default_factory=dict)
    imaging: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExamView:
    physical_exam: PhysicalExam = field(default_factory=PhysicalExam)
    test_results: TestResults = field(default_factory=TestResults)
    tests: Dict[str, str] = field(default_factory=dict)
    test_catalog: List[str] = field(default_factory=list)


@dataclass
class DiagnosisGroundTruth:
    primary_label: str = ""
    display_name: str = ""
    acceptable_names: List[str] = field(default_factory=list)
    partial_credit_names: List[str] = field(default_factory=list)


@dataclass
class EvidenceRubric:
    key_points: List[str] = field(default_factory=list)
    error_points_examples: List[str] = field(default_factory=list)


@dataclass
class ManagementRubric:
    key_points: List[str] = field(default_factory=list)
    danger_points_examples: List[str] = field(default_factory=list)


@dataclass
class GroundTruth:
    diagnosis: DiagnosisGroundTruth = field(default_factory=DiagnosisGroundTruth)
    evidence_rubric: EvidenceRubric = field(default_factory=EvidenceRubric)
    management_rubric: ManagementRubric = field(default_factory=ManagementRubric)


@dataclass
class ManagerView:
    ground_truth: GroundTruth = field(default_factory=GroundTruth)

    @property
    def ground_truth_diagnosis(self) -> str:
        return self.ground_truth.diagnosis.display_name

    @property
    def diagnosis_aliases(self) -> List[str]:
        candidates = [
            self.ground_truth.diagnosis.display_name,
            self.ground_truth.diagnosis.primary_label,
            *self.ground_truth.diagnosis.acceptable_names,
        ]
        seen = set()
        aliases: List[str] = []
        for item in candidates:
            text = str(item or "").strip()
            if not text or text.lower() in seen:
                continue
            seen.add(text.lower())
            aliases.append(text)
        return aliases

    @property
    def diagnosis_partial_aliases(self) -> List[str]:
        candidates = list(self.ground_truth.diagnosis.partial_credit_names)
        seen = {item.lower() for item in self.diagnosis_aliases}
        aliases: List[str] = []
        for item in candidates:
            text = str(item or "").strip()
            lowered = text.lower()
            if not text or lowered in seen:
                continue
            seen.add(lowered)
            aliases.append(text)
        return aliases

    @property
    def evidence_key_points(self) -> List[str]:
        return list(self.ground_truth.evidence_rubric.key_points)

    @property
    def treatment_keypoints(self) -> List[str]:
        return list(self.ground_truth.management_rubric.key_points)


@dataclass
class ClinicalCase:
    case_id: str
    source: CaseSource
    patient_view: PatientView
    exam_view: ExamView
    manager_view: ManagerView
    onset_context: str
    triage_payload: Dict[str, Any]
    registration_payload: Dict[str, Any]
    presenting_problem: str
    difficulty_tags: List[str]
    available_test_catalog: List[str] = field(default_factory=list)
    test_alias_to_category: Dict[str, str] = field(default_factory=dict)
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def patient_profile(self) -> Dict[str, Any]:
        return self.patient_view.patient_profile

    @property
    def history_facts(self) -> List[str]:
        return self.patient_view.history_facts

    @property
    def orderable_tests(self) -> Dict[str, Any]:
        return self.exam_view.tests

    @property
    def gold_diagnosis(self) -> str:
        return self.manager_view.ground_truth_diagnosis

    @property
    def gold_diagnosis_aliases(self) -> List[str]:
        return self.manager_view.diagnosis_aliases

    @property
    def treatment_plan_template(self) -> List[str]:
        return self.manager_view.treatment_keypoints

    @property
    def evidence_rubric(self) -> List[str]:
        return self.manager_view.evidence_key_points

    @property
    def followup_template(self) -> List[str]:
        return [
            "按病情安排门诊复诊或专科随访。",
            "如症状加重或出现新的危险信号，及时复诊或急诊就医。",
        ]


@dataclass
class DoctorCaseView:
    case_id: str
    arrival_mode: str
    opening_instruction: str
    available_test_catalog: List[str]
    difficulty_tags: List[str] = field(default_factory=list)
    test_alias_to_category: Dict[str, str] = field(default_factory=dict)


@dataclass
class DialogueLine:
    speaker: str
    text: str
    phase: str
    visit_index: int = 1
    turn_index: Optional[int] = None


@dataclass
class DialogueSegment:
    phase: str
    lines: List[DialogueLine] = field(default_factory=list)


@dataclass
class PhaseRecord:
    phase: str
    status: str
    summary: str
    visit_index: int = 1


@dataclass
class TestRequestRecord:
    visit_index: int
    turn_index: int
    test_name: str
    returned: bool
    result_if_any: Optional[str] = None
    match_type: str = "not_allowed"


@dataclass
class FinalClinicalDecision:
    diagnosis: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    followup: List[str] = field(default_factory=list)
    disposition: str = "outpatient"
    patient_instruction: str = ""
    final_diagnosis: List[str] = field(default_factory=list)
    diagnostic_basis: List[str] = field(default_factory=list)
    orders_or_prescriptions: List[str] = field(default_factory=list)
    followup_plan: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.diagnosis and self.final_diagnosis:
            self.diagnosis = list(self.final_diagnosis)
        if not self.final_diagnosis and self.diagnosis:
            self.final_diagnosis = list(self.diagnosis)
        if not self.evidence and self.diagnostic_basis:
            self.evidence = list(self.diagnostic_basis)
        if not self.diagnostic_basis and self.evidence:
            self.diagnostic_basis = list(self.evidence)
        if not self.plan and self.orders_or_prescriptions:
            self.plan = list(self.orders_or_prescriptions)
        if not self.orders_or_prescriptions and self.plan:
            self.orders_or_prescriptions = list(self.plan)
        if not self.followup and self.followup_plan:
            self.followup = list(self.followup_plan)
        if not self.followup_plan and self.followup:
            self.followup_plan = list(self.followup)


@dataclass
class DoctorAction:
    action: str
    utterance: str
    question: str = ""
    requested_history: List[str] = field(default_factory=list)
    test_name: str = ""
    rationale: List[str] = field(default_factory=list)
    working_diagnoses: List[str] = field(default_factory=list)
    needs_mdt: bool = False
    diagnosis: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    followup: List[str] = field(default_factory=list)
    interface_normalization_audit: Dict[str, Any] = field(default_factory=dict)

    def to_final_decision(self) -> FinalClinicalDecision:
        return FinalClinicalDecision(
            diagnosis=list(self.diagnosis),
            evidence=list(self.evidence),
            plan=list(self.plan),
            followup=list(self.followup),
            patient_instruction="请按医嘱处理，如症状加重及时就医。",
        )


@dataclass
class PatientTurn:
    utterance: str
    revealed_facts: List[str] = field(default_factory=list)
    emotion: str = "骞抽潤"
    cooperation: str = "閰嶅悎"


@dataclass
class TestServiceResponse:
    test_name: str
    returned: bool
    result_text: str
    evidence_tags: List[str] = field(default_factory=list)
    match_type: str = "not_allowed"


@dataclass
class ExpertOpinion:
    specialty: str
    assessment: str
    supporting_evidence: List[str] = field(default_factory=list)
    recommended_next_step: str = ""
    confidence: float = 0.0


@dataclass
class EvidenceScoreStats:
    matched: int = 0
    incorrect: int = 0
    total_key_points: int = 0


@dataclass
class ManagementScoreStats:
    matched: int = 0
    incorrect: int = 0
    total_key_points: int = 0


@dataclass
class ManagerAssessment:
    correct_diagnosis: bool
    effective_treatment: bool
    cured: bool
    feedback_text: str
    missing_points: List[str] = field(default_factory=list)
    suggested_questions: List[str] = field(default_factory=list)
    suggested_tests: List[str] = field(default_factory=list)
    corrected_diagnosis: List[str] = field(default_factory=list)
    corrected_plan: List[str] = field(default_factory=list)
    needs_revisit: bool = False
    confidence: float = 0.0
    diagnosis_grade: str = "wrong_low_risk"
    diagnosis_score: float = 0.0
    evidence_score: float = 0.0
    plan_score: float = 0.0
    total_score: float = 0.0
    evidence_stats: EvidenceScoreStats = field(default_factory=EvidenceScoreStats)
    management_stats: ManagementScoreStats = field(default_factory=ManagementScoreStats)
    matched_evidence_points: List[str] = field(default_factory=list)
    incorrect_evidence_points: List[str] = field(default_factory=list)
    matched_management_points: List[str] = field(default_factory=list)
    incorrect_management_points: List[str] = field(default_factory=list)


@dataclass
class DoctorDecisionRecord:
    visit_index: int
    turn_index: int
    action: str
    question_count: int
    test_count: int
    question: str = ""
    test_name: str = ""
    working_diagnoses: List[str] = field(default_factory=list)
    rationale: List[str] = field(default_factory=list)
    forced: bool = False
    utterance: str = ""


@dataclass
class EventRecord:
    phase: str
    actor: str
    event_type: str
    details: Dict[str, Any] = field(default_factory=dict)
    visit_index: int = 1


@dataclass
class KBWriteRecord:
    type: str
    content_summary: str
    timestamp: str
    visit_index: int


@dataclass
class EpisodeMetrics:
    num_turns: int = 0
    num_tests: int = 0
    llm_error_turns: int = 0
    raw_interface_error_turns: int = 0
    repaired_interface_turns: int = 0
    residual_interface_error_turns: int = 0
    deterministic_repair_turns: int = 0
    format_reprompt_turns: int = 0
    invalid_tests: int = 0
    invalid_test_rate: float = 0.0
    utility: float = 0.0
    visit_count: int = 0
    diagnosis_correct: bool = False
    treatment_effective: bool = False
    cured: bool = False
    mdt_used: bool = False
    diagnosis_score: float = 0.0
    evidence_score: float = 0.0
    plan_score: float = 0.0
    total_score: float = 0.0
    token_usage: Optional[int] = None
    cost: Optional[float] = None


@dataclass
class VisitRecord:
    visit_index: int
    turn_count: int = 0
    test_count: int = 0
    question_count: int = 0
    dialogue_lines: List[DialogueLine] = field(default_factory=list)
    tests_requested: List[TestRequestRecord] = field(default_factory=list)
    doctor_decisions: List[DoctorDecisionRecord] = field(default_factory=list)
    doctor_finalize: Optional[FinalClinicalDecision] = None
    manager_feedback: Optional[ManagerAssessment] = None
    termination_reason: Optional[str] = None


@dataclass
class SimulacraFlowAction:
    flow: str
    address: str
    duration: int
    description: str
    status: str
    speech: str
    pronunciatio: str
    predicate: str
    object: str
    stage_id: str = ""
    stage_index: int = 0


@dataclass
class SimulacraDialogueEvent:
    speaker: str
    text: str
    phase: str
    visit_index: int = 1
    turn_index: Optional[int] = None
    unlock_stage_id: str = ""
    unlock_stage_index: int = 0


@dataclass
class SimulacraCameraPlan:
    mode: str = "follow_then_free"
    initial_focus: Optional[str] = None
    release_stage_id: str = "consulting_pre_exam"
    release_stage_index: int = 4


@dataclass
class SimulacraReplay:
    sim_code: Optional[str]
    focus_personas: List[str]
    consultation_lines: List[DialogueLine] = field(default_factory=list)
    persona_actions: Dict[str, List[SimulacraFlowAction]] = field(default_factory=dict)
    dialogue_panel: Dict[str, Any] = field(default_factory=dict)
    cast: Dict[str, str] = field(default_factory=dict)
    dialogue_events: List[SimulacraDialogueEvent] = field(default_factory=list)
    camera_plan: SimulacraCameraPlan = field(default_factory=SimulacraCameraPlan)


@dataclass
class EpisodeResult:
    episode_id: str
    scenario_id: int
    case: ClinicalCase
    config_snapshot: Dict[str, Any]
    run_id: Optional[str] = None
    visit_records: List[VisitRecord] = field(default_factory=list)
    visit_count: int = 0
    full_dialogue: List[DialogueLine] = field(default_factory=list)
    tests_requested: List[TestRequestRecord] = field(default_factory=list)
    doctor_decisions: List[DoctorDecisionRecord] = field(default_factory=list)
    doctor_outputs: List[FinalClinicalDecision] = field(default_factory=list)
    manager_feedbacks: List[ManagerAssessment] = field(default_factory=list)
    kb_writes: List[KBWriteRecord] = field(default_factory=list)
    mdt_used: bool = False
    mdt_called: bool = False
    mdt_trigger_turn: Optional[int] = None
    mdt_summary: str = ""
    consultant_opinion: Optional[Dict[str, Any]] = None
    consultant_advice_action: Optional[str] = None
    consultant_confidence: Optional[float] = None
    termination_reason: str = "other"
    metrics: EpisodeMetrics = field(default_factory=EpisodeMetrics)
    phase_trace: List[PhaseRecord] = field(default_factory=list)
    dialogue_segments: List[DialogueSegment] = field(default_factory=list)
    final_clinical_decision: FinalClinicalDecision = field(default_factory=FinalClinicalDecision)
    learning_trace: Dict[str, Any] = field(default_factory=dict)
    simulacra_cast: Dict[str, str] = field(default_factory=dict)
    completion_status: str = "success"
    event_log: List[EventRecord] = field(default_factory=list)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    simulacra_replay: Optional[SimulacraReplay] = None
    legacy_dialogue_history: List[Dict[str, Any]] = field(default_factory=list)
