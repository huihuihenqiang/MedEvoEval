from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from medsim.clinical.agent_npc import (
    AttendingDoctorAgent,
    ConsultantAgent,
    DoctorActionError,
    ManagerAgent,
    PatientAgentNPC,
    TestService,
)
from medsim.clinical.case_adapter import ScenarioLoaderMedQA, build_doctor_case_view
from medsim.clinical.config import load_clinical_config
from medsim.clinical.context_builder import build_doctor_context, dialogue_lines_to_text
from medsim.clinical.llm_io import StructuredLLMClient
from medsim.clinical.memory import ClinicalMemoryManager
from medsim.clinical.metrics import compute_objective_metrics
from medsim.clinical.prompts import (
    format_consultant_feedback_for_doctor,
    should_allow_call_mdt,
)
from medsim.clinical.schema import (
    ClinicalRuntimeConfig,
    DialogueLine,
    DialogueSegment,
    DoctorAction,
    DoctorDecisionRecord,
    EpisodeMetrics,
    EpisodeResult,
    EventRecord,
    FinalClinicalDecision,
    KBWriteRecord,
    ManagerAssessment,
    PhaseRecord,
    TestRequestRecord,
    TestServiceResponse,
    VisitRecord,
    dataclass_to_dict,
)
from medsim.clinical.simulacra_bridge import build_simulacra_replay
from medsim.clinical.writers import write_episode_artifacts


PRIMARY_DOCTOR = "Maria Lopez"
PATIENT_NAME = "Klaus Mueller"
EXAMINER = "Rajiv Patel"
MANAGER_NAME = "管理者"
logger = logging.getLogger(__name__)


def _compact_for_log(text: str, limit: int = 220) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "...<truncated>"


def _append_phase(
    episode: EpisodeResult,
    visit: VisitRecord,
    phase: str,
    summary: str,
    lines: List[DialogueLine],
    actor: str,
    event_type: str,
    details: Dict[str, Any],
) -> None:
    if not lines:
        return
    episode.phase_trace.append(PhaseRecord(phase=phase, status="completed", summary=summary, visit_index=visit.visit_index))
    episode.dialogue_segments.append(DialogueSegment(phase=phase, lines=lines))
    episode.full_dialogue.extend(lines)
    visit.dialogue_lines.extend(lines)
    episode.event_log.append(
        EventRecord(
            phase=phase,
            actor=actor,
            event_type=event_type,
            details=details,
            visit_index=visit.visit_index,
        )
    )


def _log_lines(lines: List[DialogueLine]) -> None:
    for line in lines:
        logger.info(
            "DIALOGUE phase=%s speaker=%s visit=%s turn=%s text=%s",
            line.phase,
            line.speaker,
            line.visit_index,
            line.turn_index if line.turn_index is not None else "-",
            _compact_for_log(line.text),
        )


def _log_decision(record: DoctorDecisionRecord) -> None:
    rationale_text = " | ".join(record.rationale) if record.rationale else "-"
    action_text = record.question if record.action == "ASK" else record.utterance
    if not str(action_text or "").strip():
        action_text = record.utterance or record.question or "-"
    logger.info(
        "DOCTOR_DECISION turn=%s action=%s asked=%s tested=%s action_text=%s test=%s forced=%s rationale=%s",
        record.turn_index,
        record.action,
        record.question_count,
        record.test_count,
        _compact_for_log(str(action_text)),
        record.test_name or "-",
        record.forced,
        _compact_for_log(rationale_text),
    )


def _log_test_request(turn_index: int, requested_name: str, normalized_name: str) -> None:
    logger.info(
        "TEST_REQUEST turn=%s requested=%s normalized=%s",
        turn_index,
        _compact_for_log(requested_name),
        _compact_for_log(normalized_name),
    )


def _log_test_result(turn_index: int, test_name: str, returned: bool, match_type: str, result_text: str) -> None:
    logger.info(
        "TEST_RESULT turn=%s test=%s returned=%s match_type=%s result=%s",
        turn_index,
        _compact_for_log(test_name),
        returned,
        match_type,
        _compact_for_log(result_text),
    )


def _record_doctor_llm_error(
    episode: EpisodeResult,
    visit: VisitRecord,
    turn_index: int,
    error_type: str,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
) -> None:
    details: Dict[str, Any] = {
        "turn_index": turn_index,
        "error_type": error_type,
        "error_message": _compact_for_log(error_message),
    }
    if error_details:
        details["interface_normalization"] = dict(error_details)
    episode.event_log.append(
        EventRecord(
            phase="consultation",
            actor=PRIMARY_DOCTOR,
            event_type="doctor_llm_error",
            details=details,
            visit_index=visit.visit_index,
        )
    )
    logger.warning(
        "DOCTOR_LLM_ERROR turn=%s type=%s detail=%s",
        turn_index,
        error_type,
        _compact_for_log(error_message),
    )


def _query_text(known_history: List[str], released_results: List[str]) -> str:
    parts = [*known_history[:6], *released_results[:3]]
    if not parts:
        return "门诊初诊患者未披露主诉"
    return " ".join(str(item) for item in parts if item)


def _run_consultant_review(
    consultant: ConsultantAgent,
    turn_index: int,
    question_count: int,
    test_count: int,
    released_results: List[str],
    tests_requested: List[str],
    context_prompt: str,
) -> Dict[str, Any]:
    return consultant.review_case(
        turn_index=turn_index,
        question_count=question_count,
        test_count=test_count,
        released_results=released_results,
        tests_requested=tests_requested,
        context_prompt=context_prompt,
    )


def _decision_dialogue_lines(decision: FinalClinicalDecision, visit_index: int) -> List[DialogueLine]:
    return [
        DialogueLine(
            speaker=PRIMARY_DOCTOR,
            text=(
                f"我的诊断是：{'；'.join(decision.diagnosis)}。"
                f" 依据：{'；'.join(decision.evidence[:3]) or '当前信息有限'}。"
                f" 处理方案：{'；'.join(decision.plan[:3]) or '对症处理'}。"
                f" 后续建议：{'；'.join(decision.followup[:2]) or '按医嘱复诊'}。"
            ),
            phase="diagnosis",
            visit_index=visit_index,
        )
    ]


def _manager_dialogue_lines(assessment: ManagerAssessment, visit_index: int) -> List[DialogueLine]:
    return [
        DialogueLine(
            speaker=MANAGER_NAME,
            text=assessment.feedback_text,
            phase="manager_review",
            visit_index=visit_index,
        )
    ]


def _record_kb_write(episode: EpisodeResult, visit_index: int, entry_type: str, content_summary: str) -> None:
    episode.kb_writes.append(
        KBWriteRecord(
            type=entry_type,
            content_summary=content_summary,
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            visit_index=visit_index,
        )
    )


def _visit_summary_for_manager(visit: VisitRecord, decision: FinalClinicalDecision) -> str:
    dialogue_text = dialogue_lines_to_text(visit.dialogue_lines[-20:])
    return "\n".join(
        [
            dialogue_text,
            f"医生诊断：{decision.diagnosis}",
            f"医生治疗：{decision.plan}",
            f"医生随访：{decision.followup}",
        ]
    ).strip()


def _record_doctor_decision(
    episode: EpisodeResult,
    visit: VisitRecord,
    action: DoctorAction,
    turn_index: int,
    forced: bool = False,
) -> DoctorDecisionRecord:
    record = DoctorDecisionRecord(
        visit_index=visit.visit_index,
        turn_index=turn_index,
        action=action.action,
        question_count=visit.question_count,
        test_count=visit.test_count,
        question=action.question,
        test_name=action.test_name,
        working_diagnoses=list(action.working_diagnoses),
        rationale=list(action.rationale),
        forced=forced,
        utterance=action.utterance,
    )
    visit.doctor_decisions.append(record)
    episode.doctor_decisions.append(record)
    return record


def _sanitize_action(raw_action: str) -> str:
    normalized = (raw_action or "").strip().upper()
    aliases = {
        "ASK": "ASK",
        "QUESTION": "ASK",
        "继续问": "ASK",
        "REQUEST_TEST": "REQUEST_TEST",
        "TEST": "REQUEST_TEST",
        "去检查": "REQUEST_TEST",
        "FINALIZE": "FINALIZE",
        "DECIDE": "FINALIZE",
        "做出判断": "FINALIZE",
        "CALL_MDT": "CALL_MDT",
    }
    return aliases.get(normalized, normalized)


def _normalize_action_fields(action: DoctorAction) -> None:
    if action.action == "ASK":
        question_text = str(action.question or "").strip()
        utterance_text = str(action.utterance or "").strip()
        if question_text:
            # Keep doctor dialogue and decision text aligned to avoid stale or repeated text mismatch.
            action.utterance = question_text
            return
        if utterance_text:
            action.question = utterance_text
        return

    # For non-ASK actions, question field should not be used in decision logs.
    action.question = ""
    utterance_text = str(action.utterance or "").strip()
    if utterance_text:
        return
    if action.action == "REQUEST_TEST":
        action.utterance = f"我建议先做 {action.test_name or '必要检查'} 以进一步明确诊断。"
    elif action.action == "CALL_MDT":
        action.utterance = "当前信息存在不确定性，我申请顾问医生复核。"
    elif action.action == "FINALIZE":
        action.utterance = "我现在基于现有证据给出最终判断。"


def run_episode_stream(
    config: ClinicalRuntimeConfig,
    scenario_id: int,
    sim_code: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
    episode_output_root: Optional[Path] = None,
) -> Iterator[Dict[str, Any]]:
    random.seed(config.runtime.seed + int(scenario_id))
    dataset_path = Path(config.runtime.dataset_path) if config.runtime.dataset_path else None
    loader = ScenarioLoaderMedQA(dataset_path=dataset_path, allowed_catalog=config.clinical_tests.allowed_catalog)
    case = loader.get_scenario(scenario_id)
    doctor_view = build_doctor_case_view(case)
    llm_client = StructuredLLMClient(config)
    patient = PatientAgentNPC(case.patient_view, config, llm_client)
    doctor = AttendingDoctorAgent(doctor_view, config, llm_client)
    consultant = ConsultantAgent(doctor_view, config, llm_client)
    test_service = TestService(case.exam_view, config, llm_client)
    manager = ManagerAgent(case.manager_view, config, llm_client)
    memory = ClinicalMemoryManager(config.memory.namespace, kb_root_dir=config.memory.kb_root_dir)
    retrieval_enabled = bool(getattr(config.memory, "enable_retrieval", True))
    read_only_memory = bool(getattr(config.memory, "read_only", False))
    logger.info(
        "MEMORY_RETRIEVAL enabled=%s read_only=%s topk=%s namespace=%s",
        retrieval_enabled,
        read_only_memory,
        config.memory.retrieval_topk,
        config.memory.namespace,
    )

    episode = EpisodeResult(
        episode_id=f"episode-{scenario_id}-{sim_code or 'cli'}",
        scenario_id=scenario_id,
        case=case,
        config_snapshot=dataclass_to_dict(config),
        run_id=run_id,
        simulacra_cast={
            PRIMARY_DOCTOR: "doctor",
            PATIENT_NAME: "patient",
            "Jane Moreno": "receptionist",
            "Ayesha Khan": "nurse",
            EXAMINER: "examiner",
            "Carmen Ortiz": "pharmacy_clerk",
        },
    )
    visit = VisitRecord(visit_index=1)
    episode.visit_records.append(visit)
    metrics = EpisodeMetrics()

    known_history: List[str] = []
    released_results: List[str] = []
    tests_requested_names: List[str] = []
    mdt_summary = ""
    mdt_called = False
    mdt_trigger_turn: Optional[int] = None
    consultant_opinion: Optional[Dict[str, Any]] = None
    finalized = False
    max_total_turns = config.workflow.max_total_turns

    yield {
        "type": "episode_started",
        "case": case,
        "episode_id": episode.episode_id,
        "sim_code": sim_code,
        "simulacra_cast": dict(episode.simulacra_cast),
    }

    for turn_index in range(max_total_turns + 1):
        if retrieval_enabled:
            query_text = _query_text(known_history, released_results)
            success_entries = memory.retrieve_success_examples(query_text, top_k=config.memory.retrieval_topk)
            correction_entries = memory.retrieve_correction_examples(query_text, top_k=config.memory.retrieval_topk)
            rule_cards = memory.retrieve_rule_cards(query_text, top_k=config.memory.retrieval_topk)
        else:
            success_entries = []
            correction_entries = []
            rule_cards = []
        context_packet = build_doctor_context(
            config=config,
            dialogue_lines=visit.dialogue_lines,
            released_results=released_results,
            success_entries=success_entries,
            correction_entries=correction_entries,
            manager_feedback=None,
            mdt_summary=mdt_summary,
            constraints=[
                f"医生总动作最多 {config.workflow.max_total_turns} 轮",
                f"最多检查 {config.workflow.max_tests_per_visit} 次",
                "只能使用已经释放的检查结果",
            ],
            rule_cards=rule_cards,
        )

        if visit.turn_count >= config.workflow.max_total_turns:
            forced_decision = doctor.build_forced_final_decision(known_history, released_results, None)
            action = DoctorAction(
                action="FINALIZE",
                utterance="已达到本次接诊总轮次上限，我现在基于现有证据做出最终判断。",
                diagnosis=list(forced_decision.diagnosis),
                evidence=list(forced_decision.evidence),
                plan=list(forced_decision.plan),
                followup=list(forced_decision.followup),
                rationale=["已达到总轮次上限，系统转为终诊。"],
            )
        else:
            try:
                action = doctor.next_action(
                    turn_index=turn_index,
                    context_packet=context_packet,
                    released_results=released_results,
                    tests_requested=tests_requested_names,
                    mdt_already_used=mdt_called,
                    question_count=visit.question_count,
                    test_count=visit.test_count,
                )
            except DoctorActionError as exc:
                visit.turn_count += 1
                metrics.llm_error_turns += 1
                normalization_enabled = bool(
                    getattr(getattr(config, "interface_normalization", None), "enabled", False)
                )
                interface_error_types = {
                    "json_parse_error",
                    "missing_action",
                    "invalid_action",
                    "missing_test_name",
                    "invalid_test_name",
                    "interface_normalization_failed",
                }
                if normalization_enabled and (
                    exc.error_type in interface_error_types
                    or bool(getattr(exc, "details", {}).get("raw_error_type"))
                ):
                    metrics.raw_interface_error_turns += 1
                    metrics.residual_interface_error_turns += 1
                _record_doctor_llm_error(
                    episode=episode,
                    visit=visit,
                    turn_index=turn_index,
                    error_type=exc.error_type,
                    error_message=str(exc),
                    error_details=getattr(exc, "details", {}),
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive catch
                visit.turn_count += 1
                metrics.llm_error_turns += 1
                _record_doctor_llm_error(
                    episode=episode,
                    visit=visit,
                    turn_index=turn_index,
                    error_type="unexpected_doctor_error",
                    error_message=str(exc),
                )
                continue

            interface_audit = dict(action.interface_normalization_audit or {})
            completion_audit = dict(interface_audit.get("completion") or {})
            raw_interface_error = str(interface_audit.get("raw_error_type") or "")
            initial_parse_error = completion_audit.get("initial_parse_valid") is False
            deterministic_json_repair = bool(completion_audit.get("deterministic_json_repair"))
            repaired_action_interface = bool(interface_audit.get("repair_applied"))
            if raw_interface_error or initial_parse_error:
                metrics.raw_interface_error_turns += 1
                if repaired_action_interface or deterministic_json_repair:
                    metrics.repaired_interface_turns += 1
                if repaired_action_interface or deterministic_json_repair:
                    metrics.deterministic_repair_turns += 1
                if completion_audit.get("format_only_reprompt"):
                    metrics.format_reprompt_turns += 1
                episode.event_log.append(
                    EventRecord(
                        phase="consultation",
                        actor=PRIMARY_DOCTOR,
                        event_type="interface_normalization",
                        details={
                            "turn_index": turn_index,
                            **interface_audit,
                        },
                        visit_index=visit.visit_index,
                    )
                )

            action.action = _sanitize_action(action.action)
            _normalize_action_fields(action)

            if action.action not in {"ASK", "REQUEST_TEST", "FINALIZE", "CALL_MDT"}:
                visit.turn_count += 1
                metrics.llm_error_turns += 1
                _record_doctor_llm_error(
                    episode=episode,
                    visit=visit,
                    turn_index=turn_index,
                    error_type="invalid_action",
                    error_message=f"Unsupported action after sanitize: {action.action}",
                )
                continue

        if action.action == "CALL_MDT":
            visit.turn_count += 1
            decision_record = _record_doctor_decision(episode, visit, action, turn_index)
            _log_decision(decision_record)
            allow_call_mdt = bool(config.mdt.enabled) and should_allow_call_mdt(
                visit.question_count,
                visit.test_count,
                mdt_called,
            )
            if not bool(config.mdt.enabled):
                block_reason = "mdt_disabled"
            elif mdt_called:
                block_reason = "mdt_already_called"
            elif not should_allow_call_mdt(visit.question_count, visit.test_count, False):
                block_reason = "below_threshold"
            else:
                block_reason = "unknown"

            episode.event_log.append(
                EventRecord(
                    phase="diagnosis",
                    actor=PRIMARY_DOCTOR,
                    event_type="call_mdt",
                    details={
                        "turn_index": turn_index,
                        "allowed": allow_call_mdt,
                        "reason": ("allowed" if allow_call_mdt else block_reason),
                        "question_count": visit.question_count,
                        "test_count": visit.test_count,
                    },
                    visit_index=visit.visit_index,
                )
            )

            if not allow_call_mdt:
                if block_reason == "mdt_already_called":
                    blocked_text = "超过次数不能调用顾问医生。本病例已完成一次顾问复核，请继续问诊或尽快 FINALIZE。"
                elif block_reason == "below_threshold":
                    blocked_text = "超过次数不能调用顾问医生。当前未达到门槛（需 question_count >= 4 或 test_count >= 1）。"
                elif block_reason == "mdt_disabled":
                    blocked_text = "超过次数不能调用顾问医生。当前配置未启用 MDT。"
                else:
                    blocked_text = "超过次数不能调用顾问医生，请继续问诊或基于现有证据尽快 FINALIZE。"

                blocked_lines = [
                    DialogueLine(
                        speaker="系统",
                        text=blocked_text,
                        phase="consultation",
                        visit_index=visit.visit_index,
                        turn_index=turn_index,
                    )
                ]
                _append_phase(
                    episode,
                    visit,
                    "consultation",
                    "CALL_MDT 被拦截",
                    blocked_lines,
                    "System",
                    "mdt_blocked",
                    {
                        "turn_index": turn_index,
                        "reason": block_reason,
                        "question_count": visit.question_count,
                        "test_count": visit.test_count,
                    },
                )
                _log_lines(blocked_lines)
                yield {
                    "type": "mdt_blocked",
                    "lines": blocked_lines,
                    "reason": block_reason,
                    "decision": decision_record,
                    "question_count": visit.question_count,
                    "test_count": visit.test_count,
                }
                continue

            consultant_opinion = _run_consultant_review(
                consultant=consultant,
                turn_index=turn_index,
                question_count=visit.question_count,
                test_count=visit.test_count,
                released_results=list(released_results),
                tests_requested=list(tests_requested_names),
                context_prompt=context_packet["rendered_prompt"],
            )
            mdt_summary = format_consultant_feedback_for_doctor(consultant_opinion)
            mdt_called = True
            mdt_trigger_turn = turn_index
            episode.mdt_used = True
            episode.mdt_called = True
            episode.mdt_trigger_turn = turn_index
            episode.mdt_summary = mdt_summary
            episode.consultant_opinion = dict(consultant_opinion)
            episode.consultant_advice_action = str(consultant_opinion.get("advice_action", "")).strip().upper() or None
            try:
                episode.consultant_confidence = float(consultant_opinion.get("confidence", 0.0))
            except (TypeError, ValueError):
                episode.consultant_confidence = 0.0
            metrics.mdt_used = True

            consultant_lines = [
                DialogueLine(
                    speaker="顾问医生",
                    text=mdt_summary,
                    phase="diagnosis",
                    visit_index=visit.visit_index,
                    turn_index=turn_index,
                )
            ]
            _append_phase(
                episode,
                visit,
                "diagnosis",
                "顾问复核",
                consultant_lines,
                "Consultant",
                "mdt_review",
                {
                    "turn_index": turn_index,
                    "advice_action": episode.consultant_advice_action,
                    "confidence": episode.consultant_confidence,
                },
            )
            _log_lines(consultant_lines)
            yield {
                "type": "mdt_review",
                "lines": consultant_lines,
                "consultant_opinion": consultant_opinion,
                "summary": mdt_summary,
            }
            continue

        if action.action == "REQUEST_TEST" and visit.test_count >= config.workflow.max_tests_per_visit:
            action = DoctorAction(
                action="ASK",
                utterance="当前检查次数已达上限，请补充最关键的症状变化和诱因信息。",
                question="请补充目前最关键的症状变化和诱发/缓解因素。",
                rationale=["已达到检查次数上限，自动改写为追问以继续收集信息。"],
            )
            _normalize_action_fields(action)

        if action.action == "ASK" and visit.turn_count < config.workflow.max_total_turns:
            visit.question_count += 1
            visit.turn_count += 1
            decision_record = _record_doctor_decision(episode, visit, action, turn_index)
            _log_decision(decision_record)

            doctor_line = DialogueLine(
                speaker=PRIMARY_DOCTOR,
                text=action.utterance,
                phase="consultation",
                visit_index=visit.visit_index,
                turn_index=turn_index,
            )
            patient_turn = patient.respond(question=action.question or action.utterance, disclosed_facts=known_history)
            for fact in patient_turn.revealed_facts:
                if fact not in known_history:
                    known_history.append(fact)
            patient_line = DialogueLine(
                speaker=PATIENT_NAME,
                text=patient_turn.utterance,
                phase="consultation",
                visit_index=visit.visit_index,
                turn_index=turn_index,
            )
            lines = [doctor_line, patient_line]
            _append_phase(
                episode,
                visit,
                "consultation",
                "医生继续问诊",
                lines,
                PRIMARY_DOCTOR,
                "doctor_turn",
                {"action": action.action, "question": action.question},
            )
            _log_lines(lines)
            yield {
                "type": "consultation_turn",
                "lines": lines,
                "decision": decision_record,
                "question_count": visit.question_count,
                "test_count": visit.test_count,
            }
            continue

        if action.action == "REQUEST_TEST" and visit.turn_count < config.workflow.max_total_turns:
            over_test_budget = visit.test_count >= config.workflow.max_tests_per_visit
            if not over_test_budget:
                visit.test_count += 1
            visit.turn_count += 1
            decision_record = _record_doctor_decision(episode, visit, action, turn_index)
            _log_decision(decision_record)

            order_line = DialogueLine(
                speaker=PRIMARY_DOCTOR,
                text=action.utterance,
                phase="consultation",
                visit_index=visit.visit_index,
                turn_index=turn_index,
            )
            _append_phase(
                episode,
                visit,
                "consultation",
                "医生申请检查",
                [order_line],
                PRIMARY_DOCTOR,
                "doctor_turn",
                {"action": action.action, "test_name": action.test_name},
            )

            normalized_request = TestService.normalize_order_name(action.test_name)
            _log_test_request(turn_index=turn_index, requested_name=action.test_name, normalized_name=normalized_request)
            episode.event_log.append(
                EventRecord(
                    phase="consultation",
                    actor=PRIMARY_DOCTOR,
                    event_type="test_request",
                    details={
                        "turn_index": turn_index,
                        "requested_test_name": action.test_name,
                        "normalized_test_name": normalized_request,
                    },
                    visit_index=visit.visit_index,
                )
            )
            if over_test_budget:
                response = TestServiceResponse(
                    test_name=action.test_name,
                    returned=False,
                    result_text="由于检查资源有限，已超过最大检查次数，本次不返回检查结果。",
                    evidence_tags=["budget_limited"],
                    match_type="budget_limited",
                )
            else:
                response = test_service.request_test(action.test_name)
            tests_requested_names.append(response.test_name)
            test_record = TestRequestRecord(
                visit_index=visit.visit_index,
                turn_index=turn_index,
                test_name=response.test_name,
                returned=response.returned,
                result_if_any=response.result_text if response.returned else None,
                match_type=response.match_type,
            )
            visit.tests_requested.append(test_record)
            episode.tests_requested.append(test_record)

            if response.returned:
                released_results.append(f"{response.test_name}: {response.result_text}")

            exam_lines = [
                DialogueLine(
                    speaker=EXAMINER,
                    text=f"{response.test_name}：{response.result_text}",
                    phase="examination",
                    visit_index=visit.visit_index,
                    turn_index=turn_index,
                )
            ]
            _append_phase(
                episode,
                visit,
                "examination",
                "检查结果返回" if response.returned else "检查请求未返回结果",
                exam_lines,
                EXAMINER,
                "exam_result",
                {"test_name": response.test_name, "returned": response.returned, "match_type": response.match_type},
            )
            _log_test_result(
                turn_index=turn_index,
                test_name=response.test_name,
                returned=response.returned,
                match_type=response.match_type,
                result_text=response.result_text,
            )
            episode.event_log.append(
                EventRecord(
                    phase="examination",
                    actor=EXAMINER,
                    event_type="test_result",
                    details={
                        "turn_index": turn_index,
                        "test_name": response.test_name,
                        "returned": response.returned,
                        "match_type": response.match_type,
                        "result": _compact_for_log(response.result_text),
                    },
                    visit_index=visit.visit_index,
                )
            )
            _log_lines([order_line] + exam_lines)
            yield {
                "type": "test_result",
                "lines": [order_line] + exam_lines,
                "decision": decision_record,
                "response": response,
                "question_count": visit.question_count,
                "test_count": visit.test_count,
            }
            continue

        forced = False
        if action.action == "FINALIZE":
            decision = action.to_final_decision()
        else:
            forced = True
            decision = doctor.build_forced_final_decision(known_history, released_results, None)
            action = DoctorAction(
                action="FINALIZE",
                utterance="我现在基于现有证据给出最终判断。",
                diagnosis=list(decision.diagnosis),
                evidence=list(decision.evidence),
                plan=list(decision.plan),
                followup=list(decision.followup),
                rationale=["达到预算上限或无法继续有效收集信息，系统强制终诊。"],
            )

        visit.turn_count += 1
        decision_record = _record_doctor_decision(episode, visit, action, turn_index, forced=forced)
        _log_decision(decision_record)

        visit.doctor_finalize = decision
        episode.doctor_outputs.append(decision)
        diagnosis_lines = _decision_dialogue_lines(decision, visit.visit_index)
        _append_phase(
            episode,
            visit,
            "diagnosis",
            "医生做出最终决断",
            diagnosis_lines,
            PRIMARY_DOCTOR,
            "final_decision",
            {"diagnosis": decision.diagnosis},
        )
        _log_lines(diagnosis_lines)
        yield {
            "type": "final_decision",
            "lines": diagnosis_lines,
            "decision": decision,
            "doctor_decision": decision_record,
        }

        visit_summary_for_manager = _visit_summary_for_manager(visit, decision)
        manager_assessment = manager.assess_case(visit_summary_for_manager, decision)
        visit.manager_feedback = manager_assessment
        episode.manager_feedbacks.append(manager_assessment)
        manager_lines = _manager_dialogue_lines(manager_assessment, visit.visit_index)
        _append_phase(
            episode,
            visit,
            "manager_review",
            "管理者裁决",
            manager_lines,
            MANAGER_NAME,
            "manager_feedback",
            {
                "correct_diagnosis": manager_assessment.correct_diagnosis,
                "effective_treatment": manager_assessment.effective_treatment,
                "approved": manager_assessment.cured,
                "diagnosis_grade": manager_assessment.diagnosis_grade,
                "diagnosis_score": manager_assessment.diagnosis_score,
                "evidence_score": manager_assessment.evidence_score,
                "plan_score": manager_assessment.plan_score,
                "total_score": manager_assessment.total_score,
                "evidence_stats": {
                    "me": manager_assessment.evidence_stats.matched,
                    "he": manager_assessment.evidence_stats.incorrect,
                    "Ke": manager_assessment.evidence_stats.total_key_points,
                },
                "management_stats": {
                    "mt": manager_assessment.management_stats.matched,
                    "ht": manager_assessment.management_stats.incorrect,
                    "Kt": manager_assessment.management_stats.total_key_points,
                },
            },
        )
        _log_lines(manager_lines)
        logger.info(
            "MANAGER_REVIEW gold=%s predicted=%s diagnosis_grade=%s diagnosis_score=%.2f evidence_score=%.2f plan_score=%.2f total_score=%.2f approved=%s",
            episode.case.gold_diagnosis,
            "；".join(decision.diagnosis) if decision.diagnosis else "-",
            manager_assessment.diagnosis_grade,
            manager_assessment.diagnosis_score,
            manager_assessment.evidence_score,
            manager_assessment.plan_score,
            manager_assessment.total_score,
            manager_assessment.cured,
        )
        yield {"type": "manager_review", "lines": manager_lines, "assessment": manager_assessment}

        metrics.diagnosis_correct = manager_assessment.correct_diagnosis
        metrics.treatment_effective = manager_assessment.effective_treatment
        metrics.diagnosis_score = manager_assessment.diagnosis_score
        metrics.evidence_score = manager_assessment.evidence_score
        metrics.plan_score = manager_assessment.plan_score
        metrics.total_score = manager_assessment.total_score

        learning_enabled = retrieval_enabled and not read_only_memory
        if learning_enabled:
            learning_experience = manager.generate_learning_experience(
                visit_summary=visit_summary_for_manager,
                decision=decision,
                assessment=manager_assessment,
            )
            rule_cards = manager.generate_rule_cards(
                visit_summary=visit_summary_for_manager,
                decision=decision,
                assessment=manager_assessment,
            )

            if manager_assessment.cured:
                success_entry = memory.write_success_entry(
                    episode,
                    visit,
                    decision,
                    manager_assessment,
                    learning_experience,
                )
                _record_kb_write(
                    episode,
                    visit.visit_index,
                    "success",
                    f"experience={success_entry.get('experience')}",
                )
                episode.termination_reason = "approved"
                episode.completion_status = "success"
                metrics.cured = True
                outcome_tag = "success"
                learning_text = "已写入成功经验库。"
                yield {"type": "kb_write", "entry_type": "success", "payload": success_entry}
            else:
                correction_entry = memory.write_correction_entry(
                    episode,
                    visit,
                    decision,
                    manager_assessment,
                    learning_experience,
                )
                _record_kb_write(
                    episode,
                    visit.visit_index,
                    "correction",
                    f"experience={correction_entry.get('experience')}",
                )
                episode.termination_reason = "rejected"
                episode.completion_status = "failed"
                metrics.cured = False
                outcome_tag = "failure"
                learning_text = "已写入失败纠正库。"
                yield {"type": "kb_write", "entry_type": "correction", "payload": correction_entry}

            rule_card_entries = memory.write_rule_card_entries(
                episode,
                visit,
                decision,
                manager_assessment,
                rule_cards,
                outcome=outcome_tag,
            )
            for card in rule_card_entries:
                _record_kb_write(
                    episode,
                    visit.visit_index,
                    f"rule_card:{card.get('card_type', 'unknown')}",
                    f"rule={card.get('rule_text', '')}",
                )
                yield {"type": "kb_write", "entry_type": "rule_card", "payload": card}

            learning_text = f"{learning_text} 已写入规则卡 {len(rule_card_entries)} 张。"

            learning_lines = [
                DialogueLine(
                    speaker="系统",
                    text=learning_text,
                    phase="learning_write",
                    visit_index=visit.visit_index,
                )
            ]
            _append_phase(
                episode,
                visit,
                "learning_write",
                "经验库写入",
                learning_lines,
                "System",
                "knowledge_write",
                {"termination_reason": episode.termination_reason},
            )
            _log_lines(learning_lines)
        else:
            episode.termination_reason = "approved" if manager_assessment.cured else "rejected"
            episode.completion_status = "success" if manager_assessment.cured else "failed"
            metrics.cured = bool(manager_assessment.cured)
        finalized = True
        break

    if not finalized:
        visit.termination_reason = "forced_finalize"
        forced_decision = doctor.build_forced_final_decision(known_history, released_results, None)
        episode.doctor_outputs.append(forced_decision)
        visit.doctor_finalize = forced_decision
        episode.final_clinical_decision = forced_decision
        episode.termination_reason = "forced_finalize"
        episode.completion_status = "failed"
    else:
        episode.final_clinical_decision = visit.doctor_finalize or FinalClinicalDecision()

    objective = compute_objective_metrics(
        decisions=episode.doctor_decisions,
        tests_requested=episode.tests_requested,
        s_result=metrics.total_score,
        max_total_turns=config.workflow.max_total_turns,
        max_tests_per_visit=config.workflow.max_tests_per_visit,
        llm_error_turns=metrics.llm_error_turns,
    )
    metrics.num_turns = objective.turns
    metrics.num_tests = objective.tests
    metrics.invalid_tests = objective.invalid_tests
    metrics.invalid_test_rate = objective.invalid_test_rate
    metrics.utility = objective.utility
    latest_assessment = episode.manager_feedbacks[-1] if episode.manager_feedbacks else None
    episode.result_summary = {
        "gold_diagnosis": episode.case.gold_diagnosis,
        "predicted_diagnosis": list(episode.final_clinical_decision.diagnosis),
        "diagnosis_grade": (latest_assessment.diagnosis_grade if latest_assessment else None),
        "mdt": {
            "mdt_called": mdt_called,
            "mdt_trigger_turn": mdt_trigger_turn,
            "consultant_advice_action": (
                str(consultant_opinion.get("advice_action", "")).strip().upper() if consultant_opinion else None
            ),
            "consultant_confidence": (
                float(consultant_opinion.get("confidence", 0.0))
                if consultant_opinion and isinstance(consultant_opinion.get("confidence"), (int, float))
                else None
            ),
        },
        "scores": {
            "diagnosis_score": metrics.diagnosis_score,
            "evidence_score": metrics.evidence_score,
            "plan_score": metrics.plan_score,
            "total_score": metrics.total_score,
        },
        "objective_metrics": {
            "turns": metrics.num_turns,
            "tests": metrics.num_tests,
            "llm_error_turns": metrics.llm_error_turns,
            "raw_interface_error_turns": metrics.raw_interface_error_turns,
            "repaired_interface_turns": metrics.repaired_interface_turns,
            "residual_interface_error_turns": metrics.residual_interface_error_turns,
            "deterministic_repair_turns": metrics.deterministic_repair_turns,
            "format_reprompt_turns": metrics.format_reprompt_turns,
            "invalid_tests": metrics.invalid_tests,
            "invalid_test_rate": metrics.invalid_test_rate,
            "utility": metrics.utility,
        },
        "termination_reason": episode.termination_reason,
    }
    logger.info(
        "OBJECTIVE_METRICS turns=%s tests=%s llm_error_turns=%s invalid_tests=%s invalid_test_rate=%.3f utility=%.3f",
        metrics.num_turns,
        metrics.num_tests,
        metrics.llm_error_turns,
        metrics.invalid_tests,
        metrics.invalid_test_rate,
        metrics.utility,
    )

    visit.termination_reason = episode.termination_reason
    metrics.visit_count = len(episode.visit_records)
    episode.visit_count = metrics.visit_count
    episode.mdt_used = metrics.mdt_used
    episode.mdt_called = mdt_called
    episode.mdt_trigger_turn = mdt_trigger_turn
    episode.consultant_opinion = dict(consultant_opinion) if consultant_opinion else None
    episode.consultant_advice_action = (
        str(consultant_opinion.get("advice_action", "")).strip().upper() if consultant_opinion else None
    )
    episode.consultant_confidence = (
        float(consultant_opinion.get("confidence", 0.0))
        if consultant_opinion and isinstance(consultant_opinion.get("confidence"), (int, float))
        else None
    )
    episode.metrics = metrics
    episode.learning_trace = memory.write_learning_trace(episode)

    if config.runtime.use_simulacra_bridge:
        episode.simulacra_replay = build_simulacra_replay(episode, sim_code)
    write_episode_artifacts(
        episode,
        episodes_root=episode_output_root,
        write_legacy_outputs=config.runtime.write_legacy_outputs,
    )
    logger.info(
        "Scenario %s finished termination=%s visits=%s turns=%s tests=%s llm_error_turns=%s invalid_rate=%.3f utility=%.3f",
        scenario_id,
        episode.termination_reason,
        episode.visit_count,
        episode.metrics.num_turns,
        episode.metrics.num_tests,
        episode.metrics.llm_error_turns,
        episode.metrics.invalid_test_rate,
        episode.metrics.utility,
    )
    yield {"type": "episode_complete", "episode": episode}


def run_episode(
    config: ClinicalRuntimeConfig,
    scenario_id: int,
    sim_code: Optional[str] = None,
    *,
    run_id: Optional[str] = None,
    episode_output_root: Optional[Path] = None,
) -> EpisodeResult:
    final_episode: Optional[EpisodeResult] = None
    for event in run_episode_stream(
        config=config,
        scenario_id=scenario_id,
        sim_code=sim_code,
        run_id=run_id,
        episode_output_root=episode_output_root,
    ):
        if event.get("type") == "episode_complete":
            final_episode = event["episode"]
    if final_episode is None:
        raise RuntimeError(f"Scenario {scenario_id} did not produce a final episode.")
    return final_episode


def prepare_simulacra_episode(
    config: ClinicalRuntimeConfig,
    scenario_id: int,
    sim_code: str,
) -> EpisodeResult:
    return run_episode(config=config, scenario_id=scenario_id, sim_code=sim_code)


def load_episode_from_default_config(scenario_id: int, sim_code: Optional[str] = None) -> EpisodeResult:
    config = load_clinical_config()
    return run_episode(config=config, scenario_id=scenario_id, sim_code=sim_code)






