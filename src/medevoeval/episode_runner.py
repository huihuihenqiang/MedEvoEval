from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from medevoeval.actions import ActionType
from medevoeval.api_client import OpenAICompatibleClient
from medevoeval.doctor_agent import MockDoctorAgent
from medevoeval.doctor_agent import APIDoctorAgent
from medevoeval.exam_service import ExamService
from medevoeval.judge import Judge
from medevoeval.mdt_consultant import MockMDTConsultant
from medevoeval.patient_simulator import PatientSimulator
from medevoeval.scoring import ScoringConfig


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_episode(case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    max_turns = int(config.get("max_turns", 6))
    max_tests = int(config.get("max_tests", 2))
    mdt_enabled = bool(config.get("mdt_enabled", False))
    scoring = ScoringConfig(**dict(config.get("scoring") or {}))

    patient = PatientSimulator(case.get("patient_view") or {})
    if str(config.get("llm_mode", "mock")).lower() == "api":
        client = OpenAICompatibleClient(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            model=config.get("model_name"),
            timeout_seconds=int(config.get("timeout_seconds", 90)),
            temperature=float(config.get("temperature", 0.0)),
        )
        doctor = APIDoctorAgent(case, client=client, mdt_enabled=mdt_enabled)
    else:
        doctor = MockDoctorAgent(case, mdt_enabled=mdt_enabled)
    exam_service = ExamService(case.get("exam_view") or {}, case.get("available_test_catalog") or [])
    consultant = MockMDTConsultant()
    judge = Judge(case, scoring)

    known_history: list[str] = []
    released_results: list[str] = []
    tests_requested: list[str] = []
    invalid_tests = 0
    events: list[dict[str, Any]] = []
    final_decision: dict[str, Any] | None = None

    for turn_index in range(max_turns):
        action = doctor.next_action(turn_index, known_history, released_results, tests_requested, max_tests)
        events.append(
            {
                "episode_id": f"demo-{case['case_id']}",
                "case_id": case["case_id"],
                "turn_index": turn_index,
                "actor": "doctor",
                "event_type": "doctor_action",
                "details": action.as_event_details(),
            }
        )
        if action.action == ActionType.ASK:
            patient_turn = patient.respond(action.question or action.utterance, known_history)
            for fact in patient_turn["revealed_facts"]:
                if fact not in known_history:
                    known_history.append(fact)
            events.append(
                {
                    "episode_id": f"demo-{case['case_id']}",
                    "case_id": case["case_id"],
                    "turn_index": turn_index,
                    "actor": "patient",
                    "event_type": "patient_response",
                    "details": patient_turn,
                }
            )
        elif action.action == ActionType.REQUEST_TEST:
            response = exam_service.request_test(action.test_name)
            tests_requested.append(response["test_name"])
            if response["returned"]:
                released_results.append(f"{response['test_name']}: {response['result_text']}")
            else:
                invalid_tests += 1
            events.append(
                {
                    "episode_id": f"demo-{case['case_id']}",
                    "case_id": case["case_id"],
                    "turn_index": turn_index,
                    "actor": "exam_service",
                    "event_type": "test_result",
                    "details": response,
                }
            )
        elif action.action == ActionType.CALL_MDT:
            events.append(
                {
                    "episode_id": f"demo-{case['case_id']}",
                    "case_id": case["case_id"],
                    "turn_index": turn_index,
                    "actor": "mdt_consultant",
                    "event_type": "mdt_review",
                    "details": consultant.review(known_history, released_results),
                }
            )
        elif action.action == ActionType.FINALIZE:
            final_decision = {
                "diagnosis": action.diagnosis,
                "evidence": action.evidence,
                "plan": action.plan,
                "followup": action.followup,
            }
            break

    if final_decision is None:
        action = doctor.next_action(max_turns, known_history, released_results, tests_requested, max_tests)
        final_decision = {
            "diagnosis": action.diagnosis,
            "evidence": action.evidence,
            "plan": action.plan,
            "followup": action.followup,
        }

    turns = sum(1 for event in events if event["event_type"] == "doctor_action" and event["details"]["action"] in {"ASK", "REQUEST_TEST"})
    tests = sum(1 for event in events if event["event_type"] == "doctor_action" and event["details"]["action"] == "REQUEST_TEST")
    scores = judge.score(final_decision, turns, tests, invalid_tests, max_turns, max_tests)
    return {
        "episode_id": f"demo-{case['case_id']}",
        "case_id": case["case_id"],
        "config": {k: v for k, v in config.items() if k not in {"api_key", "base_url"}},
        "events": events,
        "final_decision": final_decision,
        "scores": scores,
    }
