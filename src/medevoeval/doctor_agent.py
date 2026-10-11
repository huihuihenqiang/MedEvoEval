from __future__ import annotations

import json

from medevoeval.actions import ActionType, DoctorAction
from medevoeval.api_client import OpenAICompatibleClient


class MockDoctorAgent:
    """Deterministic mock doctor for offline artifact review."""

    def __init__(self, case: dict, mdt_enabled: bool = False) -> None:
        self.case = dict(case)
        self.mdt_enabled = bool(mdt_enabled)
        self.called_mdt = False

    def next_action(
        self,
        turn_index: int,
        known_history: list[str],
        released_results: list[str],
        tests_requested: list[str],
        max_tests: int,
    ) -> DoctorAction:
        case = self.case
        test_catalog = list(case.get("available_test_catalog") or [])
        if turn_index == 0:
            return DoctorAction(
                action=ActionType.ASK,
                utterance="Please describe the symptom timeline and associated findings.",
                question="What symptoms changed over time?",
                rationale=["Initial history is required before diagnosis."],
            )
        if len(tests_requested) < max_tests and test_catalog:
            next_test = test_catalog[min(len(tests_requested), len(test_catalog) - 1)]
            return DoctorAction(
                action=ActionType.REQUEST_TEST,
                utterance=f"I would like to order {next_test}.",
                test_name=next_test,
                rationale=["Request the highest-yield available test."],
            )
        if self.mdt_enabled and not self.called_mdt and turn_index >= 2:
            self.called_mdt = True
            return DoctorAction(
                action=ActionType.CALL_MDT,
                utterance="I request a consultant review before finalizing.",
                rationale=["MDT is enabled and the case has enough context."],
            )
        diagnosis = [case.get("demo_gold_diagnosis") or "undifferentiated condition"]
        evidence = known_history[:2] + released_results[:2]
        plan = list(case.get("demo_plan") or ["Provide condition-specific management and follow-up."])
        return DoctorAction(
            action=ActionType.FINALIZE,
            utterance="I will finalize based on the available evidence.",
            diagnosis=diagnosis,
            evidence=evidence,
            plan=plan,
            followup=["Arrange follow-up and safety-net instructions."],
            rationale=["Mock mode uses the toy case reference label."],
        )


class APIDoctorAgent:
    """OpenAI-compatible live doctor agent for reviewer-owned API reruns."""

    def __init__(self, case: dict, client: OpenAICompatibleClient, mdt_enabled: bool = False) -> None:
        self.case = dict(case)
        self.client = client
        self.mdt_enabled = bool(mdt_enabled)
        self.called_mdt = False

    def next_action(
        self,
        turn_index: int,
        known_history: list[str],
        released_results: list[str],
        tests_requested: list[str],
        max_tests: int,
    ) -> DoctorAction:
        payload = {
            "case_id": self.case.get("case_id"),
            "task": self.case.get("task") or self.case.get("presenting_problem"),
            "chief_complaint": (self.case.get("patient_view") or {}).get("chief_complaint"),
            "known_history": known_history,
            "released_test_results": released_results,
            "tests_requested": tests_requested,
            "available_tests": self.case.get("available_test_catalog") or (self.case.get("exam_view") or {}).get("test_catalog") or [],
            "turn_index": turn_index,
            "max_tests": max_tests,
            "mdt_enabled": self.mdt_enabled,
            "mdt_already_called": self.called_mdt,
        }
        system = (
            "You are a careful outpatient doctor agent in a simulated Chinese medical evaluation. "
            "Choose exactly one next action. Return only valid JSON. "
            "Allowed actions are ASK, REQUEST_TEST, CALL_MDT, FINALIZE. "
            "Use REQUEST_TEST only with a test name exactly from available_tests. "
            "Use FINALIZE when evidence is sufficient or no useful action remains."
        )
        schema = {
            "action": "ASK | REQUEST_TEST | CALL_MDT | FINALIZE",
            "utterance": "Chinese text to say to the patient or system",
            "question": "question text when action is ASK",
            "test_name": "test name when action is REQUEST_TEST",
            "working_diagnoses": ["optional working diagnoses"],
            "rationale": ["brief rationale strings"],
            "diagnosis": ["final diagnoses when action is FINALIZE"],
            "evidence": ["supporting evidence when action is FINALIZE"],
            "plan": ["management plan when action is FINALIZE"],
            "followup": ["follow-up/safety-net instructions when action is FINALIZE"],
        }
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "Return JSON matching this schema:\n"
                + json.dumps(schema, ensure_ascii=False)
                + "\n\nCurrent state:\n"
                + json.dumps(payload, ensure_ascii=False),
            },
        ]
        response = self.client.chat_json(messages)
        action_text = str(response.get("action") or "ASK").strip().upper()
        if action_text not in {item.value for item in ActionType}:
            action_text = "ASK"
        if action_text == "CALL_MDT":
            if not self.mdt_enabled or self.called_mdt:
                action_text = "FINALIZE"
            self.called_mdt = True
        action = ActionType(action_text)
        test_name = str(response.get("test_name") or "").strip()
        if action == ActionType.REQUEST_TEST:
            available = list(payload["available_tests"])
            if test_name not in available and available:
                test_name = available[0]
        return DoctorAction(
            action=action,
            utterance=str(response.get("utterance") or response.get("question") or action.value),
            question=str(response.get("question") or ""),
            test_name=test_name,
            diagnosis=[str(item) for item in response.get("diagnosis") or []],
            evidence=[str(item) for item in response.get("evidence") or []],
            plan=[str(item) for item in response.get("plan") or []],
            followup=[str(item) for item in response.get("followup") or []],
            rationale=[str(item) for item in response.get("rationale") or []],
            working_diagnoses=[str(item) for item in response.get("working_diagnoses") or []],
        )
