from __future__ import annotations

from enum import Enum
from typing import Any


class ActionType(str, Enum):
    ASK = "ASK"
    REQUEST_TEST = "REQUEST_TEST"
    FINALIZE = "FINALIZE"
    CALL_MDT = "CALL_MDT"


ACTION_ALIASES = {
    "ASK": ActionType.ASK,
    "QUESTION": ActionType.ASK,
    "REQUEST_TEST": ActionType.REQUEST_TEST,
    "TEST": ActionType.REQUEST_TEST,
    "FINALIZE": ActionType.FINALIZE,
    "DECIDE": ActionType.FINALIZE,
    "CALL_MDT": ActionType.CALL_MDT,
}


class DoctorAction:
    def __init__(
        self,
        action: ActionType,
        utterance: str,
        question: str = "",
        test_name: str = "",
        diagnosis: list[str] | None = None,
        evidence: list[str] | None = None,
        plan: list[str] | None = None,
        followup: list[str] | None = None,
        rationale: list[str] | None = None,
        working_diagnoses: list[str] | None = None,
    ) -> None:
        self.action = action
        self.utterance = utterance
        self.question = question
        self.test_name = test_name
        self.diagnosis = list(diagnosis or [])
        self.evidence = list(evidence or [])
        self.plan = list(plan or [])
        self.followup = list(followup or [])
        self.rationale = list(rationale or [])
        self.working_diagnoses = list(working_diagnoses or [])

    def as_event_details(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "utterance": self.utterance,
            "question": self.question,
            "test_name": self.test_name,
            "diagnosis": list(self.diagnosis),
            "evidence": list(self.evidence),
            "plan": list(self.plan),
            "followup": list(self.followup),
            "rationale": list(self.rationale),
            "working_diagnoses": list(self.working_diagnoses),
        }


def parse_action(value: str) -> ActionType:
    normalized = str(value or "").strip().upper()
    if normalized not in ACTION_ALIASES:
        raise ValueError(f"Unsupported action: {value!r}")
    return ACTION_ALIASES[normalized]
