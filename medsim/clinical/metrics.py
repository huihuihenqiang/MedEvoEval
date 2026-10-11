from __future__ import annotations

from dataclasses import dataclass
from typing import List

from medsim.clinical.schema import DoctorDecisionRecord, TestRequestRecord


@dataclass
class ObjectiveMetrics:
    turns: int
    tests: int
    invalid_tests: int
    invalid_test_rate: float
    utility: float


def _is_invalid_test_record(record: TestRequestRecord) -> bool:
    # Invalid test is defined as "no returned result" or "blank returned text".
    if not record.returned:
        return True
    return not str(record.result_if_any or "").strip()


def _count_turns(decisions: List[DoctorDecisionRecord]) -> int:
    # Turns only include ASK + REQUEST_TEST; FINALIZE/CALL_MDT are excluded.
    return sum(1 for item in decisions if item.action in {"ASK", "REQUEST_TEST"})


def _count_tests(decisions: List[DoctorDecisionRecord]) -> int:
    # Tests are exactly the number of REQUEST_TEST doctor actions.
    return sum(1 for item in decisions if item.action == "REQUEST_TEST")


def compute_objective_metrics(
    decisions: List[DoctorDecisionRecord],
    tests_requested: List[TestRequestRecord],
    s_result: float,
    max_total_turns: int,
    max_tests_per_visit: int,
    llm_error_turns: int = 0,
) -> ObjectiveMetrics:
    turns = _count_turns(decisions) + max(int(llm_error_turns), 0)
    tests = _count_tests(decisions)
    invalid_tests = sum(1 for item in tests_requested if _is_invalid_test_record(item))
    invalid_test_rate = (invalid_tests / tests) if tests else 0.0

    # Match the action budget: up to B counted attempts can precede finalization.
    t_max = max(int(max_total_turns), 1)
    # E_max uses configured max tests with lower-bound protection.
    e_max = max(int(max_tests_per_visit), 1)

    utility = (
        0.6 * float(s_result)
        - 0.2 * (turns / t_max)
        - 0.1 * (tests / e_max)
        - 0.1 * invalid_test_rate
    )
    return ObjectiveMetrics(
        turns=turns,
        tests=tests,
        invalid_tests=invalid_tests,
        invalid_test_rate=invalid_test_rate,
        utility=utility,
    )
