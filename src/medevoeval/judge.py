from __future__ import annotations

from medevoeval.scoring import ScoringConfig, objective_utility, score_fraction, total_score


def _norm(text: str) -> str:
    return " ".join(str(text or "").strip().lower().split())


class Judge:
    def __init__(self, case: dict, scoring_config: ScoringConfig | None = None) -> None:
        self.case = dict(case)
        self.config = scoring_config or ScoringConfig()
        self.gt = (self.case.get("manager_view") or {}).get("ground_truth") or {}

    def score(self, decision: dict, turns: int, tests: int, invalid_tests: int, max_turns: int, max_tests: int) -> dict:
        diagnosis = self.gt.get("diagnosis") or {}
        aliases = [
            diagnosis.get("display_name"),
            diagnosis.get("primary_label"),
            *(diagnosis.get("acceptable_names") or []),
        ]
        predicted = decision.get("diagnosis") or []
        exact = any(_norm(item) and _norm(item) in {_norm(alias) for alias in aliases if alias} for item in predicted)
        diagnosis_score = 1.0 if exact else 0.25

        evidence_points = list((self.gt.get("evidence_rubric") or {}).get("key_points") or [])
        management_points = list((self.gt.get("management_rubric") or {}).get("key_points") or [])
        evidence_text = " ".join(decision.get("evidence") or [])
        plan_text = " ".join((decision.get("plan") or []) + (decision.get("followup") or []))
        matched_evidence = sum(1 for item in evidence_points if _norm(item) and _norm(item) in _norm(evidence_text))
        matched_management = sum(1 for item in management_points if _norm(item) and _norm(item) in _norm(plan_text))
        evidence_score = score_fraction(matched_evidence, 0, len(evidence_points), self.config.lambda_evidence)
        plan_score = score_fraction(matched_management, 0, len(management_points), self.config.mu_management)
        total = total_score(diagnosis_score, evidence_score, plan_score, self.config)
        invalid_rate = float(invalid_tests) / float(tests) if tests else 0.0
        utility = objective_utility(total, turns, tests, invalid_rate, max_turns, max_tests)
        return {
            "diagnosis_score": diagnosis_score,
            "evidence_score": evidence_score,
            "plan_score": plan_score,
            "total_score": total,
            "turns": int(turns),
            "tests": int(tests),
            "invalid_tests": int(invalid_tests),
            "invalid_test_rate": invalid_rate,
            "utility": utility,
            "strict_correct": int(exact),
            "pass_rate": int(total >= self.config.pass_threshold),
            "nondanger_rate": int(diagnosis_score > 0.0),
        }

