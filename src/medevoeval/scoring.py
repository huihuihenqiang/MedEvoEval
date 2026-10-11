from __future__ import annotations

class ScoringConfig:
    def __init__(
        self,
        lambda_evidence: float = 0.5,
        mu_management: float = 1.0,
        weight_diag: float = 0.5,
        weight_evidence: float = 0.3,
        weight_plan: float = 0.2,
        pass_threshold: float = 0.7,
        effective_treatment_threshold: float = 0.5,
    ) -> None:
        self.lambda_evidence = float(lambda_evidence)
        self.mu_management = float(mu_management)
        self.weight_diag = float(weight_diag)
        self.weight_evidence = float(weight_evidence)
        self.weight_plan = float(weight_plan)
        self.pass_threshold = float(pass_threshold)
        self.effective_treatment_threshold = float(effective_treatment_threshold)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def score_fraction(matched: int, incorrect: int, total: int, penalty: float) -> float:
    if total <= 0:
        return 0.0
    return clamp((float(matched) - float(penalty) * float(incorrect)) / float(total))


def total_score(
    diagnosis_score: float,
    evidence_score: float,
    plan_score: float,
    config: ScoringConfig | None = None,
) -> float:
    cfg = config or ScoringConfig()
    return clamp(
        cfg.weight_diag * diagnosis_score
        + cfg.weight_evidence * evidence_score
        + cfg.weight_plan * plan_score
    )


def objective_utility(
    total_score_value: float,
    turns: int,
    tests: int,
    invalid_test_rate: float,
    max_turns: int,
    max_tests: int,
) -> float:
    t_max = max(int(max_turns), 1)
    e_max = max(int(max_tests), 1)
    return (
        0.6 * float(total_score_value)
        - 0.2 * (float(turns) / t_max)
        - 0.1 * (float(tests) / e_max)
        - 0.1 * float(invalid_test_rate)
    )
