from __future__ import annotations

import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from medsim.clinical.experiment4 import build_case_metadata_payload
from medsim.clinical.schema import EpisodeResult, dataclass_to_dict


logger = logging.getLogger(__name__)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Invalid JSONL row ignored path=%s", path)
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_completed_scenario_ids(case_summaries_path: Path, legacy_cases_path: Optional[Path] = None) -> Set[int]:
    completed: Set[int] = set()
    for row in read_jsonl(case_summaries_path):
        scenario_id = _safe_int(row.get("scenario_id"))
        if scenario_id is not None:
            completed.add(scenario_id)
    if legacy_cases_path is not None:
        for row in read_jsonl(legacy_cases_path):
            scenario_id = _safe_int(row.get("scenario_id"))
            if scenario_id is not None:
                completed.add(scenario_id)
    return completed


def build_case_summary_from_episode(
    episode: EpisodeResult,
    run_id: str,
    include_case_metadata: bool = False,
) -> Dict[str, Any]:
    assessment = episode.manager_feedbacks[-1] if episode.manager_feedbacks else None
    llm_status = (episode.learning_trace or {}).get("llm_runtime", {}).get("status")
    llm_feedback = (episode.learning_trace or {}).get("llm_feedback")
    payload = {
        "episode_id": episode.episode_id,
        "scenario_id": episode.scenario_id,
        "run_id": run_id,
        "termination_reason": episode.termination_reason,
        "llm_status": llm_status,
        "llm_feedback": llm_feedback,
        "diagnosis": {
            "gold": episode.case.gold_diagnosis,
            "predicted": list(episode.final_clinical_decision.diagnosis),
            "diagnosis_grade": assessment.diagnosis_grade if assessment else None,
        },
        "scoring": {
            "diagnosis_score": episode.metrics.diagnosis_score,
            "evidence_score": episode.metrics.evidence_score,
            "plan_score": episode.metrics.plan_score,
            "total_score": episode.metrics.total_score,
        },
        "objective_metrics": {
            "turns": episode.metrics.num_turns,
            "tests": episode.metrics.num_tests,
            "llm_error_turns": episode.metrics.llm_error_turns,
            "raw_interface_error_turns": episode.metrics.raw_interface_error_turns,
            "repaired_interface_turns": episode.metrics.repaired_interface_turns,
            "residual_interface_error_turns": episode.metrics.residual_interface_error_turns,
            "deterministic_repair_turns": episode.metrics.deterministic_repair_turns,
            "format_reprompt_turns": episode.metrics.format_reprompt_turns,
            "invalid_tests": episode.metrics.invalid_tests,
            "invalid_test_rate": episode.metrics.invalid_test_rate,
            "utility": episode.metrics.utility,
        },
    }
    if include_case_metadata:
        payload.update(build_case_metadata_payload(episode.case))
    return payload


def build_case_summary_from_error(
    *,
    scenario_id: int,
    run_id: str,
    error_type: str,
    error_message: str,
    case_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "episode_id": f"episode-{scenario_id}-cli",
        "scenario_id": int(scenario_id),
        "run_id": run_id,
        "termination_reason": "error",
        "llm_status": None,
        "llm_feedback": None,
        "diagnosis": {
            "gold": None,
            "predicted": [],
            "diagnosis_grade": None,
        },
        "scoring": {
            "diagnosis_score": None,
            "evidence_score": None,
            "plan_score": None,
            "total_score": None,
        },
        "objective_metrics": {
            "turns": None,
            "tests": None,
            "llm_error_turns": None,
            "raw_interface_error_turns": None,
            "repaired_interface_turns": None,
            "residual_interface_error_turns": None,
            "deterministic_repair_turns": None,
            "format_reprompt_turns": None,
            "invalid_tests": None,
            "invalid_test_rate": None,
            "utility": None,
        },
        "error_type": str(error_type or "runtime_error"),
        "error_message": str(error_message or ""),
    }
    if case_metadata:
        payload.update(case_metadata)
    return payload


def append_success_episode(legacy_cases_path: Path, episode: EpisodeResult) -> None:
    append_jsonl(legacy_cases_path, dataclass_to_dict(episode))


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_legacy_summary_json(run_id: str, legacy_cases_path: Path, legacy_summary_path: Path) -> Dict[str, Any]:
    cases = read_jsonl(legacy_cases_path)
    metrics_list = [item.get("metrics", {}) for item in cases]
    summary = {
        "run_id": run_id,
        "num_cases": len(cases),
        "cured_cases": sum(1 for item in metrics_list if bool(item.get("cured", False))),
        "avg_visits": _mean([float(item.get("visit_count", 0) or 0) for item in cases]),
        "avg_turns": _mean([float(item.get("num_turns", 0) or 0) for item in metrics_list]),
        "avg_tests": _mean([float(item.get("num_tests", 0) or 0) for item in metrics_list]),
        "avg_llm_error_turns": _mean([float(item.get("llm_error_turns", 0) or 0) for item in metrics_list]),
        "avg_raw_interface_error_turns": _mean(
            [float(item.get("raw_interface_error_turns", 0) or 0) for item in metrics_list]
        ),
        "avg_repaired_interface_turns": _mean(
            [float(item.get("repaired_interface_turns", 0) or 0) for item in metrics_list]
        ),
        "avg_residual_interface_error_turns": _mean(
            [float(item.get("residual_interface_error_turns", 0) or 0) for item in metrics_list]
        ),
        "avg_invalid_test_rate": _mean([float(item.get("invalid_test_rate", 0) or 0) for item in metrics_list]),
        "avg_utility": _mean([float(item.get("utility", 0) or 0) for item in metrics_list]),
        "avg_diag_score": _mean([float(item.get("diagnosis_score", 0) or 0) for item in metrics_list]),
        "avg_evidence_score": _mean([float(item.get("evidence_score", 0) or 0) for item in metrics_list]),
        "avg_plan_score": _mean([float(item.get("plan_score", 0) or 0) for item in metrics_list]),
        "avg_total_score": _mean([float(item.get("total_score", 0) or 0) for item in metrics_list]),
        "pass_rate": _mean([1.0 if bool(item.get("cured", False)) else 0.0 for item in metrics_list]),
    }
    write_json(legacy_summary_path, summary)
    return summary


def build_batch_summary(
    *,
    run_id: str,
    target_cases: int,
    attempted: int,
    succeeded: int,
    failed: int,
    skipped: int,
    rows: List[Dict[str, Any]],
    legacy_success_count: int,
) -> Dict[str, Any]:
    outcome_counter = Counter(str(row.get("termination_reason") or "unknown") for row in rows)
    return {
        "run_id": run_id,
        "target_cases": int(target_cases),
        "attempted": int(attempted),
        "succeeded": int(succeeded),
        "failed": int(failed),
        "skipped": int(skipped),
        "records_total": len(rows),
        "records_success": int(legacy_success_count),
        "records_error": int(outcome_counter.get("error", 0)),
        "outcome_distribution": dict(outcome_counter),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
    }


def _value_or_nan(row: Dict[str, Any], *keys: str) -> float:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict):
            return math.nan
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return math.nan


def _sorted_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=lambda item: (_safe_int(item.get("scenario_id")) is None, _safe_int(item.get("scenario_id")) or 0))


def _plot_placeholder(plt, title: str, output_path: Path, message: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(title)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def render_batch_plots(rows: List[Dict[str, Any]], run_dir: Path) -> Dict[str, str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        logger.warning("Plot generation skipped: matplotlib unavailable (%s)", exc)
        return {}

    run_dir.mkdir(parents=True, exist_ok=True)
    sorted_rows = _sorted_rows(rows)
    x_values = [_safe_int(item.get("scenario_id")) for item in sorted_rows]

    score_path = run_dir / "plot_scores.png"
    objective_path = run_dir / "plot_objective.png"
    outcome_path = run_dir / "plot_outcomes.png"

    valid_points = [idx for idx, scenario_id in enumerate(x_values) if scenario_id is not None]
    if not valid_points:
        _plot_placeholder(plt, "Clinical Scores", score_path, "No scenario data")
        _plot_placeholder(plt, "Objective Metrics", objective_path, "No scenario data")
    else:
        x_plot = [int(x_values[idx]) for idx in valid_points]

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "scoring", "diagnosis_score") for idx in valid_points], marker="o", label="diagnosis_score")
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "scoring", "evidence_score") for idx in valid_points], marker="o", label="evidence_score")
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "scoring", "plan_score") for idx in valid_points], marker="o", label="plan_score")
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "scoring", "total_score") for idx in valid_points], marker="o", label="total_score")
        ax.set_title("Clinical Scores by Scenario")
        ax.set_xlabel("scenario_id")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(score_path, dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "objective_metrics", "turns") for idx in valid_points], marker="o", label="turns")
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "objective_metrics", "tests") for idx in valid_points], marker="o", label="tests")
        ax.plot(x_plot, [_value_or_nan(sorted_rows[idx], "objective_metrics", "utility") for idx in valid_points], marker="o", label="utility")
        ax.set_title("Objective Metrics by Scenario")
        ax.set_xlabel("scenario_id")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(objective_path, dpi=150)
        plt.close(fig)

    outcomes = Counter(str(item.get("termination_reason") or "unknown") for item in rows)
    if not outcomes:
        _plot_placeholder(plt, "Termination Outcomes", outcome_path, "No outcome data")
    else:
        labels = list(outcomes.keys())
        values = [outcomes[label] for label in labels]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, values)
        ax.set_title("Termination Outcome Distribution")
        ax.set_xlabel("termination_reason")
        ax.set_ylabel("count")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(outcome_path, dpi=150)
        plt.close(fig)

    return {
        "plot_scores": score_path.name,
        "plot_objective": objective_path.name,
        "plot_outcomes": outcome_path.name,
    }
