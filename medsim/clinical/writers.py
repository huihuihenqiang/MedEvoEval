from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from medsim.clinical.schema import EpisodeResult, dataclass_to_dict
from medsim.clinical.simulacra_bridge import (
    normalize_simulacra_cast,
    sim_code_dialogue_history_path,
    sim_code_replay_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPISODES_ROOT = PROJECT_ROOT / "output" / "episodes"
RUNS_ROOT = PROJECT_ROOT / "output" / "runs"
LIVE_DIALOGUE_FILE = PROJECT_ROOT / "Simulacra" / "environment" / "frontend_server" / "temp_storage" / "live_dialogue.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, items: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_legacy_dialogue_history(episode: EpisodeResult) -> List[dict]:
    history: List[dict] = []
    for line in episode.full_dialogue:
        history.append(
            {
                "speaker": line.speaker,
                "text": line.text,
                "phase": line.phase,
                "visit_index": line.visit_index,
                "turn_index": line.turn_index,
            }
        )
    return history


def _build_live_dialogue_payload(episode: EpisodeResult) -> dict:
    base_panel = {}
    cast = normalize_simulacra_cast(episode.simulacra_cast)
    if episode.simulacra_replay is not None and episode.simulacra_replay.dialogue_panel:
        base_panel = dict(episode.simulacra_replay.dialogue_panel)

    recent_lines = base_panel.get("recent_lines") or build_legacy_dialogue_history(episode)
    latest_line = base_panel.get("latest_line") or (recent_lines[-1] if recent_lines else None)
    llm_runtime = episode.learning_trace.get("llm_runtime", {}) if episode.learning_trace else {}
    llm_feedback = episode.learning_trace.get("llm_feedback") if episode.learning_trace else None
    return {
        "mode": base_panel.get("mode", "completed"),
        "stage": base_panel.get("stage", "completed_consultation"),
        "focus_personas": base_panel.get("focus_personas", [cast["patient"], cast["doctor"]]),
        "latest_line": latest_line,
        "recent_lines": recent_lines,
        "page_size": base_panel.get("page_size", 3),
        "doctor_decision": dataclass_to_dict(episode.doctor_decisions[-1]) if episode.doctor_decisions else None,
        "medical_summary": {
            "diagnosis": episode.final_clinical_decision.diagnosis,
            "orders": episode.final_clinical_decision.plan,
            "followup": episode.final_clinical_decision.followup,
            "termination_reason": episode.termination_reason,
            "visit_count": episode.visit_count,
            "llm_status": llm_runtime.get("status"),
            "warning": llm_feedback if llm_runtime.get("status") not in {None, "ready"} else None,
        },
        "session_id": episode.episode_id,
        "sim_code": episode.simulacra_replay.sim_code if episode.simulacra_replay else None,
        "camera_plan": base_panel.get("camera_plan", {}),
        "cast": base_panel.get("cast", cast),
        "persona_roles": base_panel.get("persona_roles", {name: role for role, name in cast.items()}),
        "scene_state": base_panel.get("scene_state"),
    }


def write_live_dialogue(episode: EpisodeResult) -> None:
    _write_json(LIVE_DIALOGUE_FILE, _build_live_dialogue_payload(episode))


def write_episode_artifacts(
    episode: EpisodeResult,
    *,
    episodes_root: Path | None = None,
    write_legacy_outputs: bool = True,
) -> Path:
    scenario_dir = (episodes_root or EPISODES_ROOT) / f"scenario_{episode.scenario_id}"
    episode.legacy_dialogue_history = build_legacy_dialogue_history(episode)

    _write_json(scenario_dir / "episode.json", dataclass_to_dict(episode))
    _write_jsonl(scenario_dir / "event_log.jsonl", [dataclass_to_dict(item) for item in episode.event_log])
    _write_json(scenario_dir / "legacy_dialogue_history.json", episode.legacy_dialogue_history)
    if episode.simulacra_replay is not None:
        _write_json(scenario_dir / "simulacra_replay.json", dataclass_to_dict(episode.simulacra_replay))
        if write_legacy_outputs and episode.simulacra_replay.sim_code:
            replay_cache = sim_code_replay_path(episode.simulacra_replay.sim_code)
            dialogue_cache = sim_code_dialogue_history_path(episode.simulacra_replay.sim_code)
            if not replay_cache.exists():
                _write_json(replay_cache, dataclass_to_dict(episode.simulacra_replay))
            if not dialogue_cache.exists():
                _write_json(dialogue_cache, episode.legacy_dialogue_history)
    if write_legacy_outputs:
        write_live_dialogue(episode)
    return scenario_dir


def write_run_artifacts(run_id: str, episodes: List[EpisodeResult]) -> Path:
    run_dir = RUNS_ROOT / run_id
    cases_payload = [dataclass_to_dict(episode) for episode in episodes]
    summary = {
        "run_id": run_id,
        "num_cases": len(episodes),
        "cured_cases": sum(1 for episode in episodes if episode.metrics.cured),
        "avg_visits": (sum(episode.visit_count for episode in episodes) / len(episodes)) if episodes else 0,
        "avg_turns": (sum(episode.metrics.num_turns for episode in episodes) / len(episodes)) if episodes else 0,
        "avg_tests": (sum(episode.metrics.num_tests for episode in episodes) / len(episodes)) if episodes else 0,
        "avg_llm_error_turns": (
            sum(episode.metrics.llm_error_turns for episode in episodes) / len(episodes)
        )
        if episodes
        else 0,
        "avg_invalid_test_rate": (
            sum(episode.metrics.invalid_test_rate for episode in episodes) / len(episodes)
        )
        if episodes
        else 0,
        "avg_utility": (sum(episode.metrics.utility for episode in episodes) / len(episodes)) if episodes else 0,
        "avg_diag_score": (sum(episode.metrics.diagnosis_score for episode in episodes) / len(episodes)) if episodes else 0,
        "avg_evidence_score": (sum(episode.metrics.evidence_score for episode in episodes) / len(episodes))
        if episodes
        else 0,
        "avg_plan_score": (sum(episode.metrics.plan_score for episode in episodes) / len(episodes)) if episodes else 0,
        "avg_total_score": (sum(episode.metrics.total_score for episode in episodes) / len(episodes)) if episodes else 0,
        "pass_rate": (sum(1 for episode in episodes if episode.metrics.cured) / len(episodes)) if episodes else 0,
    }
    _write_jsonl(run_dir / "cases.jsonl", cases_payload)
    _write_json(run_dir / "summary.json", summary)
    return run_dir
