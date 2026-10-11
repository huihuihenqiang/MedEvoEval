from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from medsim.clinical.schema import (
    DialogueLine,
    EpisodeResult,
    SimulacraCameraPlan,
    SimulacraDialogueEvent,
    SimulacraFlowAction,
    SimulacraReplay,
    dataclass_to_dict,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EPISODES_ROOT = PROJECT_ROOT / "output" / "episodes"
SIMULACRA_TEMP_ROOT = PROJECT_ROOT / "Simulacra" / "environment" / "frontend_server" / "temp_storage"
SIM_REPLAY_ROOT = SIMULACRA_TEMP_ROOT / "sim_replays"
SIM_DIALOGUE_ROOT = SIMULACRA_TEMP_ROOT / "sim_dialogues"
LIVE_DIALOGUE_FILE = SIMULACRA_TEMP_ROOT / "live_dialogue.json"
CURR_SIM_CODE_FILE = SIMULACRA_TEMP_ROOT / "curr_sim_code.json"
CURR_STEP_FILE = SIMULACRA_TEMP_ROOT / "curr_step.json"
RUNTIME_SESSION_FILE = SIMULACRA_TEMP_ROOT / "runtime_session.json"

OUTSIDE_HOSPITAL = "the Ville:Outdoors:Outside Hospital"
REGISTRATION_DESK = "the Ville:Hospital:Registration Window:desk"
TRIAGE_DESK = "the Ville:Hospital:Triage Station:desk"
PRIMARY_PATIENT_STATION = "the Ville:Hospital:Dentistry Consultation Room:chair"
PRIMARY_DOCTOR_STATION = "the Ville:Hospital:Dentistry Consultation Room:desk"
RADIOLOGY_CHAIR = "the Ville:Hospital:Radiology Department:chair"
RADIOLOGY_DESK = "the Ville:Hospital:Radiology Department:desk"
PHARMACY_DESK = "the Ville:Hospital:Pharmacy:desk"
PANEL_PAGE_SIZE = 3

ROLE_KEYS = ("doctor", "patient", "receptionist", "nurse", "examiner", "pharmacy_clerk", "manager")
DEFAULT_CAST = {
    "doctor": "Maria Lopez",
    "patient": "Klaus Mueller",
    "receptionist": "Jane Moreno",
    "nurse": "Ayesha Khan",
    "examiner": "Rajiv Patel",
    "pharmacy_clerk": "Carmen Ortiz",
    "manager": "管理者",
}
ROLE_LABELS = {
    "doctor": "门诊医生",
    "patient": "患者",
    "receptionist": "导诊/收费",
    "nurse": "分诊护士",
    "examiner": "检查医生",
    "pharmacy_clerk": "药房人员",
    "manager": "管理者",
}
STAGE_TO_FLOW = {
    "consulting_pre_exam": "consulting",
    "examination": "examination",
    "consulting_post_exam": "consulting",
}
PHASE_STAGE_DEFAULTS = {
    "consultation": "consulting_pre_exam",
    "consulting": "consulting_pre_exam",
    "examination": "examination",
    "diagnosis": "consulting_pre_exam",
    "manager_review": "consulting_pre_exam",
    "learning_write": "consulting_pre_exam",
}
MAIN_FLOW_PHASES = {"consultation", "examination", "diagnosis", "manager_review"}
MAIN_FLOW_ROLES = {"doctor", "patient", "examiner", "manager"}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    last_error: Optional[OSError] = None
    for _ in range(5):
        try:
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
            os.replace(temp_path, path)
            return
        except OSError as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.05)

    # Windows may hold the destination file open while Django is reading it.
    # Fall back to in-place overwrite instead of killing the clinical thread.
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    if last_error is not None:
        return


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def runtime_session_path() -> Path:
    return RUNTIME_SESSION_FILE


def read_runtime_session() -> Optional[Dict[str, Any]]:
    payload = _read_json(RUNTIME_SESSION_FILE)
    return payload if isinstance(payload, dict) else None


def write_runtime_session(
    *,
    sim_code: str,
    started_at: Optional[datetime] = None,
    sec_per_step: int = 10,
    current_step: int = 0,
) -> Dict[str, Any]:
    existing = read_runtime_session() or {}
    payload = {
        "sim_code": sim_code,
        "started_at_iso": (started_at or datetime.now()).isoformat(timespec="seconds"),
        "sec_per_step": sec_per_step,
        "current_step": current_step,
    }
    if existing.get("sim_code") == sim_code and existing.get("started_at_iso"):
        payload["started_at_iso"] = existing["started_at_iso"]
    try:
        _write_json(RUNTIME_SESSION_FILE, payload)
    except OSError:
        return payload
    return payload


def update_runtime_session_step(sim_code: str, current_step: int) -> None:
    payload = read_runtime_session()
    if not payload or payload.get("sim_code") != sim_code:
        return
    payload["current_step"] = current_step
    try:
        _write_json(RUNTIME_SESSION_FILE, payload)
    except OSError:
        return


def normalize_simulacra_cast(cast: Optional[Dict[str, str]]) -> Dict[str, str]:
    normalized = dict(DEFAULT_CAST)
    if not cast:
        return normalized
    keys = {str(key) for key in cast.keys()}
    values = {str(value) for value in cast.values()}
    if keys.issubset(ROLE_KEYS):
        for role, name in cast.items():
            if role in ROLE_KEYS and name:
                normalized[str(role)] = str(name)
        return normalized
    if values.issubset(ROLE_KEYS):
        for name, role in cast.items():
            if role in ROLE_KEYS and name:
                normalized[str(role)] = str(name)
        return normalized
    for role in ROLE_KEYS:
        if cast.get(role):
            normalized[role] = str(cast[role])
    return normalized


def cast_role_by_name(cast: Optional[Dict[str, str]]) -> Dict[str, str]:
    normalized = normalize_simulacra_cast(cast)
    return {name: role for role, name in normalized.items()}


def resolve_primary_patient_name(cast: Optional[Dict[str, str]]) -> str:
    return normalize_simulacra_cast(cast)["patient"]


def resolve_primary_doctor_name(cast: Optional[Dict[str, str]]) -> str:
    return normalize_simulacra_cast(cast)["doctor"]


def _focus_personas(cast: Dict[str, str]) -> List[str]:
    return [name for name in (cast.get("patient"), cast.get("doctor")) if name]


def _stage_index(stage_id: str, actions: Iterable[Dict[str, Any]]) -> int:
    for item in actions:
        if item.get("stage_id") == stage_id:
            return int(item.get("stage_index", 0))
    return 0


def _patient_actions(include_exam: bool) -> List[SimulacraFlowAction]:
    defs = [
        (
            "consulting_pre_exam",
            PRIMARY_PATIENT_STATION,
            4,
            "进入诊室开始主流程问诊",
            "诊室问诊",
            "",
            "CONS",
            "问诊",
            "诊室",
        ),
    ]
    if include_exam:
        defs.extend(
            [
                (
                    "examination",
                    RADIOLOGY_CHAIR,
                    4,
                    "前往检查诊室完成检查",
                    "检查中",
                    "",
                    "IMG",
                    "检查",
                    "检查诊室",
                ),
                (
                    "consulting_post_exam",
                    PRIMARY_PATIENT_STATION,
                    4,
                    "带检查结果返回诊室继续主流程",
                    "返回诊室",
                    "",
                    "CONS",
                    "复诊",
                    "诊室",
                ),
            ]
        )

    actions: List[SimulacraFlowAction] = []
    for index, item in enumerate(defs):
        stage_id, address, duration, description, status, speech, pronunciatio, predicate, object_name = item
        actions.append(
            SimulacraFlowAction(
                flow=STAGE_TO_FLOW[stage_id],
                address=address,
                duration=duration,
                description=description,
                status=status,
                speech=speech,
                pronunciatio=pronunciatio,
                predicate=predicate,
                object=object_name,
                stage_id=stage_id,
                stage_index=index,
            )
        )
    return actions


def _staff_actions(cast: Dict[str, str]) -> Dict[str, List[SimulacraFlowAction]]:
    return {
        cast["doctor"]: [
            SimulacraFlowAction(
                "consulting",
                PRIMARY_DOCTOR_STATION,
                5,
                "在诊室内等待主流程继续",
                "诊室待诊",
                "",
                "DOC",
                "等待",
                "患者",
                "consulting_pre_exam",
                0,
            )
        ],
        cast["receptionist"]: [
            SimulacraFlowAction(
                "registration",
                REGISTRATION_DESK,
                5,
                "停留在导诊台",
                "导诊台待命",
                "",
                "REG",
                "待命",
                "患者",
                "registration",
                0,
            )
        ],
        cast["nurse"]: [
            SimulacraFlowAction(
                "triage",
                TRIAGE_DESK,
                5,
                "停留在分诊台",
                "分诊台待命",
                "",
                "TRI",
                "待命",
                "患者",
                "triage",
                0,
            )
        ],
        cast["examiner"]: [
            SimulacraFlowAction(
                "examination",
                RADIOLOGY_DESK,
                5,
                "停留在检查诊室等待检查",
                "检查诊室待命",
                "",
                "IMG",
                "检查",
                "患者",
                "examination",
                1,
            )
        ],
        cast["pharmacy_clerk"]: [
            SimulacraFlowAction(
                "pharmacy_checkout",
                PHARMACY_DESK,
                5,
                "停留在药房窗口",
                "药房待命",
                "",
                "RX",
                "待命",
                "药品",
                "pharmacy_checkout",
                0,
            )
        ],
    }


def _dialogue_event(
    speaker: str,
    text: str,
    phase: str,
    stage_id: str,
    stage_index: int,
    visit_index: int = 1,
    turn_index: Optional[int] = None,
) -> SimulacraDialogueEvent:
    return SimulacraDialogueEvent(speaker, text, phase, visit_index, turn_index, stage_id, stage_index)


def _intro_events(cast: Dict[str, str]) -> List[SimulacraDialogueEvent]:
    del cast
    return []


def _dialogue_line(event: SimulacraDialogueEvent) -> DialogueLine:
    return DialogueLine(event.speaker, event.text, event.phase, event.visit_index, event.turn_index)


def _speaker_role(cast: Dict[str, str], speaker: Optional[str]) -> Optional[str]:
    if not speaker:
        return None
    return cast_role_by_name(cast).get(speaker)


def _is_main_flow_event(item: Dict[str, Any], cast: Dict[str, str]) -> bool:
    role = _speaker_role(cast, item.get("speaker"))
    phase = str(item.get("phase") or "")
    if phase == "manager_review":
        return True
    return role in MAIN_FLOW_ROLES and phase in MAIN_FLOW_PHASES


def _visible_lines(events: Iterable[Dict[str, Any]], stage_index: int, cast: Dict[str, str]) -> List[Dict[str, Any]]:
    return [
        {
            "speaker": item["speaker"],
            "text": item["text"],
            "phase": item.get("phase"),
            "visit_index": item.get("visit_index"),
            "turn_index": item.get("turn_index"),
        }
        for item in events
        if _is_main_flow_event(item, cast)
        if int(item.get("unlock_stage_index", 0)) <= int(stage_index)
    ]


def _patient_action_payloads(replay_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    cast = normalize_simulacra_cast(replay_payload.get("cast"))
    return list((replay_payload.get("persona_actions") or {}).get(cast["patient"]) or [])


def scene_state_from_replay(
    replay_payload: Dict[str, Any],
    *,
    patient_stage_index: Optional[int] = None,
    patient_stage_id: Optional[str] = None,
) -> Dict[str, Any]:
    cast = normalize_simulacra_cast(replay_payload.get("cast"))
    actions = _patient_action_payloads(replay_payload)
    if actions:
        if patient_stage_id is not None:
            stage_index = _stage_index(patient_stage_id, actions)
            stage_id = patient_stage_id
        else:
            stage_index = max(0, min(int(patient_stage_index or 0), len(actions) - 1))
            stage_id = str(actions[stage_index].get("stage_id") or "consulting_pre_exam")
    else:
        stage_id = patient_stage_id or "consulting_pre_exam"
        stage_index = int(patient_stage_index or 0)

    camera_plan = replay_payload.get("camera_plan") or {}
    release_index = camera_plan.get("release_stage_index")
    if release_index is None:
        release_index = _stage_index(str(camera_plan.get("release_stage_id") or "consulting_pre_exam"), actions)
    unlocked = len(_visible_lines(replay_payload.get("dialogue_events") or [], stage_index, cast))
    return {
        "patient_name": cast["patient"],
        "patient_stage_id": stage_id,
        "patient_stage_index": stage_index,
        "patient_flow": STAGE_TO_FLOW.get(stage_id, "consulting"),
        "camera_release_ready": stage_index >= int(release_index or 0),
        "unlocked_dialogue_count": unlocked,
        "cast": cast,
        "persona_roles": cast_role_by_name(cast),
    }


def build_dialogue_panel_from_replay(
    replay_payload: Dict[str, Any],
    *,
    patient_stage_index: Optional[int] = None,
    patient_stage_id: Optional[str] = None,
) -> Dict[str, Any]:
    panel = dict(replay_payload.get("dialogue_panel") or {})
    cast = normalize_simulacra_cast(replay_payload.get("cast"))
    scene_state = scene_state_from_replay(
        replay_payload,
        patient_stage_index=patient_stage_index,
        patient_stage_id=patient_stage_id,
    )
    visible = _visible_lines(replay_payload.get("dialogue_events") or [], scene_state["patient_stage_index"], cast)
    mode = panel.get("mode", "active")
    if not visible and mode == "active":
        mode = "idle"
    return {
        "mode": mode,
        "stage": scene_state["patient_stage_id"],
        "focus_personas": panel.get("focus_personas") or _focus_personas(cast),
        "latest_line": visible[-1] if visible else None,
        "recent_lines": visible,
        "page_size": panel.get("page_size", PANEL_PAGE_SIZE),
        "doctor_decision": panel.get("doctor_decision"),
        "medical_summary": panel.get("medical_summary"),
        "session_id": panel.get("session_id"),
        "sim_code": replay_payload.get("sim_code"),
        "camera_plan": replay_payload.get("camera_plan") or {},
        "cast": cast,
        "persona_roles": cast_role_by_name(cast),
        "role_labels": dict(ROLE_LABELS),
        "scene_state": scene_state,
    }


def _replay_dict(replay: SimulacraReplay) -> Dict[str, Any]:
    return dataclass_to_dict(replay)


def _line_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        item.get("speaker"),
        item.get("text"),
        item.get("phase"),
        item.get("visit_index"),
        item.get("turn_index"),
        item.get("unlock_stage_id"),
        item.get("unlock_stage_index"),
    )


def _should_keep_dialogue_line(line: DialogueLine, cast: Dict[str, str]) -> bool:
    if line.phase == "manager_review":
        return True
    return _speaker_role(cast, line.speaker) in MAIN_FLOW_ROLES and line.phase in MAIN_FLOW_PHASES


def _stage_id_for_dialogue_line(line: DialogueLine, *, consultation_after_exam: bool = False) -> str:
    if line.phase == "consultation":
        return "consulting_post_exam" if consultation_after_exam else "consulting_pre_exam"
    if line.phase == "examination":
        return "examination"
    if line.phase == "diagnosis":
        return "consulting_post_exam" if consultation_after_exam else "consulting_pre_exam"
    if line.phase == "manager_review":
        return "consulting_post_exam" if consultation_after_exam else "consulting_pre_exam"
    return PHASE_STAGE_DEFAULTS.get(line.phase, "consulting_pre_exam")


class SimulacraReplayStreamer:
    def __init__(self, sim_code: str, episode_id: str, cast: Optional[Dict[str, str]] = None) -> None:
        self.sim_code = sim_code
        self.episode_id = episode_id
        self.cast = normalize_simulacra_cast(cast)
        self.patient_actions = _patient_actions(False)
        self.dialogue_events = _intro_events(self.cast)
        self.exam_appended = False
        self.panel_stage_index = 0
        self.panel_mode = "active"
        self.panel_doctor_decision = None
        self.panel_medical_summary = {"diagnosis": [], "orders": [], "followup": [], "termination_reason": None}
        self.camera_plan = SimulacraCameraPlan("follow_then_free", self.cast["patient"], "consulting_pre_exam", 0)
        self.replay: Dict[str, Any] = {}
        self._refresh_panel()
        self._write_runtime_files()

    def _sync_replay(self) -> None:
        self.replay = _replay_dict(
            SimulacraReplay(
                sim_code=self.sim_code,
                focus_personas=_focus_personas(self.cast),
                consultation_lines=[_dialogue_line(item) for item in self.dialogue_events],
                persona_actions={self.cast["patient"]: self.patient_actions, **_staff_actions(self.cast)},
                dialogue_panel={},
                cast=self.cast,
                dialogue_events=self.dialogue_events,
                camera_plan=self.camera_plan,
            )
        )

    def _refresh_panel(self, *, stage_index: Optional[int] = None, mode: Optional[str] = None) -> None:
        if stage_index is not None:
            self.panel_stage_index = stage_index
        if mode is not None:
            self.panel_mode = mode
        self._sync_replay()
        panel = build_dialogue_panel_from_replay(self.replay, patient_stage_index=self.panel_stage_index)
        panel["mode"] = self.panel_mode
        panel["doctor_decision"] = self.panel_doctor_decision
        panel["medical_summary"] = dict(self.panel_medical_summary)
        panel["session_id"] = self.episode_id
        panel["sim_code"] = self.sim_code
        self.replay["dialogue_panel"] = panel

    def _write_runtime_files(self) -> None:
        _write_json(sim_code_replay_path(self.sim_code), self.replay)
        _write_json(sim_code_dialogue_history_path(self.sim_code), self.replay.get("dialogue_events", []))
        _write_json(LIVE_DIALOGUE_FILE, self.replay["dialogue_panel"])
        _write_json(CURR_SIM_CODE_FILE, {"sim_code": self.sim_code})
        _write_json(CURR_STEP_FILE, {"step": 0})
        current_step = (read_runtime_session() or {}).get("current_step", 0)
        write_runtime_session(sim_code=self.sim_code, current_step=current_step)

    def _ensure_exam(self) -> None:
        if not self.exam_appended:
            self.exam_appended = True
            self.patient_actions = _patient_actions(True)

    def _append_lines(self, lines: Iterable[DialogueLine], *, consultation_after_exam: bool = False) -> None:
        self._sync_replay()
        seen = {_line_key(item) for item in self.replay.get("dialogue_events", [])}
        actions = [dataclass_to_dict(item) for item in self.patient_actions]
        for line in lines:
            if not _should_keep_dialogue_line(line, self.cast):
                continue
            stage_id = _stage_id_for_dialogue_line(line, consultation_after_exam=consultation_after_exam)
            stage_index = _stage_index(stage_id, actions)
            payload = dataclass_to_dict(
                _dialogue_event(line.speaker, line.text, line.phase, stage_id, stage_index, line.visit_index, line.turn_index)
            )
            if _line_key(payload) in seen:
                continue
            self.dialogue_events.append(SimulacraDialogueEvent(**payload))
            seen.add(_line_key(payload))

    def _update_cast(self, cast: Optional[Dict[str, str]]) -> None:
        merged = normalize_simulacra_cast(cast)
        if merged == self.cast:
            return
        old = dict(self.cast)
        self.cast = merged
        rename = {old[role]: merged[role] for role in ROLE_KEYS if old.get(role) and merged.get(role)}
        for item in self.dialogue_events:
            item.speaker = rename.get(item.speaker, item.speaker)
        self.camera_plan.initial_focus = self.cast["patient"]

    def handle_event(self, event: Dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "episode_started":
            self._update_cast(event.get("simulacra_cast"))
            self._refresh_panel(stage_index=0, mode="active")
            self._write_runtime_files()
            return

        if kind == "consultation_turn":
            self._append_lines(event.get("lines", []), consultation_after_exam=self.exam_appended)
            decision = event.get("decision")
            self.panel_doctor_decision = {
                "action": {"ASK": "继续问诊", "REQUEST_TEST": "安排检查", "FINALIZE": "做出诊断", "CALL_MDT": "请求 MDT"}.get(
                    getattr(decision, "action", ""),
                    getattr(decision, "action", ""),
                ),
                "question_count": event.get("question_count"),
                "test_count": event.get("test_count"),
            }
            stage_id = "consulting_post_exam" if self.exam_appended else "consulting_pre_exam"
            self._refresh_panel(stage_index=_stage_index(stage_id, [dataclass_to_dict(item) for item in self.patient_actions]), mode="active")
        elif kind == "test_result":
            self._ensure_exam()
            self._append_lines(event.get("lines", []))
            decision = event.get("decision")
            self.panel_doctor_decision = {
                "action": {"ASK": "继续问诊", "REQUEST_TEST": "安排检查", "FINALIZE": "做出诊断", "CALL_MDT": "请求 MDT"}.get(
                    getattr(decision, "action", ""),
                    getattr(decision, "action", ""),
                ),
                "question_count": event.get("question_count"),
                "test_count": event.get("test_count"),
            }
            self._refresh_panel(stage_index=_stage_index("examination", [dataclass_to_dict(item) for item in self.patient_actions]), mode="active")
        elif kind == "final_decision":
            self._append_lines(event.get("lines", []), consultation_after_exam=self.exam_appended)
            decision = event.get("doctor_decision")
            final_decision = event.get("decision")
            self.panel_doctor_decision = {
                "action": {"ASK": "继续问诊", "REQUEST_TEST": "安排检查", "FINALIZE": "做出诊断", "CALL_MDT": "请求 MDT"}.get(
                    getattr(decision, "action", "FINALIZE"),
                    getattr(decision, "action", "FINALIZE"),
                ),
                "question_count": getattr(decision, "question_count", None),
                "test_count": getattr(decision, "test_count", None),
            }
            self.panel_medical_summary = {
                "diagnosis": getattr(final_decision, "diagnosis", []),
                "orders": getattr(final_decision, "plan", []),
                "followup": getattr(final_decision, "followup", []),
                "termination_reason": None,
            }
            stage_id = "consulting_post_exam" if self.exam_appended else "consulting_pre_exam"
            self._refresh_panel(stage_index=_stage_index(stage_id, [dataclass_to_dict(item) for item in self.patient_actions]), mode="active")
        elif kind == "manager_review":
            self._append_lines(event.get("lines", []), consultation_after_exam=self.exam_appended)
            stage_id = "consulting_post_exam" if self.exam_appended else "consulting_pre_exam"
            self._refresh_panel(stage_index=_stage_index(stage_id, [dataclass_to_dict(item) for item in self.patient_actions]), mode="active")
        elif kind == "episode_complete":
            episode = event.get("episode")
            if episode is not None:
                self._update_cast(getattr(episode, "simulacra_cast", None))
                self.panel_medical_summary = {
                    "diagnosis": getattr(episode.final_clinical_decision, "diagnosis", []),
                    "orders": getattr(episode.final_clinical_decision, "plan", []),
                    "followup": getattr(episode.final_clinical_decision, "followup", []),
                    "termination_reason": getattr(episode, "termination_reason", None),
                }
            stage_id = "consulting_post_exam" if self.exam_appended else "consulting_pre_exam"
            self._refresh_panel(stage_index=_stage_index(stage_id, [dataclass_to_dict(item) for item in self.patient_actions]), mode="completed")
        else:
            return

        self._write_runtime_files()


def _episode_dialogue_events(
    episode: EpisodeResult,
    patient_actions: List[SimulacraFlowAction],
    cast: Dict[str, str],
) -> List[SimulacraDialogueEvent]:
    events = _intro_events(cast)
    actions = [dataclass_to_dict(item) for item in patient_actions]
    exam_seen = False
    for line in episode.full_dialogue:
        if not _should_keep_dialogue_line(line, cast):
            continue
        stage_id = _stage_id_for_dialogue_line(line, consultation_after_exam=exam_seen)
        if line.phase == "examination":
            exam_seen = True
        events.append(
            _dialogue_event(
                line.speaker,
                line.text,
                line.phase,
                stage_id,
                _stage_index(stage_id, actions),
                line.visit_index,
                line.turn_index,
            )
        )
    return events


def build_simulacra_replay(episode: EpisodeResult, sim_code: Optional[str]) -> SimulacraReplay:
    cast = normalize_simulacra_cast(episode.simulacra_cast)
    patient_actions = _patient_actions(any(item.phase == "examination" for item in episode.full_dialogue))
    dialogue_events = _episode_dialogue_events(episode, patient_actions, cast)
    replay = SimulacraReplay(
        sim_code=sim_code,
        focus_personas=_focus_personas(cast),
        consultation_lines=[_dialogue_line(item) for item in dialogue_events],
        persona_actions={cast["patient"]: patient_actions, **_staff_actions(cast)},
        dialogue_panel={},
        cast=cast,
        dialogue_events=dialogue_events,
        camera_plan=SimulacraCameraPlan(
            "follow_then_free",
            cast["patient"],
            "consulting_pre_exam",
            _stage_index("consulting_pre_exam", [dataclass_to_dict(item) for item in patient_actions]),
        ),
    )
    replay.dialogue_panel = build_dialogue_panel_from_replay(_replay_dict(replay), patient_stage_index=len(patient_actions) - 1)
    replay.dialogue_panel["mode"] = "completed"
    replay.dialogue_panel["doctor_decision"] = dataclass_to_dict(episode.doctor_decisions[-1]) if episode.doctor_decisions else None
    replay.dialogue_panel["medical_summary"] = {
        "diagnosis": episode.final_clinical_decision.diagnosis,
        "orders": episode.final_clinical_decision.plan,
        "followup": episode.final_clinical_decision.followup,
        "termination_reason": episode.termination_reason,
    }
    replay.dialogue_panel["session_id"] = episode.episode_id
    replay.dialogue_panel["sim_code"] = sim_code
    return replay


def sim_code_replay_path(sim_code: str) -> Path:
    return SIM_REPLAY_ROOT / f"{sim_code}.json"


def sim_code_dialogue_history_path(sim_code: str) -> Path:
    return SIM_DIALOGUE_ROOT / f"{sim_code}.json"


def replay_path(scenario_id: int) -> Path:
    return EPISODES_ROOT / f"scenario_{scenario_id}" / "simulacra_replay.json"


def load_simulacra_replay(scenario_id: Optional[int] = None, sim_code: Optional[str] = None) -> Dict[str, Any]:
    if sim_code:
        payload = _read_json(sim_code_replay_path(sim_code))
        if payload is not None:
            return payload
    if scenario_id is None:
        raise FileNotFoundError("Simulacra replay is unavailable because neither sim_code nor scenario_id resolved.")
    payload = _read_json(replay_path(scenario_id))
    if payload is None:
        raise FileNotFoundError(f"Simulacra replay not found: {replay_path(scenario_id)}")
    return payload
