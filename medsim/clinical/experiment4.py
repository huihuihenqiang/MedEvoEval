from __future__ import annotations

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from medsim.clinical.schema import ClinicalCase


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP4_ROOT = PROJECT_ROOT / "shiyan4"
EXP4_DATA_DIR = EXP4_ROOT / "data"
EXP4_OUTPUT_DIR = EXP4_ROOT / "output"
EXP4_DATASET_PATH = EXP4_DATA_DIR / "medqa_exp4.jsonl"
EXP4_MANIFEST_PATH = EXP4_DATA_DIR / "medqa_exp4_manifest.json"

EXP4_SEED = 7
EXP4_STAGE_SIZE = 50
EXACT_DX_THRESHOLD = 1.0
EXP4_METADATA_KEYS = (
    "stage_id",
    "phase",
    "source_pool",
    "order_in_stage",
    "global_order",
)
EXP4_STAGE_SEQUENCE = [
    ("T1", "pre", "stable"),
    ("T2", "pre", "stable"),
    ("T3", "pre", "stable"),
    ("T4", "pre", "stable"),
    ("T5", "shock", "shifted"),
    ("T6", "shock", "shifted"),
    ("T7", "post", "stable"),
    ("T8", "post", "stable"),
    ("T9", "post", "stable"),
    ("T10", "post", "stable"),
]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as file:
        for line in file:
            row = str(line or "").strip()
            if not row:
                continue
            payload = json.loads(row)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def extract_dataset_metadata(raw_case: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key in ("stage_id", "phase", "source_pool"):
        text = str(raw_case.get(key) or "").strip()
        if text:
            metadata[key] = text
    for key in ("order_in_stage", "global_order"):
        if raw_case.get(key) is None:
            continue
        metadata[key] = _safe_int(raw_case.get(key), 0)
    return metadata


def build_case_metadata_payload(case: ClinicalCase) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"case_id": case.case_id}
    for key in EXP4_METADATA_KEYS:
        value = case.dataset_metadata.get(key)
        if key in {"order_in_stage", "global_order"}:
            if value is None:
                continue
            payload[key] = _safe_int(value, 0)
            continue
        text = str(value or "").strip()
        if text:
            payload[key] = text
    return payload


def build_exp4_dataset(
    source_path: Path,
    dataset_path: Path = EXP4_DATASET_PATH,
    manifest_path: Path = EXP4_MANIFEST_PATH,
    *,
    seed: int = EXP4_SEED,
) -> Dict[str, Any]:
    rows = _read_jsonl(source_path)
    if len(rows) < 600:
        raise ValueError("Experiment 4 source dataset must contain at least 600 cases.")

    stable_pool = list(rows[:500])
    shifted_pool = list(rows[500:])
    if len(shifted_pool) < 100:
        raise ValueError("Experiment 4 source dataset must contain at least 100 shifted cases after the first 500 rows.")

    rng = random.Random(int(seed))
    stable_selected = rng.sample(stable_pool, 400)
    shifted_selected = rng.sample(shifted_pool, 100)

    ordered_cases: List[Dict[str, Any]] = []
    manifest_cases: List[Dict[str, Any]] = []
    global_order = 1

    selections = [
        ("T1", "pre", "stable", stable_selected[0:50]),
        ("T2", "pre", "stable", stable_selected[50:100]),
        ("T3", "pre", "stable", stable_selected[100:150]),
        ("T4", "pre", "stable", stable_selected[150:200]),
        ("T5", "shock", "shifted", shifted_selected[0:50]),
        ("T6", "shock", "shifted", shifted_selected[50:100]),
        ("T7", "post", "stable", stable_selected[200:250]),
        ("T8", "post", "stable", stable_selected[250:300]),
        ("T9", "post", "stable", stable_selected[300:350]),
        ("T10", "post", "stable", stable_selected[350:400]),
    ]

    source_lookup = {id(item): index + 1 for index, item in enumerate(rows)}
    for stage_id, phase, source_pool, stage_cases in selections:
        for order_in_stage, raw_case in enumerate(stage_cases, start=1):
            payload = dict(raw_case)
            payload["case_id"] = str(payload.get("case_id") or f"medqa_case_{global_order:03d}")
            payload["stage_id"] = stage_id
            payload["phase"] = phase
            payload["source_pool"] = source_pool
            payload["order_in_stage"] = order_in_stage
            payload["global_order"] = global_order
            ordered_cases.append(payload)
            manifest_cases.append(
                {
                    "case_id": payload["case_id"],
                    "stage_id": stage_id,
                    "phase": phase,
                    "source_pool": source_pool,
                    "order_in_stage": order_in_stage,
                    "global_order": global_order,
                    "source_line_number": source_lookup.get(id(raw_case)),
                }
            )
            global_order += 1

    _write_jsonl(dataset_path, ordered_cases)
    manifest = {
        "seed": int(seed),
        "source_path": str(source_path.resolve()),
        "dataset_path": str(dataset_path.resolve()),
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "counts": {
            "source_total": len(rows),
            "stable_pool_total": len(stable_pool),
            "shifted_pool_total": len(shifted_pool),
            "stable_selected": len(stable_selected),
            "shifted_selected": len(shifted_selected),
            "dataset_total": len(ordered_cases),
            "stage_size": EXP4_STAGE_SIZE,
        },
        "stage_counts": {
            stage_id: len([item for item in ordered_cases if str(item.get("stage_id")) == stage_id])
            for stage_id, _, _ in EXP4_STAGE_SEQUENCE
        },
        "cases": manifest_cases,
    }
    _write_json(manifest_path, manifest)
    return manifest


def default_stage_specs() -> List[Dict[str, Any]]:
    return [
        {
            "stage_id": stage_id,
            "phase": phase,
            "source_pool": source_pool,
            "target_cases": EXP4_STAGE_SIZE,
        }
        for stage_id, phase, source_pool in EXP4_STAGE_SEQUENCE
    ]


def stage_sort_key(stage_id: Any) -> tuple[int, str]:
    text = str(stage_id or "").strip()
    match = re.fullmatch(r"T(\d+)", text)
    if match:
        return (int(match.group(1)), text)
    return (10_000, text)


def build_stage_specs_from_scenarios(scenarios: Sequence[ClinicalCase]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for case in scenarios:
        metadata = dict(case.dataset_metadata or {})
        stage_id = str(metadata.get("stage_id") or "").strip()
        if not stage_id:
            continue
        row = grouped.setdefault(
            stage_id,
            {
                "stage_id": stage_id,
                "phase": str(metadata.get("phase") or "").strip(),
                "source_pool": str(metadata.get("source_pool") or "").strip(),
                "target_cases": 0,
            },
        )
        row["target_cases"] = _safe_int(row.get("target_cases"), 0) + 1
        if not row["phase"]:
            row["phase"] = str(metadata.get("phase") or "").strip()
        if not row["source_pool"]:
            row["source_pool"] = str(metadata.get("source_pool") or "").strip()
    return sorted(grouped.values(), key=lambda item: stage_sort_key(item.get("stage_id")))


def validate_exp4_stage_specs(stage_specs: Sequence[Dict[str, Any]]) -> None:
    normalized = [
        (
            str(item.get("stage_id") or "").strip(),
            str(item.get("phase") or "").strip(),
            str(item.get("source_pool") or "").strip(),
            _safe_int(item.get("target_cases"), 0),
        )
        for item in stage_specs
    ]
    expected = [
        (stage_id, phase, source_pool, EXP4_STAGE_SIZE)
        for stage_id, phase, source_pool in EXP4_STAGE_SEQUENCE
    ]
    if normalized != expected:
        raise ValueError(
            "Experiment 4 dataset metadata is invalid. Expected stages T1-T10 with phase/source_pool metadata and 50 cases each."
        )


def _nested_number(row: Dict[str, Any], *keys: str) -> float:
    current: Any = row
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    try:
        if current is None:
            return 0.0
        return float(current)
    except (TypeError, ValueError):
        return 0.0


def build_stage_summary_rows(
    rows: Sequence[Dict[str, Any]],
    stage_specs: Sequence[Dict[str, Any]],
    *,
    pass_threshold: float,
) -> List[Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {
        str(spec.get("stage_id")): {
            "stage_id": str(spec.get("stage_id") or "").strip(),
            "phase": str(spec.get("phase") or "").strip(),
            "source_pool": str(spec.get("source_pool") or "").strip(),
            "target_cases": _safe_int(spec.get("target_cases"), 0),
            "n_cases": 0,
            "error_cases": 0,
            "total_score_sum": 0.0,
            "utility_sum": 0.0,
            "turns_sum": 0.0,
            "tests_sum": 0.0,
            "pass_cases": 0,
            "exact_dx_cases": 0,
        }
        for spec in stage_specs
    }

    for row in rows:
        stage_id = str(row.get("stage_id") or "").strip()
        if not stage_id:
            continue
        bucket = aggregated.setdefault(
            stage_id,
            {
                "stage_id": stage_id,
                "phase": str(row.get("phase") or "").strip(),
                "source_pool": str(row.get("source_pool") or "").strip(),
                "target_cases": EXP4_STAGE_SIZE,
                "n_cases": 0,
                "error_cases": 0,
                "total_score_sum": 0.0,
                "utility_sum": 0.0,
                "turns_sum": 0.0,
                "tests_sum": 0.0,
                "pass_cases": 0,
                "exact_dx_cases": 0,
            },
        )
        total_score = _nested_number(row, "scoring", "total_score")
        utility = _nested_number(row, "objective_metrics", "utility")
        turns = _nested_number(row, "objective_metrics", "turns")
        tests = _nested_number(row, "objective_metrics", "tests")
        diagnosis_score = _nested_number(row, "scoring", "diagnosis_score")
        termination_reason = str(row.get("termination_reason") or "").strip().lower()

        bucket["n_cases"] += 1
        bucket["error_cases"] += 1 if termination_reason == "error" else 0
        bucket["total_score_sum"] += total_score
        bucket["utility_sum"] += utility
        bucket["turns_sum"] += turns
        bucket["tests_sum"] += tests
        bucket["pass_cases"] += 1 if total_score >= float(pass_threshold) else 0
        bucket["exact_dx_cases"] += 1 if diagnosis_score >= EXACT_DX_THRESHOLD else 0

    summary_rows: List[Dict[str, Any]] = []
    for stage_id in sorted(aggregated.keys(), key=stage_sort_key):
        bucket = aggregated[stage_id]
        n_cases = _safe_int(bucket.get("n_cases"), 0)
        divisor = float(n_cases) if n_cases else 1.0
        summary_rows.append(
            {
                "stage_id": bucket["stage_id"],
                "phase": bucket["phase"],
                "source_pool": bucket["source_pool"],
                "target_cases": _safe_int(bucket.get("target_cases"), 0),
                "n_cases": n_cases,
                "error_cases": _safe_int(bucket.get("error_cases"), 0),
                "total_score_mean": float(bucket["total_score_sum"]) / divisor if n_cases else 0.0,
                "utility_mean": float(bucket["utility_sum"]) / divisor if n_cases else 0.0,
                "pass_rate": float(bucket["pass_cases"]) / divisor if n_cases else 0.0,
                "exact_dx_rate": float(bucket["exact_dx_cases"]) / divisor if n_cases else 0.0,
                "turns_mean": float(bucket["turns_sum"]) / divisor if n_cases else 0.0,
                "tests_mean": float(bucket["tests_sum"]) / divisor if n_cases else 0.0,
            }
        )
    return summary_rows


def write_stage_summary_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    _write_jsonl(path, rows)


def find_snapshot_ready_stages(stage_rows: Sequence[Dict[str, Any]], snapshots_root: Path) -> List[Dict[str, Any]]:
    ready: List[Dict[str, Any]] = []
    sorted_rows = sorted(stage_rows, key=lambda item: stage_sort_key(item.get("stage_id")))
    for index, row in enumerate(sorted_rows):
        target_cases = _safe_int(row.get("target_cases"), EXP4_STAGE_SIZE)
        n_cases = _safe_int(row.get("n_cases"), 0)
        if n_cases < target_cases:
            continue
        snapshot_dir = snapshots_root / f"memory_after_{row.get('stage_id')}"
        if snapshot_dir.exists():
            continue
        later_started = any(_safe_int(item.get("n_cases"), 0) > 0 for item in sorted_rows[index + 1 :])
        if later_started:
            continue
        ready.append(row)
    return ready


def resolve_memory_root(namespace: str, kb_root_dir: Optional[str]) -> Path:
    text = str(kb_root_dir or "").strip()
    if text:
        return Path(text)
    clean_namespace = str(namespace or "default").strip() or "default"
    return PROJECT_ROOT / "output" / "knowledge" / clean_namespace


def write_memory_snapshot(
    snapshot_dir: Path,
    *,
    memory_root: Path,
    config_snapshot: Dict[str, Any],
    stage_row: Dict[str, Any],
) -> Path:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    knowledge_dir = snapshot_dir / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    if memory_root.exists():
        for source_path in sorted(memory_root.rglob("*")):
            if not source_path.is_file():
                continue
            relative_path = source_path.relative_to(memory_root)
            target_path = knowledge_dir / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied_files.append(str(relative_path).replace("\\", "/"))

    for default_name in ("success_kb.jsonl", "correction_kb.jsonl", "rule_cards_kb.jsonl"):
        placeholder = knowledge_dir / default_name
        if placeholder.exists():
            continue
        placeholder.parent.mkdir(parents=True, exist_ok=True)
        placeholder.touch()
        copied_files.append(default_name)

    config_path = snapshot_dir / "config_snapshot.json"
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    _write_json(config_path, config_snapshot)
    _write_json(
        manifest_path,
        {
            "snapshot_name": snapshot_dir.name,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
            "memory_root": str(memory_root.resolve()),
            "knowledge_dir": str(knowledge_dir.resolve()),
            "files": sorted(set(copied_files)),
            "stage": {
                "stage_id": str(stage_row.get("stage_id") or ""),
                "phase": str(stage_row.get("phase") or ""),
                "source_pool": str(stage_row.get("source_pool") or ""),
                "target_cases": _safe_int(stage_row.get("target_cases"), 0),
                "n_cases": _safe_int(stage_row.get("n_cases"), 0),
                "error_cases": _safe_int(stage_row.get("error_cases"), 0),
            },
        },
    )
    return snapshot_dir
