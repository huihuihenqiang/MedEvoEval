from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from medsim.clinical.schema import (
    APIBaseOptions,
    APIOptions,
    ClinicalTestsOptions,
    ClinicalRuntimeConfig,
    InterfaceNormalizationOptions,
    MDTOptions,
    MemoryOptions,
    ModelOptions,
    NetworkOptions,
    RuntimeOptions,
    ScoringOptions,
    SimulacraOptions,
    WorkflowOptions,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "medsim" / "configs" / "config_clinic.yaml"


def _merge_section(instance: Any, payload: Dict[str, Any]) -> Any:
    current = instance.__dict__.copy()
    current.update(payload or {})
    return replace(instance, **current)


def _normalize_scoring_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    defaults = ScoringOptions()

    def _coerce(name: str) -> float:
        value = payload.get(name, getattr(defaults, name))
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(getattr(defaults, name))

    normalized = {
        "lambda_evidence": _coerce("lambda_evidence"),
        "mu_management": _coerce("mu_management"),
        "weight_diag": _coerce("weight_diag"),
        "weight_evidence": _coerce("weight_evidence"),
        "weight_plan": _coerce("weight_plan"),
        "pass_threshold": _coerce("pass_threshold"),
        "effective_treatment_threshold": _coerce("effective_treatment_threshold"),
    }

    if normalized["lambda_evidence"] < 0 or normalized["mu_management"] < 0:
        raise ValueError("scoring.lambda_evidence and scoring.mu_management must be non-negative.")
    if any(normalized[name] < 0 for name in ("weight_diag", "weight_evidence", "weight_plan")):
        raise ValueError("scoring weights must be non-negative.")
    weight_sum = normalized["weight_diag"] + normalized["weight_evidence"] + normalized["weight_plan"]
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError("scoring weights must sum to 1.0.")
    for threshold_name in ("pass_threshold", "effective_treatment_threshold"):
        value = normalized[threshold_name]
        if value < 0 or value > 1:
            raise ValueError(f"scoring.{threshold_name} must be in [0, 1].")
    return normalized


def _normalize_clinical_tests_section(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_catalog = payload.get("allowed_catalog", [])
    if raw_catalog is None:
        raw_catalog = []
    if not isinstance(raw_catalog, list):
        raise ValueError("clinical_tests.allowed_catalog must be a list of strings.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in raw_catalog:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)

    return {"allowed_catalog": cleaned}


def _resolve_optional_path(value: Any, base_dir: Path) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _normalize_path_sections(raw: Dict[str, Any], config_dir: Path) -> Dict[str, Dict[str, Any]]:
    runtime = dict(raw.get("runtime", {}) or {})
    memory = dict(raw.get("memory", {}) or {})

    runtime["dataset_path"] = _resolve_optional_path(runtime.get("dataset_path"), config_dir)
    memory["kb_root_dir"] = _resolve_optional_path(memory.get("kb_root_dir"), config_dir)

    return {
        "runtime": runtime,
        "memory": memory,
    }


def load_clinical_config(path: str | Path | None = None) -> ClinicalRuntimeConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    config_path = config_path.resolve()
    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    path_sections = _normalize_path_sections(raw, config_path.parent)

    config = ClinicalRuntimeConfig(
        runtime=_merge_section(RuntimeOptions(), path_sections["runtime"]),
        models=_merge_section(ModelOptions(), raw.get("models", {})),
        workflow=_merge_section(WorkflowOptions(), raw.get("workflow", {})),
        clinical_tests=_merge_section(
            ClinicalTestsOptions(),
            _normalize_clinical_tests_section(raw.get("clinical_tests", {}) or {}),
        ),
        interface_normalization=_merge_section(
            InterfaceNormalizationOptions(),
            raw.get("interface_normalization", {}) or {},
        ),
        mdt=_merge_section(MDTOptions(), raw.get("mdt", {})),
        memory=_merge_section(MemoryOptions(), path_sections["memory"]),
        simulacra=_merge_section(SimulacraOptions(), raw.get("simulacra", {})),
        api_keys=_merge_section(APIOptions(), raw.get("api_keys", {})),
        api_base=_merge_section(APIBaseOptions(), raw.get("api_base", {})),
        network=_merge_section(NetworkOptions(), raw.get("network", {})),
        scoring=_merge_section(ScoringOptions(), _normalize_scoring_section(raw.get("scoring", {}) or {})),
    )
    os.environ["MEDSIM_CONFIG_PATH"] = str(config_path)
    os.environ["MEDSIM_USE_SYSTEM_PROXY"] = "true" if config.network.use_system_proxy else "false"
    return config


def resolve_scenario_count(config: ClinicalRuntimeConfig, dataset_size: int) -> int:
    configured = config.runtime.run_n_cases
    if configured is None:
        return dataset_size
    return max(0, min(int(configured), dataset_size))
