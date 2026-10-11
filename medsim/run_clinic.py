from __future__ import annotations

import argparse
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from medsim.clinical.batch_reporting import (
    append_jsonl,
    append_success_episode,
    build_batch_summary,
    build_case_summary_from_episode,
    build_case_summary_from_error,
    load_completed_scenario_ids,
    read_jsonl,
    render_batch_plots,
    write_json,
    write_legacy_summary_json,
)
from medsim.clinical.case_adapter import ScenarioLoaderMedQA
from medsim.clinical.config import load_clinical_config, resolve_scenario_count
from medsim.clinical.experiment4 import (
    EXP4_OUTPUT_DIR,
    build_case_metadata_payload,
    build_stage_specs_from_scenarios,
    build_stage_summary_rows,
    find_snapshot_ready_stages,
    resolve_memory_root,
    validate_exp4_stage_specs,
    write_memory_snapshot,
    write_stage_summary_jsonl,
)
from medsim.clinical.orchestrator import run_episode
from medsim.clinical.schema import dataclass_to_dict
from medsim.clinical.writers import RUNS_ROOT


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def _validate_experiment_config(config, experiment: str | None) -> None:
    if experiment != "exp4":
        return
    if not config.runtime.dataset_path:
        raise ValueError("Experiment 4 requires runtime.dataset_path to be set explicitly.")
    if bool(getattr(config.memory, "enable_retrieval", True)) and not config.memory.kb_root_dir:
        raise ValueError("Experiment 4 memory runs require memory.kb_root_dir to be set explicitly.")


def _write_exp4_stage_outputs(
    run_dir: Path,
    rows: List[Dict[str, Any]],
    stage_specs: List[Dict[str, Any]],
    config,
) -> List[Dict[str, Any]]:
    stage_rows = build_stage_summary_rows(rows, stage_specs, pass_threshold=config.scoring.pass_threshold)
    write_stage_summary_jsonl(run_dir / "stage_summary.jsonl", stage_rows)
    if bool(getattr(config.memory, "enable_retrieval", True)):
        snapshots_root = run_dir / "memory_snapshots"
        memory_root = resolve_memory_root(config.memory.namespace, config.memory.kb_root_dir)
        config_snapshot = dataclass_to_dict(config)
        for stage_row in find_snapshot_ready_stages(stage_rows, snapshots_root):
            snapshot_dir = snapshots_root / f"memory_after_{stage_row['stage_id']}"
            write_memory_snapshot(
                snapshot_dir,
                memory_root=memory_root,
                config_snapshot=config_snapshot,
                stage_row=stage_row,
            )
    return stage_rows


# 中文说明：main 是闭环门诊仿真的 CLI 主入口，支持单病例与批量运行两种模式。
def main() -> int:
    parser = argparse.ArgumentParser(description="Closed-loop clinical simulation runner")
    parser.add_argument(
        "--config",
        type=str,
        default="medsim/configs/config_clinic.yaml",
        help="Path to the clinical runtime YAML configuration.",
    )
    parser.add_argument("--scenario_id", type=int, default=None, help="Run only one scenario by id.")
    parser.add_argument("--sim_code", type=str, default=None, help="Optional Simulacra simulation code.")
    parser.add_argument("--run_id", type=str, default=None, help="Optional run identifier for batch outputs.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override.")
    parser.add_argument("--run_n_cases", type=int, default=None, help="Optional number of cases override.")
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="Optional start scenario id (inclusive) for batch run.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Optional end scenario id (inclusive) for batch run.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=["exp4"],
        help="Optional experiment mode.",
    )
    args = parser.parse_args()

    config = load_clinical_config(args.config)
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if args.run_n_cases is not None:
        config.runtime.run_n_cases = int(args.run_n_cases)
    random.seed(config.runtime.seed)
    _validate_experiment_config(config, args.experiment)

    dataset_path = Path(config.runtime.dataset_path) if config.runtime.dataset_path else None
    loader = ScenarioLoaderMedQA(dataset_path=dataset_path, allowed_catalog=config.clinical_tests.allowed_catalog)
    run_id = args.run_id or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S")
    exp4_mode = args.experiment == "exp4"
    exp4_run_dir = (EXP4_OUTPUT_DIR / "runs" / run_id) if exp4_mode else None
    episode_output_root = (exp4_run_dir / "episodes") if exp4_run_dir is not None else None
    include_case_metadata = exp4_mode
    stage_specs: List[Dict[str, Any]] = []
    if exp4_mode:
        stage_specs = build_stage_specs_from_scenarios(loader.scenarios)
        validate_exp4_stage_specs(stage_specs)

    if args.scenario_id is not None:
        episode = run_episode(
            config=config,
            scenario_id=args.scenario_id,
            sim_code=args.sim_code,
            run_id=run_id,
            episode_output_root=episode_output_root,
        )
        print(
            json.dumps(
                build_case_summary_from_episode(
                    episode,
                    run_id,
                    include_case_metadata=include_case_metadata,
                ),
                ensure_ascii=False,
            )
        )
        return 0

    dataset_size = loader.num_scenarios
    configured_count = resolve_scenario_count(config, dataset_size)
    has_explicit_range = args.start is not None or args.end is not None
    if not has_explicit_range:
        range_start = 0
        range_end_exclusive = configured_count
    else:
        range_start = int(args.start or 0)
        if args.end is None:
            range_end_exclusive = dataset_size
        else:
            if args.end < 0:
                raise ValueError("--end must be >= 0.")
            range_end_exclusive = min(int(args.end) + 1, dataset_size)
        if range_start < 0:
            raise ValueError("--start must be >= 0.")
        if args.end is not None and range_start > int(args.end):
            raise ValueError("--start must be <= --end when both are provided.")
        range_start = min(range_start, dataset_size)
    count = max(0, range_end_exclusive - range_start)
    run_dir = exp4_run_dir or (Path(RUNS_ROOT) / run_id)
    case_summaries_path = run_dir / "case_summaries.jsonl"
    legacy_cases_path = run_dir / "cases.jsonl"
    legacy_summary_path = run_dir / "summary.json"
    batch_summary_path = run_dir / "batch_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    case_summaries_path.touch(exist_ok=True)
    legacy_cases_path.touch(exist_ok=True)

    completed_scenario_ids = load_completed_scenario_ids(case_summaries_path, legacy_cases_path)
    rows = read_jsonl(case_summaries_path)
    if exp4_mode:
        _write_exp4_stage_outputs(run_dir, rows, stage_specs, config)

    attempted = 0
    succeeded = 0
    failed = 0
    skipped = 0

    with logging_redirect_tqdm():
        with tqdm(total=count, desc="Batch cases", unit="case", dynamic_ncols=True) as progress:
            for scenario_id in range(range_start, range_end_exclusive):
                if scenario_id in completed_scenario_ids:
                    skipped += 1
                else:
                    attempted += 1
                    case_metadata = None
                    if include_case_metadata:
                        case_metadata = build_case_metadata_payload(loader.get_scenario(scenario_id))
                    try:
                        episode = run_episode(
                            config=config,
                            scenario_id=scenario_id,
                            sim_code=args.sim_code,
                            run_id=run_id,
                            episode_output_root=episode_output_root,
                        )
                        payload = build_case_summary_from_episode(
                            episode,
                            run_id,
                            include_case_metadata=include_case_metadata,
                        )
                        append_jsonl(case_summaries_path, payload)
                        rows.append(payload)
                        append_success_episode(legacy_cases_path, episode)
                        completed_scenario_ids.add(scenario_id)
                        succeeded += 1
                    except Exception as exc:  # pragma: no cover - runtime resilience
                        failed += 1
                        logger.exception("Batch scenario failed scenario_id=%s error=%s", scenario_id, exc)
                        payload = build_case_summary_from_error(
                            scenario_id=scenario_id,
                            run_id=run_id,
                            error_type=exc.__class__.__name__,
                            error_message=str(exc),
                            case_metadata=case_metadata,
                        )
                        append_jsonl(case_summaries_path, payload)
                        rows.append(payload)
                        completed_scenario_ids.add(scenario_id)

                    if exp4_mode:
                        _write_exp4_stage_outputs(run_dir, rows, stage_specs, config)

                progress.update(1)
                progress.set_postfix(
                    {
                        "attempted": attempted,
                        "succeeded": succeeded,
                        "failed": failed,
                        "skipped": skipped,
                    },
                    refresh=False,
                )

    legacy_summary = write_legacy_summary_json(run_id, legacy_cases_path, legacy_summary_path)
    plot_outputs = render_batch_plots(rows, run_dir)
    batch_summary = build_batch_summary(
        run_id=run_id,
        target_cases=count,
        attempted=attempted,
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
        rows=rows,
        legacy_success_count=int(legacy_summary.get("num_cases", 0)),
    )
    if plot_outputs:
        batch_summary["plots"] = plot_outputs
    if exp4_mode:
        _write_exp4_stage_outputs(run_dir, rows, stage_specs, config)
        batch_summary["stage_summary_path"] = str((run_dir / "stage_summary.jsonl").resolve())
    write_json(batch_summary_path, batch_summary)

    response: Dict[str, Any] = {
        "run_id": run_id,
        "target_cases": count,
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "case_summaries_path": str(case_summaries_path),
        "batch_summary_path": str(batch_summary_path),
    }
    if exp4_mode:
        response["stage_summary_path"] = str(run_dir / "stage_summary.jsonl")
    print(json.dumps(response, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
