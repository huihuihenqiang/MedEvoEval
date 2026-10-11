#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from package_paths import data_root,CASE_FILE
from medevoeval.api_client import load_env_file
from medevoeval.episode_runner import load_config, read_jsonl, run_episode


def sanitize_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in {"api_key", "base_url"}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a lightweight API-doctor example (toy patient, mock MDT, rule judge; not the paper engine).")
    parser.add_argument("--config", default="configs/live_api.yaml", help="YAML config relative to repo root.")
    parser.add_argument("--input", default=None, help="Optional input JSONL relative to repo root.")
    parser.add_argument("--output-dir", default=None, help="Output directory relative to repo root.")
    parser.add_argument("--case-id", default=None, help="Run exactly one release case_id, such as case_001.")
    parser.add_argument("--n-cases", type=int, default=1, help="Number of cases to run when --case-id is not set.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index when --case-id is not set.")
    parser.add_argument("--data-dir")
    parser.add_argument("--execute",action="store_true",help="Execute the lightweight API-doctor example, not the real engine")
    args = parser.parse_args()
    if not args.execute:
        print("This is the lightweight demo. Use run_paper.py for the real engine. Add --execute only for this API-doctor demo.")
        return 0

    load_env_file(ROOT / ".env")
    config = load_config(ROOT / args.config)
    config["llm_mode"] = "api"
    config["api_key"] = os.environ.get("LLM_API_KEY", "")
    config["base_url"] = os.environ.get("LLM_BASE_URL", "")
    config["model_name"] = os.environ.get("MODEL_NAME") or config.get("model_name")

    input_path = (ROOT / args.input).resolve() if args.input else data_root(args.data_dir)/CASE_FILE
    output_dir = ROOT / (args.output_dir or config.get("output_path", "outputs/live_api"))
    cases = read_jsonl(input_path)
    if args.case_id:
        selected = [row for row in cases if row.get("case_id") == args.case_id]
        if not selected:
            raise SystemExit(f"Case not found: {args.case_id}")
    else:
        selected = cases[args.start : args.start + args.n_cases]
    if not selected:
        raise SystemExit("No cases selected.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_out = output_dir / "cases.jsonl"
    scores_out = output_dir / "scores.jsonl"
    total_scores: list[float] = []
    with cases_out.open("w", encoding="utf-8") as cases_f, scores_out.open("w", encoding="utf-8") as scores_f:
        for index, case in enumerate(selected, start=1):
            try:
                result = run_episode(case, config)
            except RuntimeError as exc:
                raise SystemExit(
                    f"Live API rerun is not configured: {exc}\n"
                    "Set LLM_API_KEY, LLM_BASE_URL, and MODEL_NAME in the environment or a local .env file."
                ) from exc
            result["run_index"] = index
            result["config"] = sanitize_config(result.get("config") or {})
            total_scores.append(float(result.get("scores", {}).get("total_score", 0.0)))
            cases_f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            score_row = {
                "episode_id": result["episode_id"],
                "case_id": result["case_id"],
                "model_name": config.get("model_name"),
                **result["scores"],
            }
            scores_f.write(json.dumps(score_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            case_dir = output_dir / result["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "episode.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            with (case_dir / "event_log.jsonl").open("w", encoding="utf-8") as event_f:
                for event in result["events"]:
                    event_f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "num_cases": len(selected),
        "model_name": config.get("model_name"),
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "avg_total_score": sum(total_scores) / len(total_scores),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
