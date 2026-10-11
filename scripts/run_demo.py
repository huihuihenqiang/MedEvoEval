#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medevoeval.episode_runner import load_config, read_jsonl, run_episode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline MedEvoEval demo episode.")
    parser.add_argument("--config", default="configs/demo.yaml", help="Path to YAML config relative to repo root.")
    parser.add_argument("--case-id", default=None, help="Optional case_id from examples/sample_cases.jsonl.")
    parser.add_argument("--output-dir", default="outputs/demo", help="Output directory.")
    args = parser.parse_args()

    config_path = ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"Missing config: {config_path}")
    config = load_config(config_path)
    cases = read_jsonl(ROOT / config.get("input_path", "../data/sample_cases.jsonl"))
    if not cases:
        raise SystemExit("No cases found. Check input_path in the config.")
    case = next((row for row in cases if row.get("case_id") == args.case_id), cases[0])

    result = run_episode(case, config)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "episode.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "event_log.jsonl").open("w", encoding="utf-8") as f:
        for event in result["events"]:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as f:
        row = {"episode_id": result["episode_id"], "case_id": result["case_id"], **result["scores"]}
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"episode_id": result["episode_id"], "output_dir": str(out_dir), "scores": result["scores"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

