"""Portable locations shared by the release adapters; no model calls."""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
CASE_FILE = "chinese_medevoeval_medqa700.jsonl"
def data_root(value=None):
    p = Path(value or os.environ.get("MEDEVOEVAL_DATA_DIR") or ROOT.parent / "MedEvoEval_data").expanduser().resolve()
    if not (p / CASE_FILE).is_file():
        raise SystemExit(f"Missing {CASE_FILE} in {p}. Use --data-dir PATH or MEDEVOEVAL_DATA_DIR.")
    return p
def config_path(value):
    p=Path(value).expanduser()
    if not p.is_absolute() and not p.is_file(): p=ROOT/p
    if not p.is_file(): raise SystemExit(f"Configuration not found: {p}")
    return p.resolve()
def check_output(output, data):
    output=Path(output).expanduser().resolve()
    if output == data or data in output.parents:
        raise SystemExit("Output must not be inside the read-only data package.")
    return output
def route_outputs(output):
    from medsim.clinical import writers, llm_io, experiment4
    writers.RUNS_ROOT=output/"runs"
    writers.EPISODES_ROOT=output/"episodes"
    llm_io.LLM_INPUT_LOG_FILE=output/"llm_inputs"/"requests.jsonl"
    experiment4.EXP4_OUTPUT_DIR=output/"exp4"
    from medsim import run_clinic
    run_clinic.RUNS_ROOT=writers.RUNS_ROOT
    run_clinic.EXP4_OUTPUT_DIR=experiment4.EXP4_OUTPUT_DIR
    return run_clinic
