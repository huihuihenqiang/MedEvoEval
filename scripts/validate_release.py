#!/usr/bin/env python
"""Offline release integrity, loader, role-flow, and memory smoke. No model calls."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
import hashlib
import importlib
import logging
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from package_paths import data_root,check_output
DATA=None
def rows(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def blocked(*args,**kwargs):raise AssertionError("Network use is forbidden in offline release verification")
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs"/"offline_smoke")
    p.add_argument("--core-only",action="store_true",help="Skip process-corpus counts for the standalone code archive.")
    p.add_argument("--data-dir")
    a=p.parse_args()
    global DATA
    DATA=data_root(a.data_dir);out=check_output(a.output_dir,DATA)
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"engine_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        assert digest(ROOT/item["path"])==item["sha256"],item["path"]
        importlib.import_module(item["path"].removesuffix(".py").replace("/",".").removesuffix(".__init__"))
    from medsim.clinical.config import load_clinical_config
    from medsim.clinical.case_adapter import ScenarioLoaderMedQA
    from medsim.clinical.llm_io import StructuredLLMClient
    from medsim.clinical.orchestrator import run_episode
    import requests
    logging.getLogger().setLevel(logging.CRITICAL)
    cases=rows(DATA/"chinese_medevoeval_medqa700.jsonl")
    assert len(cases)==700
    assert [r["case_id"] for r in cases]==[f"case_{i:03d}" for i in range(1,701)]
    fps=[hashlib.sha256(json.dumps({k:r[k] for k in ["task","patient_view","exam_view","manager_view"]},ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf-8")).hexdigest() for r in cases]
    assert len(set(fps))==688
    counts=Counter(fps);assert sorted(v for v in counts.values() if v>1)==[2]*12
    assert not set(fps[:500])&set(fps[500:])
    with (DATA/"case_scenario_map.csv").open(encoding="utf-8",newline="") as f: mapping=list(csv.DictReader(f))
    assert len(mapping)==700
    assert all(r["content_sha256"]==fps[i] and int(r["scenario_id"])==i for i,r in enumerate(mapping))
    trajectory_count=step_count=None
    if not a.core_only:
        trajectory_count=sum(bool(l.strip()) for l in (DATA/"chinese_medevoeval_process_trajectories.jsonl").open(encoding="utf-8"))
        step_count=sum(bool(l.strip()) for l in (DATA/"chinese_medevoeval_process_steps.jsonl").open(encoding="utf-8"))
        assert (trajectory_count,step_count)==(3103,3386)
    configs=sorted((ROOT/"configs"/"paper").glob("*.yaml"))
    for f in configs:
        c=load_clinical_config(f)
        assert not c.api_keys.dashscope
        assert not c.runtime.use_simulacra_bridge and not c.runtime.write_legacy_outputs
        c.runtime.dataset_path=str(DATA/"chinese_medevoeval_medqa700.jsonl")
    cfg=load_clinical_config(ROOT/"configs"/"paper"/"h3.yaml")
    cfg.runtime.dataset_path=str(DATA/"chinese_medevoeval_medqa700.jsonl")
    loader=ScenarioLoaderMedQA(Path(cfg.runtime.dataset_path),cfg.clinical_tests.allowed_catalog)
    assert loader.num_scenarios==700
    cfg.memory.kb_root_dir=str(out/"controlled_memory")
    if Path(cfg.memory.kb_root_dir).exists():
        raise SystemExit("Use a fresh output-dir for the controlled memory smoke.")
    cfg.mdt.enabled=True
    trace_counts=Counter();state={"doctor_count":0,"scenario":0}
    def controlled(self,model,system,user,fallback,trace_label="",**kwargs):
        trace_counts[trace_label]+=1
        if trace_label!="doctor":return copy.deepcopy(fallback)
        n=state["doctor_count"];state["doctor_count"]+=1
        if n==0:return {"action":"REQUEST_TEST","utterance":"受控测试：请求未定义检查。","test_name":"NOT_IN_PUBLIC_CATALOG"}
        if n==1:return {"action":"REQUEST_TEST","utterance":"受控测试：请求体格检查。","test_name":cfg.clinical_tests.allowed_catalog[0]}
        if n==2:return {"action":"CALL_MDT","utterance":"受控测试：请求顾问复核。"}
        case=loader.get_scenario(state["scenario"])
        return {"action":"FINALIZE","utterance":"受控测试终止。","diagnosis":[case.gold_diagnosis],"evidence":["受控测试"],"plan":["受控测试"],"followup":["受控测试"]}
    def bank_rows():
        return sum(len(rows(f)) for f in (out/"controlled_memory").glob("*.jsonl"))
    episode_checks=[]
    with patch.object(socket.socket,"connect",blocked),patch.object(socket,"create_connection",blocked),patch.object(requests.Session,"request",blocked),patch.object(StructuredLLMClient,"is_available",lambda self:True),patch.object(StructuredLLMClient,"status_snapshot",lambda self:{"status":"ready","configured":True,"offline_stub":True}),patch.object(StructuredLLMClient,"complete_json",controlled):
        for sid,mode in [(0,"writeback"),(1,"writeback"),(2,"readonly"),(3,"disabled")]:
            state.update(doctor_count=0,scenario=sid)
            cfg.memory.enable_retrieval=mode!="disabled"
            cfg.memory.read_only=mode=="readonly"
            before=bank_rows()
            ep=run_episode(cfg,sid,run_id="controlled_smoke",episode_output_root=out/"controlled_episodes")
            after=bank_rows()
            assert len(ep.tests_requested)==1
            assert ep.tests_requested[0].test_name==cfg.clinical_tests.allowed_catalog[0]
            assert ep.tests_requested[0].returned is True
            assert ep.metrics.llm_error_turns>=1
            assert ep.mdt_called
            assert bool(ep.manager_feedbacks)
            if mode=="writeback":assert after>before
            else:assert after==before
            episode_checks.append({"scenario_id":sid,"mode":mode,"invalid_test_rejected_without_substitution":True,"manager_ran":True,"mdt_ran":True,"memory_rows_before":before,"memory_rows_after":after})
    assert {"patient","doctor","consultant","manager_extraction","manager_learning","manager_rule_cards"}<=set(trace_counts)
    report={"status":"passed","validation_kind":"offline_controlled_response_smoke","api_calls":0,"engine_files_verified":len(manifest["files"]),"turn_divisor":"action budget","configs_loaded":len(configs),"records":700,"distinct_content":688,"duplicate_pairs":12,"stream_heldout_content_overlap":0,"process_trajectories":trajectory_count,"process_steps":step_count,"controlled_episodes":episode_checks,"stub_completion_counts":dict(trace_counts),"limits":["No live model rerun","No statistical equivalence or model-quality claim","Memory construction uses the public stream","Provider output rights require author review"]}
    # Exercise the public adapter's execute branch under the same network block.
    import os
    import run_paper
    state.update(doctor_count=0,scenario=0)
    adapter_out=out/"adapter_execution"
    old_argv=sys.argv[:]
    try:
        sys.argv=["run_paper.py","--execute","--data-dir",str(DATA),"--output-dir",str(adapter_out),"--config",str(ROOT/"configs/paper/mdt_qwen.yaml"),"--start","0","--end","0","--run-id","controlled_adapter"]
        with patch.dict(os.environ,{"DASHSCOPE_API_KEY":"offline-controlled-fixture"}),patch.object(socket.socket,"connect",blocked),patch.object(socket,"create_connection",blocked),patch.object(requests.Session,"request",blocked),patch.object(StructuredLLMClient,"is_available",lambda self:True),patch.object(StructuredLLMClient,"status_snapshot",lambda self:{"status":"ready","configured":True,"offline_stub":True}),patch.object(StructuredLLMClient,"complete_json",controlled):
            assert run_paper.main()==0
    finally:
        sys.argv=old_argv
    adapter_records=rows(adapter_out/"runs/controlled_adapter/cases.jsonl")
    assert len(adapter_records)==1 and adapter_records[0]["mdt_called"]
    import jsonschema
    jsonschema.validate(adapter_records[0],json.loads((ROOT/"schemas/real_engine_episode_schema.json").read_text(encoding="utf-8")))
    report["adapter_execute_branch"]={"controlled_episodes":1,"requested_api_calls":0,"custom_output_routing":True,"real_episode_schema_validated":True}
    (out/"validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__":main()
