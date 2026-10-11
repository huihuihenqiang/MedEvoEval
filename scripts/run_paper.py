#!/usr/bin/env python
"""Real MedEvoEval engine. Default: check configuration and data without API calls."""
from __future__ import annotations
import argparse,json,os,re,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from package_paths import data_root,config_path,check_output,route_outputs,CASE_FILE

def main_max_tokens(label):
    label=(label or "").strip().lower()
    if label=="doctor":return 1200
    if label.startswith("mdt"):return 700
    if label=="manager_learning":return 1800
    if label.startswith("manager"):return 1200
    return 500

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",default="configs/paper/benchmark_qwen-turbo.yaml")
    p.add_argument("--data-dir",help="Dataset root; otherwise MEDEVOEVAL_DATA_DIR or sibling MedEvoEval_data")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs"/"engine")
    p.add_argument("--memory-dir",type=Path,help="Explicit writable bank, required for initialized-memory profiles")
    p.add_argument("--execute",action="store_true",help="Make real provider calls; charges may apply")
    p.add_argument("--token-profile",choices=["main","extended_manager"],default=None)
    p.add_argument("--run-id","--run_id",dest="run_id",default=None)
    a,engine_args=p.parse_known_args()
    if any(x.startswith("--config") or x.startswith("--run_id") for x in engine_args):p.error("Use adapter configuration/run-id arguments")
    data=data_root(a.data_dir);out=check_output(a.output_dir,data);profile=config_path(a.config)
    run_id=a.run_id or datetime.utcnow().strftime("local-%Y%m%d-%H%M%S")
    if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):p.error("run-id must use letters, digits, underscore or hyphen")
    import yaml
    raw=yaml.safe_load(profile.read_text(encoding="utf-8-sig"))
    token_profile=a.token_profile or raw.get("release",{}).get("token_profile","main")
    if token_profile not in {"main","extended_manager"}:p.error("Invalid release.token_profile")
    if raw.get("api_keys",{}).get("dashscope"):p.error("Keep credentials out of YAML; use DASHSCOPE_API_KEY")
    raw["runtime"]["dataset_path"]=str(data/CASE_FILE)
    if raw["runtime"].get("use_simulacra_bridge") or raw["runtime"].get("write_legacy_outputs"):
        p.error("Public profiles must disable Simulacra and legacy UI output")
    bank=check_output(a.memory_dir or out/"memory"/run_id,data)
    raw["memory"]["kb_root_dir"]=str(bank)
    from medsim.clinical.config import load_clinical_config
    from medsim.clinical.case_adapter import ScenarioLoaderMedQA
    cfg=load_clinical_config(profile)
    cfg.runtime.dataset_path=str(data/CASE_FILE);cfg.memory.kb_root_dir=str(bank)
    loader=ScenarioLoaderMedQA(Path(cfg.runtime.dataset_path),cfg.clinical_tests.allowed_catalog)
    facts={"status":"checked_without_api","n_records":loader.num_scenarios,"data_dir":str(data),"output_dir":str(out),"profile":profile.name,"configured_models":vars(cfg.models),"effective_single_consultant":cfg.models.mdt_expert if cfg.mdt.enabled else None,"memory":{"retrieval":cfg.memory.enable_retrieval,"writeback":cfg.memory.enable_retrieval and not cfg.memory.read_only,"bank":str(bank)},"token_profile":token_profile,"engine_args":engine_args}
    from medsim.clinical import llm_io
    caps=main_max_tokens if token_profile=="main" else llm_io._max_tokens_for_trace
    facts["output_caps"]={label:caps(label) for label in ["doctor","patient","consultant","manager_extraction","manager_learning","manager_rule_cards"]}
    if not a.execute:print(json.dumps(facts,ensure_ascii=False,indent=2));return 0
    if not os.environ.get("DASHSCOPE_API_KEY","").strip():p.error("Set DASHSCOPE_API_KEY before --execute")
    if profile.stem in {"e2","e3","h3"} and (a.memory_dir is None or not bank.is_dir() or not any(bank.iterdir())):
        p.error("This profile needs an initialized --memory-dir; use run_longitudinal.py to reconstruct memory")
    if (out/"runs"/run_id).exists():p.error("Choose a fresh run-id; existing runs are preserved")
    effective=out/"effective_configs"/(run_id+".json")
    effective.parent.mkdir(parents=True,exist_ok=True)
    effective.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.environ["MEDSIM_CONFIG_PATH"]=str(effective)
    from medsim.clinical import llm_io
    engine_token_caps=llm_io._max_tokens_for_trace
    if token_profile=="main":llm_io._max_tokens_for_trace=main_max_tokens
    engine=route_outputs(out)
    sys.argv=["medsim.run_clinic","--config",str(effective),"--run_id",run_id,*engine_args]
    try:return engine.main()
    finally:llm_io._max_tokens_for_trace=engine_token_caps
if __name__=="__main__":raise SystemExit(main())
