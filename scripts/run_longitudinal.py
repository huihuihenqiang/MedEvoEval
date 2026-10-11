#!/usr/bin/env python
"""Rebuild E1/E2/E3 memory from the public stream, then run H0/H3. Default: print plan."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
from package_paths import data_root,check_output,CASE_FILE
STAGES = [("e0",0,499), ("e1",0,499), ("e2",0,499), ("e3",0,499), ("h0",500,699), ("h3",500,699)]

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="longitudinal_rebuild")
    p.add_argument("--execute", action="store_true", help="Run 2,400 paid model episodes; never used during release verification.")
    p.add_argument("--data-dir")
    p.add_argument("--output-dir",type=Path,default=ROOT/"outputs"/"protocols")
    a=p.parse_args()
    data=data_root(a.data_dir);DATA=data/CASE_FILE
    output=check_output(a.output_dir,data)
    if not a.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in a.run_name):
        p.error("run-name must use letters, numbers, underscore, or hyphen")
    work=output/a.run_name
    plan={"mode":"continual_writeback", "record_order":"ascending scenario_id",
        "stream":[0,499],"heldout":[500,699],"memory_initialization":"E1 empty; E2 and E3 inherit previous pass; H3 copies the post-E3 bank",
        "memory_initialization_source":"public stream episodes",
        "stages":[{"stage":n,"start":s,"end":e,"n":e-s+1} for n,s,e in STAGES]}
    if not a.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return 0
    if not os.environ.get("DASHSCOPE_API_KEY", "").strip():
        raise SystemExit("Set DASHSCOPE_API_KEY before --execute.")
    if work.exists():
        raise SystemExit("Choose a fresh run-name. Partial runs are preserved; automatic memory resume is intentionally disabled.")
    work.mkdir(parents=True)
    (work/"plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    stream_bank=work/"memory_stream"
    for stage,start,end in STAGES:
        cfg=json.loads((ROOT/"configs"/"paper"/(stage+".yaml")).read_text(encoding="utf-8"))
        cfg["runtime"]["dataset_path"]=str(DATA.resolve())
        cfg["runtime"]["run_n_cases"]=end-start+1
        bank=stream_bank if stage in {"e1","e2","e3"} else work/("memory_"+stage)
        if stage=="h3":
            if not stream_bank.exists():raise RuntimeError("post-E3 memory bank missing")
            shutil.copytree(stream_bank,bank)
        cfg["memory"]["kb_root_dir"]=str(bank.resolve())
        cfg_path=work/(stage+".yaml")
        cfg_path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        run_id=a.run_name+"_"+stage
        subprocess.run([sys.executable,"-B",str(ROOT/"scripts"/"run_paper.py"),"--execute",
            "--token-profile","main","--data-dir",str(data),"--output-dir",str(work/"engine"),"--memory-dir",str(bank),"--config",str(cfg_path),"--start",str(start),"--end",str(end),"--run_id",run_id],cwd=ROOT,check=True)
        summary=json.loads((work/"engine"/"runs"/run_id/"batch_summary.json").read_text(encoding="utf-8"))
        # Abort on any recorded episode error; never advance a contaminated memory stage.
        rows=[json.loads(l) for l in (work/"engine"/"runs"/run_id/"case_summaries.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(rows)!=end-start+1 or any(r.get("error_type") or r.get("status")=="error" for r in rows):
            raise RuntimeError("Incomplete stage; inspect the preserved local outputs before restarting with a fresh run-name.")
        if stage in {"e1","e2","e3"}:
            shutil.copytree(stream_bank,work/("snapshot_"+stage))
    (work/"completed.json").write_text(json.dumps({"status":"completed","stages":[s[0] for s in STAGES]},indent=2)+"\n",encoding="utf-8")
    return 0
if __name__=="__main__":raise SystemExit(main())
