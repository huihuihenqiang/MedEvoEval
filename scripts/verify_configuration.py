"""Check public configuration selection and displayed role/cap values, offline."""
import argparse, contextlib, io, json, sys
from pathlib import Path
import run_paper

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data-dir')
args=parser.parse_args()
checks=[]
for path in sorted((ROOT/'configs/paper').glob('*.yaml')):
    cfg=json.loads(path.read_text(encoding='utf8'))
    old=sys.argv[:]
    sys.argv=['run_paper.py','--config',str(path),'--run-id','configuration_check']
    if args.data_dir:sys.argv+=['--data-dir',args.data_dir]
    output=io.StringIO()
    try:
        with contextlib.redirect_stdout(output):assert run_paper.main()==0
    finally:sys.argv=old
    facts=json.loads(output.getvalue())
    expected=cfg['models']['mdt_expert'] if cfg['mdt']['enabled'] else None
    assert facts['effective_single_consultant']==expected,path.name
    extended=path.stem.startswith('strong_primary')
    assert facts['token_profile']==('extended_manager' if extended else 'main'),path.name
    assert facts['output_caps']==dict(doctor=1200,patient=500,consultant=500,manager_extraction=2400 if extended else 1200,manager_learning=3600 if extended else 1800,manager_rule_cards=2400 if extended else 1200),path.name
    checks.append({'config':path.name,'consultant':expected,'token_profile':facts['token_profile']})
print(json.dumps({'status':'passed','api_calls':0,'checks':checks}))
