"""Single-arm percentile bootstrap intervals for the four-model table."""
from pathlib import Path
import argparse,csv,json
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'results/inputs/dynamic_models.json')
p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/reproduction/results')
a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
rows=json.loads(a.input.read_text(encoding='utf8'));out=[]
for model in ['Qwen-Turbo','Qwen3.5-122B-A10B','GLM-5','Kimi-K2.5']:
 rr=[r for r in rows if r['condition']==model]
 assert len(rr)==80
 for metric in ['S','U']:
  x=np.array([float(r['total_score']) if metric=='S' else .6*float(r['total_score'])-.2*float(r['turns'])/10-.1*float(r['tests'])/3-.1*float(r['invalid_test_rate']) for r in rr])
  rng=np.random.default_rng(7)
  means=x[rng.integers(0,len(x),(20000,len(x)))].mean(axis=1)
  lo,hi=np.quantile(means,[.025,.975])
  out.append(dict(model=model,metric=metric,n=len(x),mean=float(x.mean()),ci95_low=float(lo),ci95_high=float(hi),bootstrap_resamples=20000,bootstrap_seed=7))
with (a.output_dir/'model_bootstrap_intervals.csv').open('w',encoding='utf8',newline='') as f:
 w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
print(json.dumps(out,indent=2))
