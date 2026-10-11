#!/usr/bin/env python3
"""Recompute camera-ready statistics and vector figures from frozen scalar inputs.
No network, model calls, training, or access to the original project is required.
Usage: python -B scripts/reproduce_results.py
Dependencies: numpy, scipy, matplotlib (versions used are recorded in output).
All outputs, including matplotlib cache, remain under this release root.
"""
from pathlib import Path
import os,csv,json,itertools,hashlib,math,re,warnings,platform
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1]
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--results-dir',type=Path,default=ROOT/'results',help='Read-only frozen results directory')
_parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/reproduction',help='Recomputed tables and figures')
_parser.add_argument('--skip-figures',action='store_true')
_args=_parser.parse_args()
SOURCE=_args.results_dir.resolve();BASE=_args.output_dir.resolve()
if BASE==SOURCE or SOURCE in BASE.parents:raise SystemExit('Choose an output directory outside the frozen results directory')
OUT=BASE/'results';INPUT=SOURCE/'inputs';FIG=BASE/'figures'
OUT.mkdir(parents=True,exist_ok=True)
DISPLAY={"A_无MDT":"No MDT","B_同模型MDT":"Qwen-Turbo consultant","C_强模型MDT":"Kimi-K2.5 consultant"}
def display(text):
 text=str(text)
 for raw,label in DISPLAY.items():text=text.replace(raw,label)
 return text

os.environ['MPLCONFIGDIR']=str(BASE/'cache/mpl')
import numpy as np
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
FIG.mkdir(parents=True,exist_ok=True)
MODELS=['Qwen-Turbo','Qwen3.5-122B-A10B','Kimi-K2.5','GLM-5']
SHORT=['Qwen-Turbo','Qwen3.5','Kimi-K2.5','GLM-5']
BINARY={'strict_accuracy','pass_rate','nondanger_rate','any_error_episode'}
SEVEN=['total_score','utility','turns','tests','strict_accuracy','pass_rate','nondanger_rate']
LABEL={'total_score':'Clinical score S','utility':'Utility U','strict_accuracy':'Strict accuracy','pass_rate':'Pass rate','nondanger_rate':'Nondanger rate','turns':'Turns','tests':'Tests'}
def load(n):return json.loads((INPUT/(n+'.json')).read_text(encoding='utf-8'))
def dump(n,x):
 (OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')
def writecsv(n,rows):
 if not rows:return
 rows=[dict(r,**{k+'_display':display(r[k]) for k in ['condition','comparison'] if k in r}) for r in rows]
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/n).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def canonical(rows):
 for r in rows:
  for k,v in list(r.items()):
   if isinstance(v,str):
    try:r[k]=float(v)
    except ValueError:pass
  r['strict_accuracy']=float(r.get('strict_accuracy',r.get('strict_correct',r.get('exact_dx_pass',r.get('exact_dx_rate',float(r.get('diagnosis_score',0))==1)))))
  r['pass_rate']=float(r.get('pass_rate',r.get('pass',float(r['total_score'])>=.7)))
  if 'diagnosis_grade' in r:r['nondanger_rate']=float(r['diagnosis_grade']!='dangerous')
  else:r['nondanger_rate']=float(r.get('nondanger_rate',r.get('nondanger',float(r.get('diagnosis_score',0))>0)))
  r['any_error_episode']=float(float(r.get('llm_error_turns',0))>0)
  if all(k in r for k in ['total_score','turns','tests','invalid_test_rate']):
   u=.6*r['total_score']-.2*(r['turns']/10)-.1*(r['tests']/3)-.1*r['invalid_test_rate']
   r['utility_recomputed']=u
   # Preserve exact recorded float when its implemented formula agrees; this avoids artificial rank ties.
   if 'utility' not in r or abs(r['utility']-u)>1e-12:r['utility']=u
 return rows
def group(rows,key='condition'):
 d=defaultdict(list)
 for r in rows:d[r[key]].append(r)
 return dict(d)
def mean(rows,m):return float(np.mean([float(x[m]) for x in rows]))
def paired(a,b,key='scenario_id'):
 def index(x):
  d={r[key]:r for r in x};assert len(d)==len(x),('duplicate pairing key',key);return d
 a=index(a);b=index(b);assert set(a)==set(b),'pair membership mismatch';ks=sorted(a);return [a[k] for k in ks],[b[k] for k in ks]
def holm(rows):
 for rr in group(rows,'family_id').values():
  order=np.argsort([float(r['raw_p']) for r in rr],kind='stable');last=0
  for i,j in enumerate(order):
   p=min(1,max(last,(len(rr)-i)*float(rr[j]['raw_p'])));last=p
   rr[j].update(family_size=len(rr),holm_p=p,holm_adjusted_p=p,significant_holm_0_05=p<=.05)
 return rows
def mcnemar(a,b):
 up=int(np.sum((a==0)&(b==1)));down=int(np.sum((a==1)&(b==0)))
 return (float(stats.binomtest(min(up,down),up+down,.5).pvalue) if up+down else 1),up,down
def boot(d,seed=7,n=20000):
 rng=np.random.default_rng(seed);vals=[]
 for start in range(0,n,1000):vals.extend(np.mean(d[rng.integers(0,len(d),size=(min(1000,n-start),len(d)))],axis=1))
 return [float(x) for x in np.quantile(vals,[.025,.975])]
def compare(a,b,metric,family,comparison,ci=True,seed=7,resamples=20000,key='scenario_id'):
 a,b=paired(a,b,key);x=np.array([float(r[metric]) for r in a]);y=np.array([float(r[metric]) for r in b]);d=y-x
 if metric in BINARY:p,up,down=mcnemar(x,y);test='exact McNemar'
 else:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore');p=float(stats.wilcoxon(x,y).pvalue) if np.any(d) else 1
  test='paired Wilcoxon'
 r=dict(family_id=family,comparison=comparison,metric=metric,n=len(x),test=test,mean_a=float(x.mean()),mean_b=float(y.mean()),delta_b_minus_a=float(d.mean()),raw_p=p,analysis_role='reported paired comparison')
 if ci:r.update(zip(['ci95_low','ci95_high'],boot(d,seed,resamples)));r.update(bootstrap_resamples=resamples,bootstrap_seed=seed)
 if metric in BINARY:r.update(discordant_improved=up,discordant_worsened=down)
 return r
def kappa(a,b):
 labels=set(a)|set(b);n=len(a);agree=sum(x==y for x,y in zip(a,b))/n;pe=sum(a.count(v)*b.count(v) for v in labels)/n**2
 return dict(n=n,agreement=agree,kappa=(agree-pe)/(1-pe))

def physician():
 # Individual completed physician ratings are confidential; reproduce published aggregates.
 out=load('physician_review_summary')
 assert out['n']==80 and out['direction_agreement']==21 and out['direction_total']==24
 writecsv('r1_direction_contrasts.csv',out['contrasts'])
 dump('r1_physician_statistics.json',out)
 return out

# Verify frozen source inputs before recomputation; never silently accept a changed cohort.
for entry in json.loads((SOURCE/'frozen_input_hashes.json').read_text(encoding='utf-8')):
 payload=(SOURCE/entry['path']).read_bytes()
 assert hashlib.sha256(payload).hexdigest()==entry['sha256'], 'Frozen input hash mismatch: '+entry['path']

DYNAMIC=canonical(load('dynamic_models'));STREAM=canonical(load('stream_memory'));HELD=canonical(load('heldout_n200'));MDT=canonical(load('mdt'));ABL=canonical(load('ablation'));AB=canonical(load('interface_ab'));BWT=canonical(load('bwt'))
for r in BWT:r['pair_key']=str(r['evaluation_stage'])+'|'+str(r['case_id'])
DG=group(DYNAMIC);SG=group(STREAM);HG=group(HELD);MG=group(MDT);AG=group(ABL);BG=group(BWT)
ALL=[]
r1=physician()
for cond in ['H0','H3']:assert len(HG[cond])==200 and set(int(r['scenario_id']) for r in HG[cond])==set(range(500,700))
ha,hb=paired(HG['H0'],HG['H3']);writecsv('heldout_n200_paired.csv',[dict(scenario_id=int(a['scenario_id']),case_id=a.get('case_id','scenario_'+str(int(a['scenario_id']))),**{'H0_'+m:a[m] for m in SEVEN},**{'H3_'+m:b[m] for m in SEVEN}) for a,b in zip(ha,hb)])
n200=[compare(HG['H0'],HG['H3'],m,'heldout_transfer','H3 - H0',True,507+i,100000) for i,m in enumerate(SEVEN)];holm(n200);ALL+=n200;writecsv('heldout_n200_statistics.csv',n200)
for model in MODELS[1:]:
 for m in ['total_score','utility','strict_accuracy']:ALL.append(compare(DG[MODELS[0]],DG[model],m,'table8_dynamic_models',model+' - '+MODELS[0]))
mc=sorted(MG)
for a,b in itertools.combinations(mc,2):
 for m in ['total_score','utility','strict_accuracy','turns','tests']:ALL.append(compare(MG[a],MG[b],m,'original_mdt_vs_nomdt' if a.startswith('A_') else 'original_mdt_consultant',b+' - '+a))
pg={c:canonical(load('placebo_'+c)) for c in ['h0','h3real','h3shuffled']}
for a,b,f in [('h3shuffled','h3real','real_vs_shuffled'),('h0','h3real','real_vs_h0'),('h0','h3shuffled','shuffled_vs_h0')]:
 for m in ['total_score','utility','turns','tests','evidence_score','plan_score','strict_accuracy','pass_rate','nondanger_rate']:ALL.append(compare(pg[a],pg[b],m,'placebo_'+f,b+' - '+a))
strong={c:canonical(load('strong_'+c)) for c in ['nomdt','mdt']}
assert sum(bool(r['mdt_used']) for r in strong['mdt'])==85
for m in ['total_score','diagnosis_score','evidence_score','plan_score','utility','turns','tests','invalid_test_rate','strict_accuracy','pass_rate','nondanger_rate']:ALL.append(compare(strong['nomdt'],strong['mdt'],m,'strong_mdt_itt','MDT - NoMDT'))
for m in ['total_score','utility','turns','tests','pass_rate','strict_accuracy']:
 ALL.append(compare(AG['EXP_TYPE_FAIL_K3'],AG['EXP_TYPE_FULL_K3'],m,'memory_content_ablation','full memory - failure-only memory'))
 names=['TOPK_FULL_K1','TOPK_FULL_K3','TOPK_FULL_K5'];arr=[]
 for c in names:arr.append([r[m] for r in sorted(AG[c],key=lambda r:r['scenario_id'])])
 if m in BINARY:
  z=np.array(arr).T;col=z.sum(0);row=z.sum(1);tot=z.sum();den=3*tot-np.sum(row**2);q=2*(3*np.sum(col**2)-tot**2)/den if den else 0;p=float(stats.chi2.sf(q,2));test='Cochran Q paired omnibus'
 else:p=float(stats.friedmanchisquare(*arr).pvalue);test='Friedman paired omnibus'
 ALL.append(dict(family_id='retrieval_scale_ablation',comparison='paired omnibus across K=1,3,5',metric=m,n=100,test=test,raw_p=p,means_by_K='; '.join(str(k)+':'+str(float(np.mean(v))) for k,v in zip([1,3,5],arr)),analysis_role='reported omnibus comparison'))
# Mann-Kendall utility trends from the published stage means.
def stage_mk(values):
    x=np.asarray(values,dtype=float);n=len(x)
    s=sum(int(np.sign(x[j]-x[i])) for i in range(n-1) for j in range(i+1,n))
    counts=Counter(x);var=(n*(n-1)*(2*n+5)-sum(k*(k-1)*(2*k+5) for k in counts.values() if k>1))/18
    z=(s-np.sign(s))/math.sqrt(var) if var>0 else 0
    return float(s/(n*(n-1)/2)),float(2*stats.norm.sf(abs(z)))
STAGE_UTILITY={}
for g,rr in group(load('stage_metrics'),'group').items():
    rr=sorted(rr,key=lambda z:int(z['stage_num']))
    STAGE_UTILITY[g]=stage_mk([float(z['utility_mean']) for z in rr])

# Utility-stage trends are recomputed; unaffected stage diagnostics retain their original values.
for r in load('source_family_template'):
 if r['family_id'].startswith('update_trend_'):
  stage_row={k:v for k,v in r.items() if k in ['family_id','family_label','comparison','metric','test','n','delta_b_minus_a','raw_p','analysis_role','notes']}
  if r['metric']=='utility_mean':
   tau,p=STAGE_UTILITY[r['family_id'].split('_')[-1].upper()];stage_row.update(delta_b_minus_a=tau,raw_p=p,notes='Mann-Kendall utility trend from stage means with continuity and tie correction.')
  ALL.append(stage_row)
for m in ['total_score','utility']:ALL.append(compare(BG['before'],BG['after'],m,'backward_transfer','snapshot T10 - original stage snapshot',key='pair_key'))
abg=defaultdict(list)
for r in AB:abg[(r['condition'],r['arm'])].append(r)
for t in load('interface_tests_cached'):
 f=t['family_id'];m=t['metric'];comp=t['comparison']
 if f in ['raw_model_ranking','normalized_model_ranking']:
  arm=f.split('_')[0];model=comp.split(' - ')[0];a=abg[('Qwen-Turbo',arm)];b=abg[(model,arm)]
 else:
  model=comp.split(':')[0];a=abg[(model,'raw')];b=abg[(model,'normalized')]
 ALL.append(compare(a,b,m,'interface_'+f,comp,True,7,20000))
holm(ALL);writecsv('final_holm_tests.csv',ALL)
families=[dict(family_id=k,n_tests=len(v),n_significant_holm=sum(bool(x['significant_holm_0_05']) for x in v),surviving=[x['comparison']+' / '+x['metric'] for x in v if x['significant_holm_0_05']]) for k,v in group(ALL,'family_id').items()]
writecsv('final_holm_families.csv',families)
# Outcome-weight simplex, including boundaries, with final 200-case held-out cohort.
weights=[];base=sorted(MODELS,key=lambda c:-mean(DG[c],'total_score'));meansrows=[]
WG=group(DYNAMIC+STREAM+HELD)
for di in range(11):
 for ei in range(11-di):
  wi=(di/10,ei/10,(10-di-ei)/10);res={}
  for c,rows in WG.items():
   ss=np.array([sum(w*float(r[m]) for w,m in zip(wi,['diagnosis_score','evidence_score','plan_score'])) for r in rows]);us=np.array([.6*s-.2*(r['turns']/10)-.1*(r['tests']/3)-.1*r['invalid_test_rate'] for s,r in zip(ss,rows)])
   res[c]={'S':float(ss.mean()),'U':float(us.mean()),'pass_rate':float(np.mean(ss>=.7)),'strict_accuracy':mean(rows,'strict_accuracy')}
   meansrows.append(dict(w_diag=wi[0],w_evidence=wi[1],w_plan=wi[2],condition=c,n=len(rows),**res[c]))
  rank=sorted(MODELS,key=lambda c:-res[c]['S']);deltaS=res['H3']['S']-res['H0']['S'];deltaU=res['H3']['U']-res['H0']['U']
  weights.append(dict(w_diag=wi[0],w_evidence=wi[1],w_plan=wi[2],interior=all(w>0 for w in wi),rank=' > '.join(rank),rank_preserved=rank==base,heldout_delta_S=deltaS,heldout_delta_U=deltaU,E_strictly_increasing=all(res['E'+str(i)]['S']<res['E'+str(i+1)]['S'] for i in range(3)),E3_minus_E0_S=res['E3']['S']-res['E0']['S']))
writecsv('weight_sensitivity_grid_66.csv',weights);writecsv('weight_sensitivity_condition_means.csv',meansrows)
ws=dict(n_grid=66,n_interior=sum(r['interior'] for r in weights),ranking_preserved=sum(r['rank_preserved'] for r in weights),interior_ranking_preserved=sum(r['rank_preserved'] and r['interior'] for r in weights),heldout_S_positive=sum(r['heldout_delta_S']>0 for r in weights),heldout_U_positive=sum(r['heldout_delta_U']>0 for r in weights),E_strictly_increasing=sum(r['E_strictly_increasing'] for r in weights),E3_gt_E0=sum(r['E3_minus_E0_S']>0 for r in weights),baseline_rank=base,swaps=[r for r in weights if not r['rank_preserved']],utility_turn_divisor=10,strict_accuracy_weight_invariant=True,heldout_n=200)
dump('weight_sensitivity_statistics.json',ws)
# Alternate judges: numeric records only, with explicit scope and fallback coverage.
rejudges=[]
for tag in ['deepseek','qwenmax','kimiretest']:
 rows=load('rejudge_'+tag);conditions=group(rows);orig=[r['orig_strict_correct'] for r in rows];new=[r['new_strict_correct'] for r in rows]
 out=dict(tag=tag,n=len(rows),condition_counts={k:len(v) for k,v in conditions.items()},extraction_ok=sum(bool(r['extraction_ok']) for r in rows),fallback=sum(not bool(r['extraction_ok']) for r in rows),strict_agreement=kappa(orig,new),score_pearson=float(stats.pearsonr([r['orig_total_score'] for r in rows],[r['new_total_score'] for r in rows]).statistic),score_spearman=float(stats.spearmanr([r['orig_total_score'] for r in rows],[r['new_total_score'] for r in rows]).statistic))
 out['model_rank']=sorted(MODELS,key=lambda c:-mean(conditions[c],'new_total_score'));out['model_kendall_tau']=float(stats.kendalltau([base.index(c) for c in MODELS],[out['model_rank'].index(c) for c in MODELS]).statistic)
 out['condition_means']={c:{'S':mean(rr,'new_total_score'),'strict_accuracy':mean(rr,'new_strict_correct')} for c,rr in conditions.items()};rejudges.append(out)
writecsv('judge_reliability.csv',[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rejudges]);dump('judge_reliability.json',dict(results=rejudges,coverage_note='Alternate judges cover the 2520 frozen outputs: four model arms at 80 each, four stream stages at 500 each, and a 100-case-per-arm held-out subset. They do not cover the complete final 200-case-per-arm held-out evaluation. Self test-retest covers 520 model and held-out-subset outputs. Extraction failure uses deterministic substring matching against evidence and management rubric points.'))
jf=load('judge_free');jg=group(jf);jfs=[dict(condition=c,n=len(rr),judge_free_strict=mean(rr,'judge_free_strict'),reported_strict=mean(rr,'reported_strict'),downgraded=sum(int(r['downgraded_by_judge']) for r in rr),upgraded=sum(int(r['upgraded_vs_direction']) for r in rr)) for c,rr in jg.items()];writecsv('judge_free_summary.csv',jfs)
# Reusable summary rows for paper tables and plotting.
summary=[]
for section,rows in [('dynamic_models',DYNAMIC),('stream_memory',STREAM),('heldout_n200',HELD),('original_mdt',MDT),('ablation',ABL),('placebo',sum(pg.values(),[])),('strong_mdt',[dict(r,condition=c) for c,rr in strong.items() for r in rr])]:
 for c,rr in group(rows).items():summary.append(dict(section=section,condition=c,n=len(rr),**{m:mean(rr,m) for m in SEVEN if m in rr[0]}))
writecsv('condition_summary.csv',summary)
final=dict(schema_version=1,software=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__),utility_formula='0.6*S - 0.2*(turns/10) - 0.1*(tests/3) - 0.1*invalid_test_rate',R1=r1,R2=dict(n_pairs=200,scenario_ids='500-699 inclusive',results=n200),A1=dict(n_families=len(families),n_tests=len(ALL),families=families,notes='One Holm family per reported scientific question; no strong-MDT triggered-only family. Utility trends are recomputed from stage means; unaffected stage diagnostics retain their original p-values.'),A2=load('transformation_audit_summary'),A3=dict(rejudges=rejudges,judge_free=jfs),A4=dict(acceptance=load('interface_acceptance'),n=640),A5=dict(n_each=100,results=[r for r in ALL if r['family_id'].startswith('placebo_')],boundary='Same card-count and truncation policy; exact token/character length matching is not established. Frozen-bank rerun does not identify a relevance benefit.'),A6=ws,A7=dict(n_pairs=100,mdt_triggered=85,files='Only shard0of6 through shard5of6',results=[r for r in ALL if r['family_id']=='strong_mdt_itt']),A8=dict(interface_measured_usage=load('interface_usage_summary')),case_record_boundary='Released 700 records represent 688 unique full contents. Stream:500 records/488 contents; heldout:200/200 with zero stream overlap; cross-sectional:100/88; model comparison:80/68. Inference pairs/resamples case records, not deduplicated independent contents.',limitations=['Physician direction contrasts are descriptive group-mean signs, not paired inferential tests.','Physician audit tables use published aggregates; completed individual ratings are confidential.','Held-out subset rejudging does not establish judge stability for every final 200-case output.','Utility sensitivity always uses turn divisor 10; total-score sensitivity changes only outcome weights.'])
dump('final_statistics.json',final)
print(json.dumps({'R1_kappa':r1['agreement'],'R1_direction':str(r1['direction_agreement'])+'/24','N200':[{k:r[k] for k in ['metric','delta_b_minus_a','raw_p','holm_p','ci95_low','ci95_high']} for r in n200],'families':len(families),'tests':len(ALL),'weights':ws},ensure_ascii=False))

# Five utility-weight settings from the original analysis, using implemented turn divisor 10.
utility_settings=[('default',.6,.2,.1,.1),('outcome_heavy',.8,.1,.1,0),('turn_cost_heavy',.4,.4,.1,.1),('test_cost_heavy',.5,.1,.3,.1),('invalid_action_heavy',.5,.1,.1,.3)]
utility_rows=[]
for name,ws,wt,we,wr in utility_settings:
 vals={c:float(np.mean([ws*r['total_score']-wt*(r['turns']/10)-we*(r['tests']/3)-wr*r['invalid_test_rate'] for r in rr])) for c,rr in DG.items()}
 for rank,c in enumerate(sorted(vals,key=vals.get,reverse=True),1):utility_rows.append(dict(setting=name,weights_S_turns_tests_invalid=f'({ws:.1f},{wt:.1f},{we:.1f},{wr:.1f})',turn_divisor=10,test_divisor=3,rank=rank,model=c,utility=vals[c]))
writecsv('utility_sensitivity.csv',utility_rows);final['R7_utility_sensitivity']=utility_rows
resource=load('resource_summary')
writecsv('resource_summary.csv',resource['rows'])
dump('resource_summary.json',resource)
final['A8']={'resource_summary':resource}
dump('final_statistics.json',final)

# Named outcome weights, including equal thirds outside the tenth-step grid.
named=[]
for label,ww in [('default',(0.5,0.3,0.2)),('balanced',(0.4,0.4,0.2)),('diagnosis_heavy',(0.6,0.2,0.2)),('equal_thirds',(1/3,1/3,1/3))]:
 for c,rr in WG.items():
  scores=[sum(w*r[m] for w,m in zip(ww,['diagnosis_score','evidence_score','plan_score'])) for r in rr]
  utilities=[.6*s-.2*(r['turns']/10)-.1*(r['tests']/3)-.1*r['invalid_test_rate'] for s,r in zip(scores,rr)]
  named.append(dict(weighting=label,w_diag=ww[0],w_evidence=ww[1],w_plan=ww[2],condition=c,n=len(rr),S=float(np.mean(scores)),U=float(np.mean(utilities)),strict_accuracy=mean(rr,'strict_accuracy'),pass_rate=float(np.mean(np.array(scores)>=.7))))
writecsv('weight_sensitivity_named.csv',named)
final['A6']['named_weightings_file']='weight_sensitivity_named.csv';dump('final_statistics.json',final)

final['display_labels']=DISPLAY
final['model_identity_notes']={'original_mdt':'The final 100-case study compares no MDT, a Qwen-Turbo consultant and a Kimi-K2.5 consultant with a Qwen-Turbo primary. The consultant is selected through models.mdt_expert.','strong_primary_mdt':'Primary and effective single consultant are glm-5; kimi-k2.5 is manager. Raw keys are retained for provenance.'}
dump('final_statistics.json',final)
import subprocess,sys
subprocess.run([sys.executable,'-B',str(ROOT/'scripts/reproduce_model_intervals.py'),'--input',str(INPUT/'dynamic_models.json'),'--output-dir',str(OUT)],check=True)
if not _args.skip_figures:
 import subprocess,sys
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/reproduce_statistics_figures.py'),'--results-dir',str(OUT),'--input-dir',str(INPUT),'--output-dir',str(FIG)],check=True)
print(json.dumps({'status':'reproduced','results_dir':str(OUT),'figures_dir':None if _args.skip_figures else str(FIG),'api_calls':0}))
