"""Deterministic vector figures from frozen scalar release inputs; no model calls."""
from pathlib import Path
import os,json,csv,argparse
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results-dir',type=Path,default=ROOT/'results')
parser.add_argument('--input-dir',type=Path,default=ROOT/'results/inputs')
parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures')
parser.add_argument('--only',nargs='+')
args=parser.parse_args();OUT=args.results_dir.resolve();I=args.input_dir.resolve();F=args.output_dir.resolve()
if F==I or I in F.parents or F==OUT or OUT in F.parents:raise SystemExit('Write figures outside read-only inputs/results')
F.mkdir(parents=True,exist_ok=True)
selected=args.only
os.environ['MPLCONFIGDIR']=str(F.parent/'cache/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':10,'savefig.bbox':'tight'})
# Preserve the submitted figure identities; metric colors are shared across panels.
C=['#4C78A8','#F58518','#54A24B','#8E6CBB']
MODEL_COLORS={'Qwen-Turbo':'#4C78A8','Qwen3.5-122B-A10B':'#2CB1A2','Kimi-K2.5':'#F58518','GLM-5':'#54A24B'}
CONDITION_COLORS={'E0':'#5F6368','E1':'#2CB1A2','E2':'#F58518','E3':'#54A24B','G0':'#5F6368','G1':'#54A24B','H0':'#5F6368','H3':'#54A24B'}
PROCESS_COLORS=['#5F6368','#E45756']
MODELS=['Qwen-Turbo','Qwen3.5-122B-A10B','GLM-5','Kimi-K2.5'];SHORT=['Qwen\nTurbo','Qwen3.5','GLM-5','Kimi\nK2.5']
def load(n):
 rows=json.loads((I/(n+'.json')).read_text(encoding='utf8'))
 if not isinstance(rows,list):return rows
 for r in rows:
  if all(k in r for k in ['total_score','turns','tests','invalid_test_rate']):
   r['utility']=.6*float(r['total_score'])-.2*(float(r['turns'])/10)-.1*(float(r['tests'])/3)-.1*float(r['invalid_test_rate'])
 return rows
def csvr(n):return list(csv.DictReader((OUT/n).open(encoding='utf8')))
def group(rows,key='condition'):
 d=defaultdict(list)
 for r in rows:d[r[key]].append(r)
 return d
def av(rows,k):return np.mean([float(r[k]) for r in rows])
def sem(rows,k):return np.std([float(r[k]) for r in rows],ddof=1)/np.sqrt(len(rows))
def save(fig,name):
 if selected and name not in selected:
  plt.close(fig);return
 fig.savefig(F/(name+'.pdf'),metadata={'Creator':'MedEvoEval frozen-result reproduction','CreationDate':None,'ModDate':None});fig.savefig(F/(name+'.png'),dpi=160);plt.close(fig)
def style(ax):ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
def stagger_labels(fig,labels):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();placed=[]
 for text in labels:
  box=text.get_window_extent(renderer).expanded(1.06,1.12)
  while any(box.overlaps(other) for other in placed):
   x,y=text.get_position();text.set_position((x,y+8));box=text.get_window_extent(renderer).expanded(1.06,1.12)
  placed.append(box)
def bars(ax,groups,names,metrics,labels,colors=None):
 x=np.arange(len(names));width=.75/len(metrics)
 for j,(m,lab) in enumerate(zip(metrics,labels)):
  ax.bar(x+(j-(len(metrics)-1)/2)*width,[av(groups[n],m) for n in names],width,label=lab,color=(colors or C)[j],edgecolor='white',linewidth=.4)
 ax.set_xticks(x);style(ax)
 ymax=max(av(groups[n],m) for n in names for m in metrics);ymin=min(0,min(av(groups[n],m) for n in names for m in metrics));ax.set_ylim(ymin*1.2-.01,ymax*1.45+.025)
D=group(load('dynamic_models'));S=group(load('stream_memory'));H=group(load('heldout_n200'));M=group(load('mdt'));A=group(load('ablation'));B=load('bwt');stage=group(load('stage_metrics'),'group')
for rr in stage.values():rr.sort(key=lambda r:float(r['stage_num']))
mdtnames=sorted(M);mlabel=['No MDT','Qwen-Turbo\nconsultant','Kimi-K2.5\nconsultant']
final=json.loads((OUT/'final_statistics.json').read_text(encoding='utf8'))
# Main dashboard: restore the submitted four-panel questions with verified inputs.
fig,ax=plt.subplots(2,2,figsize=(6.8,5.1),layout='constrained')
def point_labels(a,x,values,dy=6,fmt='.2f',color='black'):
 for k,(xx,v) in enumerate(zip(x,values)):
  offset=dy[k] if isinstance(dy,(list,tuple)) else dy
  a.annotate(format(v,fmt),(xx,v),xytext=(0,offset),textcoords='offset points',ha='center',fontsize=8,color=color)
bars(ax[0,0],D,MODELS,['total_score','strict_correct'],['Clinical score S','Strict accuracy'],colors=[C[0],C[2]])
ax[0,0].set_xticklabels(SHORT);ax[0,0].set_title('(A) Model discriminability',loc='left');ax[0,0].legend(fontsize=8);ax[0,0].set_ylabel('Mean value');ax[0,0].set_ylim(0,.78)
for container in ax[0,0].containers:ax[0,0].bar_label(container,fmt='%.2f',padding=3,fontsize=8)
x=np.arange(2)
for j,(metric,label,color) in enumerate(zip(['turns','tests'],['Turns vs no MDT','Tests vs no MDT'],PROCESS_COLORS)):
 vals=[av(M[g],metric)-av(M[mdtnames[0]],metric) for g in mdtnames[1:]]
 container=ax[0,1].bar(x+(j-.5)*.34,vals,.34,label=label,color=color)
 ax[0,1].bar_label(container,fmt='%+.2f',padding=3,fontsize=8)
ax[0,1].axhline(0,color='grey',lw=.8);ax[0,1].set_xticks(x);ax[0,1].set_xticklabels(['Qwen-Turbo\nconsultant','Kimi-K2.5\nconsultant']);ax[0,1].set_ylim(-1.0,2.5);ax[0,1].set_title('(B) MDT process shift',loc='left');ax[0,1].set_ylabel('Mean difference');ax[0,1].legend(fontsize=8);style(ax[0,1])
panel=ax[1,0]
for metric,label,color,marker in [('total_score','Clinical score S',C[0],'o'),('utility','Utility U',C[1],'s')]:
 vals=[av(S[g],metric) for g in ['E0','E1','E2','E3']]
 panel.plot(range(4),vals,color=color,marker=marker,ms=4,lw=1.6,label=label)
 point_labels(panel,range(4),vals,dy=6 if metric=='total_score' else [-10,-10,6,6])
 offset=-.17 if metric=='total_score' else .17
 container=panel.bar(np.array([4.8,5.8])+offset,[av(H[g],metric) for g in ['H0','H3']],width=.32,color=color,edgecolor='white',linewidth=.4)
 labels=panel.bar_label(container,fmt='%.2f',padding=3,fontsize=8)
 if metric=='utility':
  for text in labels:text.set_position((5,3))
panel.axvspan(-.45,3.45,color='#EAF1F8',alpha=.65,zorder=0)
panel.axvspan(4.3,6.55,color='#EEF5EC',alpha=.8,zorder=0)
panel.axhline(0,color='#777777',lw=.8,ls='--',zorder=1)
panel.set_xlim(-.45,6.55);panel.set_ylim(-.14,.66)
panel.set_xticks([0,1,2,3,4.8,5.8]);panel.set_xticklabels(['E0','E1','E2','E3','H0','H3'])
panel.set_title('(C) Stream and held-out',loc='left');panel.set_ylabel('Mean score');panel.legend(fontsize=8,loc='upper left',ncol=2,handlelength=1.4,columnspacing=.9)
panel.set_xlabel('Stream: 500/condition     Held-out: 200/condition',fontsize=8);style(panel)
x=np.arange(3)
for g,marker,dy in [('G0','o',[7,-13,-13]),('G1','s',[-13,7,7])]:
 vals=[np.mean([float(r['total_score_mean']) for r in stage[g][a:b]]) for a,b in [(0,4),(4,6),(6,10)]]
 ax[1,1].plot(x,vals,marker=marker,ms=4,ls='--' if g=='G0' else '-',color=CONDITION_COLORS[g],label=g);point_labels(ax[1,1],x,vals,dy,color=CONDITION_COLORS[g])
ax[1,1].axvspan(.75,1.25,color=C[1],alpha=.12,zorder=0)
bwtvals={metric:av([r for r in B if r['condition']=='after'],metric)-av([r for r in B if r['condition']=='before'],metric) for metric in ['total_score','utility']}
ax[1,1].text(.97,.08,f"BWT: S {bwtvals['total_score']:+.4f}, U {bwtvals['utility']:+.4f}",transform=ax[1,1].transAxes,ha='right',fontsize=8,color='#5F6368')
ax[1,1].set_xticks(x);ax[1,1].set_xticklabels(['Pre','Update','Post']);ax[1,1].set_ylim(.18,.41);ax[1,1].set_title('(D) Update response and retention',loc='left');ax[1,1].set_ylabel('Clinical score S');ax[1,1].legend(fontsize=8);style(ax[1,1]);save(fig,'main_experimental_dashboard')
# Model appendix: components, process, distributions, score vs tests.
fig,ax=plt.subplots(2,2,figsize=(6.8,4.9),layout='constrained')
bars(ax[0,0],D,MODELS,['diagnosis_score','evidence_score','plan_score'],['Diagnosis','Evidence','Plan'],colors=[C[0],'#2CB1A2',C[1]]);ax[0,0].set_xticklabels(SHORT,rotation=0);ax[0,0].set_ylim(0,1);ax[0,0].legend(fontsize=8,loc='upper left');ax[0,0].set_title('(A) Outcome components',loc='left')
component_labels=[]
for container in ax[0,0].containers:component_labels.extend(ax[0,0].bar_label(container,fmt='%.2f',padding=2,fontsize=7))
bars(ax[0,1],D,MODELS,['turns','tests'],['Turns','Tests'],colors=PROCESS_COLORS);ax[0,1].set_xticklabels(SHORT,rotation=0);ax[0,1].legend(fontsize=9);ax[0,1].set_title('(B) Resource use',loc='left')
for container in ax[0,1].containers:ax[0,1].bar_label(container,fmt='%.2f',padding=2,fontsize=8)
box=ax[1,0].boxplot([[float(r['total_score']) for r in D[m]] for m in MODELS],patch_artist=True,showfliers=True)
ax[1,0].set_xticks(range(1,len(SHORT)+1),SHORT)
for b in box['boxes']:b.set_facecolor('#DCE8F5');b.set_edgecolor(C[0])
ax[1,0].set_title('(C) Clinical score distributions',loc='left');ax[1,0].set_ylabel('Clinical score S');ax[1,0].tick_params(axis='x',rotation=0);style(ax[1,0])
for m,l,c in zip(MODELS,SHORT,[MODEL_COLORS[m] for m in MODELS]):ax[1,1].scatter(av(D[m],'tests'),av(D[m],'total_score'),s=65,color=c,label=l)
ax[1,1].set_xlabel('Mean tests');ax[1,1].set_ylabel('Mean clinical score S');ax[1,1].set_title('(D) Outcomes and testing',loc='left');ax[1,1].legend(fontsize=9);style(ax[1,1]);stagger_labels(fig,component_labels);save(fig,'fig_app_model_comparison')
# MDT: full assigned-arm means; paired grade/advice breakdown among called cases.
behavior=group(load('mdt_behavior'));base_mdt={int(r['scenario_id']):r for r in M[mdtnames[0]]}
fig,ax=plt.subplots(2,2,figsize=(6.8,5.1),layout='constrained')
bars(ax[0,0],M,mdtnames,['total_score','utility','strict_correct'],['Clinical score S','Utility U','Strict accuracy']);ax[0,0].legend(fontsize=8);ax[0,0].set_title('(A) Outcomes (100/condition)',loc='left');ax[0,0].set_ylabel('Mean / proportion')
bars(ax[0,1],M,mdtnames,['turns','tests'],['Turns','Tests'],colors=PROCESS_COLORS);ax[0,1].legend(fontsize=8);ax[0,1].set_title('(B) Resource use',loc='left');ax[0,1].set_ylabel('Mean count')
for a in ax[0]:
 a.set_xticklabels(mlabel)
 for container in a.containers:a.bar_label(container,fmt='%.2f',padding=3,fontsize=8)
for j,g in enumerate(mdtnames[1:]):
 active=[r for r in behavior[g] if r['mdt_called']==1];n=len(active);lookup={int(r['scenario_id']):r for r in M[g]}
 delta=[float(lookup[int(r['scenario_id'])]['diagnosis_score'])-float(base_mdt[int(r['scenario_id'])]['diagnosis_score']) for r in active]
 counts=[sum(v>1e-9 for v in delta),sum(abs(v)<=1e-9 for v in delta),sum(v< -1e-9 for v in delta)]
 bottom=0
 for count,label,color in zip(counts,['Higher','Same','Lower'],[C[2],'#B8BEC8','#E45756']):
  height=100*count/n;ax[1,0].bar(j,height,bottom=bottom,width=.58,color=color,label=label if j==0 else None);ax[1,0].text(j,bottom+height/2,str(count),ha='center',va='center',fontsize=9,color='white' if color==C[0] else 'black');bottom+=height
 bottom=0
 for action,color in zip(['ASK','FINALIZE'],['#2CB1A2','#8E6CBB']):
  count=sum(r['consultant_advice_action']==action for r in active);height=100*count/n;ax[1,1].bar(j,height,bottom=bottom,width=.58,color=color,label=action if j==0 else None);ax[1,1].text(j,bottom+height/2,str(count),ha='center',va='center',fontsize=9,color='white' if color==C[0] else 'black');bottom+=height
for a,title in zip(ax[1],['(C) Diagnosis grade vs. no MDT','(D) Consultant advice']):
 a.set_title(title,loc='left');a.set_ylim(0,126);a.set_yticks([0,25,50,75,100]);a.set_ylabel('Called cases (%)');a.set_xticks([0,1]);a.set_xticklabels(['Qwen-Turbo\n98/100 called','Kimi-K2.5\n99/100 called']);a.legend(loc='upper center',ncol=3 if a==ax[1,0] else 2,fontsize=8,frameon=False);style(a)
save(fig,'fig_app_mdt_analysis')
# Six endpoints in fixed consecutive windows of 50 records.
fig,ax=plt.subplots(2,3,figsize=(6.8,3.9),layout='constrained')
metrics=['total_score','utility','exact_dx_pass','pass_rate','turns','tests']
for g,c,marker in zip(['E0','E1','E2','E3'],[CONDITION_COLORS[g] for g in ['E0','E1','E2','E3']],['o','s','^','D']):
 rr=sorted(S[g],key=lambda r:int(r['scenario_id']));assert len(rr)==500
 chunks=[rr[k:k+50] for k in range(0,500,50)]
 for a,metric in zip(ax.flat,metrics):
  factor=100 if metric in ['exact_dx_pass','pass_rate'] else 1
  a.plot(range(1,11),[factor*av(v,metric) for v in chunks],linestyle='-',marker=marker,ms=3,color=c,label=g)
for a,title,unit in zip(ax.flat,['(A) Clinical score','(B) Utility','(C) Strict accuracy','(D) Pass rate','(E) Counted turns','(F) Requested tests'],['Mean S','Mean U','Cases (%)','Cases (%)','Mean count','Mean count']):
 a.set_title(title,loc='left',fontsize=9);a.set_ylabel(unit,fontsize=9);a.set_xlabel('Stage (50 records)',fontsize=9);a.set_xticks([1,3,5,7,9,10]);a.tick_params(labelsize=8);style(a)
handles,labels=ax[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',fontsize=9,ncol=4,frameon=False)
save(fig,'fig_app_learning_curves')
# Held-out: separate process scale, and directly display paired score/rate CIs.
fig=plt.figure(figsize=(6.8,4.6),layout='constrained');gs=fig.add_gridspec(2,2);ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
rows=final['R2']['results'];rate=[r for r in rows if r['metric'] not in ['turns','tests']]
x=np.arange(len(rate));ax[0].bar(x-.17,[r['mean_a'] for r in rate],.34,color=CONDITION_COLORS['H0'],label='H0');ax[0].bar(x+.17,[r['mean_b'] for r in rate],.34,color=CONDITION_COLORS['H3'],label='H3');ax[0].set_xticks(x);ax[0].set_xticklabels(['S','U','Strict','Pass','Non-\ndanger'],rotation=20);ax[0].legend(fontsize=9);ax[0].set_title('(A) Held-out means (200 pairs)',loc='left');style(ax[0])
bars(ax[1],H,['H0','H3'],['turns','tests'],['Turns','Tests'],colors=PROCESS_COLORS);ax[1].set_xticklabels(['H0','H3']);ax[1].legend(fontsize=9);ax[1].set_title('(B) Process means',loc='left')
for a in ax[:2]:
 for container in a.containers:a.bar_label(container,fmt='%.2f',padding=3,fontsize=7 if a==ax[0] else 8)
ax[0].set_ylim(0,.85)
for j,r in enumerate(rate):
 v=r['delta_b_minus_a'];sig=r['significant_holm_0_05'];col=C[2] if sig else C[0]
 ax[2].errorbar(v,j,xerr=[[v-r['ci95_low']],[r['ci95_high']-v]],fmt='o',color=col,markerfacecolor=col if sig else 'white',markeredgecolor=col,markeredgewidth=1.1,capsize=3)
for j,r in enumerate(rate):ax[2].text(.163,j,format(r['delta_b_minus_a'],'+.3f'),ha='left',va='center',fontsize=8)
ax[2].set_xlim(-.055,.20);ax[2].axvline(0,color='grey',lw=1,ls='--');ax[2].set_yticks(range(len(rate)));ax[2].set_yticklabels(['S','U','Strict','Pass','Nondanger']);ax[2].invert_yaxis();ax[2].set_xlabel('H3 − H0 (95% paired bootstrap confidence interval)');ax[2].set_title('(C) Score / rate differences',loc='left');save(fig,'fig_app_external_transfer')
# Update diagnostics: preserve phases, paired BWT bars, and both S/U CUSUM curves.
fig,ax=plt.subplots(2,2,figsize=(6.8,4.4),layout='constrained')
for g,c,marker in zip(['G0','G1'],[CONDITION_COLORS[g] for g in ['G0','G1']],['o','s']):
 for a,m in zip(ax[0],['total_score_mean','utility_mean']):a.plot(range(1,11),[float(r[m]) for r in stage[g]],linestyle='--' if g=='G0' else '-',marker=marker,ms=3,color=c,label=g)
for a,title,lab in zip(ax[0],['(A) Stage clinical score','(B) Stage utility'],['S','U']):
 for left,right,color,label in [(.5,4.5,C[0],'pre'),(4.5,6.5,C[1],'update'),(6.5,10.5,C[2],'post')]:
  a.axvspan(left,right,color=color,alpha=.07 if label!='update' else .12,zorder=0);a.text((left+right)/2,.04,label,transform=a.get_xaxis_transform(),ha='center',fontsize=8,color='#5F6368')
 a.set_title(title,loc='left');a.set_ylabel(lab);a.set_xlabel('Stage');a.set_xticks(range(1,11));a.legend(fontsize=8);style(a)
for j,(m,c,lab) in enumerate(zip(['total_score','utility'],C,['Clinical score S','Utility U'])):
 vals=[]
 for stage_id in range(1,10):
  before=[r for r in B if r['evaluation_stage']==stage_id and r['condition']=='before'];after=[r for r in B if r['evaluation_stage']==stage_id and r['condition']=='after'];vals.append(av(after,m)-av(before,m))
 ax[1,0].bar(np.arange(1,10)+(j-.5)*.35,vals,.35,color=c,label=lab)
ax[1,0].axhline(0,color='grey',lw=1);ax[1,0].set_title('(C) Backward-transfer differences',loc='left');ax[1,0].set_xlabel('Prior stage (50 pairs each)');ax[1,0].set_ylabel('Final snapshot − original snapshot');ax[1,0].set_xticks(range(1,10));ax[1,0].set_ylim(-.085,.07);ax[1,0].legend(fontsize=8);style(ax[1,0])
for metric,col,lab,marker in [('total_score_mean',C[0],'Clinical score S','o'),('utility_mean',C[1],'Utility U','s')]:
 vals=np.array([float(r[metric]) for r in stage['G1']]);ax[1,1].plot(range(1,11),np.cumsum(vals-vals.mean()),marker=marker,ms=3,color=col,label=lab)
ax[1,1].axhline(0,color='grey',ls='--',lw=1);ax[1,1].axvline(4,color='grey',ls='--',lw=.8);ax[1,1].set_title('(D) G1 CUSUM diagnostics',loc='left');ax[1,1].set_xlabel('Stage');ax[1,1].set_ylabel('Cumulative centered deviation');ax[1,1].legend(fontsize=8);style(ax[1,1]);save(fig,'fig_app_update_diagnostics')
# Ablations: restore the submitted content endpoints and K-line presentation.
fig,ax=plt.subplots(1,2,figsize=(6.8,2.9),layout='constrained')
names=['EXP_TYPE_FULL_K3','EXP_TYPE_FAIL_K3']
bars(ax[0],A,names,['total_score','exact_dx_rate','tests'],['Clinical score S','Strict accuracy','Tests'],colors=[C[0],C[1],PROCESS_COLORS[1]])
ax[0].set_xticklabels(['Success + failure','Failure only']);ax[0].set_title('(A) Memory content at K = 3',loc='left');ax[0].set_ylabel('Mean / proportion');ax[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.02),frameon=False);ax[0].set_ylim(-.01,2.15);ax[0].set_xlabel('100 paired records/condition')
for container in ax[0].containers:ax[0].bar_label(container,fmt='%.2f',padding=2,fontsize=8)
names=['TOPK_FULL_K1','TOPK_FULL_K3','TOPK_FULL_K5'];x=[1,3,5]
for metric,lab,col,marker,dy in [('total_score','Clinical score S',C[0],'o',6),('exact_dx_rate','Strict accuracy',C[1],'s',6),('utility','Utility U',C[2],'^',-11)]:
 vals=[av(A[g],metric) for g in names];ax[1].plot(x,vals,color=col,marker=marker,ms=4,label=lab);point_labels(ax[1],x,vals,dy,fmt='.3f' if metric=='utility' else '.2f')
ax[1].set_xticks(x);ax[1].set_ylim(-.065,.50);ax[1].set_title('(B) Retrieval scale',loc='left');ax[1].set_ylabel('Mean / proportion');ax[1].set_xlabel('Retrieved records K');ax[1].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,1.02),frameon=False);ax[1].set_ylim(-.075,.58);style(ax[1]);save(fig,'fig_app_ablation_results')
# Complete 66-point triangular grid, with values directly inspectable in the CSV.
grid=csvr('weight_sensitivity_grid_66.csv');fig,ax=plt.subplots(1,3,figsize=(6.5,3.25),layout='constrained')
for a,m,title in zip(ax,['rank_preserved','heldout_delta_S','heldout_delta_U'],['(A) Model rank','(B) Held-out ΔS','(C) Held-out ΔU']):
 mat=np.full((11,11),np.nan)
 for r in grid:mat[round(float(r['w_diag'])*10),round(float(r['w_evidence'])*10)]=(1 if r[m]=='True' else 0) if m=='rank_preserved' else float(r[m])
 im=a.pcolormesh(np.arange(-.05,1.1001,.1),np.arange(-.05,1.1001,.1),np.ma.masked_invalid(mat),shading='flat',rasterized=False,cmap=ListedColormap(['#CB7062','#377D92']) if m=='rank_preserved' else 'viridis',vmin=0 if m=='rank_preserved' else None,vmax=1 if m=='rank_preserved' else None)
 a.set_title(title,loc='left');a.set_xlabel('Evidence weight');a.set_ylabel('Diagnosis weight');a.set_xticks([0,.5,1]);a.set_yticks([0,.5,1]);a.set_aspect('equal');a.plot(.3,.5,'*',ms=12,color='white',markeredgecolor='black',markeredgewidth=.5)
 cb=fig.colorbar(im,ax=a,orientation='horizontal',shrink=.95,pad=.03,aspect=15)
 cb.solids.set_rasterized(False)
 if m=='rank_preserved':cb.set_ticks([0,1]);cb.set_ticklabels(['Swap','Preserved'])
fig.supxlabel('Plan weight = 1 − diagnosis − evidence; star: default (0.5, 0.3, 0.2).',fontsize=9);save(fig,'weight_sensitivity')
print('Generated requested vector PDFs and PNG previews under figures/generated.')

if not selected:
 import subprocess,sys
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts/reproduce_appendix_visuals.py'),'--results-dir',str(OUT),'--output-dir',str(F)],check=True)
