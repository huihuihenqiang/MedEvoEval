"""Two appendix visualizations of existing release tables; no new experiments."""
from pathlib import Path
import csv, json, os, hashlib
from decimal import Decimal, ROUND_HALF_UP
import argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--results-dir',type=Path,default=ROOT/'results')
p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures')
a=p.parse_args();DATA=a.results_dir.resolve();OUT=a.output_dir.resolve()
if OUT==DATA or DATA in OUT.parents:raise SystemExit('Choose an output directory outside results')
OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT.parent/'cache/mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
def read(n): return list(csv.DictReader((DATA/n).open(encoding='utf8')))
def fmt(v,sign=False):
 d=Decimal(str(v)).quantize(Decimal('.001'),rounding=ROUND_HALF_UP)
 return format(d,'+.3f' if sign else '.3f')
def save(fig,n):
 fig.savefig(OUT/(n+'.pdf'),metadata={'CreationDate':None,'ModDate':None})
 fig.savefig(OUT/(n+'.png'),dpi=200)
 plt.close(fig)
rows=read('r1_direction_contrasts.csv')
comparisons=list(dict.fromkeys(r['comparison'] for r in rows))
dims=['diagnosis_score','evidence_score','plan_score']
fields=['delta_A','delta_B','delta_auto']
fig,axes=plt.subplots(1,3,figsize=(6.8,3.75),layout='constrained',sharey=True)
cmap=LinearSegmentedColormap.from_list('direction_difference',['#F58518','#FFFFFF','#4C78A8'])
for ax,dim,title in zip(axes,dims,['(A) Diagnosis','(B) Evidence','(C) Plan']):
 rr=[next(r for r in rows if r['comparison']==c and r['dimension']==dim) for c in comparisons]
 vals=np.array([[float(r[k]) for k in fields] for r in rr])
 im=ax.pcolormesh(np.arange(4)-.5,np.arange(9)-.5,vals,cmap=cmap,norm=TwoSlopeNorm(0,-.65,.65),edgecolors='white',linewidth=.7,rasterized=False)
 for i,r in enumerate(rr):
  for j,k in enumerate(fields):ax.text(j,i,fmt(r[k],True),ha='center',va='center',fontsize=7.5,color='white' if abs(float(r[k]))>.48 else '#192336')
  if r['all_directions_agree']=='False':ax.add_patch(Rectangle((-.49,i-.49),2.98,.98,fill=False,edgecolor='#A5678E',lw=1.5,clip_on=False))
 ax.set_xticks([0,1,2],['Physician\nA','Physician\nB','Automatic']);ax.tick_params(axis='both',length=0,labelsize=8);ax.set_title(title,loc='left',fontsize=10,pad=10)
 ax.set_yticks(range(8));ax.invert_yaxis()
 for sp in ax.spines.values():sp.set_visible(False)
labels=[c.replace('Qwen3.5-122B-A10B','Qwen3.5').replace(' - ',' − ') for c in comparisons]
axes[0].set_yticklabels(labels,fontsize=8)
cb=fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.06,pad=.055,shrink=.76,aspect=38)
cb.set_label('Difference in group mean score',fontsize=8);cb.ax.tick_params(labelsize=8);cb.solids.set_rasterized(False)
fig.supxlabel('Outlined rows: scorer directions differ; agreement in 21/24 contrasts.',fontsize=8)
save(fig,'fig_app_physician_directions')
rows=[r for r in read('judge_free_summary.csv') if r['condition'] not in ['H0','H3']]
assert len(rows)==8
fig,axes=plt.subplots(1,2,figsize=(6.8,2.85),layout='constrained')
for ax,rr,title in zip(axes,[rows[:4],rows[4:]],['(A) Model outputs (80/model)','(B) Stream outputs (500/condition)']):
 for y,r in enumerate(rr):
  a=float(r['reported_strict']);b=float(r['judge_free_strict'])
  ax.plot([a,b],[y,y],color='#BBC2CA',lw=2,zorder=1)
  ax.scatter(a,y,s=35,c='#4C78A8',marker='o',label='Strict accuracy' if y==0 else None,zorder=3)
  ax.scatter(b,y,s=30,facecolors='none',edgecolors='#F58518',linewidth=1.5,marker='s',label='Diagnosis-match rate' if y==0 else None,zorder=4)
  ax.annotate(fmt(r['reported_strict']),(a,y),xytext=(0,6),textcoords='offset points',ha='center',va='bottom',fontsize=8,color='#335C87')
  ax.annotate(fmt(r['judge_free_strict']),(b,y),xytext=(0,-7),textcoords='offset points',ha='center',va='top',fontsize=8,color='#A3500A')
 ax.set_yticks(range(4),[r['condition'].replace('Qwen3.5-122B-A10B','Qwen3.5') for r in rr]);ax.set_ylim(3.6,-.6);ax.set_xlim(0,.46)
 ax.set_title(title,fontsize=9.5,loc='left');ax.set_xlabel('Rate');ax.set_xticks([0,.1,.2,.3,.4]);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2,frameon=False,fontsize=9)
save(fig,'fig_app_diagnosis_matching')
print('Rendered two new vector appendix visualizations from existing CSV tables.')
