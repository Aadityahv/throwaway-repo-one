"""Score immutable published-method predictions once per cell set. No fitting or variant selection."""
import argparse,csv,json,math,subprocess,sys,warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import predict_static as P
HERE=P.HERE
sys.path.insert(0,str(P.ROOT/'analysis'))
from paired_comparison import paired_delta
from shared_evaluator import ModelResult

def positive(v):return isinstance(v,(float,int)) and math.isfinite(v) and v>0

def failure_quantile(finite,failures,q):
 values=sorted(finite)+[math.inf]*failures
 if not values:return None
 h=(len(values)-1)*q;lo=values[math.floor(h)];hi=values[math.ceil(h)]
 if math.isinf(hi):return 'failure'
 return float(lo+(hi-lo)*(h-math.floor(h)))

def stats(ids,preds,actuals):
 ok=[c for c in ids if positive(preds.get(c)) and positive(actuals.get(c))]
 err=[100*(preds[c]/actuals[c]-1) for c in ok];ape=[abs(e) for e in err];failures=len(ids)-len(ok)
 return dict(declared_cells=len(ids),supported_paired_cells=len(ok),failures=failures,median_ape_pct=float(np.median(ape)) if ape else None,p90_ape_pct=float(np.percentile(ape,90)) if ape else None,signed_median_pct=float(np.median(err)) if err else None,failure_aware_median_pct=failure_quantile(ape,failures,.5),failure_aware_p90_pct=failure_quantile(ape,failures,.9))

def rho(x,y):
 if len(x)<2 or len(set(x))<2 or len(set(y))<2:return None
 return float(spearmanr(x,y).statistic)

def ranking(ids,rows,pred,actual):
 groups=defaultdict(list)
 for cid in ids:groups[(rows[cid]['family'],rows[cid]['regime'])].append(cid)
 detail=[];px=[];py=[]
 for (family,regime),cs in sorted(groups.items()):
  cs=sorted(cs,key=lambda c:rows[c]['candidate'])
  if len(cs)!=2 or any(not positive(pred.get(c)) or not positive(actual.get(c)) for c in cs):
   detail.append(dict(family=family,regime=regime,status='failure'));continue
  chosen=min(cs,key=lambda c:(pred[c],rows[c]['candidate']));best=min(actual[c] for c in cs)
  regrets=100*(actual[chosen]/best-1);spearman=rho([pred[c] for c in cs],[actual[c] for c in cs])
  normp=np.mean([pred[c] for c in cs]);normy=np.mean([actual[c] for c in cs]);px.extend(pred[c]/normp for c in cs);py.extend(actual[c]/normy for c in cs)
  detail.append(dict(family=family,regime=regime,status='ok',spearman=spearman,prediction_tie=pred[cs[0]]==pred[cs[1]],chosen_candidate=rows[chosen]['candidate'],top1_regret_pct=regrets,top1_optimal=actual[chosen]==best))
 valid=[r for r in detail if r['status']=='ok'];defined=[r['spearman'] for r in valid if r['spearman'] is not None]
 return dict(declared_pairs=len(detail),scored_pairs=len(valid),failed_pairs=len(detail)-len(valid),undefined_spearman_pairs=len(valid)-len(defined),mean_pair_spearman=float(np.mean(defined)) if defined else None,pair_normalized_spearman=rho(px,py),top1_optimal_fraction=sum(r['top1_optimal'] for r in valid)/len(valid) if valid else None,mean_top1_regret_pct=float(np.mean([r['top1_regret_pct'] for r in valid])) if valid else None,median_top1_regret_pct=float(np.median([r['top1_regret_pct'] for r in valid])) if valid else None,per_pair=detail)

def load_labels(path,ids,prefix):
 E={};W={};T={};provenance={};unexpected=[]
 with Path(path).open(newline='') as stream:
  for r in csv.DictReader(stream):
   if prefix and not r['run_id'].startswith(prefix):continue
   cid='blackwell/%s/%s/%s'%(r['parent_id'],r['regime'],r['candidate_id'])
   if cid not in ids:unexpected.append(cid);continue
   if cid in E:raise ValueError('duplicate measured cell '+cid)
   energy=float(r['board_energy_j_per_launch']);t=float(r['counted_launch_interval_s'])/int(r['launch_count'])
   if not positive(energy) or not positive(t):raise ValueError('invalid label '+cid)
   E[cid]=energy;W[cid]=energy/t;T[cid]=t;provenance[cid]={k:r[k] for k in ('run_id','gpu_uuid','correctness_check','trace_dir','launch_count')}
 if unexpected:raise ValueError('unexpected cells in requested run prefix: '+str(unexpected))
 return E,W,T,provenance

def score(scope,energy_csv,prefix):
 doc=P.read(HERE/'predictions_runtime.json');rows={c:r for c,r in doc['rows'].items() if r['scope']==scope};ids=sorted(rows)
 E,W,T,prov=load_labels(energy_csv,ids,prefix);below=[c for c in ids if c not in W or W[c]<.95*600]
 report=dict(schema='published_fresh_energy_score/1',scope=scope,label_policy='accepted application_energy_raw rows; window energy per launch, no idle subtraction; rejected or unmeasured cells remain failures',declared_cells=len(ids),measured_cells=len(E),missing_cells=sorted(set(ids)-set(E)),below_cap_cells=sum(c in W for c in below),cap_power_w=600,cap_fraction=.95,methods=doc['methods'],blocks={},paired={},per_cell={},label_provenance=prov)
 for label,cs in [('all',ids),('below_cap_or_missing_label',below)]:
  block={}
  for method in doc['methods']:
   pred={c:rows[c]['energy_j'].get(method) for c in ids};power={c:rows[c]['native_power_w'].get(method) for c in ids}
   block[method]=dict(energy=stats(cs,pred,E),ranking=ranking(cs,rows,pred,E),per_family={},per_tier={},per_family_tier={})
   if any(positive(power[c]) for c in ids):block[method]['native_power']=stats(cs,power,W)
   for group,col in [('per_family','family'),('per_tier','tier')]:
    for v in sorted({rows[c][col] for c in cs}):
     subset=[c for c in cs if rows[c][col]==v];block[method][group][v]=dict(energy=stats(subset,pred,E),ranking=ranking(subset,rows,pred,E))
   for f,t in sorted({(rows[c]['family'],rows[c]['tier']) for c in cs}):
    subset=[c for c in cs if rows[c]['family']==f and rows[c]['tier']==t];block[method]['per_family_tier'][f+' / '+t]=stats(subset,pred,E)
  report['blocks'][label]=block
  paired={}
  for method in doc['methods']:
   if method.startswith('ours_'):continue
   keys=[c for c in cs if c in E and positive(rows[c]['energy_j'].get(method)) and positive(rows[c]['energy_j'].get('ours_static'))]
   if len({rows[c]['family'] for c in keys})<2:paired[method]=dict(status='insufficient independent families');continue
   make=lambda name:ModelResult(name=name,predictions={(('cell_id',c),):rows[c]['energy_j'][name] for c in keys},actuals={(('cell_id',c),):E[c] for c in keys},metadata={(('cell_id',c),):{'family':rows[c]['family']} for c in keys})
   result=paired_delta(make(method),make('ours_static'),metric='median_ape',samples=2000,seed=0)
   result['per_group']={' / '.join(g):v for g,v in result['per_group'].items()};result['units']='fraction; candidate (our model) minus published baseline; multiply by 100 for percentage points';paired[method]=result
  report['paired'][label]=paired
 for c in ids:report['per_cell'][c]=dict(family=rows[c]['family'],tier=rows[c]['tier'],regime=rows[c]['regime'],candidate=rows[c]['candidate'],measured_energy_j=E.get(c),measured_window_power_w=W.get(c),measured_window_runtime_s=T.get(c),measured_short_runtime_s=rows[c]['measured_runtime_s'],predicted_runtime_s=rows[c]['predicted_runtime_s'],predictions_energy_j=rows[c]['energy_j'],native_power_w=rows[c]['native_power_w'])
 report['inputs_sha256']={str(Path(energy_csv).resolve().relative_to(P.ROOT)):P.sha(energy_csv),str((HERE/'predictions_runtime.json').relative_to(P.ROOT)):P.sha(HERE/'predictions_runtime.json'),str(Path(__file__).relative_to(P.ROOT)):P.sha(__file__)}
 return report

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--scope',choices=['prospective','retrospective'],required=True);ap.add_argument('--energy-csv',type=Path,required=True);ap.add_argument('--run-prefix',required=True);a=ap.parse_args();out=HERE/('score_'+a.scope+'.json')
 if out.exists():raise SystemExit('REFUSED: this cell set has already been scored')
 # Exact remote freeze is checked BEFORE any energy bytes are read.
 freeze=P.read(HERE/'PREDICTION_FREEZE.json')
 for path,digest in freeze['files_sha256'].items():
  if P.sha(P.ROOT/path)!=digest:raise SystemExit('REFUSED: frozen input changed: '+path)
 for path,digest in freeze['files_sha256'].items():
  remote=subprocess.check_output(['git','show','origin/cross-gpu-prep:'+path],cwd=P.ROOT)
  if P.hashlib.sha256(remote).hexdigest()!=digest:raise SystemExit('REFUSED: freeze is not on origin: '+path)
 rep=score(a.scope,a.energy_csv,a.run_prefix);out.write_text(json.dumps(rep,indent=2,sort_keys=True,allow_nan=False)+'\n');print('Scored once:',a.scope,rep['measured_cells'],'/',rep['declared_cells'])
if __name__=='__main__':main()
