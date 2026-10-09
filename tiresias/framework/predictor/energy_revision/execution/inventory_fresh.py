"""Pin candidate source lineages and audit metadata/history; NEVER opens measurement files.

Source preparation is not freshness admission or a numeric prediction freeze.
"""
import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
REV='5443602d89ed99aede2e4b7bf329daddeadb320e'
CANDIDATES={
    'dct8x8':'discrete cosine transform', 'dwtHaar1D':'Haar wavelet transform',
    'recursiveGaussian':'recursive Gaussian filter', 'SobelFilter':'Sobel edge filter',
    'nbody':'N-body interaction', 'quasirandomGenerator':'quasi-random generation',
    'binomialOptions':'binomial option pricing', 'histogram':'byte histogram',
}
METADATA=(
    'tiresias/app_runners/workload_catalog.csv',
    'tiresias/app_runners/source_equivalences.json',
    'tiresias/framework/predictor/fresh_e/fresh_cells_e.json',
    'tiresias/framework/predictor/unseen_kernels/frozen/cells_unseen.json',
    'tiresias/framework/compile_evidence/fresh_extra/HANDOFF.md',
    'tiresias/framework/compile_evidence/fresh_wrappers/HANDOFF.md',
    'tiresias/planning/CURRENT_PLAN.md',
    'the booking log','CHANGELOG.md',
)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'GPU-energy-research-source-inventory'})
    with urllib.request.urlopen(req,timeout=30) as r:return r.read()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out-dir',type=Path,default=HERE/'fresh_source_inventory');a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=False)
    tree_url=f'https://api.github.com/repos/NVIDIA/cuda-samples/git/trees/{REV}?recursive=1'
    raw=fetch(tree_url);tree=json.loads(raw)
    if tree.get('truncated'):raise ValueError('REFUSED: incomplete source tree')
    (a.out_dir/'upstream_tree.json').write_bytes(raw)
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    lineage=[]
    for name,title in CANDIDATES.items():
        files=[x for x in tree['tree'] if f'/{name}/' in x['path'] and x['type']=='blob' and x['path'].endswith(('.cu','.cuh','.h'))]
        if not files:raise ValueError('REFUSED: pinned source lineage absent: '+name)
        sources=[]
        for f in files:
            url=f'https://raw.githubusercontent.com/NVIDIA/cuda-samples/{REV}/{f["path"]}'
            contents=fetch(url)
            blob=hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()
            if blob!=f['sha']:raise ValueError('REFUSED: upstream Git blob mismatch')
            dest=a.out_dir/'sources'/f['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(contents)
            sources.append(dict(path=f['path'],raw_url=url,git_blob_sha1=blob,sha256=sha(dest),bytes=len(contents)))
        path_hits=[s for s in tracked if name.casefold() in s.casefold()]
        # Search only declared history/static metadata. Never recursively search
        # raw outputs, scores, energy traces, sealed labels or timing records.
        q=subprocess.run(['rg','-n','-i',re.escape(name),*METADATA],cwd=ROOT,capture_output=True,text=True)
        if q.returncode not in (0,1):raise ValueError('history audit failed: '+q.stderr)
        hits=q.stdout.splitlines()
        lineage.append(dict(name=title,upstream_sample=name,source_files=sources,existing_tracked_path_hits=path_hits,
            declared_metadata_history_hits=hits,status='candidate_only_history_review_and_static_support_pending',
            qualification='No absence proof for undocumented/remote measurements; not admitted as genuinely fresh. Existing proposal references are retained, never silently treated as new.'))
        print(title,len(sources),'source files',len(path_hits),'path hits',len(hits),'metadata/history hits',flush=True)
    out=dict(schema='fresh_energy_source_candidate_inventory/1',status='SOURCE_PINNED_NOT_ADMITTED_OR_MEASURED',
        upstream_revision=REV,upstream_tree_url=tree_url,upstream_tree_sha256=sha(a.out_dir/'upstream_tree.json'),
        candidate_lineages=lineage,metadata_sha256={s:sha(ROOT/s) for s in METADATA},
        audit='Only tracked paths and explicit static/history metadata searched; no measured energy/runtime, score or sealed-label file opened.',
        exposed_families_excluded=['scalar product','fast Walsh transform','matrix multiply','Black-Scholes','prefix-sum scan','separable convolution'],
        prospective_target='At least eight admitted source lineages, four size regimes and two equivalent-work implementations each; full relevant grid, one frozen runtime vector and all numeric predictions pushed before target access.',
        blockers=['Review metadata hits and remote acquisition ownership','Derive full equivalent-work geometry/input/oracle grid','Compile and audit static runtime/activity support','Freeze candidate/calibration/runtime profiles and all numeric predictions','Separate shared-machine target measurement approval/booking'])
    (a.out_dir/'inventory.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':main()
