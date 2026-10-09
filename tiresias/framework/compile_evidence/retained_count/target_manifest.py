"""Inventory exact target evidence gaps without reading energy labels."""
import json
from pathlib import Path
from collections import Counter
from derive import HERE, BASE, sha, require

def build():
    root=BASE.parents[2]
    inventory_path=BASE/'counts/outputs/operator_count_inventory.json'
    archived_path=BASE/'counts/outputs/archived_compile_inventory.json'
    inventory=json.loads(inventory_path.read_text())
    archival=json.loads(archived_path.read_text())
    require(inventory['total_cells']==336 and len(inventory['rows'])==336,'target denominator drift')
    compiled={(r['gpu'],r['operator_id'],r['cell']):r for r in archival}
    by_hash={}
    for path in root.rglob('*.cubin'): by_hash.setdefault(sha(path),[]).append(str(path.relative_to(root)))
    rows=[]
    for row in inventory['rows']:
        archived=compiled.get((row['gpu'],row['operator_id'],row['regime']+'/'+row['candidate_id']))
        wanted=archived['cubin_sha256'] if archived else None
        local=by_hash.get(wanted,[]) if wanted else []
        rows.append({'cell_id':row['cell_id'],'gpu':row['gpu'],'operator_id':row['operator_id'],'operator_group':row['operator_group'],'operator_description':row['operator_description'],'family':row['family'],'shape':row['shape'],'launch':row['launch'],'architecture':row['architecture'],'analytical_tier':row['analytical_tier'],'archived_cubin_sha256':wanted,'archived_source_sha256':archived['source_sha256'] if archived else None,'exact_local_cubin_paths':local,'executed_count_status':'unknown','derived_counts':None,'reference_counts':None,'required_evidence':['exact cubin and full exact-symbol SASS/PTX','retained source/header/compiler closure and ordered host dispatch/launch arguments','source/compiled path proof for every actually timed kernel and all control/predicates','separate compatible per-opcode predicate-true thread profiler references across full declared grid'],'blockers':row['unsupported_reasons'],'priority':'first reduction transfer proof' if 'reduction' in row['family'] or 'reduction' in row['operator_id'] else 'complete target coverage'})
    require(len({r['cell_id'] for r in rows})==336,'duplicate target cell')
    return {'schema':'compiled_target_evidence_request_v1','origin_inventory_sha256':sha(inventory_path),'origin_archival_sha256':sha(archived_path),'scope':'Exposed development grid; no sealed labels selected or opened. CPU local-file inventory only; remote availability not inferred.','declared_cells':336,'full_target_admitted_cells':0,'exact_local_binary_cells':sum(bool(r['exact_local_cubin_paths']) for r in rows),'archived_hash_cells':sum(r['archived_cubin_sha256'] is not None for r in rows),'cells_per_gpu':dict(Counter(r['gpu'] for r in rows)),'proposed_first_adapter':'CUDA Samples reduction at all declared shapes/candidates, all timed dispatch kernels. Primitive specialization does not cover operator dispatch.','fresh_operator_grid_gap':'This exposed inventory does not contain the separately frozen fresh operators; their source/shape/dispatch bindings must also be acquired before prospective model validation.','acquisition_authorized_by_manifest':False,'rows':rows}

if __name__=='__main__':
    report=build(); (HERE/'target_evidence_manifest.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ('declared_cells','archived_hash_cells','exact_local_binary_cells','full_target_admitted_cells')}))
