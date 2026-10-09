"""Report only completed full-inventory CPU suites; preserve all pending costs."""
import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True, help='Collected private results directory')
parser.add_argument('--out', type=Path, default=HERE.parent)
args = parser.parse_args()
sources = {}
def read(path):
    raw = path.read_bytes(); sources[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
cached = read(HERE / 'evidence/cached/one_worker.json')
oracle = {r['cell_id']: r['prediction'] for r in cached['rows']}
assert len(oracle) == 167
report = {'scope': 'Actual complete-inventory CPU benchmarks of frozen Blackwell predictions. Compilation/disassembly is a separate shared-unit cost. GPU cost scenarios are explicitly estimates based on the recorded calibration scope and median label cadence, not measured end-to-end campaigns.',
          'configurations': 167, 'cold': {}, 'pending_cold_workers': [], 'cached': {}, 'compilation': {},
          'calibration_scope': 'Blackwell reported 52-minute calibration and seven-minute tensor booking; excludes pair-program setup/compilation/staging. These records are not a complete invocation timer.',
          'measurement_reference': '151 seconds is the median archived operator-window cadence (121 within-stream gaps), not the measured total for this exact 167-configuration campaign.',
          'shared_host_scope': 'One/four/eight-worker runs overlap; four/eight start during the initial 55-worker tail. Disjoint one/four/eight CPU masks, with initial mask overlap. Recorded host load is part of the actual throughput. No isolated scaling guarantee.',
          'memory_scope': 'RSS is the sampled sum across the coordinator/process tree and may double-count shared pages; no unique-memory or host-energy claim.'}
for workers in [1, 4, 8, 55]:
    path = args.results / f'cold_workers_{workers}/report.json'
    if not path.exists(): report['pending_cold_workers'].append(workers); continue
    raw = read(path)
    assert raw['configurations'] == 167 and raw['predictions_match_frozen']
    assert len(raw['rows']) == 167 and {r['cell_id'] for r in raw['rows']} == set(oracle)
    for row in raw['rows']:
        for key in ['runtime_s', 'energy_j']:
            a, b = row['prediction'][key], oracle[row['cell_id']][key]
            assert (a is None and b is None) or (a is not None and b is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12))
    report['cold'][workers] = {k: raw[k] for k in ['wall_s', 'throughput_cells_per_min', 'worker_cpu_s', 'child_cpu_s',
        'peak_aggregate_rss_bytes', 'max_cell_rss_kib', 'cell_wall_s', 'predictions_match_frozen']}
    report['cold'][workers]['slowest_configurations'] = sorted(raw['rows'], key=lambda r: r['wall_s'], reverse=True)[:10]
for workers, filename in [(1, 'one_worker'), (4, 'four_workers'), (8, 'eight_workers'), (56, 'fifty_six_workers')]:
    raw = read(HERE / f'evidence/cached/{filename}.json')
    assert raw['configurations'] == 167 and raw['predictions_match_frozen']
    report['cached'][workers] = {k: raw[k] for k in ['wall_s', 'throughput_cells_per_min', 'worker_cpu_s', 'peak_aggregate_rss_bytes']}
path = HERE / 'evidence/compile_results.tar.gz'
sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
with tarfile.open(path) as archive:
    for workers in [1, 4, 8]:
        raw = json.load(archive.extractfile(f'compile_results/workers_{workers}/report.json'))
        assert raw['units'] == 13 and raw['all_encoded_sass_exact']
        report['compilation'][workers] = {k: raw[k] for k in ['units', 'wall_s', 'cpu_s', 'all_encoded_sass_exact', 'all_cubins_exact']}
        if workers in report['cold']:
            wall = raw['wall_s'] + report['cold'][workers]['wall_s']
            report['cold'][workers]['compile_plus_analysis_wall_s'] = wall
            report['cold'][workers]['recorded_calibration_plus_cpu_scenario_s'] = 3540 + wall
            report['cold'][workers]['median_label_cadence_times_inventory_scenario_s'] = 151 * 167
            report['cold'][workers]['scenario_predicted_pipeline_less_elapsed'] = 3540 + wall < 151 * 167
if (args.results / 'concurrent_campaign_state.json').exists():
    report['controller'] = read(args.results / 'concurrent_campaign_state.json')
report['complete_required_worker_comparisons'] = all(w in report['cold'] for w in [1, 4, 8])
report['sources_sha256'] = sources
args.out.mkdir(parents=True, exist_ok=True)
(args.out / 'cpu_cost_results.json').write_text(json.dumps(report, indent=2) + '\n')
md = ['# Measured CPU throughput and cost', '', report['scope'], '', report['shared_host_scope'], '',
      'Compilation/disassembly (13 unique units; encoded instructions all match retained code, cubins not all byte-identical):']
for workers, row in report['compilation'].items():
    md.append(f"- {workers} workers: {row['wall_s']:.2f} seconds elapsed; {row['cpu_s']:.2f} process CPU-seconds.")
md += ['', 'Prediction from retained derived statistics (all 167 configurations; includes process startup and fresh prediction computation):']
for workers, row in report['cached'].items():
    md.append(f"- {workers} workers: {row['wall_s']:.3f} seconds elapsed.")
md += ['', 'Full analysis from retained compiled code, empty derived-output cache, fresh interpreter per configuration:']
for workers, row in report['cold'].items():
    wall = row['cell_wall_s']
    md.append(f"- {workers} workers: {row['wall_s']/60:.2f} minutes elapsed; {row['worker_cpu_s']/3600:.2f} worker CPU-hours; {row['throughput_cells_per_min']:.2f} configurations/minute. Cell median/p90/maximum: {wall['median']:.1f}/{wall['p90']:.1f}/{wall['max']:.1f} seconds. Peak aggregate RSS: {row['peak_aggregate_rss_bytes']/2**30:.2f} GiB. All 167 frozen predictions/refusals match.")
if report['pending_cold_workers']:
    md += ['', 'Pending complete cold suites: ' + ', '.join(map(str, report['pending_cold_workers'])) + ' workers. No timing is inferred from partial progress or substituted from cached prediction.']
md += ['', report['memory_scope'], '', report['calibration_scope'], '', report['measurement_reference'], '',
       'Do not assert complete pair-inclusive calibration cost, a measured exact-inventory energy-label total, isolated scaling, CPU energy saved or serial median-cost break-even from these records.', '',
       'The manuscript and its existing cost table remain unchanged pending exact-edit approval. Detailed outputs and source hashes: cpu_cost_results.json.']
(args.out / 'CPU_COST_RESULTS.md').write_text('\n'.join(md) + '\n')
print('Completed cold suites:', sorted(report['cold']), '; pending:', report['pending_cold_workers'])
