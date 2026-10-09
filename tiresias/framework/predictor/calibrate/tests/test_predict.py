"""CPU-only tests of calibrate/predict.py on the committed Blackwell fresh-set features."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
import predict as P  # noqa: E402
from cal import energy as En  # noqa: E402

SR = HERE.parent
FEATURES = SR / 'fresh_e' / 'features_fresh_e.json'
RUNTIME = SR / 'fresh_e' / 'predictions_fresh_e_v3i.json'


def make_doc(path, complete=True, unidentified=None):
    status = {c: 'ok' for c in En.COLUMNS}
    if unidentified: status[unidentified] = 'UNIDENTIFIED (fewer than 2 admitted windows)'
    prof = dict(base_power_w=150.0, rates_pJ=dict(B_l2=60., B_dr=170., B_wr=0., ffma=9., fpo=8., intc=3., sfu=12., shm=38., shfl=70., bar=50.), status=status)
    doc = dict(complete=complete, warnings=[] if complete else ['energy: incomplete'], device=dict(name='TestGPU', sm_count=188), constants=dict(energy=dict(cap_w=600.0, profile=prof)))
    Path(path).write_text(json.dumps(doc)); return doc


@pytest.mark.skipif(not FEATURES.exists(), reason='committed features not present')
def test_energy_matches_the_formula_and_marks_capped_cells(tmp_path):
    cal = tmp_path / 'cal.json'; doc = make_doc(cal); out = tmp_path / 'out.json'
    assert P.main(['--calibration', str(cal), '--features', str(FEATURES), '--runtime-json', str(RUNTIME), '--out', str(out)]) == 0
    res = json.loads(out.read_text())['cells']; feats = {r['cell_id']: r for r in json.loads(FEATURES.read_text())['rows']}; rt = json.loads(RUNTIME.read_text())
    ok = [c for c, v in res.items() if v['status'] == 'ok']; assert len(ok) >= 10
    c = ok[0]; r = feats[c]; t = rt[c]['primary_s']; tot = r['per_launch_totals']
    cols = En.columns_from_feature_row(tot, r['memory']['logical_bytes_per_launch'], r['memory']['tier'], tot['executed_global_store_bytes_lane_level'])
    expect = min(600 * t, doc['constants']['energy']['profile']['base_power_w'] * t + sum(doc['constants']['energy']['profile']['rates_pJ'][k] * cols[k] * 1e-12 for k in En.COLUMNS))
    assert res[c]['energy_j'] == pytest.approx(expect) and res[c]['mean_power_w'] <= 600 + 1e-9


def test_refuses_incomplete_and_unidentified_calibrations(tmp_path):
    cal = tmp_path / 'cal.json'; make_doc(cal, complete=False); rt = tmp_path / 'rt.json'; rt.write_text('{}')
    assert P.main(['--calibration', str(cal), '--features', str(tmp_path / 'f.json'), '--runtime-json', str(rt), '--out', str(tmp_path / 'o.json')]) == 1
    make_doc(cal, unidentified='shfl')
    assert P.main(['--calibration', str(cal), '--features', str(tmp_path / 'f.json'), '--runtime-json', str(rt), '--out', str(tmp_path / 'o.json')]) == 1


def test_refuses_without_a_runtime_source(tmp_path):
    cal = tmp_path / 'cal.json'; make_doc(cal); (tmp_path / 'f.json').write_text('{"rows": []}')
    assert P.main(['--calibration', str(cal), '--features', str(tmp_path / 'f.json'), '--out', str(tmp_path / 'o.json')]) == 1
