"""Regression test: the packaged application-fit tool reproduces the Blackwell results of 2 Oct 2026 (exact per-launch counts, leave-one-operator-out)."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / 'tools')); sys.path.insert(0, str(HERE))
import fit_application_model as F  # noqa: E402

SR = HERE.parent
LEGACY = SR / 'energy_revision' / 'execution' / 'runs' / 'native_5a5e881d'
LABELS = HERE / 'data' / 'application_cells_blackwell.csv'
FEATS = [SR / 'fresh_e/features_fresh_e.json', SR / 'unseen_kernels/frozen/features_unseen.json', SR / 'features_blackwell.json']


@pytest.mark.skipif(not (LEGACY.exists() and LABELS.exists() and all(f.exists() for f in FEATS)), reason='committed Blackwell data not present')
def test_reproduces_the_blackwell_results():
    app = F.load_cells(LABELS, FEATS); cal = F.windows_from_legacy(LEGACY / 'compiled_design.json', LEGACY / 'energy/result.json')
    ev = F.evaluate(app, cal, measured_base_w=152.529134822)
    s, m = ev['static']['target'], ev['measured']['target']
    assert (s['below']['median'], s['below']['p90']) == pytest.approx((9.9, 36.5), abs=0.06)
    assert (s['all']['median'], s['all']['p90']) == pytest.approx((11.2, 35.6), abs=0.06)
    assert (m['below']['median'], m['below']['p90']) == pytest.approx((9.7, 30.9), abs=0.06)
    assert (m['all']['median'], m['all']['p90']) == pytest.approx((9.8, 24.2), abs=0.06)
    assert s['below']['n'] == 35 and s['all']['n'] == 48


def test_final_profile_has_every_rate_and_ties_shuffle_and_barrier():
    app = [dict(id=str(i), op='o%d' % (i % 3), src='dev', E=1e-3 * (1 + i % 5), t=1e-5 * (1 + i % 5), tp=1e-5 * (1 + i % 5), P=300.0, B_l2=1e8 * (1 + i % 4), B_dr=1e7 * (i % 3), ffma=1e8 * (i % 5), shm=1e7 * (i % 4), shb=1e6 * (i % 2), sfu=1e6 * (i % 3)) for i in range(40)]
    prof = F.final_profile(app, [])
    assert set(prof['rates_pJ']) == set(F.En.COLUMNS) and prof['rates_pJ']['shfl'] == prof['rates_pJ']['bar'] and prof['base_power_w'] >= 0
