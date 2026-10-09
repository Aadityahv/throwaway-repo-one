#!/usr/bin/env python3
"""Predict runtime and energy of kernels from a calibration document. No kernel is executed.

    python3 calibrate/predict.py --calibration calibrate/runs/<run>/calibration_<arch>_<uuid8>.json --features features.json \\
        [--phases phases.json --unique unique.json --bank bank.json | --runtime-json runtime.json] --out predictions.json

`features.json` is the output of the static analysis of the compiled kernels (extract_features rows: per-launch instruction-family counts, logical bytes, memory tier).
Runtime comes from the calibrated runtime model when --phases/--unique/--bank are given (the legacy constants written next to the calibration document are used), or from
--runtime-json ({cell_id: {"primary_s": seconds}} or {cell_id: seconds}), e.g. a measured time for the "measured runtime" variant.
Energy = min(cap * t, base * t + sum(rate * count)) with the rates fitted by the calibration's energy stage on THIS board. The command refuses an incomplete calibration unless
--allow-incomplete is given explicitly (the output is then marked), and a rate that the calibration could not identify is never replaced by another board's number.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cal import energy as En  # noqa: E402


class PredictRefusal(RuntimeError):
    pass


def load_calibration(path, allow_incomplete=False):
    doc = json.loads(Path(path).read_text())
    if 'energy' not in doc.get('constants', {}): raise PredictRefusal('the calibration has no energy constants (run the packaged calibrator with the energy stage)')
    if not doc.get('complete') and not allow_incomplete:
        raise PredictRefusal('the calibration is incomplete (%s); fix it or pass --allow-incomplete explicitly' % '; '.join(doc.get('warnings') or ['see its stages']))
    bad = [c for c, s in doc['constants']['energy']['profile']['status'].items() if s.startswith('UNIDENTIFIED')]
    if bad and not allow_incomplete: raise PredictRefusal('energy rates not identified by this calibration: ' + ', '.join(bad))
    return doc


def supported(row):
    return row.get('status') in ('supported', 'supported_with_assumptions')


def runtime_from_model(doc, calib_path, features, phases, unique, bank):
    import cal.portable_predict as PP
    legacy = Path(calib_path).resolve().parent / 'legacy_constants'
    if not legacy.is_dir(): raise PredictRefusal('legacy_constants/ not found next to the calibration document: %s' % legacy)
    out = PP.predict(features, phases, unique, bank, PP.load_constants(legacy), doc['device']['sm_count'])
    return {cid: (v.get('primary_s') if isinstance(v, dict) else v) for cid, v in out.items()}


def energy_rows(doc, features, runtime):
    e = doc['constants']['energy']; profile, cap = e['profile'], e['cap_w']; rows = {}
    for r in features['rows']:
        cid = r['cell_id']
        if not supported(r): rows[cid] = dict(status='unsupported', reason='static analysis status %s' % r.get('status')); continue
        t = runtime.get(cid)
        if t is None or not t > 0: rows[cid] = dict(status='no_runtime', reason='no positive runtime for this cell'); continue
        tot = r.get('per_launch_totals') or r['work']; mem = r['memory']
        cols = En.columns_from_feature_row(tot, mem['logical_bytes_per_launch'], mem['tier'], tot.get('executed_global_store_bytes_lane_level', mem.get('executed_global_store_bytes_lane_level', 0)))
        terms = {c: profile['rates_pJ'].get(c, 0.0) * cols[c] * 1e-12 for c in En.COLUMNS}
        if cols.get('tc', 0) > 0:   # tensor-core instructions: priced only by a calibrated tensor rate (optional tensor stage); otherwise the cell is unsupported, never charged at another class's rate
            te = (doc['constants'].get('tensor') or {}).get('energy') or {}
            if te.get('status') != 'ok': rows[cid] = dict(status='unsupported', reason='the kernel has tensor-core instructions and the calibration has no usable tensor energy rate (%s)' % (te.get('status') or 'tensor stage not run')); continue
            terms['tc'] = te['rate_pJ_per_lane_instruction'] * cols['tc'] * 1e-12
        uncapped = profile['base_power_w'] * t + sum(terms.values()); en = min(cap * t, uncapped)
        rows[cid] = dict(status='ok', runtime_s=t, energy_j=en, mean_power_w=en / t, capped=uncapped > cap * t, base_term_j=profile['base_power_w'] * t, term_j=terms)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--calibration', type=Path, required=True); ap.add_argument('--features', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--phases', type=Path); ap.add_argument('--unique', type=Path); ap.add_argument('--bank', type=Path); ap.add_argument('--runtime-json', type=Path)
    ap.add_argument('--allow-incomplete', action='store_true')
    a = ap.parse_args(argv)
    try:
        doc = load_calibration(a.calibration, a.allow_incomplete); features = json.loads(a.features.read_text())
        if a.runtime_json:
            raw = json.loads(a.runtime_json.read_text()); runtime = {k: (v.get('primary_s') if isinstance(v, dict) else v) for k, v in raw.items()}; source = 'runtime-json'
        elif a.phases and a.unique and a.bank:
            ph = json.loads(a.phases.read_text()); ph = ph.get('rows', ph); uq = json.loads(a.unique.read_text()); uq = uq.get('rows', uq); bk = json.loads(a.bank.read_text())['rows']
            runtime = runtime_from_model(doc, a.calibration, features, ph, uq, bk); source = 'calibrated runtime model'
        else:
            raise PredictRefusal('give either --runtime-json or all of --phases, --unique and --bank')
        rows = energy_rows(doc, features, runtime)
    except PredictRefusal as ex:
        print('REFUSED: %s' % ex, file=sys.stderr); return 1
    out = dict(calibration=str(a.calibration), calibration_complete=bool(doc.get('complete')), runtime_source=source, device=doc['device']['name'], cells=rows,
               note='No kernel was executed. Rates are this board\'s own; an incomplete calibration is marked, never completed from another board.')
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    ok = sum(1 for v in rows.values() if v['status'] == 'ok'); print(json.dumps(dict(cells=len(rows), predicted=ok, runtime_source=source)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
