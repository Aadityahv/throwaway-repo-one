"""Candidate (v3j) traffic-based energy columns and the portable wrapper of the candidate runtime model. New file; nothing existing is edited (cal/energy.py, cal/portable_predict.py
and predict.py stay byte-identical, so every frozen hash still verifies).

Why the columns change. The calibrator fits two byte rates from synthetic windows: `B_l2` (bytes served by the L2) and `B_dr` (bytes moved to or from DRAM). In every DRAM window
(`rd_dram_*`, `wr_dram_*`, `copy_dram_*`) the bytes appear ONLY in `B_dr`, so the fitted DRAM rate already contains the L2 pass of those bytes; in every L2 window the bytes are all the loads
and stores the L2 served. The application columns of `energy.columns_from_feature_row` charge the cell's compulsory (logical) bytes once, at the rate of the cell's tier. That omits the L2
traffic of tile re-reads (a block's loads that hit sectors another block already fetched). The candidate columns charge the bytes the runtime model itself says each level serves:

    B_dr = DRAM bytes of the runtime model (grid-wide unique footprint reads + writes, DRAM tier only)
    B_l2 = L2-served bytes of the runtime model (first touches + L2-served re-reads + writes) - B_dr      (the DRAM rate already includes one L2 pass of the DRAM bytes)
    B_wr = executed global store bytes, unchanged

No rate is refitted: the same calibration document is used. Counting only; applies identically to every kernel.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cal import energy as En  # noqa: E402
from cal import portable_predict as PP  # noqa: E402


def columns_from_traffic(work, l2_served_bytes, dram_bytes, store_bytes):
    """Fit columns of an application kernel (instruction columns as in energy.columns_from_feature_row; byte columns from the runtime model's traffic)."""
    c = En.columns_from_feature_row(work, 0.0, 'L2', store_bytes)
    c['B_dr'] = float(dram_bytes); c['B_l2'] = max(0.0, float(l2_served_bytes) - float(dram_bytes)); c['B_wr'] = float(store_bytes or 0)
    return c


def energy_rows_traffic(doc, features, runtime, traffic, columns=columns_from_traffic, dram_bytes_override=None):
    """Same as predict.energy_rows (same capping, tensor rule and statuses) with the traffic columns. traffic: {cell_id: dict(l2_read_bytes, l2_write_bytes, dram_read_bytes, dram_write_bytes)};
    a cell without traffic is reported unsupported with the reason (never charged at logical bytes)."""
    import predict as P
    e = doc['constants']['energy']; profile, cap = e['profile'], e['cap_w']; rows = {}
    for r in features['rows']:
        cid = r['cell_id']
        if not P.supported(r): rows[cid] = dict(status='unsupported', reason='static analysis status %s' % r.get('status')); continue
        t = runtime.get(cid)
        if t is None or not t > 0: rows[cid] = dict(status='no_runtime', reason='no positive runtime for this cell'); continue
        tr = traffic.get(cid)
        if not tr: rows[cid] = dict(status='unsupported', reason='the runtime model gave no traffic for this cell (refused or unsupported)'); continue
        tot = r.get('per_launch_totals') or r['work']; mem = r['memory']
        store = tot.get('executed_global_store_bytes_lane_level', mem.get('executed_global_store_bytes_lane_level', 0))
        dram = tr['dram_read_bytes'] + tr['dram_write_bytes'] if dram_bytes_override is None else dram_bytes_override[cid]   # override: ablation only
        cols = columns(tot, tr['l2_read_bytes'] + tr['l2_write_bytes'], dram, store)
        terms = {c: profile['rates_pJ'].get(c, 0.0) * cols[c] * 1e-12 for c in En.COLUMNS}
        if cols.get('tc', 0) > 0:
            te = (doc['constants'].get('tensor') or {}).get('energy') or {}
            if te.get('status') != 'ok': rows[cid] = dict(status='unsupported', reason='tensor-core instructions and no usable tensor energy rate'); continue
            terms['tc'] = te['rate_pJ_per_lane_instruction'] * cols['tc'] * 1e-12
        uncapped = profile['base_power_w'] * t + sum(terms.values()); en = min(cap * t, uncapped)
        rows[cid] = dict(status='ok', runtime_s=t, energy_j=en, mean_power_w=en / t, capped=uncapped > cap * t, base_term_j=profile['base_power_w'] * t, term_j=terms, columns=cols)
    return rows


class _Shim:
    """Stands in for predict_runtime_v3i inside portable_predict.predict: same call, the candidate model behind it."""
    def __init__(self, J, read_footprint, write_footprint=False, l2_rule='wave'): self.J, self.read_footprint, self.write_footprint, self.l2_rule = J, read_footprint, write_footprint, l2_rule
    def build(self, features, phase_rows, unique_rows, bank_rows, K, v3c, V3C, V3E, OV):
        return self.J.build(features, phase_rows, unique_rows, bank_rows, K, v3c, V3C, V3E, OV, read_footprint=self.read_footprint, write_footprint=self.write_footprint, l2_rule=self.l2_rule)


def predict_candidate(features, phase_rows, unique_rows, bank_rows, constants, sm_count, read_footprint=True, write_footprint=False, l2_rule='wave'):
    """portable_predict.predict with the candidate runtime model (all its patches for the SM count, tensor class and shared-memory cost are applied unchanged)."""
    import predict_runtime_v3j as J
    old = PP.V3I
    PP.V3I = _Shim(J, read_footprint, write_footprint, l2_rule)
    try: return PP.predict(features, phase_rows, unique_rows, bank_rows, constants, sm_count)
    finally: PP.V3I = old
