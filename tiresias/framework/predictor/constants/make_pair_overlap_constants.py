"""Derive constants/pair_overlap_constants.json from the dependent fragment-load microbenchmark (attention_diagnosis/raw/dep_frag.jsonl, run 3): median over consumer shapes of the
shared-load/MMA pair overlap coefficient beta at each measured resident-warps-per-SM value. No operator cell is used.   python3 make_pair_overlap_constants.py"""
import hashlib, json, statistics
from pathlib import Path
import argparse
HERE = Path(__file__).resolve().parent
_ap = argparse.ArgumentParser(); _ap.add_argument('--raw', default='attention_diagnosis/raw/dep_frag.jsonl', help='path relative to predictor/ (default: the Blackwell run)')
_ap.add_argument('--out', default='pair_overlap_constants.json', help='file name in constants/ (default: the Blackwell constants)'); _a = _ap.parse_args()
RAW = HERE.parent / _a.raw
rows = [json.loads(l) for l in RAW.read_text().splitlines() if l.strip()]
by = {}
for r in rows:
    if r['consumer'] == 'hmma': by.setdefault(r['warps_per_sm'], []).append(r['beta'])
xs = sorted(by)
doc = dict(schema='pair_overlap_constants/1', rule='phase time of the shared-load pipe (s) and the MMA pipe (m): max(s, m) + beta * min(s, m); beta linearly interpolated in resident warps per SM, held at the end values outside the measured range',
           resident_warps_per_sm=xs, beta_hmma=[statistics.median(by[x]) for x in xs], configurations_per_point=[len(by[x]) for x in xs],
           source=_a.raw, source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest())
(HERE / _a.out).write_text(json.dumps(doc, indent=1) + '\n'); print(json.dumps(doc))
