#!/usr/bin/env python3
"""Re-derive the runtime model's traffic rates (`constants.store_legacy`) of an archived calibration document from its archived raw stage outputs (CPU only, no GPU).

    python3 calibrate/tools/rederive_store.py --doc <calibration_*.json> --method legacy|stream_rates|guarded [--out <new document>] [--legacy-dir <dir>]

Reads `stages/store/stdout.jsonl` and `stages/stream/stdout.jsonl` next to the document (their SHA-256 must equal the hashes the document recorded), recomputes the stream
curves with the tool's own fit and refuses unless they equal the document's, then derives the traffic rates with `cal.fits.fit_store_detailed`. Writes a NEW document next to the original
(default `<stem>_store_<method>.json`; an existing file is never overwritten): the original document with `constants.store_legacy` replaced, the store-related warnings recomputed
under the new derivation, `store_derivation` (method requested and used, identification diagnostics, window fit) and `rederived_from` (original file name and SHA-256). With --legacy-dir
also exports the legacy constants files (`stream_constants.json` and the others) from the new document."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cal import document, fits  # noqa: E402

STORE_WARNING_PREFIXES = ('store fit is ill-conditioned', 'a stream coefficient was driven to zero', 'UNIDENTIFIED TRAFFIC RATES', 'IMPLAUSIBLE DRAM RATES', 'store refit is ill-conditioned', 'stream_rates refit')


CRLF, LF = bytes([13, 10]), bytes([10])


def sha256_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_jsonl(text): return [json.loads(l) for l in text.splitlines() if l.strip().startswith('{')]


def rederive(doc_path, method, out_path=None, legacy_dir=None, dram_peak_TBps=None):
    doc_path = Path(doc_path); doc = json.loads(doc_path.read_text()); base = doc_path.parent
    raw = {}
    for name in ('store', 'stream'):
        p = base / 'stages' / name / 'stdout.jsonl'
        if not p.exists(): raise SystemExit('REFUSED: raw stage output %s is not archived next to the document' % p)
        rec = next((s for s in doc['stages'] if s['name'] == name), None)
        want = (rec or {}).get('stdout_sha256')
        data = p.read_bytes()
        # a Windows checkout (core.autocrlf) turns the archived LF line endings into CRLF; the hash recorded at run time is of the LF text
        if want and hashlib.sha256(data).hexdigest() != want and hashlib.sha256(data.replace(CRLF, LF)).hexdigest() != want:
            raise SystemExit('REFUSED: %s does not match the SHA-256 recorded in the document (raw or with CRLF read as LF)' % p)
        raw[name] = parse_jsonl(data.decode('utf-8'))
    curves = fits.fit_stream(raw['stream'])
    if curves != doc['constants']['stream_curves']: raise SystemExit('REFUSED: stream curves recomputed from the archived raw rows differ from the document\'s stream_curves')
    c = doc['constants']
    store, warnings, derivation = fits.fit_store_detailed(raw['store'], doc['device']['sm_count'], c['chase_latency_ns'], curves, method, doc['device'].get('l2_bytes'), dram_peak_TBps)
    new = json.loads(json.dumps(doc))
    new['constants']['store_legacy'] = store
    new['warnings'] = [w for w in doc['warnings'] if not w.startswith(STORE_WARNING_PREFIXES)] + warnings
    new['store_derivation'] = derivation
    new['rederived_from'] = dict(original_document=doc_path.name, original_document_sha256=sha256_file(doc_path), original_store_legacy=doc['constants']['store_legacy'],
                                 original_warnings=doc['warnings'], tool='calibrate/tools/rederive_store.py', rederived_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'))
    out_path = Path(out_path) if out_path else doc_path.with_name('%s_store_%s.json' % (doc_path.stem, method))
    if out_path.exists(): raise SystemExit('REFUSED: %s exists; this tool never overwrites' % out_path)
    out_path.write_text(json.dumps(new, indent=1, sort_keys=True) + '\n')
    if legacy_dir: document.export_legacy(new, legacy_dir)
    return out_path, new


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--doc', type=Path, required=True); ap.add_argument('--method', required=True, choices=fits.STORE_METHODS); ap.add_argument('--out', type=Path); ap.add_argument('--legacy-dir', type=Path); ap.add_argument('--dram-peak-TBps', type=float, help='verified DRAM peak of the board from HARDWARE_GROUND_TRUTH.md (optional; never computed here)')
    a = ap.parse_args(argv)
    out, new = rederive(a.doc, a.method, a.out, a.legacy_dir, a.dram_peak_TBps)
    d = new['store_derivation']
    print('wrote %s (method used %s; identification passes: %s)' % (out, d['method_used'], d['identification']['passes']))
    print(json.dumps(new['constants']['store_legacy'], sort_keys=True))
    for w in new['warnings']: print('WARNING:', w)
    return 0


if __name__ == '__main__':
    sys.exit(main())
