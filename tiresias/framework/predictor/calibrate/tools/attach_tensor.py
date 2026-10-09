#!/usr/bin/env python3
"""Attach the result of a `--stages tensor` calibration run to the calibration document of a full run of the SAME board (CPU only).

    python3 calibrate/tools/attach_tensor.py --tensor-run <run dir of --stages tensor> --energy-doc <calibration_*.json of a full run> --out <new document> [--legacy-dir <dir>]

The tensor energy rate is derived from the base power and rates of the energy stage, so a tensor-only run cannot derive it by itself. This tool re-derives it from the archived tensor windows and the energy
document, writes a copy of the energy document with `constants.tensor` added (and the hashes of both inputs in `tensor_attachment`), and with --legacy-dir exports the runtime constants with the
`tensor_mma` issue class added. Refuses unless both runs name the same device UUID. The document stays `complete: false` if the energy document was incomplete."""
import argparse, hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from cal import document, energy as En, tensor as Tn  # noqa: E402


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tensor-run', type=Path, required=True); ap.add_argument('--energy-doc', type=Path, required=True); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--legacy-dir', type=Path)
    a = ap.parse_args()
    tdoc_path = next(a.tensor_run.glob('calibration_*.json')); tdoc = json.loads(tdoc_path.read_text()); edoc = json.loads(a.energy_doc.read_text())
    if tdoc['device']['uuid'] != edoc['device']['uuid']: print('REFUSED: the two runs are of different devices (%s, %s)' % (tdoc['device']['uuid'], edoc['device']['uuid'])); return 1
    sdir = a.tensor_run / 'stages' / 'tensor'
    windows = json.loads((sdir / 'windows.json').read_text()); issue = json.loads((sdir / 'issue' / 'issue.json').read_text())
    funcs = En.split_functions((sdir / 'sass.txt').read_text()); sym = next(s for s in funcs if s.startswith('_Z10mma_energyILi%dE' % Tn.M_PER_TRIP))
    verified = {}
    for w in windows:
        cnt = En.count_kernel(funcs[sym], json.loads((sdir / w['id'] / 'window.json').read_text())['trips_per_launch']); bad = Tn.verify_design(cnt['in_loop'])
        if bad: print('REFUSED: SASS mismatch for %s: %s' % (w['id'], bad)); return 1
        verified[w['id']] = dict(in_loop=cnt['in_loop'], out_of_loop=cnt['out_of_loop'], loop_opcodes=cnt['loop_opcodes'])
    c = Tn.finalize(dict(issue=issue, windows=windows, sass_verified=verified), edoc['constants']['energy'], edoc['device']['sm_count'])
    out = dict(edoc); out['constants'] = dict(edoc['constants'], tensor=c)
    out['tensor_attachment'] = dict(tensor_run_document=str(tdoc_path), tensor_run_document_sha256=sha(tdoc_path), energy_document=str(a.energy_doc), energy_document_sha256=sha(a.energy_doc),
                                    tensor_stage_status=c['status'], note='constants.tensor was derived by calibrate/tools/attach_tensor.py from the archived tensor windows and the energy stage of the energy document; nothing else in the document changed')
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + '\n')
    if a.legacy_dir: document.export_legacy(out, a.legacy_dir)
    print(json.dumps(dict(status=c['status'], issue=c['issue'], energy={k: c['energy'].get(k) for k in ('status', 'rate_pJ_per_lane_instruction', 'max_window_error_pct', 'admitted', 'not_admitted', 'reason')}), indent=1)); return 0 if c['status'] == 'ok' else 3


if __name__ == '__main__':
    sys.exit(main())
