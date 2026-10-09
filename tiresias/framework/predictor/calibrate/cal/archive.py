"""Slim a run directory for committing: the pipes cells hold hundreds of MB of raw per-lane outputs (sample*.bin, input_pattern.bin). Before they are removed, a manifest records
the SHA-256 of every raw file and a summary records, per cell, the quantities the fits use (per-sample device-cycle spans, event times, pointer-chase cycles per step), so the
fits can be re-run from the committed summary and the raw bins can be checked against the manifest on the host that retains them."""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import fits


def slim(run_dir, remove_raw=True):
    run_dir = Path(run_dir); cells = run_dir / 'stages' / 'pipes' / 'cells'
    grid = {r['slot']: r for r in json.loads((run_dir / 'stages' / 'pipes' / 'grid.json').read_text())}
    manifest, summary = {}, {}
    for slot, row in grid.items():
        d = cells / slot
        files = sorted(p for p in d.iterdir() if p.suffix == '.bin')
        manifest[slot] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        res = json.loads((d / 'result.json').read_text())
        if row['kind'] == 'compute':
            span, ms = fits.load_cell(d, row); summary[slot] = dict(kind='compute', device_cycle_span=span, event_ms=ms, event_ms_samples=res['event_ms'])
        else:
            summary[slot] = dict(kind='memory', tier=row['tier'], cycles_per_step=fits.chase_cycles_per_step(d, row), event_ms_samples=res['event_ms'])
        if remove_raw:
            for p in files: p.unlink()
    (run_dir / 'stages' / 'pipes' / 'raw_files_sha256.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    (run_dir / 'stages' / 'pipes' / 'cells_summary.json').write_text(json.dumps(summary, indent=1, sort_keys=True) + '\n')
    return len(manifest)
