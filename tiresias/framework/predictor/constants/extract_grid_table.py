"""Compact per-cell table of the isolated microbenchmark grid, so fits are reproducible without the raw files."""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_microbench as F
def main(campaign_dir, packet, out):
    rows = json.loads(Path(packet).read_text())['rows']
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slot', 'kind', 'family', 'streams', 'threads', 'blocks', 'unroll', 'loops', 'table_bytes', 'width_bytes', 'cache_policy', 'stride',
                    'last_warp_cycles_mean_over_blocks', 'event_ms_median'])
        for r in rows:
            cyc, ms = F.load_cell(Path(campaign_dir) / 'cells' / r['slot'], r)
            w.writerow([r['slot'], r['kind'], r['family'], r['streams'], r['threads'], r['blocks'], r['unroll'], r['loops'], r['table_bytes'],
                        r['width_bytes'], r['cache_policy'], r['stride'], f'{cyc:.1f}', f'{ms:.9f}'])
if __name__ == '__main__':
    main(*sys.argv[1:4])
