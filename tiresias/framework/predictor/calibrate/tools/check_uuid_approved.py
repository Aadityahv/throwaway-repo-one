#!/usr/bin/env python3
"""Early allow-list check used by energy_harness/run_calibration_cluster.sh before any GPU work (CPU only, no CUDA).

    python3 calibrate/tools/check_uuid_approved.py --uuid GPU-... --section H100 [--require-idle] [--allow-list approved_devices.json]

Exit 0: the UUID is in the allow-list (placeholders never count), its `ground_truth_section` is the expected one, and with --require-idle the GPU has no compute
process and is idle (the same guard the calibrator itself applies before every stage). Exit 1: refused, with the exact reason; when the UUID is merely unlisted the
message names the approved UUIDs of that section so a cluster job can be pinned to a node that has one (--nodelist). Matching is by UUID only: `machine` is a label, never
compared with the host name, so a compute node such as node-6.cluster.example.org is not refused on its name.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cal import device as D  # noqa: E402


def check(uuid, section, allow_list=None, require_idle=False):
    allow = D.load_allow_list(allow_list)
    try:
        entry = D.require_approved(uuid, allow)
    except D.Refusal as ex:
        same = sorted(u for u, e in allow.items() if e.get('ground_truth_section') == section)
        raise D.Refusal('%s\nApproved %s UUIDs (%d): %s\nList the node UUIDs with energy_harness/cluster_list_gpu_uuids.sh, add them with calibrate/tools/add_cluster_uuids.py, '
                        'or resubmit with --nodelist=<a node that has an approved UUID>.' % (ex, section, len(same), ', '.join(same) or 'none yet'))
    if entry.get('ground_truth_section') != section:
        raise D.Refusal('device %s is approved for section %r, not %r' % (uuid, entry.get('ground_truth_section'), section))
    idle = D.wait_idle(uuid) if require_idle else None
    return entry, idle


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--uuid', required=True); ap.add_argument('--section', required=True); ap.add_argument('--allow-list', type=Path); ap.add_argument('--require-idle', action='store_true')
    a = ap.parse_args(argv)
    try:
        entry, idle = check(a.uuid, a.section, a.allow_list, a.require_idle)
    except D.Refusal as ex:
        print('REFUSED: %s' % ex, file=sys.stderr); return 1
    print('APPROVED: %s' % json.dumps(entry, sort_keys=True))
    if idle is not None: print('IDLE: %s' % json.dumps(idle, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
