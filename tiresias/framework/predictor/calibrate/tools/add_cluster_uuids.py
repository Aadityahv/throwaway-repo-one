#!/usr/bin/env python3
"""Turn the output of energy_harness/cluster_list_gpu_uuids.sh into allow-list entries (and the rows of the table in HARDWARE_GROUND_TRUTH.md). CPU only.

    python3 calibrate/tools/add_cluster_uuids.py ~/gpu_uuids_h100_<jobid> [more directories ...]     # print what would be added
    python3 calibrate/tools/add_cluster_uuids.py ~/gpu_uuids_h100_<jobid> --write                    # add to approved_devices.json and HARDWARE_GROUND_TRUTH.md

For every GPU row in `gpu_query.csv` (all GPUs nvidia-smi showed to the job; `--allocated-only` keeps just the job's own) it writes one entry with
machine=cluster, the section taken from the directory's arch_label (or --section), and compute_capability and sm_count copied from the entries already present for
that section in approved_devices.json (the placeholders carry the values HARDWARE_GROUND_TRUTH.md verified). It refuses when the nvidia-smi name does not match the
section (an H100 UUID can never be filed under A100), when the section has no entry to copy from, when a UUID is malformed or the job did not finish cleanly, and it
skips (and reports) UUIDs already listed. Several entries with the same section are normal: the allow-list is keyed by UUID. Entries are inserted on their own lines,
in the file's existing one-entry-per-line style. Nothing is invented: every value comes from the job output or from the existing placeholder entry.
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from cal import device as D  # noqa: E402

NAME_MARK = {'H100': 'H100', 'A100': 'A100'}
ARCH_SECTION = {'h100': 'H100', 'a100': 'A100'}
UUID_RE = re.compile(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
PENDING_ROW = re.compile(r'^\|\s*pending: filled from energy_harness/cluster_list_gpu_uuids\.sh')


def read_dir(d):
    d = Path(d); meta = {}
    for l in (d / 'meta.txt').read_text().splitlines():
        if '=' in l:
            k, v = l.split('=', 1); meta[k.strip()] = v.strip()
    if not (d / 'COMPLETE').is_file() or not (d / 'COMPLETE').read_text().startswith('ok'):
        raise D.Refusal('%s has no COMPLETE marker starting with "ok": the UUID job did not finish cleanly' % d)

    def rows(name):
        out = []
        for r in csv.reader((d / name).read_text().splitlines(), skipinitialspace=True):
            if r and r[0].strip() != 'index': out.append([c.strip() for c in r])
        return out
    return meta, rows('gpu_query.csv'), rows('allocated_gpu.csv')


def build_entries(dirs, section, allow_path, allocated_only=False, approved_by=None):
    """Return (entries, markdown_rows, skipped). Raises Refusal on any identity problem."""
    devices = json.loads(Path(allow_path).read_text())['devices']
    seen = set(x['uuid'] for x in devices)
    entries, md, skipped = [], [], []
    for d in dirs:
        meta, all_rows, alloc_rows = read_dir(d)
        sec = section or ARCH_SECTION.get(meta.get('arch_label', ''))
        if sec not in NAME_MARK: raise D.Refusal('%s: cannot determine the section (arch_label=%r); pass --section H100|A100' % (d, meta.get('arch_label')))
        tmpl = [x for x in devices if x.get('ground_truth_section') == sec and x.get('machine') == 'cluster']
        if not tmpl: raise D.Refusal('approved_devices.json has no cluster entry for section %s to copy compute capability and SM count from' % sec)
        cc, sm = tmpl[0]['compute_capability'], tmpl[0]['sm_count']
        node = meta.get('hostname', 'unknown'); job = meta.get('slurm_job_id', 'unknown')
        for r in (alloc_rows if allocated_only else all_rows):
            if len(r) < 4: raise D.Refusal('%s: malformed GPU row %r' % (d, r))
            idx, uuid, pci, name = r[0], r[1], r[2], r[3]
            if not UUID_RE.fullmatch(uuid): raise D.Refusal('%s: %r is not a full GPU UUID' % (d, uuid))
            if NAME_MARK[sec] not in name: raise D.Refusal('%s: GPU %s is named %r, which does not match section %s; refusing to file it' % (d, uuid, name, sec))
            if uuid in seen:
                skipped.append(uuid); continue
            seen.add(uuid)
            e = dict(uuid=uuid, machine='cluster', gpu='%s (Cluster %s, index %s)' % (re.sub(r'^NVIDIA\s+', '', name), node.split('.')[0], idx), ground_truth_section=sec,
                     compute_capability=cc, sm_count=sm, node=node, pci_bus=pci, source_job=job)
            if approved_by: e['approved_by'] = approved_by
            entries.append(e)
            md.append('| %s | %s | %s | %s | %s | cluster job %s (energy_harness/cluster_list_gpu_uuids.sh) |' % (node, idx, uuid, pci, name, job))
    return entries, md, skipped


def entry_line(e): return ' ' + json.dumps(e, separators=(', ', ': '))


def insert_entries(text, entries):
    """Insert entries after the last device line, keeping one entry per line and valid JSON."""
    lines = text.rstrip('\n').split('\n')
    last = max(i for i, l in enumerate(lines) if l.lstrip().startswith('{"uuid"'))
    lines[last] = lines[last].rstrip().rstrip(',') + ','
    new = [entry_line(e) + ',' for e in entries]; new[-1] = new[-1].rstrip(',')
    out = '\n'.join(lines[:last + 1] + new + lines[last + 1:]) + '\n'
    json.loads(out)
    return out


def insert_table_rows(gt_text, md_rows):
    m = re.search(r'^## Cluster GPU UUIDs\b.*?(?=^## |\Z)', gt_text, re.S | re.M)
    if not m: raise D.Refusal('HARDWARE_GROUND_TRUTH.md has no "## Cluster GPU UUIDs" section')
    sec_lines = [l for l in m.group(0).split('\n') if not PENDING_ROW.match(l)]
    last_row = max(i for i, l in enumerate(sec_lines) if l.startswith('|'))
    sec_lines[last_row + 1:last_row + 1] = md_rows
    return gt_text[:m.start()] + '\n'.join(sec_lines) + gt_text[m.end():]


def read_keep_eol(path):
    with open(path, encoding='utf-8', newline='') as f: raw = f.read()
    return raw.replace('\r\n', '\n'), ('\r\n' if '\r\n' in raw else '\n')


def write_with_eol(path, text, eol):
    with open(path, 'w', encoding='utf-8', newline='') as f: f.write(text.replace('\n', eol))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dirs', nargs='+', type=Path); ap.add_argument('--section', choices=sorted(NAME_MARK)); ap.add_argument('--write', action='store_true')
    ap.add_argument('--allocated-only', action='store_true'); ap.add_argument('--approved-by', help='free text stored as approved_by in each new entry (used by the APPROVE_ALLOCATED opt-in)'); ap.add_argument('--allow-list', type=Path, default=HERE / 'approved_devices.json'); ap.add_argument('--ground-truth', type=Path, default=D.GROUND_TRUTH)
    a = ap.parse_args(argv)
    try:
        entries, md, skipped = build_entries(a.dirs, a.section, a.allow_list, a.allocated_only, a.approved_by)
        if skipped: print('already listed, skipped: ' + ', '.join(skipped))
        if not entries:
            print('nothing to add'); return 0
        print('allow-list entries%s:' % (' (written)' if a.write else ' (dry run; pass --write)'))
        for e in entries: print(entry_line(e))
        print('\nHARDWARE_GROUND_TRUTH.md rows:'); print('\n'.join(md))
        if a.write:
            (allow_text, allow_eol), (gt_text, gt_eol) = read_keep_eol(a.allow_list), read_keep_eol(a.ground_truth)
            new_allow = insert_entries(allow_text, entries); new_gt = insert_table_rows(gt_text, md)
            write_with_eol(a.allow_list, new_allow, allow_eol); write_with_eol(a.ground_truth, new_gt, gt_eol)
    except (D.Refusal, OSError, ValueError) as ex:
        print('REFUSED: %s' % ex, file=sys.stderr); return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
