"""Device guard: allow-list, live facts, cross-check against HARDWARE_GROUND_TRUTH.md, idle check, booking reference. CPU-testable with stubbed inputs."""
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[3]
GROUND_TRUTH = REPO / 'HARDWARE_GROUND_TRUTH.md'
PLACEHOLDER = 'TO-BE-ADDED'
# Opt-in (--allow-passive-display-context, default off): foreign processes that may stay on the approved GPU when they are only a desktop/display context. Names are compared by basename.
DISPLAY_PROCESS_NAMES = frozenset({'snapd-desktop-integration',  # desktop helper the user accepted on Ada, 4 Oct 2026 (C+G context, ~18 MiB, 0% use)
    'Xorg', 'X', 'Xwayland', 'gnome-shell', 'gnome-session-binary', 'gdm-x-session', 'gdm-wayland-session', 'gdm-session-worker', 'kwin_x11', 'kwin_wayland', 'plasmashell',
                                   'sddm-greeter', 'lightdm', 'mutter', 'cinnamon', 'xfwm4', 'Xvnc', 'weston', 'sway'})
MAX_DISPLAY_CONTEXT_MIB = 64


class Refusal(RuntimeError):
    """Raised for every refusal; the message is the exact reason shown to the user."""


def load_allow_list(path=None):
    d = json.loads(Path(path or HERE / 'approved_devices.json').read_text())
    return {x['uuid']: x for x in d['devices'] if not x['uuid'].startswith(PLACEHOLDER)}


def require_approved(uuid, allow):
    if not uuid or not re.fullmatch(r'GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', uuid):
        raise Refusal('--device-uuid must be a full GPU UUID (GPU-xxxxxxxx-...), got %r' % (uuid,))
    if uuid not in allow:
        raise Refusal('device %s is not in calibrate/approved_devices.json; add it (with its verified HARDWARE_GROUND_TRUTH.md section) first' % uuid)
    return allow[uuid]


def require_booking(ref):
    if not ref or len(ref.strip()) < 8:
        raise Refusal('--booking-ref must quote the the booking log booking entry (at least 8 characters)')
    return ref.strip()


def parse_ground_truth(section, text=None):
    """Return {'sm_count': int, 'l2_bytes': int, 'compute_capability': 'x.y'} from the named section of HARDWARE_GROUND_TRUTH.md, or raise."""
    text = text if text is not None else GROUND_TRUTH.read_text()
    m = re.search(r'^## (%s)\b.*?(?=^## |\Z)' % re.escape(section), text, re.S | re.M)
    if not m:
        raise Refusal('HARDWARE_GROUND_TRUTH.md has no section starting with %r; verify the GPU and add it first' % section)
    body = m.group(0)
    out = {}
    r = re.search(r'\|\s*SM count\s*\|\s*([\d,]+)', body)
    if r: out['sm_count'] = int(r.group(1).replace(',', ''))
    r = re.search(r'\|\s*L2 cache size\s*\|\s*([\d,]+)\s*bytes', body)
    if r: out['l2_bytes'] = int(r.group(1).replace(',', ''))
    r = re.search(r'\|\s*Compute capability\s*\|[^|]*?(\d+)\.(\d+)|\|\s*Compute capability\s*\|\s*sm_(\d)(\d)', body)
    if r:
        g = [x for x in r.groups() if x is not None]
        out['compute_capability'] = '%s.%s' % (g[0], g[1])
    missing = [k for k in ('sm_count', 'l2_bytes', 'compute_capability') if k not in out]
    if missing:
        raise Refusal('HARDWARE_GROUND_TRUTH.md section %r lacks verified %s' % (section, ', '.join(missing)))
    return out


def cross_check(facts, entry, ground_truth):
    """facts: parsed device_facts JSON from the live device. Refuse on any disagreement with the allow-list entry or the ground-truth file."""
    cc = facts['compute_capability']
    problems = []
    if facts['sm_count'] != ground_truth['sm_count']: problems.append('SM count live %s vs HARDWARE_GROUND_TRUTH.md %s' % (facts['sm_count'], ground_truth['sm_count']))
    if facts['l2_bytes'] != ground_truth['l2_bytes']: problems.append('L2 size live %s vs HARDWARE_GROUND_TRUTH.md %s' % (facts['l2_bytes'], ground_truth['l2_bytes']))
    if cc != ground_truth['compute_capability']: problems.append('compute capability live %s vs HARDWARE_GROUND_TRUTH.md %s' % (cc, ground_truth['compute_capability']))
    if entry.get('sm_count') and facts['sm_count'] != entry['sm_count']: problems.append('SM count live %s vs approved_devices.json %s' % (facts['sm_count'], entry['sm_count']))
    if entry.get('compute_capability') and cc != entry['compute_capability']: problems.append('compute capability live %s vs approved_devices.json %s' % (cc, entry['compute_capability']))
    if problems:
        raise Refusal('device identity mismatch: ' + '; '.join(problems))


def smi(args, run=subprocess.run):
    r = run(['nvidia-smi'] + args, capture_output=True, text=True)
    if r.returncode != 0:
        raise Refusal('nvidia-smi %s failed (rc=%s): %s' % (' '.join(args), r.returncode, (r.stderr or '')[-200:]))
    return r.stdout


def passive_display_processes(uuid, run=subprocess.run, max_mib=MAX_DISPLAY_CONTEXT_MIB):
    """Every foreign process on the approved GPU as [{pid, name, used_mib}], or a Refusal unless ALL of them are display-server/compositor processes using at most `max_mib` MiB."""
    out = smi(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], run)
    procs, bad = [], []
    for line in out.splitlines():
        c = [x.strip() for x in line.split(',')]
        if len(c) < 4 or c[0] != uuid: continue
        name = c[2].replace('\\', '/').rsplit('/', 1)[-1]
        try: mib = int(c[3])
        except ValueError: mib = None
        procs.append(dict(pid=c[1], name=name, used_mib=mib))
        if name not in DISPLAY_PROCESS_NAMES: bad.append('%s (pid %s) is not a display-server or compositor process' % (name, c[1]))
        elif mib is None or mib > max_mib: bad.append('%s (pid %s) uses %s MiB, more than the %d MiB allowed for a passive display context' % (name, c[1], c[3], max_mib))
    if not procs or bad:
        raise Refusal('the approved GPU has compute processes that are not a passive display context (never kill one this project did not start):\n' + '\n'.join(bad or ['no process could be identified']))
    return procs


def require_idle(uuid, run=subprocess.run, allow_passive_display=False):
    """Refuse unless the approved GPU has no compute process and is at 0% utilisation (the caller re-checks before every stage).
    allow_passive_display (explicit opt-in, default off): foreign processes are tolerated only if every one is a display-server/compositor process using at most MAX_DISPLAY_CONTEXT_MIB
    MiB, and then GPU utilisation must be exactly 0%; the accepted processes are returned in `foreign_display_processes_allowed`. Anything else refuses exactly as without the opt-in."""
    out = smi(['--query-gpu=uuid,utilization.gpu,memory.used', '--format=csv,noheader,nounits'], run)
    rows = [[c.strip() for c in l.split(',')] for l in out.splitlines() if l.strip()]
    mine = [r for r in rows if r[0] == uuid]
    if not mine:
        raise Refusal('nvidia-smi does not list the approved GPU %s' % uuid)
    apps = smi(['--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], run)
    busy = [l for l in apps.splitlines() if l.strip().startswith(uuid)]
    foreign = None
    if busy:
        if not allow_passive_display:
            raise Refusal('the approved GPU has compute processes (never kill one this project did not start):\n' + '\n'.join(busy))
        foreign = passive_display_processes(uuid, run)
        if int(mine[0][1]) != 0:
            raise Refusal('the approved GPU is at %s%% utilisation with a foreign display context present, not idle' % mine[0][1])
    elif int(mine[0][1]) > 5:
        raise Refusal('the approved GPU is at %s%% utilisation, not idle' % mine[0][1])
    rec = dict(utilization_pct=int(mine[0][1]), memory_used_mib=int(mine[0][2]))
    if foreign is not None: rec['foreign_display_processes_allowed'] = foreign
    return rec


def wait_idle(uuid, run=subprocess.run, sleep=None, timeout_s=40.0, readings=3, interval_s=2.0, allow_passive_display=False):
    """require_idle with a bounded settle window: the driver reports stale utilisation for a few seconds after the tool's OWN previous process exits (seen right after a
    bandwidth stage: 99% at the instant of exit, 0% two seconds later). A compute process on the approved GPU refuses at once; otherwise `readings` consecutive idle readings
    are required within `timeout_s`, else refuse."""
    import time
    sleep = sleep or time.sleep
    waited, streak, last = 0.0, 0, None
    while True:
        try:
            last = require_idle(uuid, run, allow_passive_display); streak += 1
            if streak >= readings: return dict(last, settled_after_s=waited)
        except Refusal as ex:
            if 'compute processes' in str(ex) or 'does not list' in str(ex): raise
            streak = 0; last_reason = str(ex)
            if waited >= timeout_s: raise Refusal('%s (did not settle within %.0f s)' % (last_reason, timeout_s))
        sleep(interval_s); waited += interval_s
