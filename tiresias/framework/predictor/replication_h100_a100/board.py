"""Explicit target-board bindings; never changes the shared generators or models."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SR = HERE.parent
REPO = SR.parents[2]
ADA = SR / 'replication_ada_rtx5000'
CONFIG = {'a100': ('## A100', 'sm_80', 'NVIDIA A100-SXM4-80GB'),
          'h100': ('## H100', 'sm_90', 'NVIDIA H100 80GB HBM3')}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def binding(label, out=None):
    if label not in CONFIG:
        raise ValueError('unsupported target board: ' + label)
    for path in (ADA, SR, SR / 'h100_eval_freeze'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import make_cells_h100 as reader
    def read(section):
        previous = reader.SECTION
        reader.SECTION = section
        try:
            # These sections already have the parser's short label. Replacing the
            # parallel long-label row would create duplicate matches and hide it.
            return reader.read_h100_hardware()
        finally:
            reader.SECTION = previous
    def blackwell_values():
        hw, _ = read('## Blackwell')
        return hw['l2_bytes'], hw['sm_count']
    section, arch, name = CONFIG[label]
    target = HERE / label if out is None else Path(out)
    return SimpleNamespace(
        HERE=target, SR=SR, REPO=REPO, FZ=SR / 'h100_eval_freeze',
        SECTION=section, ARCH=arch, PREFIX=label, GPU_NAME=name,
        SASS_BASE=(target / 'compiled_base' if (target / 'compiled_base').is_dir()
                   else SR / ('port_' + label) / 'compiled_cluster_cuda12.1'),
        SASS_EVAL=target / 'compiled',
        PAIR_CONSTANTS=target / 'pair_overlap_constants.json',
        read_ada_hardware=lambda: read(section),
        forbidden_blackwell_values=blackwell_values)


def cells_module(label, out=None):
    """Reuse the complete, label-free generator with explicit board globals."""
    b = binding(label, out)
    mod = load_module('_cluster_cells_' + label, ADA / 'make_cells_ada.py')
    mod.B, mod.HERE = b, b.HERE
    mod.SCHEMA = 'cluster_replication_cells/1'
    mod.FILES = {g: 'cells_' + g + '.json' for g in mod.GROUPS}
    return mod


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
