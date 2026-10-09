"""Registers the kernels of the prospective test (src/prosp_kernels.cuh and the existing tc/attn headers, compiled with CUDA 13.2.78 for sm_120 into compiled/) with the unseen-kernel static pipeline
(imported, never edited), on top of fresh_h_lib (which carries the in-memory interpreter extensions for HMMA, STS.128, STS.U16, F2FP and the control-flow helpers). Only the kernel registry and the
SASS/resource directories are replaced; extra opcodes needed by the two new kernels, if any, are added to the in-memory whitelist below."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
sys.path.insert(0, str(SR / 'fresh_h')); sys.path.insert(0, str(SR / 'fresh_g'))
import fresh_h_lib as L  # noqa: E402
UP = L.UP

UP.HERE = HERE; UP.SASS_DIR = HERE / 'compiled/sass'; UP.LOG_DIR = HERE / 'compiled/log'; UP.ISO_DIR = HERE / 'isolated'
G = [('A', 'ptr'), ('B', 'ptr'), ('C', 'ptr'), ('N', 'i32'), ('K', 'i32')]
GB = [('A', 'ptr'), ('B', 'ptr'), ('bias', 'ptr'), ('C', 'ptr'), ('N', 'i32'), ('K', 'i32')]
AT = [('Q', 'ptr'), ('K', 'ptr'), ('V', 'ptr'), ('O', 'ptr'), ('S', 'i32')]
UP.KERNELS.clear()
UP.KERNELS.update({
    'tc128x64': dict(sass='prosp', needle='tc_gemmILi128ELi64E', role='main', params=G),
    'tc256x128': dict(sass='prosp', needle='tc_gemmILi256E', role='main', params=G),
    'at2': dict(sass='prosp', needle='attn_fwdILi2E', role='main', params=AT),
    'at16': dict(sass='prosp', needle='attn_fwdILi16E', role='main', params=AT),
    'nm4': dict(sass='prosp', needle='attn_nomaxILi4E', role='main', params=AT),
    'nm8': dict(sass='prosp', needle='attn_nomaxILi8E', role='main', params=AT),
    'tcr128': dict(sass='prosp', needle='tc_gemm_bias_reluILi128E', role='main', params=GB),
    'tcr64': dict(sass='prosp', needle='tc_gemm_bias_reluILi64E', role='main', params=GB),
})
for _kid, _spec in UP.KERNELS.items():
    UP.A.PARAM_FIELDS[_kid] = _spec['params']
UP._KERNEL_CACHE.clear()
