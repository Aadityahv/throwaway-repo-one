"""Cell definitions of the prospective test (24 cells), fixed in DESIGN.md (addendum of 3 October 2026) before any cell was built. Family builders of the evaluation sets are used unchanged (tier rule:
L2 iff footprint / L2 < 1, no cell between 0.4 and 1.5 of the verified L2; shapes tile exactly; logical bytes = compulsory bytes).
  P1 tensor-core matrix multiply, new tile configurations: c1 128x64 tile 4 warps, c2 256x128 tile 16 warps   (family tcgemm2)
  P2 fused attention, new block sizes: c1 2 warps (32 query rows), c2 16 warps (256 query rows)               (family attn2)
  P3 new fused kernel A, attention without running maximum: c1 4 warps, c2 8 warps                           (family attnnm)
  P4 new fused kernel B, matrix multiply with bias and ReLU epilogue: c1 128x128 8 warps, c2 64x64 4 warps    (family tcrelu)
Sizes: matrix multiply M=N x K: small 1024 x 2048, medium 3072 x 1024, large 8192 x 768; attention batch*heads x S: small 32x768, medium 72x1024, large 384x1024."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; SR = HERE.parent
for d in ('fresh_g', 'fresh_h'): sys.path.insert(0, str(SR / d))
import cells_g as G, cells_h as H  # noqa

GEMM = {'small': (1024, 1024, 2048), 'medium': (3072, 3072, 1024), 'large': (8192, 8192, 768)}
ATT = {'small': (32, 768), 'medium': (72, 1024), 'large': (384, 1024)}
D = 64


def define_cells(hw):
    l2 = hw['l2_bytes']; assert l2 == 134217728; cells = []
    for regime, (m, n, k) in GEMM.items():
        fp = (m * k + n * k) * 2 + m * n * 4
        for cand, kid, bm, bn, thr in (('c1', 'tc128x64', 128, 64, 128), ('c2', 'tc256x128', 256, 128, 512)):
            assert m % bm == 0 and n % bn == 0 and k % 32 == 0
            cells.append(G._cell('tcgemm2', 'prosp_tensor_matmul_new_tiles', regime, cand, [dict(kid=kid, grid=[n // bn, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k))], fp, fp, l2, m, n, dict(M=m, N=n, K=k, tile_m=bm, tile_n=bn)))
        for cand, kid, bm, thr in (('c1', 'tcr128', 128, 256), ('c2', 'tcr64', 64, 128)):
            assert m % bm == 0 and n % bm == 0
            cells.append(G._cell('tcrelu', 'prosp_tensor_matmul_bias_relu', regime, cand, [dict(kid=kid, grid=[n // bm, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k))], fp, fp, l2, m, n, dict(M=m, N=n, K=k, tile=bm)))
    for regime, (bh, s) in ATT.items():
        fp = 3 * bh * s * D * 2 + bh * s * D * 4
        for fam, op, kids in (('attn2', 'prosp_fused_attention_new_blocks', (('c1', 'at2', 2), ('c2', 'at16', 16))), ('attnnm', 'prosp_attention_no_max', (('c1', 'nm4', 4), ('c2', 'nm8', 8)))):
            for cand, kid, warps in kids:
                qt = 16 * warps; assert s % qt == 0 and s % 64 == 0
                cells.append(H._cell(fam, op, regime, cand, [dict(kid=kid, grid=[s // qt, bh, 1], block=[warps * 32, 1, 1], args=dict(S=s))], fp, fp, l2, bh, s, dict(BH=bh, S=s, D=D, query_rows_per_block=qt)))
    return cells


if __name__ == '__main__':
    import prosp_lib as L
    hw = L.UP.X.load_hardware(L.UP.X.read_text(L.UP.X.GROUND_TRUTH))
    for c in define_cells(hw): print('%-62s %-4s %.3f fp=%.1f MiB blocks=%d' % (c['cell_id'], c['tier'], c['l2_ratio'], c['footprint_bytes'] / 2**20, c['kernels'][-1]['grid'][0] * c['kernels'][-1]['grid'][1]))
