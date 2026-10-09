"""Cell definitions of the prospective validation set (6 new shapes x 2 candidates = 12 cells), defined before any measurement and using the family cell builders of the
evaluation sets unchanged (so the tier rule, the 0.4 to 1.5 exclusion band and the logical-byte rule are the same). No shape here is an evaluation shape.

  V1 FP32 matrix multiply (register tiled)        M=N=2048, K=512        L2
  V2 tensor-core matrix multiply (bf16)           M=12288, N=4096, K=512 DRAM
  V3 fused attention (bf16)                       BH=256 heads, S=1536   DRAM
  V4 scalar product (batched dot products; reduction)  vectorN=768, elementN=40960  DRAM
  V5 separable convolution (rows then columns)    5120 x 3584 image      DRAM
  V6 RMSNorm                                      12288 rows x 2560      DRAM
c1 / c2 are the two candidates of the family (tile sizes, grid or thread-count variants), as in the evaluation sets."""
import sys
from pathlib import Path
SR = Path(__file__).resolve().parents[2]
for d in ('fresh_f', 'fresh_g', 'fresh_h', 'fresh_e', 'unseen_kernels'): sys.path.insert(0, str(SR / d))

LIBS = {'f': 'fresh_f_lib', 'g': 'fresh_g_lib', 'h': 'fresh_h_lib', 'e': 'fresh_e_lib', 'classic': None}


def define(lib, hw):
    l2 = hw['l2_bytes']; FLOAT = 4; cells = []
    if lib == 'f':
        import cells_f as F
        m, n, kk = 2048, 2048, 512
        for cand, kid, bm, bk in (('c1', 'sg64', 64, 16), ('c2', 'sg128', 128, 8)):
            assert m % bm == 0 and n % bm == 0 and kk % bk == 0
            k = dict(kid=kid, grid=[n // bm, m // bm, 1], block=[256, 1, 1], args=dict(N=n, K=kk)); fp = (m * kk + kk * n + m * n) * FLOAT
            cells.append(F._cell('sgemm', 'validation_fp32_matmul', 'v1', cand, [k], fp, fp, l2, m, n, dict(M=m, N=n, K=kk, tile=bm)))
        r, c = 12288, 2560
        for cand, kid, thr, args in (('c1', 'rms_s', 256, dict(cols=c, inv_cols=1.0 / c, eps=F.EPS)), ('c2', 'rms_v4', 128, dict(cols4=c // 4, inv_cols=1.0 / c, eps=F.EPS))):
            assert c % (4 * thr) == 0 if kid == 'rms_v4' else c % thr == 0
            k = dict(kid=kid, grid=[r, 1, 1], block=[thr, 1, 1], args=args); fp = (2 * r * c + c) * FLOAT
            cells.append(F._cell('rmsnorm', 'validation_rmsnorm', 'v6', cand, [k], fp, fp, l2, r, c, dict(rows=r, cols=c, threads=thr)))
    elif lib == 'g':
        import cells_g as G
        m, n, k_ = 12288, 4096, 512
        for cand, kid, bm, thr in (('c1', 'tc128', 128, 256), ('c2', 'tc64', 64, 128)):
            assert m % bm == 0 and n % bm == 0 and k_ % 32 == 0
            kern = dict(kid=kid, grid=[n // bm, m // bm, 1], block=[thr, 1, 1], args=dict(N=n, K=k_)); fp = (m * k_ + n * k_) * 2 + m * n * 4
            cells.append(G._cell('tcgemm', 'validation_tensor_matmul', 'v2', cand, [kern], fp, fp, l2, m, n, dict(M=m, N=n, K=k_, tile=bm)))
    elif lib == 'h':
        import cells_h as H
        bh, s = 256, 1536
        for cand, kid, warps in (('c1', 'at4', 4), ('c2', 'at8', 8)):
            qt = 16 * warps; assert s % qt == 0 and s % 64 == 0
            kern = dict(kid=kid, grid=[s // qt, bh, 1], block=[warps * 32, 1, 1], args=dict(S=s)); fp = 3 * bh * s * H.D * 2 + bh * s * H.D * 4
            cells.append(H._cell('attn', 'validation_fused_attention', 'v3', cand, [kern], fp, fp, l2, bh, s, dict(BH=bh, S=s, D=H.D, query_rows_per_block=qt)))
    elif lib == 'e':
        import cells_e as E
        v, e = 768, 40960
        for cand, g in E.SP_GRID.items():
            assert v % g == 0 and e % 1024 == 0
            k = dict(kid='sp', grid=[g, 1, 1], block=[256, 1, 1], args=dict(vectorN=v, elementN=e)); fp = (2 * v * e + v) * FLOAT
            cells.append(E._cell('sp', 'validation_scalar_product', 'v4', cand, [k], fp, fp, l2, v, e, dict(vectorN=v, elementN=e, grid=g)))
    elif lib == 'classic':
        import cells as K
        w0, h0 = 5120, 3584
        for cand, (w, h) in (('c1', (w0, h0)), ('c2', (h0, w0))):
            assert w % 128 == 0 and h % 64 == 0
            ks = [dict(kid='conv_rows', grid=[w // 128, h // 4, 1], block=[16, 4, 1], sampling='classes', args=dict(imageW=w, imageH=h, pitch=w)),
                  dict(kid='conv_cols', grid=[w // 16, h // 64, 1], block=[16, 8, 1], sampling='classes', args=dict(imageW=w, imageH=h, pitch=w))]
            fp = 3 * w * h * FLOAT; logical = 4 * w * h * FLOAT
            cells.append(K._cell('conv', 'validation_separable_convolution', 'v5', cand, ks, fp, logical, l2, w, h, dict(imageW=w, imageH=h)))
    return cells
