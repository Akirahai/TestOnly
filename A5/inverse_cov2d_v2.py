"""2x2 PSD inverse on a5: mirrors 3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::inverse_cov2d_v2.

Contract:
    cov2_00, cov2_01, cov2_11 : float32 [N]   (1D batch of upper-triangle entries)
    inv_x_0,  inv_x_1,  inv_x_2 : float32 [N]

    det   = cov2_00 * cov2_11 - cov2_01 * cov2_01
    inv_0 =  cov2_11 / det
    inv_1 = -cov2_01 / det
    inv_2 =  cov2_00 / det

Topology: vec-only.
Tail-safety: arbitrary N.
Device: a5 (950).

The @kernel form internally takes [N, 1] GMTensors (matches the build_rotation
narrow-second-dim pattern); the public python entry point inverse_cov2d_v2 keeps
the 1D in / 1D out signature of the PyTorch reference by unsqueezing before
dispatch and squeezing the three returned tensors.
"""

from easyasc.a5 import *


CHUNK = 32      # rows per @vf call; UB tile height
IN_COLS = 1     # logical width of each input
PAD = 8         # float32 C0; one C0 block per row of UB


@vf()
def inverse_cov2d_vf(c00b: Tensor, c01b: Tensor, c11b: Tensor,
                     i0b: Tensor, i1b: Tensor, i2b: Tensor, rows: Var):
    c00 = Reg(DT.float)
    c01 = Reg(DT.float)
    c11 = Reg(DT.float)
    det = Reg(DT.float)
    tmp = Reg(DT.float)
    out = Reg(DT.float)

    for i in range(rows):
        c00 <<= c00b[i:i + 1, 0:1].single()
        c01 <<= c01b[i:i + 1, 0:1].single()
        c11 <<= c11b[i:i + 1, 0:1].single()

        # det = c00 * c11 - c01 * c01
        det <<= c00 * c11
        tmp <<= c01 * c01
        det <<= det - tmp

        # inv_0 = c11 / det
        out <<= c11 / det
        i0b[i:i + 1, 0:1] <<= out.single_value()

        # inv_1 = -c01 / det  (compute c01/det then negate via *-1.0)
        tmp <<= c01 / det
        out <<= tmp * -1.0
        i1b[i:i + 1, 0:1] <<= out.single_value()

        # inv_2 = c00 / det
        out <<= c00 / det
        i2b[i:i + 1, 0:1] <<= out.single_value()


@kernel()
def inverse_cov2d_v2_kernel(
    cov2_00: GMTensor, cov2_01: GMTensor, cov2_11: GMTensor,
    inv_00: GMTensor, inv_01: GMTensor, inv_11: GMTensor,
    N: Var,
):
    c00_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    c01_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    c11_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    i0_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    i1_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)
    i2_ub = DBuff(DT.float, [CHUNK, PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 1] -> UB [valid_rows, 8] (1 real + 7 junk per row).
            c00_ub[buf_cnt] <<= cov2_00[row0:row0 + valid_rows, 0:IN_COLS]
            c01_ub[buf_cnt] <<= cov2_01[row0:row0 + valid_rows, 0:IN_COLS]
            c11_ub[buf_cnt] <<= cov2_11[row0:row0 + valid_rows, 0:IN_COLS]

            inverse_cov2d_vf(
                c00_ub[buf_cnt], c01_ub[buf_cnt], c11_ub[buf_cnt],
                i0_ub[buf_cnt], i1_ub[buf_cnt], i2_ub[buf_cnt],
                valid_rows,
            )

            # UB [valid_rows, 8] -> GM [valid_rows, 1] (drop the 7 junk cols).
            inv_00[row0:row0 + valid_rows, 0:IN_COLS] <<= i0_ub[buf_cnt][0:valid_rows, 0:IN_COLS]
            inv_01[row0:row0 + valid_rows, 0:IN_COLS] <<= i1_ub[buf_cnt][0:valid_rows, 0:IN_COLS]
            inv_11[row0:row0 + valid_rows, 0:IN_COLS] <<= i2_ub[buf_cnt][0:valid_rows, 0:IN_COLS]

            buf_cnt += 1

    return inv_00, inv_01, inv_11


def inverse_cov2d_v2(cov2_00, cov2_01, cov2_11):
    """Public 1D wrapper: takes three 1D float32 tensors of shape [N], returns three 1D float32 tensors of shape [N].

    Uses shape-only unsqueeze/squeeze to bridge to the [N, 1] kernel layout. Mirrors
    `3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::inverse_cov2d_v2(c00, c01, c11, scale=1.0)`,
    with `scale` fixed to the only value used at the existing call site.
    """
    import torch

    assert cov2_00.shape == cov2_01.shape == cov2_11.shape, "all three inputs must share shape [N]"
    assert cov2_00.dim() == 1, "expected 1D inputs"
    N = cov2_00.shape[0]

    c00_2d = cov2_00.unsqueeze(-1).contiguous()
    c01_2d = cov2_01.unsqueeze(-1).contiguous()
    c11_2d = cov2_11.unsqueeze(-1).contiguous()

    i0_2d = torch.zeros_like(c00_2d)
    i1_2d = torch.zeros_like(c00_2d)
    i2_2d = torch.zeros_like(c00_2d)

    i0_out, i1_out, i2_out = OpExec(inverse_cov2d_v2_kernel, simulator=True)(
        c00_2d, c01_2d, c11_2d, i0_2d, i1_2d, i2_2d, N,
    )
    return i0_out.squeeze(-1), i1_out.squeeze(-1), i2_out.squeeze(-1)


if __name__ == "__main__":
    import torch

    def inverse_cov2d_v2_torch(c00, c01, c11):
        # Mirrors 3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::inverse_cov2d_v2 with scale=1.0.
        det = c00 * c11 - c01 * c01
        return c11 / det, -c01 / det, c00 / det

    torch.manual_seed(0)

    # Mix aligned, tail, and tiny shapes (same spread as build_rotation.py).
    for N in [64, 128, 100, 17, 1025]:
        # Build positive-definite-ish covariances with non-tiny det.
        c00 = (1.0 + torch.rand(N, dtype=torch.float32))
        c11 = (1.0 + torch.rand(N, dtype=torch.float32))
        c01 = 0.1 * torch.randn(N, dtype=torch.float32)

        i0_ref, i1_ref, i2_ref = inverse_cov2d_v2_torch(c00, c01, c11)

        i0_k, i1_k, i2_k = inverse_cov2d_v2(c00, c01, c11)

        torch.testing.assert_close(i0_k, i0_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(i1_k, i1_ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(i2_k, i2_ref, rtol=1e-4, atol=1e-4)

        d0 = torch.abs(i0_k - i0_ref).max().item()
        d1 = torch.abs(i1_k - i1_ref).max().item()
        d2 = torch.abs(i2_k - i2_ref).max().item()
        print(f"N={N:>5}  max_abs_diff: inv_0={d0:.3e}  inv_1={d1:.3e}  inv_2={d2:.3e}")
