"""Bounding-circle radius of a projected 2D Gaussian on a5: mirrors
3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::get_radius.

Contract:
    cov2d : float32 [N, 2, 2]   per-Gaussian 2x2 covariance
    radii : float32 [N]         3 * sqrt(max(lambda1, lambda2)) then ceil

    det  = c00*c11 - c01*c10
    mid  = 0.5 * (c00 + c11)
    d    = mid*mid - det
    root = sqrt(max(d, 0.1))
    sqrt_lambda_max = sqrt(mid + root)             # kernel writes this
    radius          = 3.0 * ceil(sqrt_lambda_max)  # applied host-side

The pytorch reference is `3.0 * torch.sqrt(torch.max(lam1, lam2)).ceil()`,
i.e. ceil(sqrt(...)) first, then * 3.0. To match exactly the kernel emits
sqrt(lambda_max) (no `* 3.0` inside) and the wrapper does `.ceil() * 3.0`.

Simplification: clip(min=0.1) makes root > 0 strictly, so lambda1 = mid + root
> mid - root = lambda2 always, and max(lambda1, lambda2) = mid + root. The
explicit two-eigenvalue max op is therefore dropped.

Topology: vec-only.
Tail-safety: arbitrary N. Second dim fixed at 4 (input flat) / 1 (output).
Device: a5 (950).

The @kernel takes a [N, 4] flattened cov2d (cols = c00, c01, c10, c11) and
writes [N, 1] pre-ceil radii. The public python entry point get_radius keeps
the [N, 2, 2] in / [N] out signature of the PyTorch reference: a shape-only
reshape feeds the kernel, then squeeze + ceil produces the final radii.
"""

from easyasc.a5 import *


CHUNK = 32      # rows per @vf call; UB tile height
IN_COLS = 4     # c00, c01, c10, c11
OUT_COLS = 1    # radius_pre_ceil
IN_PAD = 8      # float32 C0; one C0 block per row of UB
OUT_PAD = 8     # one C0 block per row of UB


@vf()
def get_radius_vf(cbuf: Tensor, rbuf: Tensor, rows: Var):
    c00 = Reg(DT.float)
    c01 = Reg(DT.float)
    c10 = Reg(DT.float)
    c11 = Reg(DT.float)
    t = Reg(DT.float)
    s = Reg(DT.float)
    out = Reg(DT.float)

    for i in range(rows):
        c00 <<= cbuf[i:i + 1, 0:1].single()
        c01 <<= cbuf[i:i + 1, 1:2].single()
        c10 <<= cbuf[i:i + 1, 2:3].single()
        c11 <<= cbuf[i:i + 1, 3:4].single()

        # det = c00*c11 - c01*c10
        t <<= c00 * c11
        s <<= c01 * c10
        t <<= t - s

        # mid = 0.5 * (c00 + c11)
        s <<= c00 + c11
        s <<= s * 0.5

        # d_safe = max(mid*mid - det, 0.1); root = sqrt(d_safe)
        out <<= s * s
        out <<= out - t
        out <<= out.vmaxs(0.1)         # clip(min=0.1) -> root > 0
        out <<= out.sqrt()

        # lambda_max = mid + root  (root > 0 makes lambda1 >= lambda2 always)
        out <<= out + s

        # sqrt(lambda_max). The * 3.0 happens host-side after .ceil() to match
        # the pytorch reference order: 3.0 * torch.sqrt(...).ceil().
        out <<= out.sqrt()

        rbuf[i:i + 1, 0:1] <<= out.single_value()


@kernel()
def get_radius_kernel(cov2d_flat: GMTensor, radii: GMTensor, N: Var):
    cbuf = DBuff(DT.float, [CHUNK, IN_PAD], Position.UB)
    rbuf = DBuff(DT.float, [CHUNK, OUT_PAD], Position.UB)

    buf_cnt = Var(0)

    total_chunks = CeilDiv(N, CHUNK)
    chunks_per_core = CeilDiv(total_chunks, GetVecNum())
    chunk_begin = Var(chunks_per_core * GetVecIdx())
    chunk_end = Min(chunk_begin + chunks_per_core, total_chunks)

    with auto_sync():
        for chunk_idx in range(chunk_begin, chunk_end):
            row0 = Var(chunk_idx * CHUNK)
            valid_rows = Min(CHUNK, N - row0)

            # GM [valid_rows, 4] -> UB [valid_rows, 8] (4 real + 4 junk per row).
            cbuf[buf_cnt] <<= cov2d_flat[row0:row0 + valid_rows, 0:IN_COLS]

            get_radius_vf(cbuf[buf_cnt], rbuf[buf_cnt], valid_rows)

            # UB [valid_rows, 8] -> GM [valid_rows, 1] (drop the 7 junk cols).
            radii[row0:row0 + valid_rows, 0:OUT_COLS] <<= rbuf[buf_cnt][0:valid_rows, 0:OUT_COLS]

            buf_cnt += 1

    return radii


def get_radius(cov2d):
    """Public wrapper: takes float32 [N, 2, 2] cov2d, returns float32 [N] radii.

    Mirrors 3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::get_radius. Uses a
    shape-only reshape ([N, 2, 2] -> [N, 4]) to bridge to the kernel layout, then
    applies the final .ceil() host-side because a5 has no scalar ceil primitive.
    """
    import torch

    assert cov2d.dim() == 3 and cov2d.shape[1] == 2 and cov2d.shape[2] == 2, \
        "expected cov2d shape [N, 2, 2]"
    assert cov2d.dtype == torch.float32, "expected float32 cov2d"
    N = cov2d.shape[0]

    cov_flat = cov2d.reshape(N, 4).contiguous()
    radii_2d = torch.zeros((N, 1), dtype=torch.float32)

    sqrt_lam = OpExec(get_radius_kernel, simulator=True)(cov_flat, radii_2d, N)
    return 3.0 * sqrt_lam.squeeze(-1).ceil()


if __name__ == "__main__":
    import torch

    def get_radius_torch(cov2d):
        # Mirrors 3DGS-opts/pytorch/EWA_fully_fused_proj_packed.py::get_radius.
        det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] * cov2d[:, 1, 0]
        mid = 0.5 * (cov2d[:, 0, 0] + cov2d[:, 1, 1])
        lam1 = mid + torch.sqrt((mid ** 2 - det).clip(min=0.1))
        lam2 = mid - torch.sqrt((mid ** 2 - det).clip(min=0.1))
        return 3.0 * torch.sqrt(torch.max(lam1, lam2)).ceil()

    torch.manual_seed(0)

    # Test budget: N <= 60, at most 2 cases. User runs larger N themselves.
    for N in [17, 60]:
        # PSD-ish symmetric 2x2: A A^T + 0.5 I keeps det non-tiny and conditioned.
        A = 0.5 * torch.randn((N, 2, 2), dtype=torch.float32)
        cov2d = A @ A.transpose(-1, -2) + 0.5 * torch.eye(2, dtype=torch.float32)

        ref = get_radius_torch(cov2d)
        out = get_radius(cov2d)

        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
        print(f"N={N:>3}  max_abs_diff={torch.abs(out - ref).max().item():.3e}")
