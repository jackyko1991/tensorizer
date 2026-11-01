r"""Physics → tensor translation.

With gradients switched on, a spin at position x precesses at a frequency that depends on x, so by time t it has
picked up the phase 2\pi\,k(t)\cdot x with k(t) = \frac{\gamma}{2\pi}\int_0^t G\,dt'. The coil adds up all spins:

    s(k) = \sum_x \rho(x)\, e^{-2\pi i\, k\cdot x}

On an N × N grid (pixels p, r; k-space lines u, samples v) this is a linear map, the encoding matrix:

    s_{uv} = E_{up} E_{vr} \rho_{pr},      E_{up} = e^{-2\pi i u p / N}          (p, r summed)

which is exactly the 2D discrete Fourier transform, so s = FFT(ρ) and the image comes back as ρ = IFFT(s).
Missing lines (sampling mask m_{uv}) give the zero-filled image IFFT(m ⊙ s). With only some lines u ∈ U,
a regularised (Tikhonov) reconstruction solves, for every image column r, the least-squares problem

    (A^H A + \lambda W_r)\, \rho_{:r} = A^H h_{Ur} / \sqrt N,    A = E_{Up}/\sqrt N,   W_r = diag(1 outside the head)
"""

import torch


def encode(rho: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    r"""s_{uv} = E_{up} E_{vr} \rho_{pr}  (p, r summed).   rho: (N, N), E: (N, N) → (N, N) complex"""
    return torch.einsum("up,vr,pr->uv", E, E, rho.to(E.dtype))


def encode_fft(rho: torch.Tensor) -> torch.Tensor:
    """The same signal with the fast Fourier transform: O(N² log N) instead of O(N³)."""
    return torch.fft.fft2(rho)


def zero_filled(s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""\hat\rho = IFFT(m_{uv} s_{uv}): unmeasured samples set to zero.   s: (N, N), mask: (..., N, N) → (..., N, N)"""
    return torch.fft.ifft2(mask * s)


def regularised(
    s: torch.Tensor, lines: torch.Tensor, E: torch.Tensor, outside: torch.Tensor, lam: float
) -> torch.Tensor:
    r"""Solve (A^H A + \lambda W_r) \rho_{:r} = A^H h_{Ur} / \sqrt N for every column r (batched).   → (N, N) complex

    h_{ur} = IFFT_v(s_{uv}) = E_{up} \rho_{pr} is the data after transforming the fully sampled readout direction.
    W_r = diag(outside_{:r}) penalises signal where we know there is no tissue (outside the head); a tiny
    multiple of the identity keeps every system solvable.
    """
    n = E.shape[0]
    h = torch.fft.ifft(s[lines], dim=1)  # (M, N): row u = acquired line, column r = image column
    A = E[lines] / n**0.5  # (M, N) encoding along the phase-encode direction p
    AhA = A.conj().T @ A  # (N, N), shared by all columns
    W = torch.diag_embed(outside.T.to(E.dtype))  # (N_r, N_p, N_p): one diagonal weight matrix per column r
    eye = torch.eye(n, dtype=E.dtype, device=E.device)
    lhs = AhA + lam * W + 1e-6 * eye  # (N_r, N, N)
    rhs = (A.conj().T @ h / n**0.5).T[..., None]  # (N_r, N, 1): A^H h for each column
    return torch.linalg.solve(lhs, rhs)[..., 0].T  # (N_p, N_r)
