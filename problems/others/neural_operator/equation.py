r"""Physics → tensor translation.

The physics is the viscous Burgers equation on a periodic domain x ∈ [0, 2π),

    u_t + u u_x = ν u_xx      ⇔      û_t(k) = -ν k² û(k) - (ik/2) FFT[u²](k)

written in Fourier space (k = wavenumber), where ∂_x becomes multiplication by ik.

A neural OPERATOR learns the map G: u_0(·) ↦ u(·, T) for a whole family of initial conditions. A Fourier
neural operator (FNO) acts on a function sampled at N grid points x, with indices
b = sample in the batch, i/o = input/output channel, x = grid point, k = Fourier mode:

    lift           v_{bcx}   = P_{cf} a_{bfx} + p_c                      f summed   (a = [u_0, x/2π])
    Fourier layer  v'_{box}  = σ( W_{oi} v_{bix} + b_o + (K v)_{box} )   i summed
    spectral conv  (K v)     = IFFT_k[ R_{iok} v̂_{bik} ]  for k < m,     i summed, modes k ≥ m dropped
    project        u_{bx}    = Q2_{h} σ( Q1_{hc} v_{bcx} + q1_h ) + q2   c, h summed

The weights R_{iok} are complex numbers that live on Fourier MODES, not on grid points: the same
weights work on any grid with more than 2m points. That is why the FNO is discretisation invariant.
"""

import torch
import torch.nn.functional as F

# --- the PDE (used by the data solver) ---------------------------------------------------------------


def burgers_nonlinear(u_hat: torch.Tensor, k: torch.Tensor, dealias: torch.Tensor, n: int) -> torch.Tensor:
    r"""-(ik/2) FFT[u²]: the advection term u u_x = (½u²)_x in Fourier space.   u_hat: (B, K) → (B, K).

    u² is formed on the grid (pseudo-spectral); modes |k| > N/3 are zeroed (2/3 rule) against aliasing.
    """
    u = torch.fft.irfft(u_hat, n=n)  # (B, N) back to grid values
    return -0.5j * k * dealias * torch.fft.rfft(u**2)


def mass(u: torch.Tensor, dx: float) -> torch.Tensor:
    r"""∫ u dx ≈ Σ_x u_x Δx — conserved: the k = 0 mode is untouched by ik(·) and by νk²."""
    return u.sum(dim=-1) * dx


# --- the Fourier neural operator ---------------------------------------------------------------------


def lift(a: torch.Tensor, P: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """v_{bcx} = P_{cf} a_{bfx} + p_c   (f summed): 2 input features → C channels at every point x."""
    return torch.einsum("cf,bfx->bcx", P, a) + p[:, None]


def spectral_conv(v: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""(K v)_{box} = IFFT[ R_{iok} v̂_{bik} ]   (i summed), keeping only the lowest m = R.shape[-1] modes.

    v: (B, C, N) real,  R: (C, C, m) complex  →  (B, C, N) real.
    """
    n, m = v.shape[-1], R.shape[-1]
    v_hat = torch.fft.rfft(v)  # (B, C, N//2 + 1) complex Fourier coefficients
    out_hat = torch.zeros_like(v_hat)  # modes k ≥ m stay zero: the truncation
    out_hat[..., :m] = torch.einsum("bik,iok->bok", v_hat[..., :m], R)  # mix channels, mode by mode
    return torch.fft.irfft(out_hat, n=n)  # back to the same N grid points


def fourier_layer(v: torch.Tensor, R: torch.Tensor, W: torch.Tensor, b: torch.Tensor, last: bool) -> torch.Tensor:
    """v' = σ(W v + b + K v): a local (pointwise) path plus a global (spectral) path.   (B, C, N) → (B, C, N)."""
    local = torch.einsum("oi,bix->box", W, v) + b[:, None]  # i summed, same W at every x
    out = local + spectral_conv(v, R)
    return out if last else F.gelu(out)  # σ = GELU (no activation after the last layer)


def project(v: torch.Tensor, Q1, q1, Q2, q2) -> torch.Tensor:
    """u_{bx} = Q2_h σ(Q1_{hc} v_{bcx} + q1_h) + q2   (c, h summed): C channels → 1 value per point."""
    h = F.gelu(torch.einsum("hc,bcx->bhx", Q1, v) + q1[:, None])
    return torch.einsum("h,bhx->bx", Q2, h) + q2


def fno(u0: torch.Tensor, x: torch.Tensor, w: dict) -> torch.Tensor:
    """The operator G_w: u_0 sampled on ANY periodic grid x → u(·, T) on the same grid.   u0: (B, N), x: (N,)."""
    grid = (x / (2 * torch.pi)).expand_as(u0)  # (B, N) the position as a second input feature
    a = torch.stack([u0, grid], dim=1)  # (B, 2, N)
    v = lift(a, w["P"], w["p"])  # (B, C, N)
    n_layers = len(w["R"])
    for layer in range(n_layers):
        v = fourier_layer(v, w["R"][layer], w["W"][layer], w["b"][layer], last=layer == n_layers - 1)
    return project(v, w["Q1"], w["q1"], w["Q2"], w["q2"])  # (B, N)


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """‖pred - true‖ / ‖true‖ per sample, averaged over the batch: the training loss.   (B, N) → ()."""
    return (torch.linalg.norm(pred - true, dim=-1) / torch.linalg.norm(true, dim=-1)).mean()
