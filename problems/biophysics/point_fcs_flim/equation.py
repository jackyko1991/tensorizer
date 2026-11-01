r"""Physics → tensor translation for confocal point FCS with lifetime (FLIM/TCSPC) information.

Detection volume. A molecule n at position x_{ni} (i = x, y, z, measured from the focus) is seen with weight

    W_n = exp(-x_{ni} A_{ij} x_{nj}),     A = diag(2/w_0², 2/w_0², 2/z_0²)        (i, j summed)

and the photon rate of species s sums the molecules that belong to it (membership S_{ns} = 1 or 0):

    I_{ts} = q_s S_{ns} W_{tn}                                                    (n summed)

Correlation. With δI_t = I_t - ⟨I⟩ the autocorrelation G(τ) = ⟨δI_t δI_{t+τ}⟩ / ⟨I⟩² is computed for every lag at once
with the FFT (Wiener–Khinchin: the autocorrelation is the inverse transform of the power spectrum |Î_ω|²).
For free 3D diffusion through the Gaussian volume

    G(τ) = (1/N) (1 + τ/τ_D)^{-1} (1 + τ/(κ²τ_D))^{-1/2},    τ_D = w_0² / (4D)

Lifetime filters (FLCS). The microtime histogram h_c is a mix of normalised decay patterns M_{cj} (one per species,
plus a flat background). Filters f_{kc} with f_{kc} M_{cj} = δ_{kj} (c summed) turn the photon stream into one
intensity trace per species.
"""

import math

import torch


def effective_volume(w0: float, z0: float) -> float:
    """V_eff = π^{3/2} w_0² z_0 (µm³): N = c V_eff molecules on average 'in' the Gaussian volume."""
    return math.pi**1.5 * w0**2 * z0


def shape_matrix(w0: float, z0: float, device: torch.device) -> torch.Tensor:
    """A_{ij} = diag(2/w_0², 2/w_0², 2/z_0²), shape (3, 3)."""
    return torch.diag(torch.tensor([2 / w0**2, 2 / w0**2, 2 / z0**2], device=device))


def detection(x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """W = exp(-x_i A_ij x_j) for positions x of shape (..., 3) → (...).   i, j summed."""
    return torch.exp(-torch.einsum("...i,ij,...j->...", x, A, x))


def species_rates(W: torch.Tensor, S: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """I_{ts} = q_s S_{ns} W_{tn}:  W (T, n), S (n, s), q (s,) → (T, s) photons per second.   n summed."""
    return torch.einsum("tn,ns,s->ts", W, S, q)


def peak_brightness(eps: torch.Tensor) -> torch.Tensor:
    """q = 2^{3/2} ε: rate at the very centre, from the usual counts per molecule ε = ⟨I⟩ / N."""
    return 2**1.5 * eps


def autocorrelation(trace: torch.Tensor) -> torch.Tensor:
    """G_k(τ) = ⟨δI_{k,t} δI_{k,t+τ}⟩_t / ⟨I_k⟩² for every lag τ = 0 … T-1.   trace I: (K, T) → (K, T).

    Zero-padding to 2T makes the circular FFT correlation equal to the ordinary (linear) one; dividing by
    T - τ, the number of pairs at lag τ, gives the unbiased average.
    """
    T = trace.shape[-1]
    mean = trace.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(trace - mean, n=2 * T)
    sums = torch.fft.irfft(spectrum * spectrum.conj(), n=2 * T)[..., :T]  # Σ_t δI_t δI_{t+τ}
    pairs = T - torch.arange(T, device=trace.device, dtype=trace.dtype)
    return sums / pairs / mean**2


def diffusion_model(lag: torch.Tensor, tau_D: torch.Tensor, kappa: float) -> torch.Tensor:
    """g_{ls} = (1 + τ_l/τ_{D,s})^{-1} (1 + τ_l/(κ²τ_{D,s}))^{-1/2}:  lag (L,), tau_D (..., S) → (..., L, S)."""
    r = lag[:, None] / tau_D[..., None, :]
    return 1 / (1 + r) / torch.sqrt(1 + r / kappa**2)


def diffusion_coefficient(w0: float, tau_D: torch.Tensor) -> torch.Tensor:
    """D = w_0² / (4 τ_D)."""
    return w0**2 / (4 * tau_D)


def background_correction(total_rate: float, background_rate: float) -> float:
    """Uncorrelated background dilutes G by ((I - B)/I)²; multiply by the inverse to undo it."""
    return (total_rate / (total_rate - background_rate)) ** 2


def decay_pattern(t: torch.Tensor, tau: torch.Tensor, t0: float, sigma: float, period: float) -> torch.Tensor:
    """Exponential decay convolved with a Gaussian IRF (centre t0, width σ), normalised over the channels.

    t: channel centres (C,), tau: (..., S) → (..., C, S). The analytic convolution is the exponentially modified
    Gaussian  p(t) ∝ exp(σ²/2τ² - (t - t0)/τ) erfc((σ²/τ - (t - t0)) / (√2 σ)); the tails of the two previous laser
    pulses (t + period, t + 2 period) are added because a long lifetime does not finish within one period.
    """
    tau = tau[..., None, :]  # (..., 1, S)
    p = 0
    for m in range(3):
        u = t[:, None] - t0 + m * period  # (C, 1)
        p = p + torch.exp(sigma**2 / (2 * tau**2) - u / tau) * torch.special.erfc(
            (sigma**2 / tau - u) / (2**0.5 * sigma)
        )
    return p / p.sum(-2, keepdim=True)


def flcs_filters(M: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Lifetime filters F = (Mᵀ D⁻¹ M)⁻¹ Mᵀ D⁻¹ with D = diag(h):  M (C, J), h (C,) → F (J, C).

    Weighted least squares with the Poisson variance h_c of every channel; by construction F M = 1, i.e.
    Σ_c f_{kc} M_{cj} = δ_{kj}: filter k counts one photon per photon of species k and zero for the others.
    """
    Dinv = 1 / h.clamp(min=1)
    MtDM = torch.einsum("cj,c,ck->jk", M, Dinv, M)  # (J, J), c summed
    return torch.linalg.solve(MtDM, M.T * Dinv)  # (J, C)
