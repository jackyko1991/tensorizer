r"""Physics → tensor translation.

Every pixel is a mix of basis materials m (water, bone, iodine) with amounts a_{myx}, so its attenuation at
energy E_e is μ_{eyx} = μ_{em} a_{myx}. Along a ray r (one angle θ, one detector bin s) only the basis line
integrals A_{rm} = ∫ a_m dl matter, and Beer–Lambert for a polychromatic beam counted in energy bin b reads

    λ_{rb} = N_0\, R_{be}\, S_e \exp(-\mu_{em} A_{rm})          (e, m summed)

Indices: r = ray, b = detector energy bin, e = photon energy, m = basis material.
S_e is the tube spectrum (sums to 1), R_{be} the probability that a photon of energy E_e lands in bin b.
The counts are Poisson, y_{rb} ~ Poisson(λ_{rb}), so the decomposition minimises the negative log-likelihood

    \mathcal L(A) = \sum_{rb} \lambda_{rb}(A) - y_{rb} \ln \lambda_{rb}(A)

ray by ray (all rays in one tensor). Its Fisher information, used for the Newton-like steps, is

    F_{rmn} = J_{rbm} J_{rbn} / \lambda_{rb},   J_{rbm} = \partial\lambda_{rb}/\partial A_{rm}   (b summed)

A conventional energy-integrating detector adds up the deposited energy instead:  I_r = N_0 E_e S_e T_{re}.
The Radon transform and filtered back-projection are the same as in radon_ct.
"""

import torch
import torch.nn.functional as F


def line_integrals_mu(mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""\int \mu(E_e)\,dl = \mu_{em} A_{rm}.   mu (E, M), A (R, M) → (R, E);  m summed."""
    return torch.einsum("em,rm->re", mu, A)


def expected_counts(n0: float, s: torch.Tensor, R: torch.Tensor, mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""Beer–Lambert per bin: λ_{rb} = N_0 R_{be} S_e \exp(-\mu_{em}A_{rm}).   → (R, B);  e summed."""
    T = torch.exp(-line_integrals_mu(mu, A))  # (R, E) transmission at each energy
    return n0 * torch.einsum("be,e,re->rb", R, s, T)


def poisson_nll(y: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    r"""Negative Poisson log-likelihood per ray, \sum_b (λ_{rb} − y_{rb} \ln λ_{rb}) (constant term dropped): (R,)."""
    return (lam - y * torch.log(lam)).sum(-1)


def jacobian(n0: float, s: torch.Tensor, R: torch.Tensor, mu: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    r"""J_{rbm} = ∂λ_{rb}/∂A_{rm} = −N_0 R_{be} S_e T_{re} μ_{em}.   → (R, B, M);  e summed."""
    T = torch.exp(-line_integrals_mu(mu, A))
    return -n0 * torch.einsum("be,e,re,em->rbm", R, s, T, mu)


def fisher(J: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    r"""F_{rmn} = J_{rbm} J_{rbn} / λ_{rb}: expected curvature of the NLL, one M×M matrix per ray.  b summed."""
    return torch.einsum("rbm,rbn,rb->rmn", J, J, 1.0 / lam)


def energy_integrating(n0: float, s: torch.Tensor, e: torch.Tensor, mu: torch.Tensor, A: torch.Tensor):
    r"""Conventional detector, signal ∝ deposited energy: I_r = N_0 E_e S_e T_{re} (e summed) → −ln(I/I_0), (R,)."""
    T = torch.exp(-line_integrals_mu(mu, A))
    signal = n0 * torch.einsum("e,e,re->r", e, s, T)
    return -torch.log(signal / (n0 * (e * s).sum()))


def monoenergetic(mu_at_e: torch.Tensor, maps: torch.Tensor) -> torch.Tensor:
    r"""Virtual monoenergetic image μ_{yx}(E) = μ_m(E) a_{myx}.   mu_at_e (M,) or (K, M), maps (M, Y, X);  m summed."""
    return torch.einsum("...m,myx->...yx", mu_at_e, maps)


def hounsfield(mu_map: torch.Tensor, mu_water: torch.Tensor) -> torch.Tensor:
    """CT number HU = 1000 (μ − μ_water) / μ_water."""
    return 1000.0 * (mu_map - mu_water) / mu_water


# --- Radon transform and filtered back-projection (as in radon_ct), batched over a leading channel axis ---


def radon(f: torch.Tensor, theta: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    r"""p_{c\theta s} = f_{c\theta ts}\,\Delta t  for C images at once.  f (C, N, N) → (C, A, S), t summed."""
    c, sn = torch.cos(theta)[:, None, None], torch.sin(theta)[:, None, None]  # (A, 1, 1)
    x = s[None, None, :] * c - s[None, :, None] * sn  # (A, T, S): sample points of every ray
    y = s[None, None, :] * sn + s[None, :, None] * c
    points = torch.stack([x, y], dim=-1)
    image = f[None].expand(len(theta), *f.shape)  # (A, C, N, N)
    samples = F.grid_sample(image, points, mode="bilinear", align_corners=False)  # (A, C, T, S)
    return torch.einsum("acts->cas", samples) * (s[1] - s[0])


def ramp_filter(n_det: int, ds: float, device=None, dtype=torch.float64) -> tuple[torch.Tensor, int]:
    """Band-limited ramp |ν| × sinc(ν Δs) (Shepp–Logan window) on a zero-padded detector of P ≥ 2S bins."""
    pad = 1 << (2 * n_det - 1).bit_length()
    idx = torch.arange(pad, device=device)
    n = torch.where(idx < pad // 2, idx, idx - pad).to(dtype)
    h = torch.where(n % 2 == 1, -1.0 / (torch.pi * n * ds) ** 2, torch.zeros_like(n))
    h[0] = 1.0 / (4 * ds**2)
    nu = torch.fft.fftfreq(pad, d=ds, device=device).to(dtype)
    return ds * torch.fft.fft(h).real * torch.sinc(nu * ds), pad


def smear_filtered(p: torch.Tensor, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""Ramp-filter each projection and spread it back: Q_{c\theta yx} = q_{c\theta}(x\cos\theta + y\sin\theta).

    p (C, A, S) → (C, A, Y, X). Summing over θ with Δθ = π/A gives the filtered back-projection.
    """
    H, pad = ramp_filter(p.shape[-1], float(x[1] - x[0]), p.device, p.dtype)
    q = torch.fft.ifft(torch.fft.fft(p, n=pad, dim=-1) * H, dim=-1).real[..., : p.shape[-1]]  # (C, A, S)
    s = x[None, None, :] * torch.cos(theta)[:, None, None] + x[None, :, None] * torch.sin(theta)[:, None, None]
    points = torch.stack([s, torch.zeros_like(s)], dim=-1)  # (A, Y, X, 2)
    q_img = q.permute(1, 0, 2)[:, :, None, :]  # (A, C, 1, S): one 1-pixel-high image per angle
    return F.grid_sample(q_img, points, mode="bilinear", align_corners=False).permute(1, 0, 2, 3)


def backproject(Q: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    r"""f_{cyx} = Q_{c\theta yx}\,\Delta\theta  (θ summed).   (C, A, Y, X) → (C, Y, X)"""
    return torch.einsum("cayx,a->cyx", Q, torch.full_like(theta, torch.pi / len(theta)))
