r"""Physics → tensor translation: incompressible Navier–Stokes in Fourier space (pseudo-spectral, rotational form).

    ∂_t û_i = P_ij(k) [u × ω]^_j − ν |k|² û_i,      P_ij = δ_ij − k_i k_j / |k|²,      ω̂_i = i ε_ijl k_j û_l

b indexes the batch of runs (one viscosity each), i, j, l the three components; repeated j, l are summed. Writing
(u·∇)u = ω × u + ∇(½|u|²) moves the gradient into the pressure, and the projector P removes every gradient, so the
pressure never has to be computed: P keeps the velocity divergence-free (k_i P_ij = 0).

Diagnostics in real space: E = ½⟨|u|²⟩, enstrophy Ω = ½⟨|ω|²⟩ (dE/dt = −2νΩ), ‖ω‖_∞ and the Beale–Kato–Majda
integral ∫₀^t ‖ω‖_∞ dt′, which must diverge if a smooth solution blows up at time t (BKM 1984).
"""

import torch

# Levi-Civita symbol ε_ijl: (3, 3, 3)
EPS = torch.zeros(3, 3, 3, dtype=torch.float64)
for _i, _j, _l in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
    EPS[_i, _j, _l], EPS[_i, _l, _j] = 1.0, -1.0


def curl_hat(u_hat: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """ω̂_bi = i ε_ijl k_j û_bl.   u_hat (B, 3, N, N, M) complex, k (3, N, N, M) → (B, 3, N, N, M)"""
    eps = EPS.to(device=k.device, dtype=u_hat.dtype)
    return 1j * torch.einsum("ijl,jxyz,blxyz->bixyz", eps, k.to(u_hat.dtype), u_hat)


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(a × b)_i = ε_ijl a_j b_l pointwise.   (B, 3, N, N, N) each"""
    return torch.einsum("ijl,bjxyz,blxyz->bixyz", EPS.to(device=a.device, dtype=a.dtype), a, b)


def project(f_hat: torch.Tensor, k: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """P_ij f̂_j = f̂_i − k_i (k_j f̂_j) / |k|²: removes the gradient (pressure) part.   (B, 3, N, N, M)"""
    kc = k.to(f_hat.dtype)
    k_dot_f = torch.einsum("jxyz,bjxyz->bxyz", kc, f_hat)  # k_j f̂_j
    return f_hat - kc[None] * (k_dot_f / torch.where(k2 == 0, 1.0, k2))[:, None]


def rhs(u_hat: torch.Tensor, nu: torch.Tensor, k: torch.Tensor, k2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """∂_t û = P[(u × ω)^] − ν|k|² û with the 2/3 dealiasing mask.   nu (B,) one viscosity per run."""
    n = k.shape[1]
    shape = (n, n, n)
    u = torch.fft.irfftn(u_hat, s=shape, dim=(-3, -2, -1))  # (B, 3, N, N, N)
    w = torch.fft.irfftn(curl_hat(u_hat, k), s=shape, dim=(-3, -2, -1))
    nonlinear = torch.fft.rfftn(cross(u, w), dim=(-3, -2, -1)) * mask  # (u × ω)^, aliased modes dropped
    return project(nonlinear, k, k2) - nu[:, None, None, None, None] * k2 * u_hat


def energy(u: torch.Tensor) -> torch.Tensor:
    """E_b = ½⟨u_i u_i⟩ over the box.   (B, 3, N, N, N) → (B,)"""
    return 0.5 * torch.einsum("bixyz,bixyz->b", u, u) / u[0, 0].numel()


def enstrophy(w: torch.Tensor) -> torch.Tensor:
    """Ω_b = ½⟨ω_i ω_i⟩.   (B, 3, N, N, N) → (B,)"""
    return energy(w)


def max_vorticity(w: torch.Tensor) -> torch.Tensor:
    """‖ω‖_∞ = max over the grid of |ω|.   (B, 3, N, N, N) → (B,)"""
    return torch.sqrt(torch.einsum("bixyz,bixyz->bxyz", w, w)).flatten(1).max(dim=1).values


def bkm_integral(t: torch.Tensor, wmax: torch.Tensor) -> torch.Tensor:
    """Beale–Kato–Majda ∫₀^t ‖ω‖_∞ dt′ by the trapezoid rule.   t (T,), wmax (T, B) → (T, B)"""
    steps = 0.5 * (wmax[1:] + wmax[:-1]) * (t[1:] - t[:-1])[:, None]
    return torch.cat([torch.zeros_like(wmax[:1]), torch.cumsum(steps, dim=0)])


def spectrum(u_hat: torch.Tensor, k: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """Shell-averaged energy spectrum E(κ) = Σ_{κ−½ ≤ |k| < κ+½} ½|û_k|² / N⁶.   (B, 3, N, N, M) → (B, K)"""
    n = k.shape[1]
    weight = torch.where((k[2] == 0) | (k[2] == n // 2), 1.0, 2.0)  # rfft stores each ±k_z pair once
    density = 0.5 * weight * (u_hat.abs() ** 2).sum(dim=1) / n**6  # (B, N, N, M)
    shell = torch.sqrt(k2).round().long().flatten()  # (N·N·M,)
    out = torch.zeros(u_hat.shape[0], int(shell.max()) + 1, dtype=density.dtype, device=density.device)
    return out.index_add_(1, shell, density.flatten(1))


def analyticity_strip(spec: torch.Tensor, k_lo: int, k_hi: int) -> torch.Tensor:
    """δ from a fit E(κ) ∝ κ^{-3} e^{−2δκ} on κ ∈ [k_lo, k_hi]: the distance from the real axis to the nearest
    complex singularity. A real singularity at time T* would need δ(T*) = 0.   (B, K) → (B,)"""
    kappa = torch.arange(k_lo, k_hi + 1, dtype=spec.dtype, device=spec.device)  # (K',)
    s = spec[:, k_lo : k_hi + 1]
    w = (s > 1e-26 * spec.max(dim=1, keepdim=True).values).to(spec.dtype)  # fit only shells above round-off
    y = torch.log(s.clamp_min(1e-300)) + 3 * torch.log(kappa)  # (B, K')
    n = w.sum(dim=1)
    k_mean = torch.einsum("bk,k->b", w, kappa) / n.clamp_min(1)
    y_mean = torch.einsum("bk,bk->b", w, y) / n.clamp_min(1)
    kc = kappa[None] - k_mean[:, None]
    slope = torch.einsum("bk,bk,bk->b", w, kc, y - y_mean[:, None]) / torch.einsum("bk,bk,bk->b", w, kc, kc)
    return torch.where(n >= 3, -slope / 2, torch.inf)  # < 3 shells excited: no measurable tail yet (δ = ∞)


def sample_periodic(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Trilinear interpolation of a periodic vector field at arbitrary points.
    f (F, 3, N, N, N) on the grid x_j = 2πj/N, x (F, S, 3) positions → (F, S, 3). The 8 corner weights are products
    w = Π_i (1 − |x_i/Δ − c_i|) over the cell corners c, gathered with wrap-around indices."""
    n = f.shape[-1]
    g = x * (n / (2 * torch.pi))  # grid units
    base = torch.floor(g)
    frac = g - base
    base = base.long()
    frames = torch.arange(f.shape[0], device=f.device)[:, None]  # (F, 1)
    out = torch.zeros_like(x)
    for corner in range(8):
        c = torch.tensor([(corner >> 2) & 1, (corner >> 1) & 1, corner & 1], device=x.device)  # (3,)
        idx = (base + c) % n  # (F, S, 3)
        w = torch.where(c.bool(), frac, 1 - frac).prod(dim=-1)  # (F, S)
        val = f[frames, :, idx[..., 0], idx[..., 1], idx[..., 2]]  # (F, S, 3)
        out = out + w[..., None] * val
    return out


def vortex_lines(w: torch.Tensor, seeds: torch.Tensor, n_steps: int, h: float) -> torch.Tensor:
    """Field lines of ω (dx/ds = ω/|ω|), traced both ways from every seed with midpoint (RK2) steps of length h.
    w (F, 3, N, N, N), seeds (F, S, 3) → (F, S, 2·n_steps + 1, 3), wrapped into [0, 2π)³; all frames and seeds at
    once."""

    def direction(x: torch.Tensor) -> torch.Tensor:
        v = sample_periodic(w, x % (2 * torch.pi))
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    branches = []
    for sign in (1.0, -1.0):
        x, path = seeds, []
        for _ in range(n_steps):
            mid = x + 0.5 * sign * h * direction(x)
            x = x + sign * h * direction(mid)
            path.append(x)
        branches.append(torch.stack(path, dim=2))  # (F, S, n_steps, 3)
    line = torch.cat([branches[1].flip(2), seeds[:, :, None], branches[0]], dim=2)
    return line % (2 * torch.pi)
