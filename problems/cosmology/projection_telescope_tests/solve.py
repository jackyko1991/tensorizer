"""Solvers: the posterior of D and its autograd fit, the screening scale, ray-traced shadows, the fold bounds and the
supernova leakage fit."""

import math

import torch

from . import equation

F64 = torch.float64
D_TABLE = torch.linspace(0.05, 200.0, 4000, dtype=F64)  # d_GW grid (Mpc) for the tabulated GW likelihood


def likelihood_table(gw: dict) -> torch.Tensor:
    """L(d_GW) on D_TABLE: the kernel density is evaluated once, then interpolated."""
    return equation.gw_likelihood(D_TABLE.to(gw["d"].device), gw["d"], gw["w"], gw["h"])


def em_grid(em: dict, n: int = 401):
    """Grid over the electromagnetic distance (±6σ) and its Gaussian weights for the marginalisation."""
    d = torch.linspace(em["d"] - 6 * em["sigma"], em["d"] + 6 * em["sigma"], n, dtype=F64).clamp_min(0.5)
    return d, equation.gaussian(d, em["d"], em["sigma"])


def posterior_D(Ds: torch.Tensor, gw: dict, em: dict, table=None) -> dict:
    """p(D) on the grid Ds (uniform prior), its MAP and minimal 68% interval (highest density)."""
    table = likelihood_table(gw) if table is None else table
    d, g = em_grid(em)
    L = equation.interp(equation.gw_distance(d[None, :], Ds[:, None]), D_TABLE, table)  # (nD, nd)
    like = torch.trapezoid(L * g, d, dim=1)  # ∫ L(d_GW(d, D)) N(d) dd
    pdf = like / torch.trapezoid(like, Ds)
    order = pdf.argsort(descending=True)  # highest-density interval: add grid points from the top down
    mass = (pdf[order] * (Ds[1] - Ds[0])).cumsum(0)
    inside = order[: int((mass < 0.6827).sum()) + 1]
    return {"D": Ds, "pdf": pdf, "map": float(Ds[pdf.argmax()]),
            "interval": (float(Ds[inside].min()), float(Ds[inside].max())), "like": like}  # fmt: skip


def log_posterior(D: torch.Tensor, gw: dict, em: dict) -> torch.Tensor:
    """Smooth log p(D) (direct kernel sum, so autograd gives first and second derivatives)."""
    d, g = em_grid(em, 201)
    logL = equation.gw_log_likelihood(equation.gw_distance(d, D), gw["d"], gw["w"], gw["h"])  # (nd,)
    return torch.logsumexp(logL + torch.log(g * (d[1] - d[0])), 0)  # log ∫ L N dd (rectangle rule)


def derivatives(fn, x: torch.Tensor):
    """Value, gradient and Hessian of a scalar function by autograd (x: (P,))."""
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    (g,) = torch.autograd.grad(y, x, create_graph=True)
    H = torch.stack([torch.autograd.grad(g[i], x, retain_graph=True)[0] for i in range(len(x))])
    return y.detach(), g.detach(), H.detach()


def newton_max(fn, x0: torch.Tensor, steps: int = 40, max_step: float = 0.2) -> torch.Tensor:
    """Maximise fn by damped Newton steps on autograd derivatives (a gradient step where fn is not concave)."""
    x = x0.clone()
    for _ in range(steps):
        _, g, H = derivatives(fn, x)
        concave = bool((torch.linalg.eigvalsh(H) < 0).all())
        step = -torch.linalg.solve(H, g) if concave else 0.05 * g / g.norm().clamp_min(1e-12)
        x = x + step.clamp(-max_step, max_step)
        if step.abs().max() < 1e-9:
            break
    return x


def fit_D(gw: dict, em: dict, D0: float = 4.5) -> dict:
    """MAP of D by Newton's method on autograd derivatives of log p(D), and its Laplace error (-d²log p/dD²)^-1/2."""
    fn = lambda x: log_posterior(x[0], gw, em)  # noqa: E731
    D = newton_max(fn, torch.tensor([D0], dtype=F64))
    _, g, H = derivatives(fn, D)
    return {"D": float(D[0]), "sigma": float((-H[0, 0]).clamp_min(1e-12).rsqrt()), "grad": float(g[0])}


def screening_bound(D_vals: torch.Tensor, n_vals: torch.Tensor, gw: dict, em: dict, table=None) -> dict:
    """Smallest allowed R_c for each (D, n): where -2 ln[L(R_c)/L(no leakage)] drops below 2.71 (95%, one-sided)."""
    table = likelihood_table(gw) if table is None else table
    d, g = em_grid(em, 201)
    Rc = torch.logspace(-1, 4, 400, dtype=F64)  # 0.1 ... 10^4 Mpc
    dgw = equation.gw_distance(d, D_vals[:, None, None, None], Rc[None, None, :, None], n_vals[None, :, None, None])
    like = torch.trapezoid(equation.interp(dgw, D_TABLE, table) * g, d, dim=-1)  # (nD, nn, nR)
    like0 = torch.trapezoid(equation.interp(d, D_TABLE, table) * g, d)
    dchi2 = -2 * torch.log(like.clamp_min(1e-300) / like0)
    excluded = dchi2 > 2.71
    last = torch.where(excluded, torch.arange(len(Rc)), torch.full_like(excluded, -1, dtype=torch.long)).amax(-1)
    bound = torch.where(last >= 0, Rc[(last + 1).clamp(max=len(Rc) - 1)], Rc[0])
    return {"Rc": Rc, "dchi2": dchi2, "bound": bound, "D": D_vals, "n": n_vals}


# ---------------------------------------------------------------- shadows
def trace(b: torch.Tensor, r_cap: torch.Tensor, phi_max: float = 6 * math.pi, dphi: float = 2e-3) -> torch.Tensor:
    """Ray-trace photons of impact parameter b (units r_s) inward from infinity; True where captured.

    b, r_cap broadcast to (F, B). Integrates (u, u') with RK4 in φ; a ray stops when it reaches u = 1/r_cap
    (captured) or comes back to u = 0 (escaped)."""
    b, r_cap = torch.broadcast_tensors(b, r_cap)
    y = torch.stack([torch.zeros_like(b), 1 / b], -1)  # (F, B, 2) start at infinity
    alive = torch.ones_like(b, dtype=torch.bool)
    captured = torch.zeros_like(alive)
    for _ in range(int(phi_max / dphi)):
        k1 = equation.photon_rhs(y)
        k2 = equation.photon_rhs(y + 0.5 * dphi * k1)
        k3 = equation.photon_rhs(y + 0.5 * dphi * k2)
        k4 = equation.photon_rhs(y + dphi * k3)
        y = torch.where(alive[..., None], y + dphi / 6 * (k1 + 2 * k2 + 2 * k3 + k4), y)
        hit = alive & (y[..., 0] >= 1 / r_cap)
        out = alive & (y[..., 0] < 0)
        captured |= hit
        alive &= ~(hit | out)
        if not alive.any():
            break
    return captured


def traced_shadow_edge(r_f: torch.Tensor, n: int = 400) -> torch.Tensor:
    """Shadow radius (units r_s) by ray tracing, for folds at r_f (<= 1: none, capture at the horizon).  (F,)

    Two passes: a coarse grid of b, then a fine one around the first escaping ray."""
    r_cap = r_f.clamp_min(1.0)[:, None]
    lo = torch.full_like(r_f, 2.3)
    hi = torch.maximum(torch.full_like(r_f, 3.0), 1.3 * r_cap[:, 0])
    for _ in range(2):
        s = torch.linspace(0, 1, n, dtype=F64)
        b = lo[:, None] + (hi - lo)[:, None] * s  # (F, n)
        cap = trace(b, r_cap)
        k = (~cap).long().argmax(1).clamp(1, n - 1)  # first escaping ray
        idx = torch.arange(len(r_f))
        lo, hi = b[idx, k - 1], b[idx, k]
    return 0.5 * (lo + hi)


def shadow_tests(eht: list[dict], spin_inc_max: float = 90.0, m87_inc: float = 17.0) -> dict:
    """δ for each EHT row, the spin band, the allowed fold growth and the bound on r_f/r_s.

    δ_up = δ + 2σ_+ (about 95%). A fold enlarges the shadow; spin can shrink it by up to |δ_Kerr,min|, so the fold
    may grow it by δ_up - δ_Kerr,min. M87*: δ = θ_g(EHT)/θ_g(dyn) - 1, inclination 17° (jet); Sgr A*: published δ,
    inclination up to spin_inc_max."""
    spins = torch.linspace(1e-3, 0.998, 40, dtype=F64)
    rows = []
    for r in eht:
        if r["delta"] is None:
            ratio = r["thetag_eht"] / r["thetag_dyn"]
            delta = ratio - 1
            sig = ratio * math.hypot(r["thetag_eht_err"] / r["thetag_eht"], r["thetag_dyn_err"] / r["thetag_dyn"])
            lo = hi = sig
            inc = torch.tensor([m87_inc], dtype=F64)
        else:
            delta, lo, hi = r["delta"], r["delta_lo"], r["delta_hi"]
            inc = torch.linspace(1.0, spin_inc_max, 30, dtype=F64)
        kerr_min = float(equation.kerr_delta(spins, inc).min())
        delta_up = delta + 2 * hi
        allowed = delta_up - kerr_min
        rows.append({**r, "delta": delta, "delta_lo": lo, "delta_hi": hi, "kerr_min": kerr_min, "delta_up": delta_up,
                     "allowed": allowed, "r_bound": float(equation.fold_bound(torch.tensor(allowed, dtype=F64))),
                     "shadow_schw": 2 * math.sqrt(27) * r["thetag_dyn"],
                     "pull": delta / (hi if delta < 0 else lo)})  # fmt: skip
    return {"rows": rows, "spins": spins}


# ---------------------------------------------------------------- supernovae
def supernova_fit(z, mu, sigma, n_a: int = 71, n_m: int = 56) -> dict:
    """Grid over (α, Ω_m) with the offset marginalised, the α posterior, ΛCDM vs DGP, and an autograd refinement."""
    nodes = torch.linspace(0, 1.01 * float(z.max()), 800, dtype=F64)
    alpha = torch.linspace(-1.0, 1.8, n_a, dtype=F64)
    Om = torch.linspace(0.05, 0.6, n_m, dtype=F64)
    chi2 = equation.chi2_offset_marginal(equation.distance_modulus(z, nodes, Om[None, :], alpha[:, None]), mu, sigma)
    post = torch.exp(-0.5 * (chi2 - chi2.min()))  # (A, M), uniform priors
    pa = torch.trapezoid(post, Om, dim=1)
    pa = pa / torch.trapezoid(pa, alpha)
    mean = torch.trapezoid(alpha * pa, alpha)
    sd = torch.trapezoid((alpha - mean) ** 2 * pa, alpha).sqrt()

    def best_at(a_val: float):
        c = equation.chi2_offset_marginal(equation.distance_modulus(z, nodes, Om, torch.tensor(a_val, dtype=F64)),
                                          mu, sigma)  # fmt: skip
        return float(c.min()), float(Om[c.argmin()])

    lcdm, dgp = best_at(0.0), best_at(1.0)
    i, j = divmod(int(chi2.argmin()), n_m)
    fn = lambda p: -0.5 * equation.chi2_offset_marginal(equation.distance_modulus(z, nodes, p[1], p[0]), mu, sigma)  # noqa: E731
    p = newton_max(fn, torch.tensor([float(alpha[i]), float(Om[j])], dtype=F64), steps=20, max_step=0.05)
    a_fit, om_fit = (float(x) for x in p)
    shape = {k: equation.distance_modulus(z, nodes, torch.tensor(o, dtype=F64), torch.tensor(a, dtype=F64)).detach()
             for k, a, o in (("lcdm", 0.0, lcdm[1]), ("dgp", 1.0, dgp[1]), ("fit", a_fit, om_fit))}  # fmt: skip
    return {"alpha": alpha, "Om": Om, "chi2": chi2, "pa": pa, "mean": float(mean), "sd": float(sd),
            "lcdm": lcdm, "dgp": dgp, "a_fit": a_fit, "om_fit": om_fit, "shape": shape,
            "z": z, "mu": mu, "sigma": sigma, "n": len(z)}  # fmt: skip
