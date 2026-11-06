r"""The finder (module 3): sparse regression of $\partial_t u$ on the library, its uncertainty, the annealed-gate
alternative, and the LaTeX writer with its reader.

STRidge (Rudy et al. 2017): with unit-norm columns and target, solve a ridge regression, zero every coefficient below
a tolerance, refit on the survivors, repeat. A sweep of tolerances gives a path from the full library to the empty
one. Paths are run for three ridge strengths $\lambda$; the model kept is the one with the fewest terms whose
validation error stays within 5 % (+ 10⁻³) of the best model found, ties broken by the lower validation error.
"""

import math
import re
import time

import torch

from . import equation as eq
from . import tensors as T

TOLS = torch.logspace(-4, 0, 41, dtype=torch.float64)  # STRidge tolerances (normalised coefficients)
LAMS = (1e-8, 1e-5, 1e-3)  # ridge strengths on unit-norm columns, one path each (collinear libraries need the larger)
LAM = 1e-5
SLACK, FLOOR = 0.05, 1e-3  # model selection: sparsest with err ≤ (1 + SLACK) min err + FLOOR
MAX_ROWS = 20_000  # strong-form rows kept (random subset) for the regression


# ---------------------------------------------------------------- STRidge
def _normalise(A: torch.Tensor, b: torch.Tensor):
    sa, sb = A.norm(dim=0), b.norm(dim=0)
    sa, sb = torch.where(sa > 0, sa, 1.0), torch.where(sb > 0, sb, 1.0)
    return A / sa, b / sb, sa, sb


def masked_ridge(G: torch.Tensor, r: torch.Tensor, mask: torch.Tensor, lam: float) -> torch.Tensor:
    """For every equation e at once: argmin ‖A ξ_e − b_e‖² + λ‖ξ_e‖² over the columns in mask_e (others 0).
    G = AᵀA (K, K), r = (Aᵀb)ᵀ (E, K), mask (E, K) → ξ (E, K)."""
    m = mask.to(G.dtype)
    Gm = G[None] * m[:, :, None] * m[:, None, :] + lam * torch.eye(len(G), dtype=G.dtype)
    return torch.linalg.solve(Gm, (r * m).unsqueeze(-1)).squeeze(-1) * m


def _stridge_normalised(G, r, tol: float, lam: float, iters: int = 10) -> torch.Tensor:
    mask = torch.ones_like(r, dtype=torch.bool)
    for _ in range(iters):
        xi = masked_ridge(G, r, mask, lam)
        new = mask & (xi.abs() >= tol)
        if torch.equal(new, mask):
            break
        mask = new
    return masked_ridge(G, r, mask, 1e-12)  # unbiased refit on the support


def stridge(A: torch.Tensor, b: torch.Tensor, tol: float, lam: float = LAM) -> torch.Tensor:
    """ξ (E, K) in physical units for A (n, K), b (n, E)."""
    An, bn, sa, sb = _normalise(A, b)
    xi = _stridge_normalised(An.T @ An, (An.T @ bn).T, tol, lam)
    return xi * sb[:, None] / sa[None, :]


def stridge_path(A, b, tols=TOLS, lam: float = LAM) -> torch.Tensor:
    """ξ along the tolerance sweep: (len(tols), E, K), physical units."""
    An, bn, sa, sb = _normalise(A, b)
    G, r = An.T @ An, (An.T @ bn).T
    return torch.stack([_stridge_normalised(G, r, float(t), lam) for t in tols]) * sb[None, :, None] / sa[None, None, :]


def select(path: torch.Tensor, A_val, b_val) -> tuple[torch.Tensor, torch.Tensor]:
    """Per equation, the index of the model on the path (M, E, K) with the fewest terms among those whose validation
    error is within (1 + SLACK) of the best (+ FLOOR), ties go to the lower error. Returns index (E,), err (M, E)."""
    err = (torch.einsum("nk,mek->mne", A_val, path) - b_val[None]).norm(dim=1) / b_val.norm(dim=0)  # (M, E)
    ok = err <= (1 + SLACK) * err.min(0).values + FLOOR
    nnz = (path != 0).sum(-1).to(err.dtype)  # (M, E)
    score = torch.where(ok, nnz, torch.inf) + err.clamp(max=0.999)  # fewest terms first, then the smaller error
    return score.argmin(0), err


def bootstrap(A: torch.Tensor, b: torch.Tensor, n_boot: int = 100, seed: int = 0):
    """Least-squares coefficients on the given columns and their spread over row resamplings (with replacement).
    The resamples are row weights w_{βn} (multinomial counts), so all n_boot fits are one batched solve:
        G_{βkl} = Σ_n w_{βn} A_{nk} A_{nl},   r_{βk} = Σ_n w_{βn} A_{nk} b_n."""
    if A.shape[1] == 0:
        return torch.zeros(0, dtype=A.dtype), torch.zeros(0, dtype=A.dtype)
    sa = A.norm(dim=0)
    An = A / sa
    est = torch.linalg.lstsq(An, b[:, None]).solution[:, 0] / sa
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(len(b), (n_boot, len(b)), generator=g)
    w = torch.zeros(n_boot, len(b), dtype=A.dtype).scatter_add_(1, idx, torch.ones(idx.shape, dtype=A.dtype))
    G = torch.einsum("bn,nk,nl->bkl", w, An, An)
    r = torch.einsum("bn,nk,n->bk", w, An, b)
    boots = torch.linalg.solve(G, r.unsqueeze(-1)).squeeze(-1) / sa
    return est, boots.std(0)


def null_vector(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The unit vector v minimising ‖Â v‖ over the column-normalised library Â (an implicit law Σ v_k Θ_k = 0), and the
    singular values in ascending order."""
    An = A / torch.where(A.norm(dim=0) > 0, A.norm(dim=0), 1.0)
    ev, V = torch.linalg.eigh(An.T @ An)
    return V[:, 0], ev.clamp(min=0).sqrt()


# ---------------------------------------------------------------- the DARTS-style alternative: annealed gates
def gate_search(A, b, steps: int = 1500, lam: float = 1e-3, temp=(1.0, 0.01), lr: float = 0.05, seed: int = 0,
                max_rows: int = 5000) -> dict:  # fmt: skip
    """A continuous relaxation of the support: ξ_ek = σ(α_ek / T) w_ek, trained with Adam on
        L = Σ_e ‖Â ξ_e − b̂_e‖² + λ Σ_ek σ(α_ek / T)
    while the temperature T falls geometrically (as in operator_search_mnist), so every gate ends open or shut. The
    support is the open gates; the coefficients are then refitted by least squares."""
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(len(A), generator=g)[:max_rows]
    An, bn, sa, sb = _normalise(A[rows], b[rows])
    w = masked_ridge(An.T @ An, (An.T @ bn).T, torch.ones(b.shape[1], A.shape[1], dtype=torch.bool), LAM)
    w = w.clone().requires_grad_(True)
    alpha = torch.zeros_like(w).requires_grad_(True)  # every gate starts half open
    opt = torch.optim.Adam([w, alpha], lr=lr)
    history = []
    for s in range(steps):
        T_s = temp[0] * (temp[1] / temp[0]) ** (s / max(steps - 1, 1))
        gate = torch.sigmoid(alpha / T_s)
        loss = ((An @ (gate * w).T - bn) ** 2).sum() + lam * gate.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % max(1, steps // 40) == 0:
            history.append(gate.detach().clone())
    support = torch.sigmoid(alpha.detach() / temp[1]) > 0.5
    xi = masked_ridge(An.T @ An, (An.T @ bn).T, support, 1e-12) * sb[:, None] / sa[None, :]
    return {"support": support, "xi": xi.detach(), "history": torch.stack(history)}


# ---------------------------------------------------------------- the LaTeX writer and reader
def _coef_text(v: float, e: float | None, digits: int) -> str:
    if e is not None and e > 0 and math.isfinite(e):
        dec = max(0, 1 - math.floor(math.log10(e)))  # two significant digits of the uncertainty
        return f"({abs(v):.{dec}f} \\pm {e:.{dec}f})"
    return f"{abs(v):.{digits}g}"


def write_equation(lhs: str, terms, xi: torch.Tensor, err: torch.Tensor | None = None, digits: int = 6) -> str:
    """'lhs = c_1\\, term_1 + c_2\\, term_2 …' over the non-zero coefficients, with ± uncertainties when given."""
    out = []
    for k in torch.nonzero(xi).flatten().tolist():
        v = float(xi[k])
        txt = _coef_text(v, None if err is None else float(err[k]), digits) + "\\, " + terms[k].latex
        out.append(("-" if v < 0 else "") + txt if not out else (" - " if v < 0 else " + ") + txt)
    return f"{lhs} = " + ("".join(out) if out else "0")


def read_equation(line: str, terms) -> torch.Tensor:
    """The coefficient vector ξ (K,) from a written equation alone (the inverse of write_equation)."""
    names = {tm.latex: k for k, tm in enumerate(terms)}
    xi = torch.zeros(len(terms), dtype=torch.float64)
    rest = line.split(" = ", 1)[1]
    if rest == "0":
        return xi
    pieces = re.split(r" ([+-]) ", rest)
    signs = ["-" if pieces[0].startswith("-") else "+"] + pieces[1::2]
    chunks = [pieces[0].lstrip("-")] + pieces[2::2]
    for sgn, chunk in zip(signs, chunks):
        coef, latex = chunk.split("\\, ", 1)
        value = float(coef.strip("()").split(" \\pm ")[0])
        xi[names[latex]] = -value if sgn == "-" else value
    return xi


# ---------------------------------------------------------------- re-simulating the found equation
SUBSTEPS = {"lorenz": 2, "heat": 30, "burgers": 15, "maxwell": 4}  # RK4 steps per data interval


def simulate_found(ds: dict, xi: torch.Tensor, n_times: int | None = None) -> torch.Tensor:
    """Integrate the found equation (RK4, spectral derivatives) from the first data snapshot; output at the data
    times, shape (F, *space, n_times)."""
    U0, hs = ds["U"][..., 0], ds["h"][:-1]
    n_times = n_times or ds["U"].shape[-1]
    sub = SUBSTEPS[ds["name"]]
    dt = ds["h"][-1] / sub
    U, out = U0.clone(), [U0]

    def f(V):
        return eq.found_rhs(V, hs, ds["terms"], xi)

    for _ in range((n_times - 1) * sub):
        k1 = f(U)
        k2 = f(U + 0.5 * dt * k1)
        k3 = f(U + 0.5 * dt * k2)
        U = U + dt / 6 * (k1 + 2 * k2 + 2 * k3 + f(U + dt * k3))
        out.append(U)
    return torch.stack(out[::sub], -1)


# ---------------------------------------------------------------- one discovery and the noise study
def build(ds: dict, U: torch.Tensor, method: str):
    """(Θ, ∂_t U) for a data set in the strong (fd, sg) or weak form."""
    if method == "weak":
        phi = eq.test_functions(U.shape[1:], ds["h"], ds["weak"]["radius"], ds["weak"]["stride"])
        return eq.weak_library(U, ds["h"], ds["terms"], phi)
    th, ut, _ = eq.strong_library(U, ds["h"], ds["terms"], method)
    return th, ut


def discover(ds: dict, method: str, noise: float, seed: int = 0, n_boot: int = 100, regressor: str = "stridge",
             gate_steps: int = 800) -> dict:  # fmt: skip
    """Add noise, build the library, find the sparse equation, refit with bootstrap error bars, write it in LaTeX."""
    t0 = time.perf_counter()
    U = T.add_noise(ds["U"], noise, seed + 1)
    th, ut = build(ds, U, method)
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(len(th), generator=g)[:MAX_ROWS]
    th, ut = th[rows], ut[rows]
    n_tr = int(0.8 * len(th))
    paths = torch.stack([stridge_path(th[:n_tr], ut[:n_tr], lam=lam) for lam in LAMS])  # (L, tols, E, K)
    pick, val_err = select(paths.flatten(0, 1), th[n_tr:], ut[n_tr:])
    lam_idx, tol_idx = pick // len(TOLS), pick % len(TOLS)
    E = ut.shape[1]
    path = torch.stack([paths[lam_idx[e], :, e] for e in range(E)], 1)  # the chosen λ's path per equation (tols, E, K)
    if regressor == "gate":
        support = gate_search(th[:n_tr], ut[:n_tr], steps=gate_steps, seed=seed)["support"]
    else:
        support = torch.stack([path[tol_idx[e], e] != 0 for e in range(E)])
    xi, err = torch.zeros_like(support, dtype=th.dtype), torch.zeros_like(support, dtype=th.dtype)
    for e in range(ut.shape[1]):
        cols = torch.nonzero(support[e]).flatten()
        xi[e, cols], err[e, cols] = bootstrap(th[:, cols], ut[:, e], n_boot=n_boot, seed=seed)
    truth = ds["truth"]
    true_terms = truth != 0
    coef_err = float(((xi - truth).abs() / truth.abs())[true_terms].max())
    return {"target": ds["name"], "method": method if regressor == "stridge" else f"gate-{method}", "noise": noise,
            "terms": ds["terms"], "xi": xi, "err": err, "support": support,
            "success": bool(torch.equal(support, true_terms)), "coef_err": coef_err,
            "lines": [write_equation(lhs, ds["terms"], xi[e], err[e]) for e, lhs in enumerate(ds["lhs"])],
            "path": path, "share": path * th[:n_tr].norm(dim=0) / ut[:n_tr].norm(dim=0)[:, None], "pick": tol_idx,
            "lam": [LAMS[i] for i in lam_idx], "rows": len(th),
            "seconds": time.perf_counter() - t0}  # fmt: skip


def noise_study(levels=T.NOISE_LEVELS, methods=("fd", "sg", "weak"), small: bool = False, n_boot: int = 100,
                gate: bool = False, gate_steps: int = 800, datasets: dict | None = None, log=None) -> list:  # fmt: skip
    """Every target × noise level × derivative method (and, with gate=True, the annealed-gate regression on the weak
    form). Returns one result dict per run."""
    datasets = datasets or {n: T.dataset(n, small) for n in T.TARGETS}
    out = []
    for ds in datasets.values():
        for lv in levels:
            runs = [("stridge", m) for m in methods] + ([("gate", "weak")] if gate else [])
            for reg, m in runs:
                r = discover(ds, m, lv, n_boot=n_boot, regressor=reg, gate_steps=gate_steps)
                out.append(r)
                if log:
                    log(r)
    return out
