r"""Physics → tensor translation: the library $\Theta(u)$ of candidate terms and the derivative engine.

The finder looks for a sparse $\xi$ in

    \partial_t u_f = \sum_k \Theta_k(u)\, \xi_{fk}

where every candidate term is a derivative of a monomial of the fields, $\Theta_k = \sum_{\text{parts}} s\, D^\alpha
\big(\prod_g u_g^{p_g}\big)$ (for example $u u_x = \tfrac12 \partial_x(u^2)$, $\nabla^2 E_z = \partial_{xx}E_z +
\partial_{yy}E_z$). Fields are tensors U of shape (F, *axes) whose last axis is time.

Two ways to turn data into the linear system:

* strong form (the PDE-FIND way): estimate every derivative on the grid, either with plain central finite differences
  (fd) or with Savitzky–Golay filters (sg: smooth the fields with a local cubic fit, then differentiate a local
  quartic fit). Both are 1D convolutions applied axis by axis.
* weak form (weak SINDy): multiply by smooth bumps $\phi_p(x)\psi_q(t)$ and integrate by parts, so all derivatives
  fall on the known test functions and none is taken of the noisy data:

    -\int \partial_t\phi\, u = \sum_k \xi_k\, (-1)^{|\alpha_k|} \int (D^{\alpha_k}\phi)\, f_k(u)

  Each integral is one tensor contraction per axis: G_{pq} = \sum_{x,t} \Phi_{px}\, \Psi_{qt}\, f(u)_{xt}\, h_x h_t
  (x and t summed).
"""

import itertools
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F


class Term(NamedTuple):
    latex: str  # how the term is written (and read back)
    parts: tuple  # ((scale, powers per field, derivative order per spatial axis), ...): Θ = Σ scale · D^α(Π u^p)


def _t(latex: str, *parts) -> Term:
    return Term(latex, tuple(parts))


# ---------------------------------------------------------------- the libraries
SCALAR_1D = (  # u(x, t): D^d(u^p) for p ≤ 3, d ≤ 3  (13 terms)
    _t("1", (1.0, (0,), (0,))),
    _t("u", (1.0, (1,), (0,))),
    _t("u^2", (1.0, (2,), (0,))),
    _t("u^3", (1.0, (3,), (0,))),
    _t("u_{x}", (1.0, (1,), (1,))),
    _t("u u_{x}", (0.5, (2,), (1,))),  # ½ ∂_x(u²)
    _t("u^2 u_{x}", (1 / 3, (3,), (1,))),  # ⅓ ∂_x(u³)
    _t("u_{xx}", (1.0, (1,), (2,))),
    _t("\\partial_{xx}(u^2)", (1.0, (2,), (2,))),
    _t("\\partial_{xx}(u^3)", (1.0, (3,), (2,))),
    _t("u_{xxx}", (1.0, (1,), (3,))),
    _t("\\partial_{xxx}(u^2)", (1.0, (2,), (3,))),
    _t("\\partial_{xxx}(u^3)", (1.0, (3,), (3,))),
)


def _monomial_latex(powers, names) -> str:
    out = [n if p == 1 else f"{n}^{p}" for n, p in zip(names, powers) if p > 0]
    return " ".join(out) if out else "1"


LORENZ = tuple(  # all monomials of x, y, z up to degree 3 (20 terms), no spatial axis
    _t(_monomial_latex(p, "xyz"), (1.0, p, ()))
    for deg in range(4)
    for p in sorted((q for q in itertools.product(range(deg + 1), repeat=3) if sum(q) == deg), reverse=True)
)


def _maxwell() -> tuple:
    fields = ("E_z", "B_x", "B_y")
    one = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    out = [_t("1", (1.0, (0, 0, 0), (0, 0)))] + [_t(f, (1.0, p, (0, 0))) for f, p in zip(fields, one)]
    out += [
        _t(f"\\partial_{a} {f}", (1.0, p, d)) for f, p in zip(fields, one) for a, d in (("x", (1, 0)), ("y", (0, 1)))
    ]
    out += [_t(f"\\nabla^2 {f}", (1.0, p, (2, 0)), (1.0, p, (0, 2))) for f, p in zip(fields, one)]
    out += [_t("E_z^2", (1.0, (2, 0, 0), (0, 0))), _t("E_z^3", (1.0, (3, 0, 0), (0, 0))),
            _t("E_z B_x", (1.0, (1, 1, 0), (0, 0))), _t("E_z B_y", (1.0, (1, 0, 1), (0, 0)))]  # fmt: skip
    return tuple(out)


MAXWELL = _maxwell()  # fields (E_z, B_x, B_y) on (x, y): 17 terms


def truth_matrix(terms, rows: list[dict]) -> torch.Tensor:
    """The true ξ as a (F, K) tensor from one {term latex: coefficient} dict per equation."""
    names = [tm.latex for tm in terms]
    xi = torch.zeros(len(rows), len(terms), dtype=torch.float64)
    for f, row in enumerate(rows):
        for k, v in row.items():
            xi[f, names.index(k)] = v
    return xi


# ---------------------------------------------------------------- derivative engine (module 2) and smoother (module 1)
SG_SMOOTH = (3, 3)  # Savitzky–Golay smoother: half-width 3 (7 samples), cubic fit
SG_DERIV = (4, 4)  # Savitzky–Golay derivative: half-width 4 (9 samples), quartic fit


def sg_kernel(m: int, p: int, d: int) -> np.ndarray:
    """Weights w_j (j = −m..m) with f^{(d)}(0) ≈ Σ_j w_j f(j) for a least-squares polynomial of degree p on 2m + 1
    samples (Savitzky and Golay 1964). Exact for polynomials of degree ≤ p."""
    j = np.arange(-m, m + 1, dtype=float)
    V = j[:, None] ** np.arange(p + 1)  # (2m+1, p+1) Vandermonde
    return math.factorial(d) * np.linalg.pinv(V)[d]


def fd_kernel(d: int) -> np.ndarray:
    """The narrowest central finite difference for the d-th derivative: polynomial interpolation through 2m+1 points."""
    m = max(1, (d + 1) // 2)
    return sg_kernel(m, 2 * m, d)


def filter_axis(f: torch.Tensor, w: np.ndarray, axis: int, scale: float = 1.0) -> torch.Tensor:
    """Cross-correlate f with the kernel w along one axis (zero padding; the edges are cropped later)."""
    m = len(w) // 2
    g = f.movedim(axis, -1)
    shape = g.shape
    k = torch.as_tensor(w, dtype=f.dtype, device=f.device).view(1, 1, -1)
    out = F.conv1d(g.reshape(-1, 1, shape[-1]), k, padding=m).reshape(shape)
    return out.movedim(-1, axis) / scale


def derivative(f: torch.Tensor, axis: int, d: int, h: float, method: str) -> torch.Tensor:
    if d == 0:
        return f
    w = fd_kernel(d) if method == "fd" else sg_kernel(SG_DERIV[0], max(SG_DERIV[1], d), d)
    return filter_axis(f, w, axis, h**d)


def margin(method: str) -> int:
    return 2 if method == "fd" else SG_SMOOTH[0] + SG_DERIV[0]


def monomial(U: torch.Tensor, powers) -> torch.Tensor:
    """Π_g U_g^{p_g}.   U: (F, *axes) → (*axes)"""
    out = torch.ones_like(U[0])
    for g, p in enumerate(powers):
        if p:
            out = out * U[g] ** p
    return out


def spatial_axes(h) -> list[int]:
    """Axes that carry a spatial derivative: every axis but the last (time) and batch axes (h = None)."""
    return [a for a in range(len(h) - 1) if h[a] is not None]


def strong_library(U: torch.Tensor, h, terms, method: str = "fd"):
    """Θ (rows, K) and ∂_t U (rows, F) at every interior grid point; crop = the slices that pick the interior.

    U: (F, *axes), h: spacing per axis (None for a batch axis), method: 'fd' or 'sg'."""
    h = [None if v is None else float(v) for v in h]
    sp, ta = spatial_axes(h), len(h) - 1
    if method == "sg":  # module 1: the smoother (local cubic fit along every axis)
        w = sg_kernel(SG_SMOOTH[0], SG_SMOOTH[1], 0)
        for a in sp + [ta]:
            U = filter_axis(U, w, a + 1)
    m = margin(method)
    crop = tuple(slice(m, -m) if (h[a] is not None) else slice(None) for a in range(len(h)))
    cols = []
    for tm in terms:
        col = 0
        for s, powers, alpha in tm.parts:
            f = monomial(U, powers)
            for a, d in zip(sp, alpha):
                f = derivative(f, a, d, h[a], method)
            col = col + s * f
        cols.append((col * torch.ones_like(U[0]))[crop].reshape(-1))
    ut = torch.stack([derivative(U[g], ta, 1, h[ta], method)[crop].reshape(-1) for g in range(len(U))], 1)
    return torch.stack(cols, 1), ut, crop


# ---------------------------------------------------------------- weak form
BUMP_POWER = 10  # φ(s) = (1 − s²)^10 on |s| ≤ 1: it and its first 9 derivatives vanish at the edge


def bump_matrices(n: int, h: float, radius: int, stride: int, orders: int = 4) -> list[torch.Tensor]:
    """[Φ^(0), …, Φ^(orders−1)]: Φ^(d)_{pn} = d^dφ/dx^d of the bump centred at c_p, radius r samples, at sample n."""
    centres = np.arange(radius, n - radius, stride)
    s = (np.arange(n)[None, :] - centres[:, None]) / radius  # (P, n)
    poly = np.polynomial.Polynomial([1.0, 0.0, -1.0]) ** BUMP_POWER
    out = []
    for d in range(orders):
        vals = poly.deriv(d)(s) if d else poly(s)
        vals = np.where(np.abs(s) <= 1, vals, 0.0) / (radius * h) ** d
        out.append(torch.tensor(vals))
    return out


def test_functions(shape, h, radius, stride) -> list:
    """Per data axis (shape = U.shape[1:]): the bump matrices, None on a batch axis. radius, stride in samples."""
    return [None if h[a] is None else bump_matrices(shape[a], float(h[a]), radius[a], stride[a]) for a in range(len(h))]


def contract(f: torch.Tensor, mats: list) -> torch.Tensor:
    """f contracted with one matrix per axis (None = keep that axis): G_{pq…} = Σ_{xt…} M1_{px} M2_{qt} … f_{xt…}."""
    for a, M in enumerate(mats):
        if M is not None:
            f = torch.tensordot(M.to(f), f, dims=([1], [a])).movedim(0, a)
    return f


def weak_library(U: torch.Tensor, h, terms, phi):
    """Weak-form Θ (rows, K) and target (rows, F): one row per test function (and batch member)."""
    h = [None if v is None else float(v) for v in h]
    sp, ta = spatial_axes(h), len(h) - 1
    vol = math.prod(v for v in h if v is not None)
    cols = []
    for tm in terms:
        col = 0
        for s, powers, alpha in tm.parts:
            mats = [None] * len(h)
            for a, d in zip(sp, alpha):
                mats[a] = phi[a][d]
            for a in sp:
                mats[a] = mats[a] if mats[a] is not None else phi[a][0]
            mats[ta] = phi[ta][0]
            f = monomial(U, powers) * torch.ones_like(U[0])
            col = col + s * (-1) ** sum(alpha) * contract(f, mats) * vol
        cols.append(col.reshape(-1))
    mats = [phi[a][0] if a in sp else None for a in range(len(h))]
    mats[ta] = phi[ta][1]
    ut = torch.stack([-contract(U[g], mats).reshape(-1) * vol for g in range(len(U))], 1)
    return torch.stack(cols, 1), ut


# ---------------------------------------------------------------- the found equation as a simulator
def spectral_derivative(f: torch.Tensor, axis: int, d: int, h: float) -> torch.Tensor:
    """d-th derivative along a periodic axis by Fourier modes: multiply by (ik)^d."""
    if d == 0:
        return f
    n = f.shape[axis]
    k = 2 * math.pi * torch.fft.fftfreq(n, d=h, dtype=f.dtype)
    if d % 2 and n % 2 == 0:
        k[n // 2] = 0  # odd derivatives of the Nyquist mode are not defined: drop it
    shape = [1] * f.dim()
    shape[axis] = n
    return torch.fft.ifft(torch.fft.fft(f, dim=axis) * (1j * k.view(shape)) ** d, dim=axis).real


def found_rhs(U: torch.Tensor, h_space, terms, xi: torch.Tensor) -> torch.Tensor:
    """∂_t U_f = Σ_k ξ_fk Θ_k(U) for one time slice, spectral derivatives (periodic space). (F, *space) → (F, *space).
    Only the terms some equation uses are evaluated."""
    sp = [a for a in range(len(h_space)) if h_space[a] is not None]
    if not sp:  # an ODE: all monomials at once, Θ_k = Π_g U_g^{P_gk}, then ξ_fk Θ_k (k summed)
        P = torch.tensor([tm.parts[0][1] for tm in terms], dtype=U.dtype).T  # (F, K)
        theta = torch.prod(U.unsqueeze(1) ** P.view(*P.shape, *[1] * (U.dim() - 1)), dim=0)  # (K, *rest)
        return torch.einsum("fk,k...->f...", xi, theta)
    out = torch.zeros_like(U)
    for k in torch.nonzero(xi.abs().sum(0)).flatten().tolist():
        col = 0
        for s, powers, alpha in terms[k].parts:
            f = monomial(U, powers) * torch.ones_like(U[0])
            for a, d in zip(sp, alpha):
                f = spectral_derivative(f, a, d, float(h_space[a]))
            col = col + s * f
        out = out + xi[:, k].view(-1, *[1] * (U.dim() - 1)) * col
    return out
