"""Postulates -> tensors: joint Born weights of two qubits, the projection race, local races, CHSH, no-signalling.

Shapes in comments. Summed indices: the qubit indices x, y, u, v in w_ij = psi*_xy (P^a_i)_xu (P^b_j)_yv psi_uv, the
outcome signs in E = s_i s_j P_ij, the race product over the other outcomes (a sum of logs) and the quadrature
variable t, and the shared hidden direction n (averaged) in the local model.
"""

import math

import torch

PAULI = torch.tensor(
    [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=torch.complex128
)  # (3, 2, 2) sigma_x, sigma_y, sigma_z
SIGNS = torch.tensor([1.0, -1.0], dtype=torch.float64)  # outcome +1, -1

# ---------- quantum mechanics of two qubits ----------


def singlet(device) -> torch.Tensor:
    """(2, 2) psi_xy of (|01> - |10>) / sqrt 2."""
    return partly_entangled(math.pi / 4, device)


def partly_entangled(eta: float, device) -> torch.Tensor:
    """(2, 2) psi_xy of cos(eta) |01> - sin(eta) |10>: the singlet at eta = pi/4."""
    psi = torch.zeros(2, 2, dtype=torch.complex128, device=device)
    psi[0, 1], psi[1, 0] = math.cos(eta), -math.sin(eta)
    return psi


def projectors(n: torch.Tensor) -> torch.Tensor:
    """(..., 2, 2, 2) P_i = (1 + s_i n.sigma) / 2 for the outcomes s = +1, -1 along unit vectors n (..., 3)."""
    ns = torch.einsum("...k,kxy->...xy", n.to(torch.complex128), PAULI.to(n.device))  # n.sigma, k summed
    eye = torch.eye(2, dtype=torch.complex128, device=n.device)
    s = SIGNS.to(n.device).to(torch.complex128)[:, None, None]
    return (eye + s * ns[..., None, :, :]) / 2


def joint_born(psi: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(..., 2, 2) w_ij = <psi| P^a_i (x) P^b_j |psi> = ||P_ij Psi||^2: the weight of each joint outcome."""
    w = torch.einsum("xy,...ixu,...jyv,uv->...ij", psi.conj(), projectors(a), projectors(b), psi)  # x y u v summed
    return w.real


def correlation(P: torch.Tensor) -> torch.Tensor:
    """(...,) E = sum_ij s_i s_j P_ij from joint outcome probabilities (..., 2, 2)."""
    s = SIGNS.to(P.device)
    return torch.einsum("i,j,...ij->...", s, s, P)


def chsh(E, a, a2, b, b2) -> torch.Tensor:
    """(B,) CHSH S = max over which term carries the minus sign of |E(a,b) + E(a,b') + E(a',b) + E(a',b') - 2 E_k|."""
    Es = torch.stack([E(a, b), E(a, b2), E(a2, b), E(a2, b2)])  # (4, B)
    return (Es.sum(0, keepdim=True) - 2 * Es).abs().amax(0)


# ---------- the projection race (step 1), run on joint outcomes ----------


def race_probabilities(w: torch.Tensor, a: torch.Tensor, n_grid: int = 4000) -> torch.Tensor:
    """P(k wins) for k* = argmax_k w_k / E_k, E_k ~ Gamma(a) i.i.d.; w (..., K), a broadcast to (...). a = 1: P = w.

    P_k = int_0^oo f_a(t; w_k) prod_{j != k} Q(a, w_j t) dt (Q: regularised upper incomplete gamma), on a log grid.
    """
    w = w.double()
    a = torch.as_tensor(a, dtype=torch.float64, device=w.device).expand(w.shape[:-1])  # (...)
    w_pos = torch.where(w > 1e-15 * w.amax(-1, keepdim=True), w, torch.zeros_like(w))
    w_min = torch.where(w_pos > 0, w_pos, torch.full_like(w, math.inf)).amin(-1)
    s_lo = math.log(1e-30) / a - torch.log(w.amax(-1))
    s_hi = torch.log((a + 60 + 12 * a.sqrt()) / w_min)
    u = torch.linspace(0, 1, n_grid, dtype=torch.float64, device=w.device)
    t = torch.exp(s_lo[..., None] + (s_hi - s_lo)[..., None] * u)  # (..., G)
    wt = w_pos[..., :, None] * t[..., None, :]  # (..., K, G)
    aa = a[..., None, None]
    log_q = torch.log(torch.special.gammaincc(aa.expand_as(wt), wt).clamp_min(1e-300))
    log_f_t = aa * torch.log(wt) - wt - torch.lgamma(aa)  # density of ln t
    integrand = torch.exp(log_f_t + log_q.sum(-2, keepdim=True) - log_q)  # product over j != k
    return torch.trapezoid(integrand, dim=-1) * ((s_hi - s_lo)[..., None] / (n_grid - 1))


def joint_race_correlation(w: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """(B,) Monte Carlo E(a, b) of the race on the joint outcomes: w (B, 4) weights, E (T, 4) hidden-block energies."""
    win = (w[:, None, :] / E[None]).argmax(-1)  # (B, T) the one event, decided in the full space
    P = torch.nn.functional.one_hot(win, 4).double().mean(1)  # (B, 4)
    return correlation(P.view(-1, 2, 2))


def joint_race_chsh(psi, m: torch.Tensor, a, a2, b, b2) -> torch.Tensor:
    """(M,) CHSH S of the joint race with m hidden complex dimensions per joint outcome, for each m."""
    w = torch.stack([joint_born(psi, a, b), joint_born(psi, a, b2), joint_born(psi, a2, b), joint_born(psi, a2, b2)])
    w = w.flatten(-2)[:, 0]  # (4, 4) setting pair, joint outcome
    P = race_probabilities(w.expand(len(m), 4, 4), m[:, None])  # (M, 4, 4)
    Es = correlation(P.view(len(m), 4, 2, 2)).T  # (4, M)
    return (Es.sum(0, keepdim=True) - 2 * Es).abs().amax(0)


def max_signalling(psi, m: torch.Tensor, generator, device, n_bob: int = 64) -> torch.Tensor:
    """(M,) largest change of Alice's P(+ | a, b) as Bob turns his setting b: zero means no signalling."""
    a = torch.nn.functional.normalize(torch.randn(1, 3, generator=generator, dtype=torch.float64), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(n_bob, 3, generator=generator, dtype=torch.float64), dim=-1)
    w = joint_born(psi, a.to(device).expand(n_bob, 3), b.to(device)).flatten(-2)  # (n_bob, 4)
    P = race_probabilities(w.expand(len(m), n_bob, 4), m[:, None])  # (M, n_bob, 4)
    pA = P[..., 0] + P[..., 1]  # Alice +, Bob either
    return pA.amax(-1) - pA.amin(-1)


# ---------- local races: each party sees only its own hidden coordinates ----------


def local_response(x: torch.Tensor, m: float) -> torch.Tensor:
    """<A> = P(+) - P(-) of a two-outcome race with weights (1 +- x)/2 and m hidden complex dimensions each.

    P(+) = I_w(m, m) = P(Binomial(2m - 1, w) >= m) for integer m; m = 1 gives x itself, m -> oo gives sign(x).
    """
    if math.isinf(m):
        return torch.sign(x)
    mi = int(m)
    if mi != m or mi < 1:
        raise ValueError("local_response needs an integer m >= 1 or inf")
    w = ((1 + x) / 2).clamp(0, 1)
    n = 2 * mi - 1
    j = torch.arange(mi, n + 1, dtype=torch.float64, device=x.device)
    log_c = math.lgamma(n + 1) - torch.lgamma(j + 1) - torch.lgamma(n - j + 1)
    terms = torch.exp(log_c + j * torch.log(w[..., None].clamp_min(1e-300)) + (n - j) * torch.log1p(-w[..., None]))
    return 2 * terms.sum(-1) - 1


def local_correlation(a: torch.Tensor, b: torch.Tensor, n: torch.Tensor, m: float) -> torch.Tensor:
    """(B,) E(a, b) = -< g_m(a.n) g_m(b.n) >_n: Alice races on (1 +- a.n)/2, Bob on (1 -+ b.n)/2, independently.

    n (N, 3) is the hidden direction both can read (set at the source); each party's hidden energies are its own.
    """
    x, y = a @ n.T, b @ n.T  # (B, N)
    return -(local_response(x, m) * local_response(y, m)).mean(-1)  # average over the shared hidden n
