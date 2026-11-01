r"""Physics → tensor translation.

Beer–Lambert law for a mixture: every species s adds its own absorbance, so at wavelength w and time t

    D_{tw} = l E_{ws} c_{ts}          (s summed)       E_{ws} = ε_s(λ_w),  l = path length

The consecutive reaction A → B → C (rate constants k1, k2) is linear in the concentrations:

    dc_s/dt = K_{sr} c_r              (r summed)       K = [[-k1, 0, 0], [k1, -k2, 0], [0, k2, 0]]

so c_s(t) = [exp(K t)]_{sr} c_r(0). Every column of K sums to zero, which is mass balance: A + B + C = const.

Going back from spectra to concentrations is least squares, solved with the pseudo-inverse E⁺ = (Eᵀ E)⁻¹ Eᵀ:

    G_{rs} = E_{wr} E_{ws}  (w summed),     E⁺_{sw} = (G⁻¹)_{sr} E_{wr}  (r summed)
    ĉ_{ts} = E⁺_{sw} D_{tw} / l  (w summed)
"""

import torch


def absorbance(E: torch.Tensor, c: torch.Tensor, path_length: float) -> torch.Tensor:
    """D_{tw} = l E_{ws} c_{ts}.   E: (W, S), c: (T, S) → (T, W).  The species index s is summed."""
    return path_length * torch.einsum("ws,ts->tw", E, c)


def rate_matrix(k1: float, k2: float, device: torch.device, dtype=torch.float64) -> torch.Tensor:
    """K_{sr} for A → B → C, shape (S, S): row s says what feeds species s and what drains it."""
    return torch.tensor([[-k1, 0.0, 0.0], [k1, -k2, 0.0], [0.0, k2, 0.0]], device=device, dtype=dtype)


def concentrations(K: torch.Tensor, c0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """c_{ts} = [exp(K t_t)]_{sr} c_r(0).   K: (S, S), c0: (S,), t: (T,) → (T, S).  r is summed."""
    propagators = torch.linalg.matrix_exp(K[None] * t[:, None, None])  # (T, S, S), exact for a linear ODE
    return torch.einsum("tsr,r->ts", propagators, c0)


def analytic(k1: float, k2: float, a0: float, t: torch.Tensor) -> torch.Tensor:
    """Textbook solution for A → B → C (k1 ≠ k2), shape (T, 3), for comparison:

    A = a0 e^{-k1 t},   B = a0 k1/(k2 - k1) (e^{-k1 t} - e^{-k2 t}),   C = a0 - A - B
    """
    A = a0 * torch.exp(-k1 * t)
    B = a0 * k1 / (k2 - k1) * (torch.exp(-k1 * t) - torch.exp(-k2 * t))
    return torch.stack([A, B, a0 - A - B], dim=1)


def pseudo_inverse(E: torch.Tensor) -> torch.Tensor:
    """E⁺_{sw} = (G⁻¹)_{sr} E_{wr} with G_{rs} = E_{wr} E_{ws}.   E: (W, S) → (S, W).  Needs independent spectra."""
    G = torch.einsum("wr,ws->rs", E, E)  # (S, S) Gram matrix: overlaps between the species' spectra
    return torch.einsum("sr,wr->sw", torch.linalg.inv(G), E)


def recover(D: torch.Tensor, E_pinv: torch.Tensor, path_length: float) -> torch.Tensor:
    """ĉ_{ts} = E⁺_{sw} D_{tw} / l: least-squares concentrations.   D: (T, W) → (T, S).  w is summed."""
    return torch.einsum("sw,tw->ts", E_pinv, D) / path_length


def crossings(lam: torch.Tensor, eps_a: torch.Tensor, eps_b: torch.Tensor, floor: float = 0.02) -> list[float]:
    """Wavelengths where two spectra cross (ε_a = ε_b): candidate isosbestic points, by linear interpolation.

    Crossings where both absorb less than `floor` × the larger peak are ignored (two tails, no signal).
    """
    diff = eps_a - eps_b
    level = floor * torch.maximum(eps_a.max(), eps_b.max())
    found = []
    for w in torch.nonzero(diff[:-1] * diff[1:] < 0).flatten().tolist():
        frac = (diff[w] / (diff[w] - diff[w + 1])).item()
        if max(eps_a[w].item(), eps_b[w].item()) > level:
            found.append(lam[w].item() + frac * (lam[w + 1] - lam[w]).item())
    return found
