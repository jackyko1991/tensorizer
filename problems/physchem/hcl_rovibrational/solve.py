"""Build the spectrum for every temperature, and fit B_e, α_e, D and ν₀ back from the line positions."""

import torch

from .equation import combination_differences, lorentzian, rot_term, spectrum


def spectra(R, P, I_R, I_P, grid: torch.Tensor, fwhm: float) -> torch.Tensor:
    """Spectrum S (T, G) from lines R, P (I, J) with strengths I_R, I_P (T, I, J).

    All isotopologues, both branches and every J are flattened into one line index ℓ, then summed in one einsum.
    """
    nu_lines = torch.cat([R.flatten(), P.flatten()])  # (L,)
    I_lines = torch.cat([I_R.flatten(1), I_P.flatten(1)], dim=1)  # (T, L)
    return spectrum(I_lines, lorentzian(nu_lines, grid, fwhm))


def fit_constants(R: torch.Tensor, P: torch.Tensor, J: torch.Tensor) -> dict[str, torch.Tensor]:
    """Least-squares fit of the measured lines R, P (I, J) by combination differences.

    y_isn = Δ₂F_s(J_n) / (4J_n + 2) = B_s − 2D (J_n² + J_n + 1)   for s = v″ = 0 (lower) and v′ = 1 (upper)
    is a straight line in x_n = J_n² + J_n + 1, so one design matrix X_nk = [1, −2x_n] serves every isotopologue
    and both states:  β_isk = X⁺_kn y_isn   (n summed; X⁺ is the pseudo-inverse = the least-squares solution).
    """
    Jm = J[1:-1]  # J = 1 … J_max − 1
    lower, upper = combination_differences(R, P)  # (I, n) each
    y = torch.stack([lower, upper], dim=1) / (4 * Jm + 2)  # (I, 2, n)
    X = torch.stack([torch.ones_like(Jm), -2 * (Jm**2 + Jm + 1)], dim=1)  # (n, 2)
    beta = torch.einsum("kn,isn->isk", torch.linalg.pinv(X), y)  # (I, 2, 2): [B_v, D] for v = 0, 1
    B0, B1, D = beta[:, 0, 0], beta[:, 1, 0], beta[:, :, 1].mean(1)
    alpha = B0 - B1
    Be = B0 + 0.5 * alpha
    # with B₀, B₁, D known, every line gives ν₀ = ν − F₁(J′) + F₀(J″); average over R(J) and P(J ≥ 1)
    F0 = rot_term(J, B0, D)
    nu0_R = R - rot_term(J + 1, B1, D) + F0
    nu0_P = (P - rot_term(J - 1, B1, D) + F0)[:, 1:]
    nu0 = torch.cat([nu0_R, nu0_P], dim=1).mean(1)
    return {"B0": B0, "B1": B1, "D": D, "ae": alpha, "Be": Be, "nu0": nu0, "x": Jm**2 + Jm + 1, "y": y}
