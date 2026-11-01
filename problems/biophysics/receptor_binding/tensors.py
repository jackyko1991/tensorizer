"""Build the concrete tensors: rate constants of each receptor variant, ligand concentrations, initial state.

Units: concentrations in nM, times in s. Rate constants are quoted in the usual M⁻¹ s⁻¹ and converted here.
"""

import torch

NANO = 1e-9  # 1 nM = 1e-9 M

# name, k_on (M⁻¹ s⁻¹), k_off (s⁻¹) — typical orders of magnitude
VARIANTS = [
    ("TCR, weak", 1e4, 1e-1),
    ("TCR, medium", 5e4, 5e-2),
    ("TCR, high-affinity", 1e5, 5e-3),
    ("antibody", 1e6, 1e-4),
]


def variant_names() -> list[str]:
    return [name for name, _, _ in VARIANTS]


def rate_constants(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """k_on (B,) in nM⁻¹ s⁻¹ and k_off (B,) in s⁻¹."""
    k_on = torch.tensor([kon for _, kon, _ in VARIANTS], device=device) * NANO  # M⁻¹ s⁻¹ → nM⁻¹ s⁻¹
    k_off = torch.tensor([koff for _, _, koff in VARIANTS], device=device)
    return k_on, k_off


def ligand_concentrations(log10_flow_uM: float, device: torch.device, per_decade: int = 4) -> torch.Tensor:
    """L_c in nM, shape (C,): a log-spaced titration from 0.01 nM to 10 µM, plus the sensorgram
    concentration as the LAST entry (that column is the one drawn as an SPR trace)."""
    low, high = -2.0, 4.0  # log10 of nM
    grid = torch.logspace(low, high, int((high - low) * per_decade) + 1, device=device)
    flow = torch.tensor([1e3 * 10**log10_flow_uM], device=device)  # µM → nM
    return torch.cat([grid, flow])


def initial_state(R_T: float, n_variants: int, n_conc: int, device: torch.device) -> torch.Tensor:
    """y0 of shape (2, B, C): all receptors free (R = R_T) and no complexes (C = 0) before the ligand arrives."""
    R = torch.full((n_variants, n_conc), R_T, device=device)
    C = torch.zeros_like(R)
    return torch.stack([R, C])
