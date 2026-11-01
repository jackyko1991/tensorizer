"""Build the concrete tensors: rate constants of each enzyme, substrate concentrations, initial state.

Units: concentrations in µM, times in s, so k1 is in µM⁻¹ s⁻¹ and k₋₁, k2 are in s⁻¹.
"""

import torch

# name, k1 (µM⁻¹ s⁻¹), k₋₁ (s⁻¹), k2 = k_cat (s⁻¹)   →   K_m = (k₋₁ + k2) / k1
ENZYMES = [
    ("A", 1.0, 1.0, 4.0),  # K_m =  5 µM, k_cat =  4 s⁻¹: binds tightly
    ("B", 0.5, 2.0, 3.0),  # K_m = 10 µM, k_cat =  3 s⁻¹: in between
    ("C", 0.5, 5.0, 10.0),  # K_m = 30 µM, k_cat = 10 s⁻¹: fast but weak binding
]

SUBSTRATE = [2.5, 5.0, 10.0, 20.0, 40.0, 80.0]  # S0 of the titration, µM


def enzyme_names() -> list[str]:
    return [name for name, *_ in ENZYMES]


def rate_constants(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """k1, k₋₁, k2, each of shape (B,): one entry per enzyme b."""
    k = torch.tensor([rates for _, *rates in ENZYMES], device=device)  # (B, 3)
    return k[:, 0], k[:, 1], k[:, 2]


def substrate_concentrations(device: torch.device) -> torch.Tensor:
    """S0_c in µM, shape (C,)."""
    return torch.tensor(SUBSTRATE, device=device)


def initial_state(E0: float, S0: torch.Tensor, n_enzymes: int) -> torch.Tensor:
    """y0 of shape (4, B, C) = (E, S, ES, P): free enzyme and substrate only, no complex or product yet."""
    S = S0.expand(n_enzymes, -1).clone()  # (B, C): every enzyme gets the whole titration
    E = torch.full_like(S, E0)
    return torch.stack([E, S, torch.zeros_like(S), torch.zeros_like(S)])
