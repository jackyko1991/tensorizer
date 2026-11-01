"""Build the concrete tensors: stoichiometry, starting moles and the batch of conditions (T, P).

Thermodynamic data are approximate standard values at 298 K (gas phase, P° = 1 bar), assumed independent of
temperature. They are good to a few percent near room temperature and only indicative at 700 K.
"""

import torch

DTYPE = torch.float64

REACTIONS = {
    "n2o4": {
        "label": "N₂O₄ ⇌ 2 NO₂",
        "species": ["N₂O₄", "NO₂"],
        "nu": [-1.0, 2.0],  # stoichiometric numbers ν_i (negative = reactant)
        "n0": [1.0, 0.0],  # starting moles
        "dH": 57.2e3,  # ΔH° ≈ +57.2 kJ/mol (endothermic: breaking the N–N bond)
        "dS": 175.8,  # ΔS° ≈ +176 J/(mol K) (one molecule → two)
        "T_curves": [280.0, 300.0, 320.0, 340.0, 360.0],
    },
    "haber": {
        "label": "N₂ + 3 H₂ ⇌ 2 NH₃",
        "species": ["N₂", "H₂", "NH₃"],
        "nu": [-1.0, -3.0, 2.0],
        "n0": [1.0, 3.0, 0.0],  # stoichiometric feed, as in an ammonia plant
        "dH": -92.2e3,  # ΔH° ≈ −92.2 kJ/mol (exothermic)
        "dS": -198.7,  # ΔS° ≈ −199 J/(mol K) (four molecules → two)
        "T_curves": [600.0, 650.0, 700.0, 750.0, 800.0],
    },
}


def reaction(name: str, device: torch.device) -> dict:
    """The reaction's data with ν (S,) and n0 (S,) as tensors."""
    r = dict(REACTIONS[name])
    r["nu"] = torch.tensor(r["nu"], device=device, dtype=DTYPE)
    r["n0"] = torch.tensor(r["n0"], device=device, dtype=DTYPE)
    return r


def pressures(device: torch.device, n: int = 61) -> torch.Tensor:
    """P_p in bar, shape (P,): log-spaced from 0.1 to 100 bar."""
    return torch.logspace(-1, 2, n, device=device, dtype=DTYPE)


def condition_grid(T_list: list[float], P: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (temperature, pressure) pair as one flat batch: T_b, P_b of shape (B,) = (len(T_list) · P,)."""
    T = torch.tensor(T_list, device=P.device, dtype=DTYPE)
    T_b, P_b = torch.meshgrid(T, P, indexing="ij")  # (nT, P) each
    return T_b.reshape(-1), P_b.reshape(-1)
