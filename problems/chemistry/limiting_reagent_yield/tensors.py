"""Build the concrete tensors: the reaction (stoichiometric numbers, molar masses) and a batch of experiments.

Units: masses in g, amounts in mol, molar masses in g/mol. Atomic masses are standard values (IUPAC, rounded).
"""

import torch

ATOMIC_MASS = {"H": 1.008, "N": 14.007, "Al": 26.982, "I": 126.904}

# Each reaction: two reactants then one product, with signed stoichiometric numbers ν_s (< 0 consumed, > 0 formed),
# the atoms in each species, and a few lab-notebook runs: (mass reactant 1, mass reactant 2, product weighed), in g.
REACTIONS = {
    "aluminium_iodine": {
        "species": ["Al", "I₂", "AlI₃"],
        "nu": [-2, -3, 2],  # 2 Al + 3 I₂ → 2 AlI₃
        "atoms": [{"Al": 1}, {"I": 2}, {"Al": 1, "I": 3}],
        "runs": [(0.50, 6.00, 4.10), (2.00, 2.00, 1.80), (1.00, 14.11, 12.90)],
    },
    "haber": {
        "species": ["N₂", "H₂", "NH₃"],
        "nu": [-1, -3, 2],  # N₂ + 3 H₂ → 2 NH₃
        "atoms": [{"N": 2}, {"H": 2}, {"N": 1, "H": 3}],
        "runs": [(28.0, 6.0, 5.1), (100.0, 10.0, 9.0), (28.0, 20.0, 6.2)],  # a single pass converts ~15 %
    },
}


def stoichiometry(reaction: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """ν_s (S,), molar masses M_s (S,) and species names.   M_s = A_{es} m_e (summed over elements e)."""
    r = REACTIONS[reaction]
    elements = sorted(ATOMIC_MASS)
    A = torch.tensor([[a.get(e, 0) for a in r["atoms"]] for e in elements], dtype=torch.float64, device=device)
    m = torch.tensor([ATOMIC_MASS[e] for e in elements], dtype=torch.float64, device=device)
    molar = torch.einsum("es,e->s", A, m)  # (S,) g/mol
    nu = torch.tensor(r["nu"], dtype=torch.float64, device=device)
    return nu, molar, r["species"]


def experiments(reaction: str, mass_1: float, mass_2: float, measured: float, device: torch.device):
    """Starting masses of all species (B, S) in g — products start at 0 — and the product weighed (B,).

    The batch is the lab-notebook runs plus "your run" from the parameters as the LAST row.
    """
    runs = REACTIONS[reaction]["runs"] + [(mass_1, mass_2, measured)]
    m0 = torch.tensor([[a, b, 0.0] for a, b, _ in runs], dtype=torch.float64, device=device)
    weighed = torch.tensor([w for _, _, w in runs], dtype=torch.float64, device=device)
    return m0, weighed
