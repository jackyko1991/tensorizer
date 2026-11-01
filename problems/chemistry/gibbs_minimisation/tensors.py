"""Build the concrete tensors: species, the element (formula) matrix, the thermochemical table and the feed.

Thermochemistry (all at the standard pressure P° = 1 bar):

* ΔfH°(298.15 K) in kJ/mol and S°(298.15 K) in J/(mol K): rounded values from the NIST-JANAF thermochemical
  tables (Chase, 4th ed., 1998).
* c_p: ILLUSTRATIVE constant heat capacities, roughly the mean of c_p(T) between 298 K and 2000 K. Real c_p rises
  with T, so H(T), S(T) and the compositions below are approximate (semi-quantitative). For real work use the
  NASA 7-coefficient polynomials (e.g. from GRI-Mech 3.0 or Cantera).
"""

import torch

T0 = 298.15  # K, reference temperature of the table
R = 8.314462618  # J/(mol K)

SPECIES = ["CO2", "H2O", "CO", "H2", "O2", "N2", "OH", "H", "O", "NO"]
ELEMENTS = ["C", "H", "O", "N"]

# species: (atoms of C, H, O, N), ΔfH°298 (kJ/mol), S°298 (J/mol/K), c_p (J/mol/K, illustrative constant)
TABLE = {
    "CO2": ((1, 0, 2, 0), -393.52, 213.79, 53.7),
    "H2O": ((0, 2, 1, 0), -241.83, 188.83, 42.7),
    "CO": ((1, 0, 1, 0), -110.53, 197.66, 33.3),
    "H2": ((0, 2, 0, 0), 0.0, 130.68, 31.1),
    "O2": ((0, 0, 2, 0), 0.0, 205.15, 34.8),
    "N2": ((0, 0, 0, 2), 0.0, 191.61, 33.0),
    "OH": ((0, 1, 1, 0), 38.99, 183.71, 30.2),
    "H": ((0, 1, 0, 0), 218.00, 114.72, 20.8),
    "O": ((0, 0, 1, 0), 249.17, 161.06, 21.0),
    "NO": ((0, 0, 1, 1), 90.29, 210.76, 34.0),
}
DH_CH4 = -74.87  # kJ/mol, ΔfH°298 of methane (NIST-JANAF): the fuel enters at 298 K
N2_PER_O2 = 3.76  # dry air: 79 % N2 / 21 % O2 (argon counted as nitrogen)


def element_matrix(device: torch.device) -> torch.Tensor:
    """A_{ij} = number of atoms of element i in species j, shape (E, S) = (4, 10)."""
    rows = [TABLE[s][0] for s in SPECIES]  # (S, E)
    return torch.tensor(rows, dtype=torch.float64, device=device).T


def thermo(device: torch.device) -> dict:
    """ΔfH°298 (kJ/mol), S°298 (J/mol/K) and c_p (J/mol/K), each of shape (S,)."""

    def column(k):
        return torch.tensor([TABLE[s][k] for s in SPECIES], dtype=torch.float64, device=device)

    return {"dH": column(1), "S": column(2), "cp": column(3)}


def feed(phi: torch.Tensor) -> torch.Tensor:
    """Element totals b_{fi} for CH4 + (2/φ)(O2 + 3.76 N2), one row per equivalence ratio φ_f: (F,) → (F, 4).

    φ = 1 is stoichiometric (CH4 + 2 O2 → CO2 + 2 H2O), φ < 1 lean (excess air), φ > 1 rich.
    """
    o2 = 2.0 / phi  # mol O2 per mol CH4
    return torch.stack([torch.ones_like(phi), 4.0 * torch.ones_like(phi), 2.0 * o2, 2.0 * N2_PER_O2 * o2], dim=-1)


def reactant_enthalpy(phi: torch.Tensor) -> torch.Tensor:
    """H of the cold reactants at 298 K in kJ per mol CH4: only CH4 has a non-zero ΔfH (O2, N2 are elements)."""
    return torch.full_like(phi, DH_CH4)


def temperatures(t_min: float, t_max: float, n: int, device: torch.device) -> torch.Tensor:
    """The batch of temperatures T_b (K), shape (B,)."""
    return torch.linspace(t_min, t_max, n, dtype=torch.float64, device=device)
