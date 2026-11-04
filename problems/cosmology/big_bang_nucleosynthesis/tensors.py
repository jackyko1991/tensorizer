"""Build the concrete tensors: species, the reaction network, rate-fit coefficients, the η batch and the T grid.

Units: MeV for temperatures and energies, seconds, cm⁻³ for number densities. float64 throughout (the network is
stiff and abundances span 30 orders of magnitude).
"""

import math

import torch

F64 = torch.float64

# physical constants (CODATA 2018 / PDG)
HBAR = 6.582119569e-22  # MeV s
HBARC = 1.973269804e-11  # MeV cm
M_PLANCK = 1.220890e22  # MeV, G = 1 / M_P²
M_E = 0.51099895  # MeV, electron mass
M_U = 931.49410242  # MeV, atomic mass unit
Q_NP = 1.29333236  # MeV, m_n − m_p
N_AVOGADRO = 6.02214076e23
ZETA3 = 1.2020569031595942
MEV_PER_T9 = 0.08617333262  # k_B × 10⁹ K in MeV

# species i: neutron, proton, deuteron, triton, helium-3, helium-4, lithium-7, beryllium-7
SPECIES = ["n", "p", "d", "t", "³He", "⁴He", "⁷Li", "⁷Be"]
MASS_NUMBER = torch.tensor([1, 1, 2, 3, 3, 4, 7, 7], dtype=F64)  # A_i
CHARGE = torch.tensor([0, 1, 1, 1, 2, 2, 3, 4], dtype=F64)  # Z_i
SPIN_STATES = torch.tensor([2, 2, 3, 2, 2, 1, 4, 4], dtype=F64)  # g_i = 2J + 1
MASS_EXCESS = torch.tensor(  # atomic mass excess Δ_i in MeV (AME 2020); Q = Σ Δ(reactants) − Σ Δ(products)
    [8.0713181, 7.2889711, 13.1357220, 14.9498060, 14.9312155, 2.4249159, 14.9071048, 15.7689990], dtype=F64
)

# the 12 reactions that matter for A ≤ 7 (Smith, Kawano & Malaney 1993): reactants → products (photons not listed)
REACTIONS = [
    (["n"], ["p"]),  # 0  weak: n + ν ↔ p + e⁻, n + e⁺ ↔ p + ν̄, n ↔ p + e⁻ + ν̄
    (["p", "n"], ["d"]),  # 1  p(n,γ)d
    (["d", "p"], ["³He"]),  # 2  d(p,γ)³He
    (["d", "d"], ["³He", "n"]),  # 3  d(d,n)³He
    (["d", "d"], ["t", "p"]),  # 4  d(d,p)t
    (["³He", "n"], ["t", "p"]),  # 5  ³He(n,p)t
    (["t", "d"], ["⁴He", "n"]),  # 6  t(d,n)⁴He
    (["³He", "d"], ["⁴He", "p"]),  # 7  ³He(d,p)⁴He
    (["³He", "⁴He"], ["⁷Be"]),  # 8  ³He(α,γ)⁷Be
    (["t", "⁴He"], ["⁷Li"]),  # 9  t(α,γ)⁷Li
    (["⁷Be", "n"], ["⁷Li", "p"]),  # 10 ⁷Be(n,p)⁷Li
    (["⁷Li", "p"], ["⁴He", "⁴He"]),  # 11 ⁷Li(p,α)⁴He
]

# JINA REACLIB fits (Cyburt et al. 2010, ApJS 189, 240; snapshot "default2" 2025-03-30 as shipped with pynucastro):
# N_A⟨σv⟩ = Σ_sets exp(a0 + a1/T9 + a2 T9^{-1/3} + a3 T9^{1/3} + a4 T9 + a5 T9^{5/3} + a6 ln T9)  [cm³ mol⁻¹ s⁻¹]
# (reaction index, set label, a0..a6). Labels: an06 Ando+ 2006, de04 Descouvemont+ 2004, gi17/go17 Gómez Iñesta+ 2017,
# cd08 Cyburt & Davids 2008, db18 Damone+ 2018.
REACLIB = [
    (1, "an06", [12.3687, 0.0, 0.0, -2.70618, 0.11718, -0.00312788, 0.469127]),
    (1, "an06", [10.7548, 0.0, 0.0, -2.30472, -0.887862, 0.137663, 0.0]),
    (1, "an06", [8.84688, 0.0, 0.0, -0.0102082, -0.0893959, 0.00696704, 1.0]),
    (2, "de04", [8.93525, 0.0, -3.7208, 0.198654, 0.0, 0.0, 0.333333]),
    (2, "de04", [7.52898, 0.0, -3.7208, 0.871782, 0.0, 0.0, -0.666667]),
    (3, "gi17", [19.0876, -0.00019002, -4.2292, 1.6932, -0.0855529, -1.35709e-25, -0.734513]),
    (4, "go17", [18.8052, 4.36209e-05, -4.32296, 1.91572, -0.081562, -3.28804e-22, -0.879518]),
    (5, "de04", [19.2762, 0.0, 0.0, 0.0438557, -0.201527, 0.0153433, 1.0]),
    (5, "de04", [20.3787, 0.0, 0.0, -0.332788, -0.700485, 0.0976521, 0.0]),
    (6, "de04", [39.3457, 0.0, -4.5244, -16.4028, 1.73103, -0.122966, 2.31304]),
    (6, "de04", [25.1794, 0.0, -4.5244, 0.350337, 0.58747, -8.84909, -0.666667]),
    (7, "de04", [41.2969, 0.0, -7.182, -17.1349, 1.36908, -0.0814423, 3.35395]),
    (7, "de04", [24.6839, 0.0, -7.182, 0.473288, 1.46847, -27.9603, -0.666667]),
    (8, "cd08", [17.7075, 0.0, -12.8271, -3.8126, 0.0942285, -0.00301018, 1.33333]),
    (8, "cd08", [15.6099, 0.0, -12.8271, -0.0308225, -0.654685, 0.0896331, -0.666667]),
    (9, "de04", [13.6162, 0.0, -8.0805, -0.217514, -0.114859, 0.0470043, -0.666667]),
    (10, "db18", [21.7899, 0.000728098, -0.30254, -0.3602, 0.17472, -0.0223, -0.4581]),
    (11, "de04", [21.8999, -26.1527, 0.0, 0.0, 0.0, 0.0, -1.5]),
    (11, "de04", [20.4438, 0.0, -8.4727, 0.297934, 0.0582335, -0.00413383, -0.666667]),
    (11, "de04", [14.2538, -4.478, 0.0, 0.0, 0.0, 0.0, -1.5]),
    (11, "de04", [11.9576, 0.0, -8.4727, 0.417943, 5.34565, -4.8684, -0.666667]),
]

# observed primordial abundances (cited constants, plotted as bands)
OBSERVED = {
    "Yp": (0.245, 0.003),  # PDG 2024 review of BBN (Fields, Molaro & Sarkar), He II/He I in metal-poor H II regions
    "DH": (2.527e-5, 0.030e-5),  # Cooke, Pettini & Steidel 2018, quasar absorption systems
    "LiH": (1.6e-10, 0.3e-10),  # Spite plateau of metal-poor halo stars (Sbordone et al. 2010)
}
ETA_PLANCK = (6.10e-10, 0.06e-10)  # from Planck 2018 Ω_b h² = 0.0224 (η₁₀ = 273.9 Ω_b h²)


def stoichiometry(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """R_{ri}, P_{ri}: how many of species i each reaction r consumes and produces, shape (Rx, N) each."""
    R = torch.zeros(len(REACTIONS), len(SPECIES), dtype=F64)
    P = torch.zeros_like(R)
    for r, (ins, outs) in enumerate(REACTIONS):
        for s in ins:
            R[r, SPECIES.index(s)] += 1
        for s in outs:
            P[r, SPECIES.index(s)] += 1
    return R.to(device), P.to(device)


def q_values(device: torch.device) -> torch.Tensor:
    """Q_r = Σ_i (R_ri − P_ri) Δ_i, energy released by each reaction (MeV), shape (Rx,). Q_0 = m_n − m_p."""
    R, P = stoichiometry(device)
    Q = (R - P) @ MASS_EXCESS.to(device)
    Q[0] = Q_NP
    return Q


def reaclib_table(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """The fit coefficients a_{sk} (S, 7) and the reaction index of every set (S,)."""
    a = torch.tensor([c for _, _, c in REACLIB], dtype=F64, device=device)
    idx = torch.tensor([r for r, _, _ in REACLIB], device=device)
    return a, idx


def eta_batch(n: int, eta_mark: float, device: torch.device) -> torch.Tensor:
    """η values: n log-spaced from 10⁻¹⁰ to 10⁻⁹, plus the marked value (Planck) as the last member, (n + 1,)."""
    grid = torch.logspace(-10, -9, n, dtype=F64)
    return torch.cat([grid, torch.tensor([eta_mark], dtype=F64)]).to(device)


def temperature_grid(T_start: float, T_end: float, n_steps: int, device: torch.device) -> torch.Tensor:
    """Photon temperatures in MeV, log-spaced from hot to cold, (S,): the integration variable is u = −ln T."""
    return torch.logspace(math.log10(T_start), math.log10(T_end), n_steps + 1, dtype=F64, device=device)


def initial_abundances(T0: float, B: int, device: torch.device) -> torch.Tensor:
    """Y_{bi} = n_i / n_baryon at T0: neutrons and protons in weak equilibrium, n/p = e^{−Q/T}; no nuclei yet."""
    Y = torch.zeros(B, len(SPECIES), dtype=F64, device=device)
    Y[:, 0] = 1 / (1 + math.exp(Q_NP / T0))
    Y[:, 1] = 1 - Y[:, 0]
    return Y
