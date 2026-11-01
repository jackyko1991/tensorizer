"""Build the concrete tensors: temperatures, rate constants, reaction orders and run lengths.

Units: concentrations in mol/L (M), times in s, energies in J/mol. The data are synthetic: a made-up reaction
with a textbook-scale activation energy (50–100 kJ/mol), "measured" with a fixed random seed.
"""

import torch

from . import equation

ORDERS = (0, 1, 2)  # the three rate laws integrated side by side
A0 = 1.0  # initial concentration [A]_0 (M), the same in every run
N_TEMPS = 6  # number of temperatures (one batch entry each)
N_SAMPLES = 20  # "measurements" per run
DTYPE = torch.float64  # least squares on ln k likes double precision


def temperatures(T_min: float, T_max: float, device: torch.device) -> torch.Tensor:
    """T_t in K, shape (T,): evenly spaced between T_min and T_max (at least 5 K apart)."""
    T_max = max(T_max, T_min + 5.0)
    return torch.linspace(T_min, T_max, N_TEMPS, device=device, dtype=DTYPE)


def orders(device: torch.device) -> torch.Tensor:
    """n_o, shape (O,) = (3,): zeroth, first and second order."""
    return torch.tensor(ORDERS, device=device, dtype=DTYPE)


def run_lengths(k: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """t_end,ot, shape (O, T): every run is followed until 90 % of A is used up ([A] = 0.1 [A]_0).

    A cold run is much slower than a hot one, so each (order, temperature) pair gets its own clock,
    just as a chemist would sample a slow run less often.
    """
    return equation.time_to_fraction(k[None, :], n[:, None], A0, 0.1)


def initial_state(n_orders: int, n_temps: int, device: torch.device) -> torch.Tensor:
    """[A]_{ot} at τ = 0, shape (O, T): every run starts at [A]_0."""
    return torch.full((n_orders, n_temps), A0, device=device, dtype=DTYPE)


def measure(conc: torch.Tensor, noise: float, seed: int = 0) -> torch.Tensor:
    """Add Gaussian measurement noise (σ = noise, in M) with a fixed seed.   conc: (T, S) → (T, S)."""
    gen = torch.Generator().manual_seed(seed)  # CPU generator: the same numbers on every device
    eps = torch.randn(conc.shape, generator=gen, dtype=DTYPE).to(conc.device)
    return conc + noise * eps
