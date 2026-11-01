r"""Physics → tensor translation.

An enzyme E binds its substrate S, forms a complex ES and turns it into product P:

    E + S  ⇌  ES  →  E + P        rates k1 E S (binding), k₋₁ ES (unbinding), k2 ES (catalysis)

For a batch of enzymes b and a titration of substrate concentrations c, mass action gives

    \dot E_{bc}  = -k^1_b E_{bc} S_{bc} + (k^{-1}_b + k^2_b) ES_{bc}
    \dot S_{bc}  = -k^1_b E_{bc} S_{bc} +  k^{-1}_b ES_{bc}
    \dot ES_{bc} = +k^1_b E_{bc} S_{bc} - (k^{-1}_b + k^2_b) ES_{bc}
    \dot P_{bc}  =  k^2_b ES_{bc}

so E + ES = E0 (enzyme is recycled) and S + ES + P = S0 (substrate atoms are conserved).
No index is summed in the kinetics: every (b, c) pair is its own test tube; the einsum calls make the
broadcasting of k_b over c explicit. The Lineweaver–Burk fit sums over the concentration index c.
"""

import torch


def reaction_rates(E, S, ES, k1, k_1, k2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binding k1 E S, unbinding k₋₁ ES and catalysis k2 ES.   E, S, ES: (B, C);  k: (B,)  →  three (B, C)."""
    binding = torch.einsum("b,bc,bc->bc", k1, E, S)  # needs an enzyme AND a substrate to meet
    unbinding = torch.einsum("b,bc->bc", k_1, ES)  # the complex falls apart unchanged
    catalysis = torch.einsum("b,bc->bc", k2, ES)  # the complex releases product and free enzyme
    return binding, unbinding, catalysis


def rhs(y: torch.Tensor, k1, k_1, k2) -> torch.Tensor:
    """dy/dt for y of shape (4, B, C) = (E, S, ES, P)."""
    E, S, ES, _ = y
    v1, v_1, v2 = reaction_rates(E, S, ES, k1, k_1, k2)
    return torch.stack([-v1 + v_1 + v2, -v1 + v_1, v1 - v_1 - v2, v2])


def michaelis_constant(k1, k_1, k2) -> torch.Tensor:
    """K_m = (k₋₁ + k2) / k1: the substrate level at which the enzyme runs at half speed.   (B,)"""
    return (k_1 + k2) / k1


def max_velocity(k2: torch.Tensor, E0: float) -> torch.Tensor:
    """V_max = k2 E0 = k_cat E0: every enzyme busy all the time.   (B,)"""
    return k2 * E0


def mm_rate(S: torch.Tensor, V_max: torch.Tensor, K_m: torch.Tensor) -> torch.Tensor:
    """Quasi-steady-state rate v_bc = V_max,b S_bc / (K_m,b + S_bc).   S: (B, C) or (C,)  →  (B, C)."""
    S = S.expand(len(V_max), -1)
    return torch.einsum("b,bc->bc", V_max, S / (K_m[:, None] + S))


def qssa_rhs(S: torch.Tensor, V_max: torch.Tensor, K_m: torch.Tensor) -> torch.Tensor:
    """dS/dt = -V_max S / (K_m + S): the reduced (Michaelis–Menten) model, one equation per test tube."""
    return -mm_rate(S, V_max, K_m)


def integrated_mm_time(S: torch.Tensor, S0: torch.Tensor, V_max: torch.Tensor, K_m: torch.Tensor) -> torch.Tensor:
    """Integrated Michaelis–Menten law: the time to go from S0 down to S,  t = (S0 − S + K_m ln(S0/S)) / V_max."""
    return (S0 - S + K_m[:, None] * torch.log(S0 / S)) / V_max[:, None]


def lineweaver_burk(S: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Fit 1/v = (K_m/V_max)(1/S) + 1/V_max by least squares for every enzyme at once.   S, v: (B, C).

    Design matrix A_{bck} = [1/S_{bc}, 1] (B, C, 2) and data y_{bc} = 1/v_{bc}. The normal equations

        G_{bkl} a_{bl} = r_{bk},   G_{bkl} = A_{bck} A_{bcl},   r_{bk} = A_{bck} y_{bc}

    sum over the concentration index c. Returns (K_m, V_max), each (B,).
    """
    A = torch.stack([1.0 / S, torch.ones_like(S)], dim=-1)  # (B, C, 2)
    G = torch.einsum("bck,bcl->bkl", A, A)  # (B, 2, 2), c summed
    r = torch.einsum("bck,bc->bk", A, 1.0 / v)  # (B, 2), c summed
    slope, intercept = torch.linalg.solve(G, r).T  # a_bl: one small 2×2 system per enzyme
    return slope / intercept, 1.0 / intercept
