"""Build the concrete tensors: the four corners of the cycle and its legs as (V, p) samples."""

import torch

from .equation import adiabat_pressure, adiabat_volume, isotherm_pressure

V1 = 1e-3  # volume at the start of the hot isotherm (m³) = 1 litre
GAMMA = {"monatomic": 5 / 3, "diatomic": 7 / 5}  # heat-capacity ratio γ = C_p / C_V


def corner_volumes(T_h: float, T_c: float, ratio: float, gamma: float) -> list[float]:
    """V1 → V2 hot isotherm (expansion), V2 → V3 adiabat, V3 → V4 cold isotherm, V4 → V1 adiabat."""
    V2 = V1 * ratio
    V3 = adiabat_volume(V2, T_h, T_c, gamma)
    V4 = adiabat_volume(V1, T_h, T_c, gamma)
    return [V1, V2, V3, V4]


def cycle(n: float, T_h: float, T_c: float, ratio: float, gamma: float, points: int = 200) -> torch.Tensor:
    """legs[l, k, :] = (V, p) of sample k on leg l.  Shape (4, K, 2), float64 for accurate areas.

    Volumes are spaced geometrically, V = V_a (V_b / V_a)^s with s ∈ [0, 1], which suits 1/V-shaped curves.
    """
    V = corner_volumes(T_h, T_c, ratio, gamma)
    T_start = [T_h, T_h, T_c, T_c]  # temperature at the first corner of each leg
    s = torch.linspace(0, 1, points, dtype=torch.float64)  # (K,)
    legs = []
    for leg in range(4):
        Va, Vb = V[leg], V[(leg + 1) % 4]
        v = Va * (Vb / Va) ** s  # (K,) volumes from Va to Vb
        if leg % 2 == 0:  # legs 0 and 2: isotherms
            p = isotherm_pressure(v, n, T_start[leg])
        else:  # legs 1 and 3: adiabats through the corner (Va, pa)
            pa = isotherm_pressure(torch.tensor(Va, dtype=torch.float64), n, T_start[leg]).item()
            p = adiabat_pressure(v, pa, Va, gamma)
        legs.append(torch.stack([v, p], dim=-1))  # (K, 2)
    return torch.stack(legs)
