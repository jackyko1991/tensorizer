"""Build the concrete tensors: gas constants, the sweep of temperatures and the volume grid (float64)."""

import math

import torch

# van der Waals constants: a in Pa m⁶ / mol², b in m³ / mol
GASES = {
    "CO2": (0.3640, 4.267e-5),
    "water": (0.5536, 3.049e-5),
    "nitrogen": (0.1370, 3.870e-5),
    "argon": (0.1355, 3.201e-5),
}


def temperatures(T_c: float, high: float, low: float, n: int) -> torch.Tensor:
    """T_t falling from high × T_c to low × T_c.  Shape (T,)."""
    return T_c * torch.linspace(high, low, n, dtype=torch.float64)


def volumes(V_c: float, points: int = 1500) -> torch.Tensor:
    """V_v from 0.4 V_c to 8 V_c, logarithmically spaced (the gas side is long and flat).  Shape (V,)."""
    return V_c * torch.logspace(math.log10(0.4), math.log10(8.0), points, dtype=torch.float64)
