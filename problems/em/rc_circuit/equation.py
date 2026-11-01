r"""Physics → tensor translation.

Kirchhoff's voltage law for a resistor R and capacitor C in series with a source V_s(t):

    V_s = R I + V,      I = C \frac{dV}{dt}     ⇒     \frac{dV}{dt} = \frac{V_s - V}{R C}

We solve B circuits at once, one for each resistance R_b. The index b only labels the
circuit — nothing is summed — so every formula is plain element-wise tensor arithmetic:

    \dot V_b = (V_s - V_b) / (R_b C),     I_b = (V_s - V_b) / R_b,     U_b = ½ C V_b^2

The time constant τ_b = R_b C sets how fast each capacitor charges: V(τ) = (1 − 1/e) V_s.
"""

import torch


def source(t: float, V0: float, shape: str, period: float) -> float:
    """V_s(t): a constant battery ("step"), or a square wave switching between V0 and 0 every half period."""
    if shape == "square" and (t % period) >= period / 2:
        return 0.0
    return V0


def rhs(V: torch.Tensor, Vs: float, R: torch.Tensor, C: float) -> torch.Tensor:
    r"""\dot V_b = (V_s − V_b)/(R_b C).   V, R: (B,) → (B,)"""
    return (Vs - V) / (R * C)


def current(V: torch.Tensor, Vs: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """I_b = (V_s − V_b)/R_b: the voltage left over for the resistor, divided by R.   → same shape as V"""
    return (Vs - V) / R


def stored_energy(V: torch.Tensor, C: float) -> torch.Tensor:
    """U_b = ½ C V_b²: energy in the capacitor's electric field.   → same shape as V"""
    return 0.5 * C * V**2
