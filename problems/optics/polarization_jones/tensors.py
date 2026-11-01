"""Build the concrete tensors: the input Jones vector, the wave plate and the analyzer angles."""

import math

import torch

from .equation import wave_plate

RETARDANCE = {"none": 0.0, "quarter": math.pi / 2, "half": math.pi}  # Γ for each plate choice


def linear_input(angle_deg: float) -> torch.Tensor:
    """Linearly polarized light at the given angle, intensity 1: E = (cos φ, sin φ), shape (2,) complex."""
    phi = math.radians(angle_deg)
    return torch.tensor([math.cos(phi), math.sin(phi)], dtype=torch.complex64)


def plate(kind: str, angle_deg: float) -> torch.Tensor:
    """Jones matrix W of the chosen wave plate (the identity for "none"): (2, 2) complex."""
    return wave_plate(RETARDANCE[kind], math.radians(angle_deg))


def analyzer_angles(n: int = 90) -> torch.Tensor:
    """Angles of the rotating polarizer (analyzer), 0 → 180°, in radians: (A,)."""
    return torch.deg2rad(torch.linspace(0.0, 180.0, n))


def phases(n: int = 100) -> torch.Tensor:
    """ωt over one optical period, used to trace the field ellipse: (T,)."""
    return torch.linspace(0.0, 2 * math.pi, n)
