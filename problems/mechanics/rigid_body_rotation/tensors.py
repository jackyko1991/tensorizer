"""Build the concrete tensors: the box's inertia tensor and the initial state (orientation, spin).

float64 keeps the conserved quantities accurate to many digits over thousands of steps.
"""

import torch

DTYPE = torch.float64
AXES = {"minor": 0, "intermediate": 1, "major": 2}  # body axis index for each choice


def box_inertia(mass: float, a: float, b: float, c: float, device: torch.device | None = None) -> torch.Tensor:
    """I_ij of a solid box with sides a > b > c along body axes 1, 2, 3: shape (3, 3).

    I = (M / 12) diag(b² + c², a² + c², a² + b²)

    The longest side a gives the smallest moment (minor axis 1), the shortest side c the largest (major axis 3).
    """
    moments = mass / 12 * torch.tensor([b**2 + c**2, a**2 + c**2, a**2 + b**2], dtype=DTYPE, device=device)
    return torch.diag(moments)


def initial_state(axis: str, spin: float, perturbation: float, device: torch.device | None = None) -> torch.Tensor:
    """y0 of shape (4, 3): orientation R = identity (rows 0–2) and body-frame ω (row 3).

    ω is mostly along the chosen axis, plus a small nudge of size perturbation · spin on the other two.
    """
    omega = perturbation * spin * torch.ones(3, dtype=DTYPE, device=device)  # (3,)
    omega[AXES[axis]] = spin
    R = torch.eye(3, dtype=DTYPE, device=device)  # (3, 3)
    return torch.cat([R, omega[None]], dim=0)
