"""Build the concrete tensors: the polynomial's coefficients, its roots and the complex pixel grids."""

import math

import torch

# Coefficients c_k from the constant term up (lowest power first), all monic.
POLYNOMIALS = {
    "z3_minus_1": [-1.0, 0.0, 0.0, 1.0],  # z³ − 1: the three cube roots of unity
    "z4_minus_1": [-1.0, 0.0, 0.0, 0.0, 1.0],  # z⁴ − 1: ±1, ±i
    "z3_minus_2z_plus_2": [2.0, -2.0, 0.0, 1.0],  # z³ − 2z + 2: has an attracting 2-cycle 0 → 1 → 0
}
LABELS = {"z3_minus_1": "z³ − 1", "z4_minus_1": "z⁴ − 1", "z3_minus_2z_plus_2": "z³ − 2z + 2"}


def coefficients(name: str, device: torch.device, dtype=torch.complex64) -> torch.Tensor:
    """c_k, shape (K,) = (degree + 1,)."""
    return torch.tensor(POLYNOMIALS[name], dtype=dtype, device=device)


def roots(c: torch.Tensor) -> torch.Tensor:
    """The roots r_j as eigenvalues of the companion matrix (works for any monic polynomial): (J,)."""
    n = len(c) - 1
    companion = torch.zeros(n, n, dtype=torch.complex128)
    companion[1:, :-1] = torch.eye(n - 1)  # ones on the sub-diagonal
    companion[:, -1] = -c[:-1].to(torch.complex128).cpu()
    r = torch.linalg.eigvals(companion)
    order = torch.argsort(torch.atan2(r.imag, r.real) % (2 * math.pi))  # sorted by angle: stable colours
    return r[order].to(c.dtype).to(c.device)


def grid(center: complex, half_width: float, n: int, device: torch.device) -> torch.Tensor:
    """Complex pixel grid z_{hw} = x_w + i y_h of n × n pixels around center: (H, W). n even, so 0 is never a pixel."""
    t = torch.linspace(-half_width, half_width, n, device=device)
    x = center.real + t  # (W,)
    y = center.imag - t  # (H,): the top row has the largest imaginary part
    return torch.complex(x[None, :].expand(n, n), y[:, None].expand(n, n))
