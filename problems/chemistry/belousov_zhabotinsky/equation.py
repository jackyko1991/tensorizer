r"""Physics → tensor translation: the two-variable Oregonator model of the Belousov–Zhabotinsky reaction.

Indices: b value of the stoichiometric factor f_b in the batch, (i, j) grid cell of the 2D medium,
α, β ∈ {x, z} the two variables of the Jacobian.

Scaled Oregonator (Field–Kőrös–Noyes mechanism, reduced to two variables by Tyson and Fife), with
x ∝ [HBrO₂] (the autocatalyst), z ∝ [Ce⁴⁺] (the oxidised catalyst); bromide is fast and has been eliminated:
    ε dx/dt = x (1 - x) - f z (x - q) / (x + q)
      dz/dt = x - z
The x equation is stiff (ε ≪ 1 and q ≪ 1). Written as production minus loss, ε dx/dt = P - L x with
    P = x + f z q / (x + q) ≥ 0,     L = x + f z / (x + q) ≥ 0,
the Patankar–Euler step x ← (x + Δt P/ε) / (1 + Δt L/ε) stays positive and stable for any Δt.

Linear stability of the steady state (x*, z* = x*): eigenvalues of the Jacobian J_αβ = ∂(dy_α/dt)/∂y_β.
Oscillations start (Hopf bifurcation) where the trace J_αα (α summed) crosses zero while det J > 0.
"""

import torch


def production_loss(x: torch.Tensor, z: torch.Tensor, f: torch.Tensor, q: float):
    """P and L of ε dx/dt = P - L x, both non-negative.   Same shape as x."""
    P = x + f * z * q / (x + q)
    L = x + f * z / (x + q)
    return P, L


def rhs(x: torch.Tensor, z: torch.Tensor, f: torch.Tensor, eps: float, q: float):
    """dx/dt = [x(1 - x) - f z (x - q)/(x + q)] / ε,   dz/dt = x - z.   Same shape as x."""
    return (x * (1 - x) - f * z * (x - q) / (x + q)) / eps, x - z


def steady_state(f: torch.Tensor, q: float) -> torch.Tensor:
    """x* = z* solves (1 - x)(x + q) = f (x - q):  x² + (f + q - 1) x - q (1 + f) = 0, positive root."""
    a = 1 - f - q
    return (a + torch.sqrt(a * a + 4 * q * (1 + f))) / 2


def jacobian(x: torch.Tensor, z: torch.Tensor, f: torch.Tensor, eps: float, q: float) -> torch.Tensor:
    """J_bαβ = ∂(ẋ, ż)/∂(x, z) at (x_b, z_b).   (B, 2, 2)"""
    Jxx = (1 - 2 * x - 2 * f * q * z / (x + q) ** 2) / eps
    Jxz = -f * (x - q) / (x + q) / eps
    one = torch.ones_like(x)
    return torch.stack([torch.stack([Jxx, Jxz], -1), torch.stack([one, -one], -1)], -2)


def trace(J: torch.Tensor) -> torch.Tensor:
    """tr J_b = J_bαα, α summed: the sum of the two eigenvalues; > 0 (with det > 0) means growing oscillations.  (B,)"""
    return torch.einsum("bii->b", J)


def x_nullcline(x: torch.Tensor, f: float, q: float) -> torch.Tensor:
    """The curve where dx/dt = 0: z = x (1 - x)(x + q) / (f (x - q)), for q < x < 1."""
    return x * (1 - x) * (x + q) / (f * (x - q))


def laplacian(u: torch.Tensor, h: float) -> torch.Tensor:
    """5-point Laplacian of u (n, n) with no-flux (Neumann) edges: a cell on the edge is its own missing neighbour."""
    up = torch.cat([u[:1], u[:-1]], 0)  # u_{i-1, j}
    down = torch.cat([u[1:], u[-1:]], 0)  # u_{i+1, j}
    left = torch.cat([u[:, :1], u[:, :-1]], 1)  # u_{i, j-1}
    right = torch.cat([u[:, 1:], u[:, -1:]], 1)  # u_{i, j+1}
    return (up + down + left + right - 4 * u) / h**2
