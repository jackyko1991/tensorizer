r"""Physics → tensor translation.

The physics is a damped mass on a spring,

    m ẍ + μ ẋ + k x = 0,      x(0) = 1,  ẋ(0) = 0,

and the unknown function x(t) is replaced by a small neural network x̂(t; w). For a batch of time points n
each layer l of the network is

    h^{(l+1)}_{ni} = tanh( W^{(l)}_{ij} h^{(l)}_{nj} + b^{(l)}_i )        j summed (the neurons of layer l)

with h^{(0)}_n = t_n and a final linear layer x̂_n = W^{(L)}_{1j} h^{(L)}_{nj} + b^{(L)}_1.

The network is trained so that it (1) passes through a few measured points and (2) satisfies the equation of
motion at collocation times t_c. The ODE residual

    r_c = m x̂''_c + μ x̂'_c + k x̂_c

needs x̂' and x̂'': torch.autograd differentiates the network itself with respect to its input t.
"""

import math

import torch


def layer(h: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pre-activation a_{ni} = W_{ij} h_{nj} + b_i.   h: (N, J), W: (I, J), b: (I,) → (N, I);  j summed."""
    return torch.einsum("ij,nj->ni", W, h) + b


def network(t: torch.Tensor, weights: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """x̂(t; w): tanh hidden layers, then one linear output layer.   t: (N,) → x̂: (N,)."""
    h = t[:, None]  # (N, 1): the only input feature is the time
    for W, b in weights[:-1]:
        h = torch.tanh(layer(h, W, b))  # h^{(l+1)}_{ni} = tanh(W_ij h_nj + b_i)
    W, b = weights[-1]
    return layer(h, W, b)[:, 0]  # (N, 1) → (N,)


def time_derivative(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """dy_n/dt_n for every point n (each y_n depends only on its own t_n).   (N,) → (N,).

    create_graph=True keeps the derivative itself differentiable, so we can take it again (x̂'')
    and later backpropagate the loss through it into the weights.
    """
    (dy,) = torch.autograd.grad(y, t, grad_outputs=torch.ones_like(y), create_graph=True)
    return dy


def residual(x: torch.Tensor, x_t: torch.Tensor, x_tt: torch.Tensor, m: float, mu, k: float) -> torch.Tensor:
    """r_c = m x''_c + μ x'_c + k x_c: zero wherever x(t) obeys the equation of motion.   (C,)."""
    return m * x_tt + mu * x_t + k * x


def network_and_derivatives(t: torch.Tensor, weights, second: bool = True):
    """x̂, x̂' and x̂'' at the times t, all (N,): differentiate the network itself with respect to its input.
    second=False skips x̂'' (returned as None) when only the slope is needed."""
    t = t.clone().requires_grad_(True)  # we differentiate with respect to t, so it must track gradients
    x = network(t, weights)
    x_t = time_derivative(x, t)
    x_tt = time_derivative(x_t, t) if second else None
    return x, x_t, x_tt


def exact_solution(t: torch.Tensor, m: float, mu: float, k: float) -> torch.Tensor:
    """Underdamped solution with x(0) = 1, ẋ(0) = 0:  x = e^{-δt} (cos ωt + (δ/ω) sin ωt),
    δ = μ / 2m,  ω = √(k/m - δ²)."""
    delta = mu / (2 * m)
    omega = math.sqrt(k / m - delta**2)
    return torch.exp(-delta * t) * (torch.cos(omega * t) + delta / omega * torch.sin(omega * t))
