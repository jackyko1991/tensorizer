"""Build the concrete tensors: random initial states, charges, network weights and a random rotation."""

import math

import torch


def initial_states(n_systems: int, n_particles: int, gen: torch.Generator, device: torch.device):
    """Random systems: positions x (B, N, 3), velocities v (B, N, 3) and charges c (B, N) in {-1, +1}.

    Positions and velocities are isotropic Gaussians, so the data set has no preferred direction.
    The centre of mass is put at rest at the origin.
    """
    x = torch.randn(n_systems, n_particles, 3, generator=gen)
    v = 0.5 * torch.randn(n_systems, n_particles, 3, generator=gen)
    x = x - x.mean(dim=1, keepdim=True)
    v = v - v.mean(dim=1, keepdim=True)
    c = torch.randint(0, 2, (n_systems, n_particles), generator=gen).float() * 2 - 1
    return x.to(device), v.to(device), c.to(device)


def mlp_weights(sizes: list[int], gen: torch.Generator, device: torch.device, last_bias: float = 0.0):
    """[(W (out, in), b (out,)), ...] with the usual 1/√in scale. All tensors require gradients."""
    weights = []
    for i, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        W = torch.randn(n_out, n_in, generator=gen) / math.sqrt(n_in)
        b = torch.zeros(n_out)
        if i == len(sizes) - 2:
            W = 0.1 * W  # start close to "no correction"
            b = b + last_bias
        weights.append((W.to(device).requires_grad_(), b.to(device).requires_grad_()))
    return weights


def egnn_weights(hidden: int, n_layers: int, gen: torch.Generator, device: torch.device) -> dict:
    """Weights of the EGNN: an embedding of (c_i, |v_i|²) and n_layers × (φ_e, φ_x, φ_v, φ_h)."""
    H = hidden
    layers = [
        {
            "phi_e": mlp_weights([2 * H + 2, H, H], gen, device),
            "phi_x": mlp_weights([H, H, 1], gen, device),
            "phi_v": mlp_weights([H, H, 1], gen, device, last_bias=1.0),  # φ_v ≈ 1: keep the velocity at first
            "phi_h": mlp_weights([2 * H, H, H], gen, device),
        }
        for _ in range(n_layers)
    ]
    return {"embed": mlp_weights([2, H], gen, device), "layers": layers}


def parameters(net) -> list[torch.Tensor]:
    """Flatten the nested weights (dict / list / tuple) into the list the optimiser needs."""
    if isinstance(net, torch.Tensor):
        return [net]
    items = net.values() if isinstance(net, dict) else net
    return [p for item in items for p in parameters(item)]


def random_rotation(gen: torch.Generator, device: torch.device) -> torch.Tensor:
    """A random proper rotation R_{ab} (3, 3) with det R = +1, from the QR decomposition of a Gaussian matrix."""
    Q, R = torch.linalg.qr(torch.randn(3, 3, generator=gen, dtype=torch.float64))
    Q = Q * torch.sign(torch.diagonal(R))  # make the decomposition unique
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.float().to(device)
