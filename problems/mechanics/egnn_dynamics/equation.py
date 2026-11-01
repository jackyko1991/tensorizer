r"""Physics → tensor translation: the true force law, an E(n)-equivariant GNN and a plain MLP.

True dynamics. N charged particles (unit mass) held together by weak springs between every pair:

    \ddot x_{bia} = \sum_j w_{bij} (x_{bia} - x_{bja}),
    w_{bij} = c_{bi} c_{bj} (d_{bij} + \varepsilon^2)^{-3/2} - k_s,     d_{bij} = r_{bija} r_{bija}

b = system in the batch, i, j = particles, a = Cartesian component. Repeated a (in d) and j are summed.

EGNN layer (Satorras, Hoogeboom & Welling 2021, the variant with velocities):

    m_{bijh} = \phi_e(h_{bi}, h_{bj}, d_{bij}, q_{bij})                      message on every pair (i ≠ j)
    v_{bia} \leftarrow \phi_v(h_{bi}) v_{bia} + \frac{1}{N-1} \sum_j r_{bija} \phi_x(m_{bij})
    x_{bia} \leftarrow x_{bia} + \tau v_{bia}
    h_{bih} \leftarrow h_{bih} + \phi_h(h_{bi}, \sum_j m_{bijh})

Only the invariant d_{bij} = |x_i - x_j|^2 enters the MLPs φ, and vectors are built only from r_{bija} = x_i - x_j
and v_{bia}. That is why rotating (and translating) the input rotates (and translates) the output exactly.
"""

import torch

Weights = list[tuple[torch.Tensor, torch.Tensor]]  # [(W (out, in), b (out,)), ...] for one small MLP


# --- true physics -----------------------------------------------------------------------------------


def separations(x: torch.Tensor) -> torch.Tensor:
    """r_{bija} = x_{bia} - x_{bja}: (B, N, 3) → (B, N, N, 3), all pairs by broadcasting."""
    return x[:, :, None, :] - x[:, None, :, :]


def squared_distances(r: torch.Tensor) -> torch.Tensor:
    """d_{bij} = r_{bija} r_{bija}  (repeated a ⇒ summed): (B, N, N, 3) → (B, N, N)."""
    return torch.einsum("bija,bija->bij", r, r)


def acceleration(x: torch.Tensor, c: torch.Tensor, k_spring: float, eps: float) -> torch.Tensor:
    r"""\ddot x_{bia} = \sum_j w_{bij} r_{bija}.   x: (B, N, 3), charges c: (B, N) → (B, N, 3)."""
    r = separations(x)
    d = squared_distances(r)
    coulomb = torch.einsum("bi,bj->bij", c, c) * (d + eps**2) ** -1.5  # like charges push apart (w > 0)
    w = coulomb - k_spring  # the springs pull every pair together; the i = j term has r = 0 and drops out
    return torch.einsum("bij,bija->bia", w, r)


def rhs(y: torch.Tensor, c: torch.Tensor, k_spring: float, eps: float) -> torch.Tensor:
    """dy/dt for the state y (B, 2, N, 3): y[:, 0] = positions, y[:, 1] = velocities."""
    return torch.stack([y[:, 1], acceleration(y[:, 0], c, k_spring, eps)], dim=1)


# --- neural networks ---------------------------------------------------------------------------------


def linear(h: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """y_{...o} = W_{oi} h_{...i} + b_o  (repeated i ⇒ summed), on any number of leading batch indices."""
    return torch.einsum("oi,...i->...o", W, h) + b


def mlp(h: torch.Tensor, weights: Weights) -> torch.Tensor:
    """Linear → SiLU → … → Linear."""
    for W, b in weights[:-1]:
        h = torch.nn.functional.silu(linear(h, W, b))
    W, b = weights[-1]
    return linear(h, W, b)


def egnn_layer(
    h: torch.Tensor, x: torch.Tensor, v: torch.Tensor, q: torch.Tensor, layer: dict, tau: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One equivariant layer. h (B, N, H), x and v (B, N, 3), q_{bij} = c_i c_j (B, N, N)."""
    B, N, H = h.shape
    r = separations(x)  # (B, N, N, 3)
    d = squared_distances(r)  # (B, N, N)   invariant
    pair = torch.cat(
        [h[:, :, None, :].expand(B, N, N, H), h[:, None, :, :].expand(B, N, N, H), d[..., None], q[..., None]], dim=-1
    )  # (B, N, N, 2H + 2)
    not_self = 1.0 - torch.eye(N, device=x.device)  # (N, N): no message from a particle to itself
    m = mlp(pair, layer["phi_e"]) * not_self[None, :, :, None]  # m_{bijh}   (B, N, N, H)

    weight = mlp(m, layer["phi_x"])[..., 0]  # φ_x(m_{bij})   (B, N, N)   a scalar per pair
    push = torch.einsum("bija,bij->bia", r, weight) / (N - 1)  # Σ_j r_{bija} φ_x(m_{bij})   (B, N, 3)
    v = mlp(h, layer["phi_v"]) * v + push  # φ_v(h_i) is a scalar per particle: (B, N, 1) * (B, N, 3)
    x = x + tau * v

    m_sum = m.sum(dim=2)  # Σ_j m_{bijh}   (B, N, H)
    h = h + mlp(torch.cat([h, m_sum], dim=-1), layer["phi_h"])
    return h, x, v


def egnn(x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, net: dict, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict (x, v) one time step dt later. The L layers each move the particles by dt / L."""
    speed2 = torch.einsum("bia,bia->bi", v, v)  # |v_i|², invariant
    h = mlp(torch.stack([c, speed2], dim=-1), net["embed"])  # (B, N, H) node features
    q = torch.einsum("bi,bj->bij", c, c)  # edge attribute: product of charges
    tau = dt / len(net["layers"])
    for layer in net["layers"]:
        h, x, v = egnn_layer(h, x, v, q, layer, tau)
    return x, v


def plain_mlp(
    x: torch.Tensor, v: torch.Tensor, c: torch.Tensor, weights: Weights, dt: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Baseline: flatten every coordinate into one vector and let an MLP predict the change. Not equivariant."""
    B, N, _ = x.shape
    flat = torch.cat([x.reshape(B, -1), v.reshape(B, -1), c], dim=-1)  # (B, 7N)
    delta = mlp(flat, weights).reshape(B, 2, N, 3)
    return x + dt * v + delta[:, 0], v + delta[:, 1]  # same "drift" head start the EGNN gets from x ← x + τv
