"""Simulate the true system with RK4, train the EGNN and the MLP on (state → next state), roll them out."""

import torch

from problems._common.integrators import integrate
from problems._common.progress import steps

from .equation import egnn, plain_mlp, rhs


def simulate(x0, v0, c, k_spring: float, eps: float, dt: float, n_steps: int, substeps: int = 10) -> torch.Tensor:
    """True trajectories sampled every dt: (n_steps + 1, B, 2, N, 3). RK4 runs with the finer step dt / substeps."""
    y0 = torch.stack([x0, v0], dim=1)  # (B, 2, N, 3)
    traj = integrate(lambda y, t: rhs(y, c, k_spring, eps), y0, dt / substeps, n_steps * substeps)
    return traj[::substeps]


def pairs(traj: torch.Tensor, c: torch.Tensor):
    """Every (state at t, state at t + dt) of every trajectory, flattened into one batch of P = T·B pairs."""
    T1, B = traj.shape[:2]
    now = traj[:-1].reshape(-1, *traj.shape[2:])  # (P, 2, N, 3)
    nxt = traj[1:].reshape(-1, *traj.shape[2:])
    charges = c[None].expand(T1 - 1, B, -1).reshape(-1, c.shape[-1])  # (P, N)
    return now, nxt, charges


def predict(model: str, net, x, v, c, dt: float):
    return egnn(x, v, c, net, dt) if model == "egnn" else plain_mlp(x, v, c, net, dt)


def mse(model: str, net, now, nxt, c, dt: float) -> torch.Tensor:
    """Mean squared error of the predicted next positions and velocities."""
    x, v = predict(model, net, now[:, 0], now[:, 1], c, dt)
    return ((x - nxt[:, 0]) ** 2).mean() + ((v - nxt[:, 1]) ** 2).mean()


def train(model: str, net, params: list, train_set, test_set, dt: float, n_iter: int, lr: float) -> dict:
    """Full-batch Adam with a cosine learning-rate decay. Returns the train and test loss histories."""
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter, eta_min=lr / 30)
    history = {"train": [], "test": []}
    for _ in steps(n_iter, f"training {model}"):
        loss = mse(model, net, *train_set, dt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history["train"].append(loss.item())
    with torch.no_grad():
        history["test_final"] = mse(model, net, *test_set, dt).item()
        history["train_final"] = mse(model, net, *train_set, dt).item()
    return history


@torch.no_grad()
def rollout(model: str, net, x0, v0, c, dt: float, n_steps: int) -> torch.Tensor:
    """Feed the model its own predictions n_steps times: (n_steps + 1, B, N, 3) positions."""
    xs, x, v = [x0], x0, v0
    for _ in range(n_steps):
        x, v = predict(model, net, x, v, c, dt)
        xs.append(x)
    return torch.stack(xs)


@torch.no_grad()
def equivariance_error(model: str, net, x, v, c, R: torch.Tensor, shift: torch.Tensor, dt: float) -> float:
    """max |f(R x + s, R v) - (R f_x + s, R f_v)| / max |f|: zero for an E(3)-equivariant model.

    Rotations act on the Cartesian index:  x'_{bia} = R_{ae} x_{bie} + s_a  (repeated e ⇒ summed).
    """

    def rot(u: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ae,bie->bia", R, u)

    x_out, v_out = predict(model, net, x, v, c, dt)
    x_rot, v_rot = predict(model, net, rot(x) + shift, rot(v), c, dt)
    err = torch.maximum((x_rot - (rot(x_out) + shift)).abs().max(), (v_rot - rot(v_out)).abs().max())
    scale = torch.maximum(x_out.abs().max(), v_out.abs().max())
    return (err / scale).item()
