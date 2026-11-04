"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import polar_order


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    n = 80 if fast else params["n_birds"]
    box, radius, v0 = params["box"], 1.0, params["speed"]
    n_steps = 40 if fast else params["steps"]
    eta = torch.cat([tensors.noise_levels(6 if fast else params["n_noise"], dev), torch.tensor([[params["eta"]]])])

    x, theta = tensors.random_flock(len(eta), n, box, dev, params["seed"])  # (B, N, 2), (B, N)
    xs, thetas, phi = solve.vicsek(x, theta, eta, n_steps, box, radius, v0, params["seed"] + 1)
    late = phi[:, n_steps // 2 :].mean(1)  # (B,) time average after the transient
    print(f"Vicsek: N = {n} birds in a {box}×{box} periodic box (density {n / box**2:.1f}), R = {radius}, v0 = {v0}")
    for e, f in zip(eta[:-1, 0].tolist(), late[:-1].tolist()):
        print(f"  η = {e:4.2f}: polar order φ = {f:.3f}")
    print(f"  maximal noise η = 2π: φ = {late[-2]:.3f}  (finite-size level 1/√N = {1 / math.sqrt(n):.3f})")
    step = (xs[1:] - xs[:-1] - box * torch.round((xs[1:] - xs[:-1]) / box)).norm(dim=-1)
    print(
        f"  speed: min {step.min():.5f}, max {step.max():.5f} (constant v0); all birds in the box: "
        f"{bool(((xs >= 0) & (xs < box)).all())}"
    )

    xb, vb = solve.boids(
        x[-1:], tensors.velocities(theta[-1:], v0), n_steps, 1.0, box, radius, tensors.reynolds_weights(v0), v0
    )
    phi_b = polar_order(vb.flatten(0, 1))  # (T + 1,)
    print(f"Reynolds boids (separation, alignment, cohesion): φ from {phi_b[0]:.3f} to {phi_b[-1]:.3f}")

    shown = {"eta": params["eta"], "x": xs[:, -1].cpu(), "theta": thetas[:, -1].cpu(), "phi": phi[-1].cpu()}
    boid = {"x": xb[:, 0].cpu(), "v": vb[:, 0].cpu(), "phi": phi_b.cpu()}
    return plot.animate(shown, boid, eta[:-1, 0].cpu(), phi[:-1].cpu(), late[:-1].cpu(), box, out_dir / "flock.gif")


if __name__ == "__main__":
    cli(run, __file__)
