"""Physical checks, written first (TDD): what any correct flocking code must satisfy.

Run with `pytest problems/others/boids_flocking/test.py`. Each test is a law or an exact property, not an
implementation detail: without noise the Vicsek flock orders (φ → 1); with maximal noise φ is only finite-size
noise O(1/√N); Vicsek birds fly at constant speed; the periodic box keeps them inside; the minimum-image
displacement is antisymmetric; and the Reynolds rules are pairwise and antisymmetric (Newton's third law), so the
flock's momentum is conserved when the speed is not renormalised.
"""

import math

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
L, R, V0 = 5.0, 1.0, 0.05


def test_minimum_image_is_antisymmetric_and_short():
    x, _ = tensors.random_flock(2, 60, L, CPU, seed=0)
    d = equation.pair_displacements(x, L)  # (B, N, N, 2) x_j − x_i
    assert torch.allclose(d, -d.transpose(1, 2), atol=1e-6)  # d_ij = −d_ji
    assert (d.abs() <= L / 2 + 1e-6).all()  # never more than half a box in either direction
    dist = d.norm(dim=-1)
    assert torch.allclose(dist, dist.transpose(1, 2))  # the distance is symmetric


def test_no_noise_orders_the_flock():
    x, theta = tensors.random_flock(1, 200, L, CPU, seed=1)
    eta = torch.zeros(1, 1)
    _, _, phi = solve.vicsek(x, theta, eta, 400, L, R, V0, seed=2)
    assert phi[0, 0] < 0.3  # starts disordered
    assert phi[0, -1] > 0.95  # η = 0: neighbours average their headings until everybody agrees


def test_maximal_noise_gives_no_order():
    n = 400
    x, theta = tensors.random_flock(1, n, L, CPU, seed=3)
    _, _, phi = solve.vicsek(x, theta, torch.full((1, 1), 2 * math.pi), 200, L, R, V0, seed=4)
    assert phi[0, 100:].mean() < 3 / math.sqrt(n)  # headings uniform at random: |mean e^{iθ}| ~ 1/√N


def test_order_decreases_with_noise_in_one_batch():
    x, theta = tensors.random_flock(4, 200, L, CPU, seed=5)
    eta = torch.tensor([0.5, 2.0, 4.0, 6.0])[:, None]  # four noise levels evolve at once
    _, _, phi = solve.vicsek(x, theta, eta, 300, L, R, V0, seed=6)
    late = phi[:, 150:].mean(1)
    assert (late.diff() < 0).all()  # the order–disorder transition
    assert late[0] > 0.8 and late[-1] < 0.2


def test_vicsek_speed_is_constant_and_birds_stay_in_the_box():
    x, theta = tensors.random_flock(3, 100, L, CPU, seed=7)
    xs, _, _ = solve.vicsek(x, theta, torch.tensor([[0.0], [1.0], [3.0]]), 50, L, R, V0, seed=8)
    step = equation.minimum_image(xs[1:] - xs[:-1], L)  # (T, B, N, 2) displacement across the boundary
    assert torch.allclose(step.norm(dim=-1), torch.tensor(V0), atol=1e-5)
    assert ((xs >= 0) & (xs < L)).all()


def test_pure_alignment_conserves_momentum():
    x, theta = tensors.random_flock(2, 150, L, CPU, seed=9)
    v = V0 * torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # (B, N, 2)
    weights = {"separation": 0.0, "alignment": 0.2, "cohesion": 0.0}
    xs, vs = solve.boids(x.double(), v.double(), 200, 0.1, L, R, weights, speed=None)
    p = vs.sum(2)  # (T, B, 2) total momentum (unit masses)
    assert torch.allclose(p, p[0].expand_as(p), atol=1e-10)  # Σ_ij A_ij (v_j − v_i) = 0 for symmetric A
    spread = (vs - vs.mean(2, keepdim=True)).pow(2).sum((-1, -2))  # (T, B) velocity variance
    assert spread[-1].max() < 0.5 * spread[0].min()  # alignment makes the velocities agree


def test_all_reynolds_rules_are_pairwise_antisymmetric():
    x, theta = tensors.random_flock(1, 120, L, CPU, seed=10)
    v = V0 * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    weights = {"separation": 0.02, "alignment": 0.5, "cohesion": 0.05}
    a = equation.reynolds_acceleration(x.double(), v.double(), L, R, weights)
    assert a.sum(1).abs().max() < 1e-12  # the forces between i and j cancel: no net force on the flock


def test_boids_with_constant_speed_flock():
    x, theta = tensors.random_flock(1, 150, L, CPU, seed=11)
    v = V0 * torch.stack([torch.cos(theta), torch.sin(theta)], -1)
    xs, vs = solve.boids(x, v, 400, 1.0, L, R, tensors.REYNOLDS, speed=V0)
    assert torch.allclose(vs[-1].norm(dim=-1), torch.tensor(V0), atol=1e-5)
    assert equation.polar_order(vs[-1])[0] > 0.9  # separation + alignment + cohesion → a moving flock
    gap = equation.pair_displacements(xs[-1], L).norm(dim=-1) + 10 * torch.eye(150)
    assert gap.min() > 0.02  # separation keeps the birds from piling up
