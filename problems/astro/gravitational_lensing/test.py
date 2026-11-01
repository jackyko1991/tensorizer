"""Physics checks, written first (TDD): what any correct gravitational-lensing code must satisfy.

Run with `pytest problems/astro/gravitational_lensing/test.py`. Each test is a physical law, not an implementation
detail: null geodesics bend by 4GM/(c²b) in the weak field, rays inside the shadow are captured, the Einstein ring
maps onto the source and the magnification matches the point lens.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_null_geodesic_deflection_approaches_einstein_value():
    b = torch.tensor([300.0, 1000.0], dtype=torch.float64)
    y0 = torch.stack([torch.zeros_like(b), 1 / b], dim=1)
    rays = solve.trace_rays(y0, 1.0, dphi=0.002, phi_max=3.3)
    alpha = solve.deflection_angles(rays)
    assert torch.allclose(alpha, equation.thin_lens_deflection(b, 1.0), rtol=0.015)  # weak field: α = 4GM/(c²b)


def test_rays_inside_shadow_are_captured():
    b, y0 = tensors.ray_fan(4.0, 6.0, 5, CPU)
    alpha = solve.deflection_angles(solve.trace_rays(y0, 1.0))
    captured = torch.isnan(alpha)
    assert torch.equal(captured, b < equation.shadow_radius(1.0))  # b < 3√3 M: the photon is captured


def test_einstein_radius_maps_to_source_on_axis():
    GM_c2, D_l, D_s = 1477.0 * 10, 1.2e20, 2.5e20  # 10 solar masses, metres
    theta_E = equation.einstein_radius(GM_c2, D_l, D_s)
    alpha = 4 * GM_c2 / (D_l * theta_E)  # deflection of the ray passing at b = D_l θ_E
    beta = theta_E - (D_s - D_l) / D_s * alpha  # lens equation in physical angles
    assert abs(beta) < 1e-12 * theta_E
    # and in θ_E units the ring θ = 1 maps onto the source position β = 0
    ring = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    assert torch.allclose(equation.lens_map(ring), torch.zeros(2, 2), atol=1e-7)


def test_image_plane_magnification_matches_point_lens():
    theta = tensors.sky_grid(400, 2.5, CPU).double()
    u = torch.tensor([0.4, 0.8, 1.2], dtype=torch.float64)
    centres = torch.stack([u, torch.zeros_like(u)], dim=-1)
    lensed, plain = solve.lensed_images(theta, centres, radius=0.03)
    # point-lens magnification μ(u)
    assert torch.allclose(solve.magnification(lensed, plain), equation.point_magnification(u), rtol=0.02)
