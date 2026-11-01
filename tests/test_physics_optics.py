import math

import torch

from problems._common.integrators import integrate
from problems.optics.diffraction_grating import equation as grating_eq
from problems.optics.diffraction_grating import solve as grating_solve
from problems.optics.double_slit import equation as slit_eq
from problems.optics.double_slit import solve as slit_solve
from problems.optics.double_slit import tensors as slit_t
from problems.optics.fourier_optics import equation as fourier_eq
from problems.optics.fourier_optics import solve as fourier_solve
from problems.optics.fourier_optics import tensors as fourier_t
from problems.optics.grin_mirage import equation as grin_eq
from problems.optics.grin_mirage import solve as grin_solve
from problems.optics.grin_mirage import tensors as grin_t
from problems.optics.lens_system import equation as lens_eq
from problems.optics.lens_system import solve as lens_solve
from problems.optics.lens_system import tensors as lens_t
from problems.optics.mirror_reflection import equation as mirror_eq
from problems.optics.mirror_reflection import solve as mirror_solve
from problems.optics.mirror_reflection import tensors as mirror_t
from problems.optics.polarization_jones import equation as jones_eq
from problems.optics.polarization_jones import solve as jones_solve
from problems.optics.polarization_jones import tensors as jones_t
from problems.optics.snell_refraction import equation as snell_eq
from problems.optics.snell_refraction import solve as snell_solve
from problems.optics.snell_refraction import tensors as snell_t
from problems.optics.thin_lens_image import equation as thin_eq
from problems.optics.thin_lens_image import solve as thin_solve
from problems.optics.thin_lens_image import tensors as thin_t

CPU = torch.device("cpu")


# --- Snell's law -------------------------------------------------------------------------------


def _incoming(angles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    d = torch.stack([torch.sin(angles), -torch.cos(angles)], dim=-1)  # going down
    n = torch.tensor([0.0, 1.0]).expand_as(d)  # normal facing the incoming ray
    return d, n


def test_snell_law_holds_below_critical_angle():
    n1, n2 = 1.0, 1.5
    angles = torch.deg2rad(torch.linspace(0.0, 85.0, 18))
    d, n = _incoming(angles)
    t, tir = snell_eq.refract(d, n, torch.full_like(angles, n1 / n2))
    assert not tir.any()
    assert torch.allclose(t.norm(dim=-1), torch.ones(len(t)), atol=1e-6)  # still unit vectors
    assert torch.allclose(n1 * snell_eq.sin_angle(d, n), n2 * snell_eq.sin_angle(t, n), atol=1e-6)
    assert (t[:, 1] < 0).all()  # transmitted, keeps going down


def test_total_internal_reflection_above_critical_angle():
    n1, n2 = 1.5, 1.0
    theta_c = snell_solve.critical_angle(n1, n2)
    assert math.isclose(theta_c, math.degrees(math.asin(n2 / n1)))
    angles = torch.deg2rad(torch.tensor([theta_c - 5, theta_c - 1, theta_c + 1, theta_c + 10]))
    d, n = _incoming(angles)
    t, tir = snell_eq.refract(d, n, torch.full_like(angles, n1 / n2))
    assert tir.tolist() == [False, False, True, True]
    # reflected rays: angle preserved, vertical component flipped
    assert torch.allclose(t[tir, 0], d[tir, 0]) and torch.allclose(t[tir, 1], -d[tir, 1])
    assert torch.allclose(n1 * snell_eq.sin_angle(d[~tir], n[~tir]), n2 * snell_eq.sin_angle(t[~tir], n[~tir]))


def test_traced_rays_obey_snell_between_layers():
    n_layer = snell_t.indices(1.6, 1.33, 1.0, CPU)
    angles = snell_t.fan_angles(4, 30.0)  # all below every critical angle
    p0, d0 = snell_t.initial_rays(angles, CPU)
    _, rays = snell_solve.solve(p0, d0, n_layer, t_max=6.0, dt=0.005)
    # x where each ray crosses y = -1 and y = -2 gives the slope in the middle layer
    mid = rays[:, :, 1]
    x1 = torch.stack([rays[(mid[:, r] <= -1.0).nonzero()[0, 0], r, 0] for r in range(len(angles))])
    x2 = torch.stack([rays[(mid[:, r] <= -2.0).nonzero()[0, 0], r, 0] for r in range(len(angles))])
    theta2 = torch.atan2(x2 - x1, torch.ones_like(x1))
    assert torch.allclose(1.6 * torch.sin(angles), 1.33 * torch.sin(theta2), atol=2e-2)


# --- Lens system -------------------------------------------------------------------------------


def test_thin_lens_equation_gives_sharp_image():
    s, f = 30.0, 10.0
    s_img = 1 / (1 / f - 1 / s)  # 1/s + 1/s' = 1/f
    m = lens_eq.chain(lens_eq.free_space(s), lens_eq.thin_lens(f), lens_eq.free_space(s_img))
    assert abs(m[0, 1]) < 1e-4  # B = 0: height independent of launch angle
    assert math.isclose(m[0, 0].item(), -s_img / s, rel_tol=1e-5)  # magnification
    single = lens_eq.chain(lens_eq.free_space(s), lens_eq.thin_lens(f))
    assert math.isclose(lens_eq.image_distance(single).item(), s_img, rel_tol=1e-5)


def test_chain_matches_multi_dot_and_det_is_one():
    for d in (5.0, 23.0, 61.0):
        m = lens_eq.system_matrix(20.0, 10.0, d, 15.0)
        chained = lens_eq.chain(
            lens_eq.free_space(20.0), lens_eq.thin_lens(10.0), lens_eq.free_space(d), lens_eq.thin_lens(15.0)
        )
        assert torch.allclose(m, chained, atol=1e-5)
        assert math.isclose(torch.linalg.det(m).item(), 1.0, abs_tol=1e-5)


def test_two_lens_image_matches_sequential_thin_lens_formula():
    s, f1, f2, d = 20.0, 10.0, 15.0, 60.0
    s1 = 1 / (1 / f1 - 1 / s)  # image of lens 1
    s2 = 1 / (1 / f2 - 1 / (d - s1))  # becomes the object of lens 2
    rays = lens_t.object_rays(1.0, 0.05, 5)
    _, image_z, _ = lens_solve.solve(rays, s, f1, f2, torch.tensor([d]), lens_t.axis(150.0))
    assert math.isclose(image_z[0].item(), s + d + s2, rel_tol=1e-4)


def test_traced_rays_meet_at_image():
    s, f1, f2, d = 20.0, 10.0, 15.0, 60.0
    rays = lens_t.object_rays(1.0, 0.05, 7)
    _, image_z, image_h = lens_solve.solve(rays, s, f1, f2, torch.tensor([d]), lens_t.axis(150.0))
    h = lens_solve.ray_heights(image_z, rays, s, f1, d, f2)  # (1, R)
    assert torch.allclose(h, image_h.expand_as(h), atol=1e-3)


# --- Double slit -------------------------------------------------------------------------------


def test_fd_laplacian_matches_analytic():
    h = 0.1
    y, x = torch.meshgrid(
        torch.arange(100, dtype=torch.float64) * h, torch.arange(120, dtype=torch.float64) * h, indexing="ij"
    )
    kx, ky = 0.7, 1.3
    u = torch.sin(kx * x) * torch.sin(ky * y)
    lap = slit_eq.laplacian(u, h)
    exact = -(kx**2 + ky**2) * u
    truncation = h**2 / 12 * (kx**4 + ky**4)  # leading error term of the 5-point stencil
    assert (lap - exact)[1:-1, 1:-1].abs().max() <= 1.01 * truncation  # edges use zero padding


def test_cfl_condition_respected_and_enforced():
    assert slit_eq.courant(1.0, 0.5, 1.0) <= slit_eq.CFL_LIMIT
    setup = slit_t.setup(48.0, 6.0, 150.0, CPU)
    try:
        slit_solve.solve(setup, 12.0, t_max=10.0, dt=0.8)
    except ValueError:
        pass
    else:
        raise AssertionError("dt above the CFL limit must be rejected")


def test_double_slit_fringe_spacing():
    lam, d, dist = 12.0, 48.0, 150.0
    setup = slit_t.setup(d, 6.0, dist, CPU)
    _, _, intensity = slit_solve.solve(setup, lam, t_max=450.0, save_every=100)
    screen = intensity[-1]  # (Ny,)
    y = torch.arange(slit_t.NY) - slit_t.NY / 2
    centre = (y.abs() < 10).nonzero().flatten()
    first = ((y > 20) & (y < 60)).nonzero().flatten()
    y0 = y[centre[screen[centre].argmax()]]
    y1 = y[first[screen[first].argmax()]]
    assert abs(y0) <= 3  # bright central fringe on the axis
    assert math.isclose((y1 - y0).item(), lam * dist / d, rel_tol=0.15)  # ≈ λL/d


# --- Mirror reflection -------------------------------------------------------------------------


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]


def test_reflection_angle_equals_incidence_angle():
    angles = torch.deg2rad(torch.linspace(-80.0, 80.0, 17))
    d = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # heading towards +x
    tilt = torch.deg2rad(torch.tensor(20.0))
    n = torch.stack([-torch.cos(tilt), torch.sin(tilt)]).expand_as(d)  # a tilted mirror facing −x
    r = mirror_eq.reflect(d, n)
    assert torch.allclose(r.norm(dim=-1), torch.ones(len(r)), atol=1e-6)
    assert torch.allclose(-mirror_eq.dot(d, n), mirror_eq.dot(r, n), atol=1e-6)  # same angle with the normal
    assert torch.allclose(_cross(d, n), _cross(r, n), atol=1e-6)  # the part along the mirror is unchanged


def _curved_crossings(radius: float, n_rays: int, aperture: float) -> tuple[torch.Tensor, torch.Tensor]:
    p, d = mirror_t.parallel_rays(radius, n_rays, aperture)
    centre = mirror_t.mirror_centre(radius)
    t = mirror_solve.hit_sphere(p, d, centre, radius)
    hit = p + t[:, None] * d
    assert torch.allclose((hit - centre).norm(dim=-1), torch.full((n_rays,), radius), atol=1e-4)
    r = mirror_eq.reflect(d, mirror_eq.sphere_normal(hit, centre, radius))
    return p[:, 1], mirror_solve.axis_crossing(hit, r)


def test_paraxial_rays_cross_near_half_radius():
    radius = 4.0
    y, x = _curved_crossings(radius, 11, 0.05)  # heights up to 0.05 R: paraxial
    off_axis = y != 0
    assert torch.allclose(x[off_axis], torch.full_like(x[off_axis], -radius / 2), atol=0.005 * radius)
    y, x = _curved_crossings(radius, 3, 0.6)  # marginal rays cross closer to the mirror (spherical aberration)
    assert x[0] > -radius / 2 + 0.05 * radius


def test_flat_mirror_rays_meet_at_virtual_image():
    radius = 4.0
    p, d = mirror_t.lamp_rays(radius, 7, 0.6)
    point, normal = mirror_t.flat_mirror()
    t = mirror_solve.hit_flat(p, d, point, normal)
    hit, r = mirror_solve.bounce(p, d, t, normal.expand_as(d))
    assert torch.allclose(hit[:, 0], torch.zeros(7), atol=1e-6)
    image = torch.tensor([0.5 * radius, 0.25 * radius])  # lamp mirrored to the other side of x = 0
    assert torch.allclose(_cross(r, image - hit), torch.zeros(7), atol=1e-5)  # image lies on every reflected line


# --- Thin lens image ---------------------------------------------------------------------------


def test_thin_lens_equation_and_magnification():
    f, h = 10.0, 2.0
    s = thin_t.object_distances(30.0, 4.0, f, 27)
    result = thin_solve.solve(s, h, f, thin_t.principal_slopes(h, s, f))
    s_img, m = result["s_img"], result["m"]
    assert torch.allclose(1 / s + 1 / s_img, torch.full_like(s, 1 / f), atol=1e-5)
    assert torch.allclose(m, -s_img / s)
    assert ((s_img > 0) == (s > f)).all()  # real image only beyond the focal length
    i = (s - 2 * f).abs().argmin()
    assert math.isclose(s[i].item(), 2 * f) and math.isclose(m[i].item(), -1.0, rel_tol=1e-5)  # 2f → 2f, m = −1


def test_principal_rays_meet_at_the_image():
    f, h = 10.0, 2.0
    s = torch.tensor([25.0, 15.0, 6.0])  # real, real, virtual
    result = thin_solve.solve(s, h, f, thin_t.principal_slopes(h, s, f))
    heights = thin_solve.height_after_lens(result["y"], result["slope_out"], result["s_img"])  # (K, 3)
    assert torch.allclose(heights, result["h_img"][:, None].expand_as(heights), atol=1e-4)
    assert torch.allclose(result["slope_out"][:, 2], torch.zeros(3), atol=1e-6)  # focal ray leaves parallel
    assert torch.allclose(thin_eq.lens_kink(torch.tensor([[0.0]]), torch.tensor([[0.3]]), f), torch.tensor([[0.3]]))


# --- Diffraction grating -----------------------------------------------------------------------


def test_grating_matches_closed_form_and_maxima():
    lam, d = 0.5, 2.0
    k = grating_eq.wavenumber(lam)
    sin_theta = torch.linspace(-0.9, 0.9, 20001, dtype=torch.float64)
    for n in (2, 5, 12):
        e = grating_eq.array_factor(n, sin_theta, k, d)
        assert torch.allclose((e.conj() * e).real, grating_eq.closed_form(sin_theta, n, k, d), atol=1e-8)
    orders = torch.tensor(grating_solve.principal_maxima(lam, d, 0.9), dtype=torch.float64)
    assert torch.allclose(orders, torch.arange(-3, 4, dtype=torch.float64) * lam / d)  # d sin θ = m λ
    e = grating_eq.array_factor(7, orders, k, d)
    assert torch.allclose(e.abs(), torch.ones_like(orders), atol=1e-9)  # all phasors aligned: |E| = 1


def test_grating_has_n_minus_2_secondary_maxima():
    lam, d = 0.5, 2.0
    k = grating_eq.wavenumber(lam)
    sin_theta = torch.linspace(0.0, lam / d, 40001, dtype=torch.float64)[1:-1]  # strictly between orders 0 and 1
    for n in (2, 3, 6, 10, 20):
        e = grating_eq.array_factor(n, sin_theta, k, d)
        i = (e.conj() * e).real
        peaks = (i[1:-1] > i[:-2]) & (i[1:-1] > i[2:])
        assert peaks.sum().item() == n - 2


# --- Polarization (Jones calculus) -------------------------------------------------------------


def test_malus_law():
    e_in = jones_t.linear_input(0.0)
    for theta in torch.linspace(0.0, math.pi, 13).tolist():
        out = jones_eq.apply(jones_eq.polarizer(theta), e_in)
        assert math.isclose(jones_eq.intensity(out).item(), math.cos(theta) ** 2, abs_tol=1e-6)
    angles = jones_t.analyzer_angles(19)
    result = jones_solve.solve(e_in, jones_t.plate("none", 0.0), angles, jones_t.phases())
    assert torch.allclose(result["intensity"], torch.cos(angles) ** 2, atol=1e-6)


def test_quarter_wave_plate_makes_circular_light():
    e_in, qwp = jones_t.linear_input(0.0), jones_t.plate("quarter", 45.0)
    ex, ey, phase = jones_solve.describe(jones_eq.apply(qwp, e_in))
    assert math.isclose(ex, ey, rel_tol=1e-5) and math.isclose(ex, 1 / math.sqrt(2), rel_tol=1e-5)
    assert math.isclose(abs(phase), 90.0, abs_tol=1e-3)
    result = jones_solve.solve(e_in, qwp, jones_t.analyzer_angles(19), jones_t.phases())
    assert torch.allclose(result["intensity"], torch.full((19,), 0.5), atol=1e-6)  # every analyzer passes I₀/2
    radius = result["ellipse_plate"].norm(dim=-1)
    assert torch.allclose(radius, torch.full_like(radius, 1 / math.sqrt(2)), atol=1e-5)  # the ellipse is a circle


def test_crossed_polarizers_and_the_45_degree_trick():
    # unpolarized light with I₀ = 1: the first polarizer passes I₀/2, crossed ones block,
    # and a 45° polarizer between them lets I₀/2 · cos²45° · cos²45° = I₀/8 through
    p0, p45, p90 = (jones_eq.polarizer(math.radians(a)) for a in (0, 45, 90))
    assert math.isclose(jones_eq.unpolarized_intensity(p0).item(), 0.5, abs_tol=1e-6)
    assert jones_eq.unpolarized_intensity(jones_eq.chain(p0, p90)).item() < 1e-10
    assert math.isclose(jones_eq.unpolarized_intensity(jones_eq.chain(p0, p45, p90)).item(), 1 / 8, abs_tol=1e-6)
    # a half-wave plate at α rotates linear light by 2α
    e = jones_eq.apply(jones_t.plate("half", 30.0), jones_t.linear_input(0.0))
    assert torch.allclose(e.abs(), torch.tensor([0.5, math.sqrt(3) / 2]), atol=1e-6)


# --- Fourier optics ----------------------------------------------------------------------------


def test_fft2_equals_einsum_dft():
    a = fourier_t.aperture("circle", 10.0, n=48)
    u_fft = fourier_eq.far_field(a).to(torch.complex128)
    u_ein = fourier_eq.far_field_einsum(a)
    assert torch.allclose(u_fft, u_ein, atol=1e-3)


def test_circular_aperture_first_dark_ring():
    for diameter in (16.0, 24.0, 32.0):
        measured = fourier_solve.first_dark_ring(diameter)
        assert math.isclose(measured, fourier_eq.airy_radius(fourier_t.N, diameter), rel_tol=0.1)


def test_square_aperture_gives_sinc_squared():
    n, side = fourier_t.N, 16
    a = fourier_t.aperture("square", side)
    assert a.sum().item() == side**2
    i = fourier_eq.intensity(fourier_eq.far_field(a))
    row = i[n // 2, n // 2 : n // 2 + 40]  # along the l axis, from the centre outwards
    expected = torch.sinc(side * torch.arange(40.0) / n) ** 2
    assert torch.allclose(row / row[0], expected, atol=0.01)
    assert (row[[16, 32]] / row[0] < 1e-6).all()  # zeros at multiples of N / D


# --- GRIN mirage -------------------------------------------------------------------------------


def _trace(dn: float, n_rays: int = 9) -> tuple[torch.Tensor, torch.Tensor]:
    angles = grin_t.viewing_angles(10.0, n_rays)
    path = grin_solve.solve(grin_t.initial_state(angles, 1.6, 400.0, dn, 0.3), dn, 0.3, 400.0)
    return angles, path


def test_rays_stay_straight_in_uniform_medium():
    angles, path = _trace(0.0)
    for s in range(1, len(path), 50):
        step = path[s] - path[0]  # (R, 2)
        moving = step.norm(dim=-1) > 0
        slope = step[moving, 1] / step[moving, 0]
        assert torch.allclose(slope, -torch.tan(angles[moving]), atol=1e-10)  # heading −x at angle α


def test_snell_invariant_conserved_along_rays():
    dn, h = 3e-5, 0.3
    angles, path = _trace(dn)
    invariant = grin_eq.snell_invariant(path, dn, h)  # (S − 1, R)
    for r in range(len(angles)):
        moving = (path[1:, r] != path[:-1, r]).any(-1)  # ignore the frozen end of each ray
        values = invariant[moving, r][:-1]  # and the last step, which may poke below the road where n is flat
        assert (values.max() - values.min()).item() < 1e-8  # n cos θ conserved ...
    n_along = grin_eq.index(path[:, 2, 1], dn, h)  # ray 2 (−7 mrad) dives into the hot layer
    assert (n_along.max() - n_along.min()).item() > 1e-6  # ... although n itself changes far more


def test_direction_norm_equals_index():
    dn, h = 3e-5, 0.3
    angles = grin_t.viewing_angles(10.0, 9)
    traj = integrate(lambda y, s: grin_eq.rhs(y, dn, h), grin_t.initial_state(angles, 1.6, 400.0, dn, h), 1.0, 150)
    p_sq = grin_eq.direction_norm_sq(traj[:, :, 1])  # p_j p_j
    n_sq = grin_eq.index(traj[:, :, 0, 1], dn, h) ** 2
    assert torch.allclose(p_sq, n_sq, atol=1e-10)


def test_mirage_image_is_inverted():
    _, path = _trace(3e-5, n_rays=25)
    seen = grin_solve.height_at_object(path)
    assert torch.isnan(seen[:3]).all()  # the steepest rays reach the road
    hit = seen[~torch.isnan(seen)]
    diffs = hit[1:] - hit[:-1]
    assert (diffs[-5:] > 0).all()  # direct view: looking higher shows higher parts (upright)
    assert (diffs[:2] < 0).all()  # mirage: looking lower shows higher parts (inverted)
