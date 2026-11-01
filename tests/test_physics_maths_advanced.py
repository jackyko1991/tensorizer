"""Checks for the advanced maths problems: Mandelbrot/Julia, spectral clustering, and ζ(s) with the primes."""

import math

import torch

from problems.maths.mandelbrot_julia import equation as mj_eq
from problems.maths.mandelbrot_julia import solve as mj_solve
from problems.maths.mandelbrot_julia import tensors as mj_tensors
from problems.maths.spectral_clustering import equation as sc_eq
from problems.maths.spectral_clustering import solve as sc_solve
from problems.maths.spectral_clustering import tensors as sc_tensors
from problems.maths.zeta_primes import equation as zp_eq
from problems.maths.zeta_primes import solve as zp_solve
from problems.maths.zeta_primes import tensors as zp_tensors

CPU = torch.device("cpu")


# --- Mandelbrot and Julia sets ---------------------------------------------------------------------------------------


def test_points_inside_the_main_cardioid_never_escape():
    c = mj_tensors.pixel_grid(complex(-0.6, 0.0), 3.2, 120, CPU)
    inside = mj_eq.in_cardioid_or_bulb(c)
    assert inside.sum() > 1000
    result = mj_solve.mandelbrot(c[inside], 2000)
    assert (result["n"] == 0).all()
    # centres of the cardioid's family: 0 (fixed point), −1 (2-cycle), and points just inside the edge
    w = 0.97 * torch.exp(1j * torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64))
    edge = w / 2 - w * w / 4
    assert mj_eq.in_cardioid_or_bulb(edge).all()
    assert (mj_solve.mandelbrot(edge, 5000)["n"] == 0).all()


def test_points_with_modulus_above_two_escape_at_the_first_step():
    angle = torch.linspace(0, 2 * math.pi, 200, dtype=torch.float64)
    radius = torch.linspace(2.01, 5.0, 200, dtype=torch.float64)
    c = radius * torch.exp(1j * angle)
    result = mj_solve.mandelbrot(c, 100, radius=2.0)
    assert (result["n"] == 1).all()
    assert not mj_eq.in_cardioid_or_bulb(c).any()


def test_distance_estimate_is_positive_outside_and_bounded_by_true_distance():
    c = mj_tensors.pixel_grid(complex(-0.6, 0.0), 3.2, 100, CPU)
    result = mj_solve.mandelbrot(c, 500)
    out = result["n"] > 0
    assert (result["d"][out] > 0).all()
    # on the real axis right of M (which ends at c = 1/4) the true distance is c − 1/4
    x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64).to(torch.complex128)
    d = mj_solve.mandelbrot(x, 500)["d"]
    true = x.real - 0.25
    assert ((d > true / 4) & (d < 4 * true)).all()


def test_cardioid_boundary_has_neutral_fixed_point_and_julia_is_connected_there():
    theta = torch.linspace(0.1, 6.0, 40, dtype=torch.float64)
    c = mj_eq.cardioid_boundary(theta)
    multiplier = (1 - torch.sqrt(1 - 4 * c)).abs()  # |2 z*| for the attracting-branch fixed point
    assert torch.allclose(multiplier, torch.ones_like(multiplier), atol=1e-12)
    # c = 0: the Julia set is the unit circle, so |z0| < 1 stays bounded and |z0| > 1 escapes
    z0 = torch.tensor([0.5, 0.9j, -0.99, 1.01, 1.5j], dtype=torch.complex128)
    n = mj_solve.julia(z0, torch.zeros(1, dtype=torch.complex128), 400)["n"]
    assert (n[:3] == 0).all() and (n[3:] > 0).all()


# --- spectral clustering ---------------------------------------------------------------------------------------------


def moons_graph(n: int = 300, noise: float = 0.08):
    x, truth = sc_tensors.points("moons", n, noise, 0, CPU)
    return x, truth, sc_eq.gaussian_knn_weights(sc_eq.squared_distances(x), 0.2, 10)


def test_laplacian_rows_sum_to_zero_and_it_is_positive_semidefinite():
    _, _, w = moons_graph()
    assert torch.allclose(w, w.T)
    lap = sc_eq.laplacian(w)
    assert lap.sum(dim=1).abs().max() < 1e-12
    assert torch.linalg.eigvalsh(lap).min() > -1e-10
    lsym = sc_eq.normalized_laplacian(w)
    evals = torch.linalg.eigvalsh(lsym)
    assert evals.min() > -1e-10 and evals.max() < 2 + 1e-10
    f = torch.randn(len(w), generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    assert torch.allclose(f @ lap @ f, sc_eq.dirichlet_energy(w, f))


def test_zero_eigenvalue_multiplicity_equals_number_of_components():
    gen = torch.Generator().manual_seed(0)
    for sizes in ([5], [4, 6], [3, 3, 7], [2, 5, 4, 6]):
        n = sum(sizes)
        w = torch.zeros(n, n, dtype=torch.float64)
        start = 0
        for s in sizes:  # a random connected block: a path plus random extra edges
            block = torch.rand(s, s, generator=gen, dtype=torch.float64) * (torch.rand(s, s, generator=gen) < 0.4)
            block = block + torch.diag(torch.full((s - 1,), 0.5, dtype=torch.float64), 1)
            block = torch.triu(block, 1)
            w[start : start + s, start : start + s] = block + block.T
            start += s
        perm = torch.randperm(n, generator=gen)
        w = w[perm][:, perm]
        evals, _ = sc_solve.spectrum(sc_eq.laplacian(w))
        assert sc_solve.count_zero_eigenvalues(evals) == len(sizes)
        assert sc_solve.connected_components(w).unique().numel() == len(sizes)


def test_grid_graph_eigenvalues_match_the_standing_wave_formula():
    g = 12
    evals, _ = sc_solve.spectrum(sc_eq.laplacian(sc_tensors.grid_graph(g, CPU)))
    one_d = 2 - 2 * torch.cos(math.pi * torch.arange(g, dtype=torch.float64) / g)
    exact = (one_d[:, None] + one_d[None, :]).flatten().sort().values
    assert torch.allclose(evals, exact, atol=1e-12)


def test_spectral_clustering_separates_moons_where_kmeans_fails():
    x, truth, w = moons_graph()
    spectral, _, _ = sc_solve.spectral_clustering(w, 2, seed=0)
    plain = sc_solve.kmeans(x, 2, seed=0)
    ari_spectral = sc_solve.adjusted_rand_index(truth, spectral)
    ari_plain = sc_solve.adjusted_rand_index(truth, plain)
    assert ari_spectral > 0.98
    assert ari_plain < 0.6
    lam2, fiedler = sc_solve.fiedler_vector(w)
    assert lam2 > 0  # the graph is connected
    assert sc_solve.adjusted_rand_index(truth, (fiedler > 0).long()) > 0.95


def test_adjusted_rand_index_limits():
    a = torch.tensor([0, 0, 1, 1, 2, 2])
    assert abs(sc_solve.adjusted_rand_index(a, torch.tensor([2, 2, 0, 0, 1, 1])) - 1) < 1e-12
    random = torch.randint(0, 3, (3000,), generator=torch.Generator().manual_seed(0))
    other = torch.randint(0, 3, (3000,), generator=torch.Generator().manual_seed(1))
    assert abs(sc_solve.adjusted_rand_index(random, other)) < 0.02


# --- ζ(s) and the primes ---------------------------------------------------------------------------------------------


def test_zeta_at_known_values():
    s = torch.tensor([2.0, 4.0, 0.0, -1.0], dtype=torch.complex128)
    z = zp_eq.zeta(s, 30).real
    expected = torch.tensor([math.pi**2 / 6, math.pi**4 / 90, -0.5, -1 / 12], dtype=torch.float64)
    assert torch.allclose(z, expected, atol=1e-12)


def test_first_three_zeta_zeros():
    t = zp_tensors.t_grid(30.0, 0.01, CPU)
    n_terms = zp_solve.n_terms_for(30.0)
    gammas = zp_solve.find_zeros(t, zp_solve.z_on_grid(t, n_terms), n_terms)
    assert len(gammas) == 3
    for g, known in zip(gammas.tolist(), [14.134725142, 21.022039639, 25.010857580]):
        assert abs(g - known) < 1e-4
    # ζ really vanishes there, and Z is real (imaginary part of e^{iθ}ζ ≈ 0) on the critical line
    assert zp_eq.zeta(0.5 + 1j * gammas, n_terms).abs().max() < 1e-8
    t_test = torch.linspace(10, 60, 50, dtype=torch.float64)
    rotated = torch.exp(1j * zp_eq.theta(t_test)) * zp_eq.zeta(0.5 + 1j * t_test, 50)
    assert rotated.imag.abs().max() < 1e-7


def test_riemann_siegel_agrees_with_euler_maclaurin():
    """With only the first correction term the error is O(t^{-3/4}): small, and shrinking with t."""
    low = torch.linspace(50, 100, 200, dtype=torch.float64)
    high = torch.linspace(400, 450, 200, dtype=torch.float64)
    err_low = (zp_eq.riemann_siegel_z(low) - zp_eq.hardy_z(low, 80)).abs().max()
    err_high = (zp_eq.riemann_siegel_z(high) - zp_eq.hardy_z(high, 250)).abs().max()
    assert err_low < 1e-2 and err_high < err_low / 2


def test_sieve_matches_known_prime_counts():
    pi = zp_tensors.prime_pi(zp_tensors.sieve(10**6, CPU))
    assert [int(pi[10**n]) for n in range(1, 7)] == [4, 25, 168, 1229, 9592, 78498]


def test_li_beats_x_over_log_x_at_one_million():
    x = torch.tensor([1e6], dtype=torch.float64)
    pi = 78498
    li = zp_eq.li(x).item()
    assert abs(li - 78627.549159) < 1e-3
    assert abs(li - pi) < abs((x / torch.log(x)).item() - pi)
    r = zp_eq.gram_r(x, zp_tensors.zeta_at_integers(100, CPU)).item()
    assert abs(r - pi) < abs(li - pi)


def test_explicit_formula_improves_with_more_zeros():
    t = zp_tensors.t_grid(zp_tensors.height_for(60), 0.01, CPU)
    n_terms = zp_solve.n_terms_for(t[-1].item())
    gammas = zp_solve.find_zeros(t, zp_solve.z_on_grid(t, n_terms), n_terms)[:60]
    x = torch.arange(2.5, 60, 1.0, dtype=torch.float64)  # half-integers, away from the jumps
    exact = zp_tensors.prime_pi(zp_tensors.sieve(60, CPU))[x.floor().long()].double()
    pi_k = zp_solve.prime_count_from_zeros(x, gammas, zp_tensors.mobius(64))
    err = (pi_k - exact).abs().mean(dim=1)
    assert err[60] < err[10] < err[0]
    assert (pi_k[60] - exact).abs().max() < 0.5  # rounding recovers π(x) exactly
