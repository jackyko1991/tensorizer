import math

import torch

from problems.quantum.double_well import equation as dw_eq
from problems.quantum.double_well import solve as dw_solve
from problems.quantum.double_well import tensors as dw_t
from problems.quantum.free_wave_packet import equation as free_eq
from problems.quantum.free_wave_packet import solve as free_solve
from problems.quantum.free_wave_packet import tensors as free_t
from problems.quantum.harmonic_oscillator import equation as ho_eq
from problems.quantum.harmonic_oscillator import solve as ho_solve
from problems.quantum.harmonic_oscillator import tensors as ho_t
from problems.quantum.hydrogen_radial import equation as hyd_eq
from problems.quantum.hydrogen_radial import solve as hyd_solve
from problems.quantum.hydrogen_radial import tensors as hyd_t
from problems.quantum.infinite_well import equation as well_eq
from problems.quantum.infinite_well import solve as well_solve
from problems.quantum.infinite_well import tensors as well_t
from problems.quantum.scattering_2d import equation as sc_eq
from problems.quantum.scattering_2d import solve as sc_solve
from problems.quantum.scattering_2d import tensors as sc_t
from problems.quantum.spin_precession import equation as spin_eq
from problems.quantum.spin_precession import solve as spin_solve
from problems.quantum.spin_precession import tensors as spin_t
from problems.quantum.tunneling import equation as tun_eq
from problems.quantum.tunneling import solve as tun_solve
from problems.quantum.tunneling import tensors as tun_t
from problems.quantum.two_qubit_entanglement import equation as qb_eq
from problems.quantum.two_qubit_entanglement import solve as qb_solve
from problems.quantum.two_qubit_entanglement import tensors as qb_t

CPU = torch.device("cpu")


# --- particle in a box -------------------------------------------------------------------------


def well(n: int = 400, L: float = 1.5):
    x, dx = well_t.grid(n, L, CPU)
    energies, phi = well_solve.eigenstates(well_t.hamiltonian(n, L, CPU), dx)
    return x, dx, energies, phi


def test_well_energies_match_n_squared_law():
    L = 1.5
    _, _, energies, _ = well(L=L)
    n = torch.arange(1, 6, dtype=torch.float64)
    exact = n**2 * math.pi**2 / (2 * L**2)
    assert torch.allclose(energies[:5], exact, rtol=1e-3)  # finite-difference error ~ (nπ dx)²/12


def test_well_eigenvectors_orthonormal():
    _, dx, _, phi = well()
    gram = torch.einsum("xn,xm->nm", phi, phi) * dx  # ∫ φ_n φ_m dx
    assert torch.allclose(gram, torch.eye(phi.shape[1], dtype=gram.dtype), atol=1e-10)


def test_well_norm_conserved_and_revival():
    x, dx, energies, phi = well()
    c = well_solve.coefficients(phi, well_t.gaussian_packet(x, 1.5, dx), dx)
    _, psi = well_solve.solve(phi, energies, c, t_max=0.7, n_steps=50)
    norms = well_eq.norm(psi, dx)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)

    # low states: grid energies ≈ n²E_1, so the initial state comes back after the revival time
    c = well_t.equal_superposition(3, len(x), CPU)
    _, psi = well_solve.solve(phi, energies, c, t_max=well_solve.revival_time(1.5), n_steps=10)
    overlap = (torch.einsum("x,x->", psi[0].conj(), psi[-1]) * dx).abs()
    assert overlap > 0.999


# --- tunnelling --------------------------------------------------------------------------------


def test_tunneling_norm_conserved():
    x, k, dx = tun_t.grid(1024, 200.0, CPU)
    V = tun_t.barrier(x, 2.5, 1.0)
    psi0 = tun_t.gaussian_packet(x, -25.0, 2.0, 2.0, dx)
    _, psi = tun_solve.solve(psi0, V, k, t_max=10.0)
    norms = tun_eq.probability(psi, dx)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10)


def test_free_packet_moves_at_k0_and_spreads():
    x, k, dx = tun_t.grid(2048, 200.0, CPU)
    x0, sigma0, k0, t = -30.0, 2.0, 3.0, 15.0
    psi0 = tun_t.gaussian_packet(x, x0, sigma0, k0, dx)
    times, psi = tun_solve.solve(psi0, torch.zeros_like(x), k, t_max=t)
    rho = psi[-1].abs() ** 2 * dx  # (N,) probability per cell
    mean = (rho * x).sum()
    width = torch.sqrt((rho * (x - mean) ** 2).sum())
    assert abs(mean - (x0 + k0 * times[-1])) < 1e-3  # group velocity = k0 (ħ = m = 1)
    expected = sigma0 * math.sqrt(1 + (times[-1] / (2 * sigma0**2)) ** 2)
    assert abs(width - expected) / expected < 1e-3


def test_wide_packet_transmission_matches_plane_wave():
    # σ0 = 12 ⇒ momentum spread 1/(2σ0) ≈ 0.04, nearly a plane wave of k0 = 2
    x, k, dx = tun_t.grid(8192, 600.0, CPU)
    k0, V0, a = 2.0, 2.5, 1.0
    psi0 = tun_t.gaussian_packet(x, -100.0, 12.0, k0, dx)
    _, psi = tun_solve.solve(psi0, tun_t.barrier(x, V0, a), k, t_max=100.0, dt=0.02, n_saved=2)
    p_trans = tun_solve.transmission(psi[-1], x, a, dx)
    expected = tun_solve.plane_wave_transmission(k0**2 / 2, V0, tun_t.grid_width(x, a, dx))
    assert abs(p_trans - expected) < 0.01


# --- spin precession ---------------------------------------------------------------------------


def test_pauli_algebra():
    sigma = spin_t.pauli(CPU)
    eps = spin_t.levi_civita(CPU)
    lhs = torch.einsum("iab,jbc->ijac", sigma, sigma)  # σ_i σ_j
    eye = torch.eye(2, dtype=sigma.dtype)
    delta = torch.eye(3, dtype=sigma.dtype)
    rhs = torch.einsum("ij,ac->ijac", delta, eye) + 1j * torch.einsum("ijk,kac->ijac", eps, sigma)
    assert torch.allclose(lhs, rhs)


def test_larmor_precession_frequency():
    sigma = spin_t.pauli(CPU)
    gamma, B0 = 1.7, 2.0
    psi0 = spin_t.tilted_spinor(90.0, CPU)  # along +x
    times, _, s = spin_solve.solve(psi0, sigma, gamma, B0, 0.0, 0.0, t_max=5.0)
    w = gamma * B0
    # ds/dt = γ s × B with B along z: s rotates clockwise seen from +z
    expected = torch.stack([torch.cos(w * times), -torch.sin(w * times), torch.zeros_like(times)], dim=1)
    assert torch.allclose(s, expected, atol=1e-8)


def test_bloch_vector_stays_unit_with_drive():
    sigma = spin_t.pauli(CPU)
    psi0 = spin_t.tilted_spinor(30.0, CPU)
    _, psi, s = spin_solve.solve(psi0, sigma, 1.0, 1.0, 0.4, 1.0, t_max=20.0)
    assert torch.allclose(s.norm(dim=1), torch.ones(len(s), dtype=s.dtype), atol=1e-10)
    assert torch.allclose(spin_eq.bloch_vector(psi[0], sigma), torch.tensor([0.5, 0.0, math.sqrt(3) / 2]).double())


def test_resonant_drive_flips_spin_at_rabi_frequency():
    sigma = spin_t.pauli(CPU)
    gamma, B0, B1 = 1.0, 1.0, 0.05
    t_flip = math.pi / (gamma * B1 / 2)  # half a Rabi period
    psi0 = spin_t.tilted_spinor(0.0, CPU)
    _, _, s = spin_solve.solve(psi0, sigma, gamma, B0, B1, gamma * B0, t_max=t_flip)
    assert s[-1, 2] < -0.99


# --- harmonic oscillator -----------------------------------------------------------------------


def oscillator(omega: float = 1.3, n: int = 600, L: float = 20.0):
    x, dx = ho_t.grid(n, L, CPU)
    energies, phi = ho_solve.eigenstates(ho_t.hamiltonian(x, dx, omega), dx)
    return x, dx, energies, phi


def test_oscillator_energy_ladder():
    omega = 1.3
    _, _, energies, _ = oscillator(omega)
    n = torch.arange(8, dtype=torch.float64)
    assert torch.allclose(energies[:8], omega * (n + 0.5), rtol=2e-3)


def test_coherent_state_follows_classical_motion():
    omega, x0 = 1.3, 2.5
    x, dx, energies, phi = oscillator(omega)
    c = ho_solve.coefficients(phi, ho_t.coherent_state(x, x0, omega), dx)
    times, psi = ho_solve.solve(phi, energies, c, t_max=2 * 2 * math.pi / omega, n_steps=80)
    x_mean = ho_eq.mean_position(psi, x, dx)
    # ~1% phase lag after two periods: finite differences make the rungs slightly narrower than ω
    assert torch.allclose(x_mean, x0 * torch.cos(omega * times), atol=0.03)
    assert torch.allclose(ho_eq.norm(psi, dx), torch.ones(len(times), dtype=torch.float64), atol=1e-8)


# --- free wave packet --------------------------------------------------------------------------


def test_free_packet_centre_width_and_norm():
    # σ = standard deviation of |ψ|²; exact: ⟨x⟩ = x0 + k0 t, σ(t) = σ0 √(1 + (t/(2σ0²))²)
    x, k, dx = free_t.grid(4096, 300.0, CPU)
    x0, sigma0, k0 = -40.0, 1.5, 2.0
    times, psi = free_solve.solve(free_t.gaussian_packet(x, x0, sigma0, k0, dx), k, t_max=30.0, n_steps=30)
    mean, width = free_eq.mean_and_width(psi, x, dx)
    assert torch.allclose(mean, free_solve.exact_centre(times, x0, k0), atol=1e-6)
    assert torch.allclose(width, free_solve.exact_width(times, sigma0), rtol=1e-6)
    assert torch.allclose(free_eq.norm(psi, dx), torch.ones(len(times), dtype=torch.float64), atol=1e-12)


# --- hydrogen ----------------------------------------------------------------------------------


def hydrogen(n_max: int = 3, n: int = 1500, r_max: float = 60.0):
    r, dr = hyd_t.radial_grid(n, r_max, CPU)
    ls = hyd_t.angular_momenta(n_max, CPU)
    H = hyd_eq.hamiltonian(hyd_t.second_derivative(n, dr, CPU), hyd_eq.effective_potential(r, ls))
    return r, dr, *hyd_solve.eigenstates(H, dr)


def test_hydrogen_energies_degeneracy_and_nodes():
    r, dr, energies, u = hydrogen()
    for n, ell in hyd_solve.orbitals(3):
        E, u_nl = hyd_solve.pick(energies, u, n, ell)
        assert abs(E + 0.5 / n**2) < 5e-4  # atomic units: E_n = −1/(2n²)
        assert hyd_solve.count_nodes(u_nl) == n - ell - 1
        assert abs((u_nl**2).sum() * dr - 1) < 1e-10
    # same n, different l: eigenvalues of different matrices, yet (nearly) equal
    assert abs(energies[0, 1] - energies[1, 0]) < 1e-4  # 2s vs 2p
    assert abs(energies[1, 1] - energies[2, 0]) < 1e-4  # 3p vs 3d


def test_hydrogen_angular_factor_normalised():
    # ∫ |Y_l0|² dΩ = 2π ∫ |Y_l0|² d(cos θ) = 1 for the Legendre factor used in the 2D slices
    cos_theta = torch.linspace(-1, 1, 20001, dtype=torch.float64)
    for ell in range(4):
        Y = hyd_eq.spherical_harmonic_m0(ell, cos_theta)
        assert abs(2 * math.pi * torch.trapezoid(Y**2, cos_theta) - 1) < 1e-6


# --- double well -------------------------------------------------------------------------------


def double_well(V0: float = 3.0, a: float = 1.5, n: int = 400):
    x, dx = dw_t.grid(n, 4 * a + 6, CPU)
    energies, phi = dw_solve.eigenstates(dw_t.hamiltonian(x, dx, dw_t.double_well(x, V0, a)), x, dx)
    return x, dx, energies, phi


def test_double_well_doublet_parity():
    _, _, energies, phi = double_well()
    assert energies[1] - energies[0] < 0.1 * (energies[2] - energies[1])  # a close pair below the rest
    assert torch.allclose(phi[:, 0].flip(0), phi[:, 0], atol=1e-8)  # even
    assert torch.allclose(phi[:, 1].flip(0), -phi[:, 1], atol=1e-8)  # odd


def test_double_well_tunnelling_period_and_cos2():
    x, dx, energies, phi = double_well()
    delta_E = (energies[1] - energies[0]).item()
    period = dw_solve.tunnelling_period(energies)
    times, psi = dw_solve.solve(phi, energies, dw_t.left_state(len(x), CPU), t_max=period, n_steps=60)
    p_left = dw_eq.left_probability(psi, x, dx)
    assert torch.allclose(p_left, dw_eq.two_state_prediction(times, delta_E), atol=0.03)
    assert p_left[0] > 0.97 and p_left[30] < 0.03  # left at t = 0, right at half a period
    assert abs(p_left[-1] - p_left[0]) < 1e-8  # back exactly after 2π/ΔE


# --- 2D scattering -----------------------------------------------------------------------------


def test_scattering_norm_conserved_without_absorber():
    X, Y, KX, KY, dx = sc_t.grid(128, 60.0, CPU)
    V = sc_t.double_slit(X, Y, 40.0, 6.0, 1.5)
    psi0 = sc_t.gaussian_packet(X, Y, -12.0, 3.0, 4.0, 3.0, dx * dx)
    result = sc_solve.solve(psi0, V, KX, KY, dx * dx, t_max=4.0, M=None, n_saved=5)
    assert torch.allclose(result["norm"], torch.ones_like(result["norm"]), atol=1e-10)


def test_free_2d_packet_moves_at_k0():
    X, Y, KX, KY, dx = sc_t.grid(128, 60.0, CPU)
    x0, k0, t = -10.0, 2.5, 6.0
    psi0 = sc_t.gaussian_packet(X, Y, x0, 3.0, 3.0, k0, dx * dx)
    result = sc_solve.solve(psi0, torch.zeros_like(X), KX, KY, dx * dx, t_max=t, n_saved=2)
    centre = sc_solve.mean_position(result["psi"].abs() ** 2, X, Y)
    assert abs(centre[0] - (x0 + k0 * t)) < 1e-3
    assert abs(centre[1]) < 1e-6


def test_kinetic_operator_diagonal_in_k_and_step_unitary():
    X, Y, KX, KY, dx = sc_t.grid(32, 20.0, CPU)
    # a plane wave on the grid is a single nonzero entry of fft2: the Fourier basis diagonalises −½∇²
    plane = torch.exp(1j * (KX[3, 0] * X + KY[0, 5] * Y))
    spectrum = torch.fft.fft2(plane).abs()
    assert (spectrum > 1e-8).sum() == 1 and spectrum[3, 5] > 1
    K = sc_eq.kinetic_phase(KX, KY, 0.05)
    assert torch.allclose(K.abs(), torch.ones_like(K.abs()))
    # one full split step (no absorber) preserves the norm of an arbitrary state
    gen = torch.Generator().manual_seed(0)
    psi = torch.randn(32, 32, dtype=torch.complex128, generator=gen)
    P = sc_eq.potential_phase(sc_t.disk(X, Y, 5.0, 4.0), 0.05)
    after = sc_eq.split_step(psi, P, K, torch.ones_like(X))
    assert torch.allclose(sc_eq.probability(after, 1.0), sc_eq.probability(psi, 1.0))


# --- two qubits --------------------------------------------------------------------------------


def test_bell_state_entropy_and_reduced_state():
    bell = qb_solve.bell_circuit(CPU)
    assert torch.allclose(bell, torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128) / math.sqrt(2))
    assert abs(qb_eq.entanglement_entropy(bell) - math.log(2)) < 1e-12
    assert abs(qb_eq.concurrence(bell) - 1) < 1e-12
    half_identity = torch.eye(2, dtype=torch.complex128) / 2
    assert torch.allclose(qb_eq.reduced_density_1(bell), half_identity)
    assert torch.allclose(qb_eq.reduced_density_2(bell), half_identity)


def test_product_state_partial_trace_and_zero_entropy():
    a = torch.tensor([0.6, 0.8j], dtype=torch.complex128)
    b = torch.tensor([1, 1 - 1j], dtype=torch.complex128) / math.sqrt(3)
    psi = torch.einsum("a,b->ab", a, b)  # |a⟩ ⊗ |b⟩
    assert qb_eq.entanglement_entropy(psi) < 1e-10
    assert qb_eq.concurrence(psi) < 1e-12
    assert torch.allclose(qb_eq.reduced_density_1(psi), torch.outer(a, a.conj()))
    assert torch.allclose(qb_eq.reduced_density_2(psi), torch.outer(b, b.conj()))


def test_gate_tensors_unitary_and_hamiltonian_matches_kron():
    sigma = qb_t.pauli(CPU)
    eye4 = torch.eye(4, dtype=torch.complex128)
    for G in [qb_t.cnot(CPU), qb_t.on_qubit_1(qb_t.hadamard(CPU)), qb_t.on_qubit_2(sigma[1])]:
        M = G.reshape(4, 4)
        assert torch.allclose(M.conj().T @ M, eye4)
    J, h = 0.7, 0.3
    X, Z, eye2 = sigma[0], sigma[2], torch.eye(2, dtype=torch.complex128)
    kron = J * torch.kron(Z, Z) + h * (torch.kron(X, eye2) + torch.kron(eye2, X))
    assert torch.allclose(qb_t.ising_hamiltonian(J, h, sigma).reshape(4, 4), kron)


def test_ising_evolution_unitary_and_entangles_product_state():
    sigma = qb_t.pauli(CPU)
    H = qb_t.ising_hamiltonian(1.0, 0.6, sigma)
    result = qb_solve.solve(qb_t.basis_state(0, 0, CPU), H, sigma, t_max=5.0, n_steps=100)
    norms = torch.einsum("tab,tab->t", result["psi"].conj(), result["psi"]).real
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-12)
    assert result["entropy"][0] < 1e-10 and result["entropy"].max() > 0.3
    # a pure global state has equally mixed halves: |r1| = |r2|
    assert torch.allclose(result["r1"].norm(dim=1), result["r2"].norm(dim=1), atol=1e-10)
