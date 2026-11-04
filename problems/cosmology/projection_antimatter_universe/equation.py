"""The criteria an antimatter universe must meet, as tensor operations.

Shapes in comments: E photon energies, Z redshifts, D domain sizes, J pion energies, B batch members, S samples along
a worldline, L time slices, W windings. Summed indices: the redshift z (photons arriving today), the pion energy j
(photon spectrum of one annihilation), the radius r (emission measure), the segment s (crossings of a slice).
Units: cgs, energies in MeV, distances in Mpc unless stated.
"""

import math

import torch

C_CM = 2.998e10  # speed of light, cm/s
MPC_CM = 3.0857e24  # one megaparsec in cm
K_B = 1.380649e-16  # Boltzmann constant, erg/K
M_P = 1.67262e-24  # proton mass, g
M_PI0 = 134.977  # neutral pion mass, MeV
N_GAMMA = 3.0  # photons per p pbar annihilation (about 1.5 neutral pions, 2 photons each)


# --- criterion 1: nothing mixed in nearby -------------------------------------------------------------------------


def emission_measure(n0: float, rc: float, beta: float, r_max: float, n: int = 4000) -> float:
    """int_0^r_max 4 pi r^2 n(r)^2 dr for the beta model n = n0 (1 + r^2/rc^2)^(-3 beta / 2) (r summed, log grid)."""
    r = torch.logspace(math.log10(1e-4 * rc), math.log10(r_max), n, dtype=torch.float64)
    f = 4 * math.pi * r**2 * n0**2 * (1 + (r / rc) ** 2) ** (-3 * beta)
    core = 4 / 3 * math.pi * (1e-4 * rc) ** 3 * n0**2  # the tiny sphere inside the first grid point
    return float(torch.trapezoid(f, r) + core)


def photons_above(E_min: float) -> float:
    """Photons above E_min (MeV) per annihilation, from the pion-decay spectrum."""
    E = torch.logspace(math.log10(E_min), 3.3, 4000, dtype=torch.float64)
    return float(torch.trapezoid(pion_decay_spectrum(E), E))


def cluster_flux(f: torch.Tensor, c: dict) -> torch.Tensor:
    """Gamma-ray flux above 100 MeV (ph cm^-2 s^-1) from a Coma-like cluster whose gas holds an antiproton fraction f:
    F = N_gamma(>100 MeV) f <sigma v> int n_p^2 dV / (4 pi D^2), with n_p = n_e / 1.17 (helium)."""
    rc = c["coma_rc_kpc"] * 1e-3 * MPC_CM
    em = emission_measure(c["coma_n0_cm3"] / 1.17, rc, c["coma_beta"], r_max=3 * MPC_CM)
    dist = C_CM / 1e5 * c["coma_z"] / (100 * c["planck_h"]) * MPC_CM  # Hubble law, cm
    return photons_above(100.0) * f * c["pbar_p_sigma_v"] * em / (4 * math.pi * dist**2)


def cluster_antimatter_limit(c: dict) -> float:
    """Largest antiproton fraction allowed by the Fermi-LAT upper limit on Coma (the flux is linear in f)."""
    one = cluster_flux(torch.ones(1, dtype=torch.float64), c)
    return float(c["coma_flux_ul"] / one)


# --- criterion 2: domains inside our horizon -----------------------------------------------------------------------


def wall_area_per_volume(d: torch.Tensor, n: int = 3) -> torch.Tensor:
    """Checkerboard of cubes of side d in n dimensions: each cell owns n faces of area d^(n-1), volume d^n -> n / d."""
    return n / d


def domains(noise: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """(D, N, N) random matter (+1) / antimatter (-1) domains of typical size d (pixels): the sign of white noise
    smoothed by a Gaussian of width d / 3 (FFT, periodic box)."""
    N = noise.shape[-1]
    k = torch.fft.fftfreq(N, dtype=torch.float64) * 2 * math.pi  # (N,) rad per pixel
    k2 = k[:, None] ** 2 + k[None, :] ** 2  # (N, N)
    sigma = (d / 3)[:, None, None]  # (D, 1, 1)
    field = torch.fft.ifft2(torch.fft.fft2(noise)[None] * torch.exp(-k2 * sigma**2 / 2)).real
    return torch.where(field >= 0, 1, -1)


def walls(sign: torch.Tensor) -> torch.Tensor:
    """(..., N, N) mask of pixels next to a neighbour of the opposite kind (where annihilation happens)."""
    m = torch.zeros_like(sign, dtype=torch.bool)
    for shift, dim in ((1, -1), (-1, -1), (1, -2), (-1, -2)):
        m |= sign != torch.roll(sign, shift, dim)
    return m


def measured_wall_per_area(sign: torch.Tensor) -> torch.Tensor:
    """(...,) wall length per unit area in pixels^-1: unlike nearest-neighbour pairs (x and y bonds) per pixel."""
    bonds = (sign != torch.roll(sign, 1, -1)).double() + (sign != torch.roll(sign, 1, -2)).double()
    return bonds.mean((-1, -2))


def z_grid(z_max: float, n: int, z_min: float = 0.0):
    """Redshifts z (Z,) uniform in ln(1 + z) from z_min to z_max, and trapezoid weights dz (Z,)."""
    x = torch.linspace(math.log1p(z_min), math.log1p(z_max), n, dtype=torch.float64)
    z = torch.expm1(x)
    w = torch.full_like(x, x[1] - x[0])
    w[[0, -1]] /= 2
    return z, w * (1 + z)


def hubble(z: torch.Tensor, c: dict) -> torch.Tensor:
    """H(z) in s^-1 for flat LCDM with radiation (Omega_r h^2 = 4.15e-5, photons + neutrinos)."""
    h = c["planck_h"]
    om, orad = c["planck_omega_m"], 4.15e-5 / h**2
    H0 = 100 * h * 1e5 / MPC_CM
    return H0 * torch.sqrt(om * (1 + z) ** 3 + orad * (1 + z) ** 4 + (1 - om - orad))


def gas_sound_speed(z: torch.Tensor, c: dict) -> torch.Tensor:
    """Adiabatic sound speed (cm/s) of the gas: T = T_CMB (1+z) above z_dec, then adiabatic cooling (1+z)^2,
    and 10^4 K, fully ionised, after reionization. c_s = sqrt(5 k T / (3 mu m_p))."""
    T0, zd, zr = c["cmb_t0_k"], c["z_dec"], c["planck_z_re"]
    T = torch.where(z >= zd, T0 * (1 + z), T0 * (1 + z) ** 2 / (1 + zd))
    ionised = z < zr
    T = torch.where(ionised, torch.clamp(T, min=1e4), T)
    mu = torch.where(ionised, 0.59, 1.22)
    return torch.sqrt(5 * K_B * T / (3 * mu * M_P))


def pion_decay_spectrum(E: torch.Tensor, n_pion: int = 120) -> torch.Tensor:
    """Photons per MeV from one p pbar annihilation (a modelling choice): N_GAMMA photons from neutral pions with total
    energies E_pi ~ Gaussian(350 MeV, 150 MeV); each pion gives a flat box of photons from E_pi(1 - beta)/2 to
    E_pi(1 + beta)/2 of width p_pi (the pion energy j is summed)."""
    Epi = torch.linspace(M_PI0 * 1.001, 1200.0, n_pion, dtype=torch.float64, device=E.device)  # (J,)
    w = torch.exp(-((Epi - 350.0) ** 2) / (2 * 150.0**2))
    w = w / w.sum()
    p = torch.sqrt(Epi**2 - M_PI0**2)
    lo, hi = (Epi - p) / 2, (Epi + p) / 2
    inside = (E[..., None] >= lo) & (E[..., None] <= hi)  # (..., J)
    return N_GAMMA * (inside.double() * (w / p)).sum(-1)


def redshifted_density(E, z, dz, rate, H, spectrum) -> torch.Tensor:
    """Photons per cm^3 per MeV today (E,): n(E) = int dz rate(z) S(E (1+z)) / H(z), summed over z.
    rate (..., Z) is per comoving volume per second, H (Z,), spectrum a function of the emitted energy."""
    S = spectrum(E[:, None] * (1 + z)[None, :])  # (E, Z)
    return torch.einsum("...z,ez->...e", rate / H * dz, S)  # repeated z => summed


def annihilation_rate(z: torch.Tensor, d_mpc: torch.Tensor, c: dict, speed_factor: float) -> torch.Tensor:
    """(D, Z) annihilations per comoving cm^3 per s: (wall area / volume) a^2 n_b(z) v = (3 / d) n_b0 (1 + z) v(z),
    with baryons flowing into the walls at v = speed_factor x the gas sound speed."""
    eta = 273.9e-10 * c["planck_omega_b_h2"]
    nb0 = eta * c["n_gamma_cm3"]
    av = wall_area_per_volume(d_mpc * MPC_CM)  # (D,) comoving cm^-1
    v = speed_factor * gas_sound_speed(z, c)  # (Z,)
    return av[:, None] * nb0 * (1 + z)[None, :] * v[None, :]


def annihilation_intensity(E, d_mpc, c, speed_factor: float, z_max: float, z_min: float = 20.0, nz: int = 1500):
    """(D, E) diffuse gamma-ray intensity dJ/dE today (ph cm^-2 s^-1 sr^-1 MeV^-1) from annihilation at domain walls
    between recombination (z_max) and z_min: J = c n(E) / (4 pi). As in Cohen, De Rujula & Glashow, later epochs
    (structure formation, reionization) are left out, which makes the bound conservative; absorption is neglected."""
    z, dz = z_grid(z_max, nz, z_min)
    R = annihilation_rate(z, d_mpc, c, speed_factor)
    n = redshifted_density(E, z, dz, R, hubble(z, c), pion_decay_spectrum)
    return C_CM * n / (4 * math.pi)


def comptel_intensity(E: torch.Tensor, c: dict) -> torch.Tensor:
    """COMPTEL diffuse background fit dJ/dE (ph cm^-2 s^-1 sr^-1 MeV^-1), NaN outside 0.8-30 MeV."""
    J = c["comptel_norm"] * E ** (-c["comptel_index"])
    return torch.where((E >= c["comptel_emin_mev"]) & (E <= c["comptel_emax_mev"]), J, torch.nan)


def fermi_intensity(E: torch.Tensor, c: dict) -> torch.Tensor:
    """Fermi-LAT isotropic background fit: K (E/100 MeV)^-Gamma exp(-E/E_cut), K fixed by the published intensity
    above 100 MeV; NaN outside 100 MeV - 820 GeV."""
    lo, hi, g, ec = c["fermi_emin_mev"], c["fermi_emax_mev"], c["fermi_index"], c["fermi_cutoff_gev"] * 1e3
    e = torch.logspace(math.log10(lo), math.log10(hi), 4000, dtype=torch.float64)
    shape = lambda x: (x / 100.0) ** (-g) * torch.exp(-x / ec)  # noqa: E731
    K = c["fermi_intensity_100mev"] / torch.trapezoid(shape(e), e)
    return torch.where((E >= lo) & (E <= hi), K * shape(E), torch.nan)


def observed_intensity(E: torch.Tensor, c: dict) -> torch.Tensor:
    """(E,) the measured diffuse background where either instrument covers E (NaN in the gap 30-100 MeV)."""
    return torch.where(torch.isnan(comptel_intensity(E, c)), fermi_intensity(E, c), comptel_intensity(E, c))


def minimum_domain_size(c: dict, speed_factor: float, z_max: float, z_min: float = 20.0) -> dict:
    """Smallest domain size the background allows. The flux is exactly proportional to 1/d, so computing it once for
    d = 1 Mpc gives d_min = 1 Mpc x max_E (predicted / observed) over the energies the instruments cover."""
    E = torch.logspace(math.log10(0.8), math.log10(8.2e5), 300, dtype=torch.float64)
    J1 = annihilation_intensity(E, torch.ones(1, dtype=torch.float64), c, speed_factor, z_max, z_min)[0]
    ratio = J1 / observed_intensity(E, c)
    ratio = torch.nan_to_num(ratio, nan=0.0)
    i = int(ratio.argmax())
    return {"d_min_mpc": float(ratio[i]), "E_worst_mev": float(E[i]), "E": E, "ratio_1mpc": ratio}


# --- criterion 3: fine-tuning of one random worldline ---------------------------------------------------------------


def random_net_charge(M: torch.Tensor, B: int, generator) -> torch.Tensor:
    """(len(M), B) net charge of M crossings with independent random signs +-1 (B samples each)."""
    signs = torch.randint(0, 2, (B, int(M.max())), generator=generator, dtype=torch.int8) * 2 - 1
    run = torch.cumsum(signs, 1, dtype=torch.int32)  # (B, M_max) partial sums
    return run[:, M - 1].T.long()


def patch_net_charge_balanced(T: int, M: torch.Tensor, B: int, generator) -> torch.Tensor:
    """(len(M), B) net charge of the first M of T crossings of a balanced loop (T/2 of each sign, random order)."""
    order = torch.rand(B, T, generator=generator).argsort(1)  # (B, T) random permutations
    signs = torch.where(order < T // 2, 1, -1)
    return torch.cumsum(signs, 1)[:, M - 1].T.long()


def bridge_variance(M: torch.Tensor, T: int) -> torch.Tensor:
    """Variance of the patch net charge for a balanced total: M (T - M) / (T - 1) (sampling without replacement)."""
    return M * (T - M) / (T - 1)


def log10_tail(Q: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """log10 P(net >= Q) for M random signs, from Cramer's rate function: -M KL((1 + q)/2 || 1/2) / ln 10, q = Q/M.
    KL = [(1+q) ln(1+q) + (1-q) ln(1-q)] / 2, with the series q^2/2 + q^4/12 for tiny q (no cancellation)."""
    q = Q / M
    exact = 0.5 * (torch.special.xlogy(1 + q, 1 + q) + torch.special.xlogy(1 - q, 1 - q))
    series = q**2 / 2 + q**4 / 12 + q**6 / 30
    kl = torch.where(q.abs() < 1e-3, series, exact)
    return -M * kl / math.log(10)


def log10_tail_exact(Q: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    """log10 P(net >= Q), exact binomial tail: sum over k >= (M + Q)/2 of C(M, k) 2^-M (lgamma, logsumexp)."""
    out = []
    for q, m in zip(Q.tolist(), M.tolist(), strict=True):
        k = torch.arange(math.ceil((m + q) / 2), int(m) + 1, dtype=torch.float64)
        logc = math.lgamma(m + 1) - torch.lgamma(k + 1) - torch.lgamma(m - k + 1) - m * math.log(2)
        out.append(torch.logsumexp(logc, 0) / math.log(10))
    return torch.stack(out)


# --- criterion 4: one worldline across the CPT mirror ---------------------------------------------------------------


def crossings(t: torch.Tensor, levels: torch.Tensor, closed: bool):
    """(B, L) forward (particle) and backward (antiparticle) crossings of every slice t = u (segments s summed)."""
    a = t
    b = torch.roll(t, -1, 1) if closed else t[:, 1:]
    a = a if closed else t[:, :-1]
    a, b = a[..., None], b[..., None]  # (B, S, 1)
    up = ((a < levels) & (levels <= b)).sum(1)
    down = ((b < levels) & (levels <= a)).sum(1)
    return up, down


def charges_across_mirror(eta: torch.Tensor, ours: torch.Tensor, period: float):
    """Net charges on our slices eta = u (0 < u < period/2) and on their mirror images eta = -u, each read with its
    own arrow of time: forward in eta is a particle for us; in the CPT mirror the arrow points to -eta, so the sign
    flips. eta (W, S + 1) is unwrapped conformal time on a circle of length `period`; every copy u + j period counts."""
    j = torch.arange(math.floor(eta.min() / period) - 1, math.ceil(eta.max() / period) + 2, dtype=torch.float64)
    out = []
    for u in (ours, -ours):
        ext = (u[None, :] + period * j[:, None]).reshape(-1)  # (J L,)
        up, down = crossings(eta, ext, closed=False)
        out.append((up - down).view(len(eta), len(j), -1).sum(1))
    q_ours, q_mirror_global = out
    return q_ours, -q_mirror_global  # the mirror reads time backwards


def masses(m_light: torch.Tensor, ordering: str, c: dict):
    """(m1, m2, m3) in eV from the lightest mass and the NuFIT splittings."""
    if ordering == "normal":
        m1 = m_light
        return m1, torch.sqrt(m1**2 + c["dm21"]), torch.sqrt(m1**2 + c["dm31_no"])
    m3 = m_light
    m2 = torch.sqrt(m3**2 - c["dm32_io"])
    return torch.sqrt(m2**2 - c["dm21"]), m2, m3


def sum_mnu(m_light: torch.Tensor, ordering: str, c: dict) -> torch.Tensor:
    """Sum of the three neutrino masses (eV), what cosmology measures."""
    m1, m2, m3 = masses(m_light, ordering, c)
    return m1 + m2 + m3


def m_beta(m_light: torch.Tensor, ordering: str, c: dict) -> torch.Tensor:
    """Effective electron-neutrino mass KATRIN measures: m_beta^2 = sum_i |U_ei|^2 m_i^2 (PDG mixing angles)."""
    m1, m2, m3 = masses(m_light, ordering, c)
    s12, s13 = c["s12sq"], c["s13sq"]
    return torch.sqrt((1 - s12) * (1 - s13) * m1**2 + s12 * (1 - s13) * m2**2 + s13 * m3**2)


def allowed(x: torch.Tensor, upper_limit: float) -> torch.Tensor:
    """A point is allowed by an upper limit when it lies strictly below it."""
    return x < upper_limit
