r"""Physics → tensor translation.

A Brownian particle of mass m in a harmonic trap (spring constant k) feels friction -γv from the
surrounding fluid and random kicks from its molecules. The Langevin equation adds this noise (the
"error term") to Newton's law:

    m dv = (-γ v - k x) dt + √(2 γ k_B T) dW,        dx = v dt

dW is a Wiener increment: a Gaussian with mean 0 and variance dt. With a finite step Δt we write
dW = √Δt ξ with ξ ~ N(0, 1). That √Δt (not Δt!) is the key point of stochastic calculus: random kicks
add up like a random walk, so their spread grows like the square root of time.

Euler–Maruyama for a batch of B particles b, time step n, noise tensor ξ_{nb}:

    v_{b,n+1} = v_{bn} + (-γ v_{bn} - k x_{bn}) Δt / m + (√(2 γ k_B T Δt) / m) ξ_{nb}
    x_{b,n+1} = x_{bn} + v_{b,n+1} Δt                          (new velocity: symplectic Euler)

    ⟨ξ_{nb} ξ_{n'b'}⟩ = δ_{nn'} δ_{bb'}      every step and every particle gets an independent kick

Ensemble averages sum over the batch index b:   ⟨x²⟩_n = (1/B) x_{nb} x_{nb}   (b summed).
"""

import math

import torch

K_B = 1.0  # reduced units: energies are measured in units where Boltzmann's constant is 1


def drift(x: torch.Tensor, v: torch.Tensor, gamma: float, k: float, m: float) -> torch.Tensor:
    """Deterministic acceleration a_b = (-γ v_b - k x_b) / m.   x, v: (B,) → (B,)."""
    return (-gamma * v - k * x) / m


def noise_amplitude(gamma: float, T: float, m: float, dt: float) -> float:
    """Standard deviation of the random velocity kick in one step: √(2 γ k_B T Δt) / m.

    Fluctuation–dissipation: the same γ that damps the motion sets the strength of the kicks,
    which is what makes the particle settle at temperature T.
    """
    return math.sqrt(2.0 * gamma * K_B * T * dt) / m


def euler_maruyama_step(x, v, xi, gamma: float, k: float, m: float, T: float, dt: float):
    """One step for every particle at once.   x, v, xi: (B,) → new (x, v), each (B,)."""
    v_new = v + drift(x, v, gamma, k, m) * dt + noise_amplitude(gamma, T, m, dt) * xi
    x_new = x + v_new * dt
    return x_new, v_new


def ensemble_mean_square(x: torch.Tensor) -> torch.Tensor:
    """⟨x²⟩_t = (1/B) Σ_b x_{tb} x_{tb}.   x: (T, B) → (T,);  the batch index b is summed."""
    return torch.einsum("tb,tb->t", x, x) / x.shape[1]


def equipartition(T: float, k: float, m: float) -> tuple[float, float]:
    """Thermal equilibrium: ½ k ⟨x²⟩ = ½ m ⟨v²⟩ = ½ k_B T   →   (⟨x²⟩, ⟨v²⟩) = (k_B T / k, k_B T / m)."""
    return K_B * T / k, K_B * T / m


def gaussian(x: torch.Tensor, variance: float) -> torch.Tensor:
    """Boltzmann distribution in the trap, p(x) ∝ exp(-k x² / 2 k_B T) = N(0, k_B T / k)."""
    return torch.exp(-(x**2) / (2 * variance)) / math.sqrt(2 * math.pi * variance)


def damped_oscillator(t: torch.Tensor, x0: float, gamma: float, k: float, m: float) -> torch.Tensor:
    """Noise-free solution m ẍ = -γ ẋ - k x released from rest at x0 (also the ensemble mean ⟨x⟩(t)).

    x(t) = x0 e^{-βt} (cos ω_d t + (β/ω_d) sin ω_d t),   β = γ / 2m,   ω_d = √(k/m - β²)
    (written with complex ω_d so the same formula covers the overdamped case).
    """
    beta = gamma / (2 * m)
    cdtype = torch.complex128 if t.dtype == torch.float64 else torch.complex64
    wd = torch.sqrt(torch.tensor(k / m - beta**2, dtype=cdtype, device=t.device))
    if wd.abs() < 1e-6:  # critical damping
        return x0 * torch.exp(-beta * t) * (1 + beta * t)
    tc = t.to(cdtype)
    x = x0 * torch.exp(-beta * tc) * (torch.cos(wd * tc) + beta / wd * torch.sin(wd * tc))
    return x.real
