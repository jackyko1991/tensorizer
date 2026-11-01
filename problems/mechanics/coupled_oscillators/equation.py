r"""Physics → tensor translation.

Newton's second law for N masses m joined by springs k between two fixed walls,

    m \ddot{\mathbf x} = -\mathsf K \mathbf x,

reads in index notation (i, j = 1 … N label the masses)

    m \ddot x_i = -K_{ij} x_j

The repeated index j is summed: each mass feels the springs of its neighbours, einsum('ij,j->i').

A normal mode is a motion where every mass oscillates with one frequency, x_i = e_{in} \cos(\omega_n t).
Plugging that in gives the eigenvalue problem (no sum over n here)

    (K_{ij} / m) \, e_{jn} = \omega_n^2 \, e_{in}

and the general motion is a superposition, summed over the mode index n:

    x_i(t) = \sum_n A_n e_{in} \cos(\omega_n t + \varphi_n)
"""

import torch


def spring_force(K: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""F_i = -K_{ij} x_j.   K: (N, N), x: (..., N) → (..., N)"""
    return -torch.einsum("ij,...j->...i", K, x)


def normal_modes(K: torch.Tensor, m: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Solve (K_{ij}/m) e_{jn} = \omega_n^2 e_{in}.   Returns ω (N,) and e (N, N); mode n is the column e[:, n]."""
    omega_sq, e = torch.linalg.eigh(K / m)  # K symmetric ⇒ real ω², orthonormal eigenvectors
    return torch.sqrt(omega_sq), e


def mode_amplitudes(
    e: torch.Tensor, omega: torch.Tensor, x0: torch.Tensor, v0: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""A_n and \varphi_n from the start state, via the modal coordinates q_n = e_{in} x_i."""
    q0 = torch.einsum("in,i->n", e, x0)  # (N,)  q_n(0), sum over masses i
    q0_dot = torch.einsum("in,i->n", e, v0)  # (N,)  \dot q_n(0)
    A = torch.sqrt(q0**2 + (q0_dot / omega) ** 2)  # q_n(t) = A_n cos(ω_n t + φ_n)
    phi = torch.atan2(-q0_dot / omega, q0)
    return A, phi


def superpose(
    e: torch.Tensor, omega: torch.Tensor, A: torch.Tensor, phi: torch.Tensor, t: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""x_{ti} = e_{in} A_n \cos(\omega_n t + \varphi_n) and its time derivative v_{ti}.   t: (T,) → (T, N) each."""
    angle = torch.einsum("t,n->tn", t, omega) + phi  # (T, N)  ω_n t + φ_n
    x = torch.einsum("in,tn->ti", e, A * torch.cos(angle))  # repeated n ⇒ sum over modes
    v = torch.einsum("in,tn->ti", e, -A * omega * torch.sin(angle))
    return x, v


def energy(x: torch.Tensor, v: torch.Tensor, K: torch.Tensor, m: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Kinetic ½ m v_i v_i and potential ½ x_i K_{ij} x_j (all indices summed).   (..., N) → (...,) each."""
    kinetic = 0.5 * m * torch.einsum("...i,...i->...", v, v)
    potential = 0.5 * torch.einsum("...i,ij,...j->...", x, K, x)
    return kinetic, potential
