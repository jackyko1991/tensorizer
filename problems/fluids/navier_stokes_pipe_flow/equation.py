r"""Physics → tensor translation.

The incompressible Navier–Stokes equations, in index form (repeated j summed):

    ρ (∂_t u_i + u_j ∂_j u_i) = -∂_i p + μ ∂_j ∂_j u_i,        ∂_i u_i = 0

Pipe flow: fluid at rest in a round pipe of radius R is suddenly pushed by a pressure gradient
G = -∂p/∂z. The flow stays axial, u = u_z(r, t) \hat z, so

    ∂_i u_i = ∂_z u_z = 0           (u_z does not depend on z: continuity holds automatically)
    u_j ∂_j u_z = u_z ∂_z u_z = 0   (the nonlinear advection term vanishes!)

and the z-component of Navier–Stokes, with the Laplacian in cylindrical coordinates, becomes linear:

    ∂_t u_z = G/ρ + ν (1/r) ∂_r (r ∂_r u_z),       u_z(R) = 0 (no slip),  ν = μ/ρ

On grid points r_i = i Δr the radial Laplacian is a matrix L_{ij}, so the PDE is

    \dot u_i = G/ρ + ν L_{ij} u_j        (j summed → einsum('ij,j->i'))

Its steady state is the Hagen–Poiseuille profile u_s = G (R^2 - r^2) / (4μ).
"""

import math

import torch


def rhs(u: torch.Tensor, L: torch.Tensor, G: float, rho: float, nu: float) -> torch.Tensor:
    r"""\dot u_i = G/ρ + ν L_{ij} u_j.   u: (N,), L: (N, N) → (N,)"""
    return G / rho + nu * torch.einsum("ij,j->i", L, u)


def hagen_poiseuille(r: torch.Tensor, G: float, mu: float, R: float) -> torch.Tensor:
    r"""Steady profile u_s = G (R^2 - r^2) / (4μ).   (N,) → (N,)"""
    return G * (R**2 - r**2) / (4 * mu)


def flow_rate(u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    r"""Q = ∫_0^R 2π r u dr (trapezoid rule on the grid).   u: (..., N), r: (N,) → (...)"""
    return torch.trapezoid(2 * math.pi * r * u, r, dim=-1)


def wall_shear_stress(u: torch.Tensor, dr: float, mu: float) -> torch.Tensor:
    r"""τ_w = -μ ∂_r u_z at r = R, from a one-sided 2nd-order difference (u at the wall is 0).   (..., N) → (...)"""
    wall, u1, u2 = u[..., -1], u[..., -2], u[..., -3]
    return -mu * (3 * wall - 4 * u1 + u2) / (2 * dr)
