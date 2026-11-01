r"""Physics → tensor translation.

After an excitation the magnetisation of every tissue s relaxes back to equilibrium M_0:

    M_{z,st}  = M_{0,s} (1 - e^{-t_t / T_{1,s}})          longitudinal recovery after a 90° pulse
    M_{z,st}  = M_{0,s} (1 - 2 e^{-t_t / T_{1,s}})        inversion recovery after a 180° pulse
    M_{xy,st} = M_{0,s} e^{-t_t / T_{2,s}}                 transverse decay

Index s labels the tissue and t the time sample; nothing is summed, the formulas broadcast over (s, t).
A spin-echo sequence repeated every TR and read out at TE gives the signal tensor over (tissue, TR, TE):

    S_{sab} = PD_s (1 - e^{-TR_a / T_{1,s}}) e^{-TE_b / T_{2,s}}

An image is the label map L_{xys} (one-hot: which tissue fills pixel x, y) contracted with S over s:

    I_{xyab} = L_{xys} S_{sab}          (s summed)
"""

import math

import torch


def longitudinal(t: torch.Tensor, T1: torch.Tensor, M0: torch.Tensor) -> torch.Tensor:
    r"""M_{z,st} = M_{0,s}(1 - e^{-t_t/T_{1,s}}).   t: (T,), T1, M0: (S,) → (S, T)"""
    return M0[:, None] * (1 - torch.exp(-t[None, :] / T1[:, None]))


def inversion_recovery(t: torch.Tensor, T1: torch.Tensor, M0: torch.Tensor) -> torch.Tensor:
    r"""M_{z,st} = M_{0,s}(1 - 2e^{-t_t/T_{1,s}}): starts at -M_0, crosses zero at T_1 \ln 2.   → (S, T)"""
    return M0[:, None] * (1 - 2 * torch.exp(-t[None, :] / T1[:, None]))


def transverse(t: torch.Tensor, T2: torch.Tensor, M0: torch.Tensor) -> torch.Tensor:
    r"""M_{xy,st} = M_{0,s} e^{-t_t/T_{2,s}}.   t: (T,), T2, M0: (S,) → (S, T)"""
    return M0[:, None] * torch.exp(-t[None, :] / T2[:, None])


def null_point(T1: torch.Tensor) -> torch.Tensor:
    r"""Inversion time where M_z = 0:  TI_s = T_{1,s} \ln 2.   (S,) → (S,)"""
    return T1 * math.log(2.0)


def spin_echo_signal(
    PD: torch.Tensor, T1: torch.Tensor, T2: torch.Tensor, TR: torch.Tensor, TE: torch.Tensor
) -> torch.Tensor:
    r"""S_{sab} = PD_s (1 - e^{-TR_a/T_{1,s}}) e^{-TE_b/T_{2,s}}.   (S,), (A,), (B,) → (S, A, B)

    The three factors depend on different index pairs; einsum multiplies them into one tensor (no sum).
    """
    recovery = 1 - torch.exp(-TR[None, :] / T1[:, None])  # (S, A)   T1 weighting
    decay = torch.exp(-TE[None, :] / T2[:, None])  # (S, B)   T2 weighting
    return torch.einsum("s,sa,sb->sab", PD, recovery, decay)


def image(labels: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    r"""I_{xy...} = L_{xys} S_{s...}  (s summed).   labels: (N, N, S), signal: (S, ...) → (N, N, ...)"""
    return torch.einsum("xys,s...->xy...", labels, signal)
