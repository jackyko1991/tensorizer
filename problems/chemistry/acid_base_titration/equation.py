r"""Chemistry → tensor translation.

An acid H_nA gives up its protons one at a time with constants K₁, K₂, …. With h = [H⁺] the fraction of the acid
that has lost j protons is

    α_{aj}(h) = β_{aj} h^{-j} / Σ_k β_{ak} h^{-k},        β_{aj} = K_{a1} K_{a2} ⋯ K_{aj}   (β_{a0} = 1)

and the average charge carried per acid molecule is  n̄_a = j α_{aj}   (summed over j).

The solution must be electrically neutral (charge balance): positive charges = negative charges,

    h + [Na⁺] = K_w / h + C_A n̄(h)       ⇔      f(h) = h + [Na⁺] - K_w/h - C_A j α_j(h) = 0

for every acid a and titrant volume n. Repeated j is summed. Everything is done in log₁₀ h = -pH, because
h spans 15 orders of magnitude; the α are a softmax over j of log β_j + j pH.
"""

import math

import torch

LN10 = math.log(10.0)


def fractions(pH: torch.Tensor, log_beta: torch.Tensor) -> torch.Tensor:
    """α_{a…j} = β_{aj} h^{-j} / Σ_k β_{ak} h^{-k}.   pH: (A, …), log β: (A, J) → (A, …, J).

    log₁₀(β_j h^{-j}) = log₁₀ β_j + j pH, and dividing by the sum is a softmax (stable even for h^{-3} ~ 10⁴²).
    """
    j = torch.arange(log_beta.shape[-1], dtype=pH.dtype, device=pH.device)  # (J,) protons lost
    shape = (log_beta.shape[0],) + (1,) * (pH.dim() - 1) + (log_beta.shape[-1],)
    logits = LN10 * (log_beta.reshape(shape) + j * pH[..., None])  # (A, …, J)
    return torch.softmax(logits, dim=-1)


def mean_charge(alpha: torch.Tensor) -> torch.Tensor:
    """n̄ = j α_j: protons released per acid molecule.   Repeated j ⇒ summed.   (…, J) → (…)."""
    j = torch.arange(alpha.shape[-1], dtype=alpha.dtype, device=alpha.device)
    return torch.einsum("...j,j->...", alpha, j)


def charge_balance(pH: torch.Tensor, log_beta: torch.Tensor, c_acid: torch.Tensor, sodium: torch.Tensor, kw: float):
    """f = h + [Na⁺] - K_w/h - C_A n̄.   pH: (A, N); C_A, [Na⁺]: (N,) → (A, N).

    f > 0: too many positive charges, so the true pH is higher; f decreases monotonically as pH rises.
    """
    h = 10.0 ** (-pH)
    return h + sodium - kw / h - c_acid * mean_charge(fractions(pH, log_beta))


def henderson_hasselbalch(pka: float, base: torch.Tensor, acid: torch.Tensor) -> torch.Tensor:
    """Buffer estimate pH ≈ pKa + log₁₀([A⁻]/[HA]), with [A⁻]/[HA] ≈ (added base)/(acid left)."""
    return pka + torch.log10(base / acid)
