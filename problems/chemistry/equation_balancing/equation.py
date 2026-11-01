r"""Chemistry → tensor translation.

A reaction  ν₁ A₁ + ν₂ A₂ + …  →  … + ν_S A_S  conserves every element e (and the charge):

    atoms of e on the left  =  atoms of e on the right      ⇔      M_{es} ν_s = 0   (summed over s)

with the signed composition matrix M_{es} = C_{es} σ_s, where C_{es} counts atoms of element e in species s and
σ_s = +1 for reactants, -1 for products. So the coefficients ν form the null space of M: a vector that M sends
to zero. Repeated indices are summed: s in the conservation law. A batch of reactions r just adds an index.
"""

import torch


def signed_composition(C: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """M_{es} = C_{es} σ_s: products count negative.   C: (E, S), σ: (S,) → (E, S).  No index summed."""
    return torch.einsum("es,s->es", C, sigma)


def imbalance(M: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """Δ_{re} = M_{res} ν_{rs}: atoms (or charge) of e created by reaction r.   Repeated s ⇒ summed.

    M: (R, E, S), ν: (R, S) → (R, E). A balanced equation has Δ = 0 for every element.
    """
    return torch.einsum("res,rs->re", M, nu)


def null_dimension(sv: torch.Tensor, tol: float = 1e-9) -> torch.Tensor:
    """Number of (numerically) zero singular values = number of independent reactions.   sv: (R, S) → (R,).

    1 → the balanced equation is unique up to scale;  2 → two reactions are mixed, the balance is not unique.
    """
    return (sv < tol * sv[..., :1]).sum(-1)
