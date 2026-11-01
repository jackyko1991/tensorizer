r"""Physics → tensor translation.

The Beer–Lambert law: a narrow X-ray beam of energy E loses photons exponentially,

    I = I_0 \exp\left(-\int \mu(E, \mathbf r)\, dl\right).

The phantom is a stack of material masks M_{myx} (1 where pixel (y, x) is material m).
A vertical ray at column x crosses a length L_{mx} of each material:

    L_{mx} = M_{myx} \Delta l_y                  (y summed: path-length weights)

and the line integral of μ splits into "how much of each material × how strongly it attenuates":

    -\ln(I_{ex}/I_0) = \mu_{em} L_{mx}            (m summed)

Indices: e = photon energy, m = material, y = depth along the ray, x = ray (detector pixel).
A polychromatic beam with spectrum S_e is a weighted sum of monochromatic beams:

    I_x / I_0 = S_e \exp(-\mu_{em} L_{mx})        (e summed, S_e sums to 1)
"""

import torch


def path_lengths(masks: torch.Tensor, dl: torch.Tensor) -> torch.Tensor:
    r"""L_{mx} = M_{myx} \Delta l_y.   masks: (M, Y, X), dl: (Y,) → (M, X) in cm"""
    return torch.einsum("myx,y->mx", masks, dl)


def line_integral(mu: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    r"""\int \mu\,dl = \mu_{em} L_{mx}.   mu: (E, M), lengths: (M, X) → (E, X), dimensionless"""
    return torch.einsum("em,mx->ex", mu, lengths)


def transmission(mu: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    r"""Beer–Lambert I/I_0 = \exp(-\mu_{em} L_{mx}) for every energy and ray: (E, X)."""
    return torch.exp(-line_integral(mu, lengths))


def polychromatic(spectrum: torch.Tensor, mu: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    r"""I_x/I_0 = S_e \exp(-\mu_{em} L_{mx}):  spectrum (E,) sums to 1 → (X,)."""
    return torch.einsum("e,ex->x", spectrum, transmission(mu, lengths))


def contrast(mu_insert: torch.Tensor, mu_background: torch.Tensor, thickness: float) -> torch.Tensor:
    r"""Subject contrast of an insert of thickness t replacing background: C = 1 - e^{-(\mu_i - \mu_b) t}."""
    return 1.0 - torch.exp(-(mu_insert - mu_background) * thickness)


def effective_mu(depth: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    r"""Local effective attenuation \mu_{eff}(d) = -\,d\ln I / dd (finite differences): (D-1,)."""
    return -torch.diff(torch.log(intensity)) / torch.diff(depth)
