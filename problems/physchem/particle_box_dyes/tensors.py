"""Build the concrete tensors: the batch of molecules, their box lengths, the level numbers and the grids.

Units: lengths in nm, energies in eV, wavelengths in nm.
"""

import torch


def molecules(series: str, n_max: int, device: torch.device) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Names, number of conjugated bonds (B,) and number of π electrons (B,) for a homologous series.

    cyanine  R₂N–(CH=CH)ₖ–CH=N⁺R₂,  k = 0 … n_max:  2k+3 chain atoms, 2k+2 bonds, 2k+4 π electrons
             (every carbon gives one π electron, the two nitrogens three between them).
    polyene  H–(CH=CH)ₖ–H,          k = 2 … n_max+2: 2k carbons, 2k−1 bonds, 2k π electrons.
    """
    if series == "cyanine":
        k = torch.arange(0, n_max + 1, device=device)
        names = [f"{2 * int(i) + 1}-methine" for i in k]  # number of CH groups between the two N atoms
        return names, 2 * k + 2, 2 * k + 4
    k = torch.arange(2, n_max + 3, device=device)
    names = [f"{int(i)} C=C" for i in k]
    return names, 2 * k - 1, 2 * k


def box_length(n_bonds: torch.Tensor, bond: float, end: float) -> torch.Tensor:
    """L_b = (n_bonds + 2·end) · l   (B,) nm.

    Kuhn's choice: the box runs over every conjugated bond and sticks out by `end` bond lengths
    past each terminal atom, because the π cloud does not stop dead at the last nucleus.
    """
    return (n_bonds.to(torch.get_default_dtype()) + 2.0 * end) * bond


def level_numbers(n_levels: int, device: torch.device) -> torch.Tensor:
    """n = 1, 2, …, n_levels   (n,)."""
    return torch.arange(1, n_levels + 1, device=device, dtype=torch.get_default_dtype())


def grids(L: torch.Tensor, n_points: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interior grid points x_bi (B, M) of each box [0, L_b] and the spacing dx_b (B,).

    ψ = 0 at the two walls, so only the M interior points are unknowns.
    """
    dx = L / (n_points + 1)  # (B,)
    i = torch.arange(1, n_points + 1, device=L.device, dtype=L.dtype)  # (M,)
    return torch.einsum("b,i->bi", dx, i), dx


def visible_wavelengths(device: torch.device) -> torch.Tensor:
    """λ grid over the visible range, 380–780 nm   (W,)."""
    return torch.linspace(380.0, 780.0, 201, device=device)


def measured_cyanine(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate measured λ_max (nm) of the simple cyanines Me₂N–(CH=CH)ₖ–CH=N⁺Me₂ for k = 1 … 4.

    Rounded literature values (about 100 nm per added CH=CH), shown only as a rough check of the model.
    """
    k = torch.tensor([1, 2, 3, 4], device=device)
    lam = torch.tensor([313.0, 416.0, 519.0, 625.0], device=device)
    return k, lam
