"""Build the concrete tensors: tissue parameters, time axes, TR/TE grids and a label-map 'brain' phantom."""

import torch

# Approximate values at 1.5 T (times in ms, proton density PD relative to CSF). For teaching only.
TISSUES = ["white matter", "grey matter", "CSF", "fat", "muscle"]
_T1 = [600.0, 950.0, 4000.0, 250.0, 900.0]
_T2 = [80.0, 100.0, 2000.0, 70.0, 50.0]
_PD = [0.70, 0.80, 1.00, 0.90, 0.75]


def tissue_parameters(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """T1_s, T2_s, PD_s for the S = 5 tissues: three tensors of shape (S,)."""
    return (
        torch.tensor(_T1, device=device),
        torch.tensor(_T2, device=device),
        torch.tensor(_PD, device=device),
    )


def time_axis(t_max: float, n: int, device: torch.device) -> torch.Tensor:
    """t_t from 0 to t_max (ms): shape (T,)."""
    return torch.linspace(0.0, t_max, n, device=device)


def protocol_grids(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Repetition times TR_a (A,) from 100 to 5000 ms and echo times TE_b (B,) from 5 to 200 ms."""
    tr = torch.arange(100.0, 5001.0, 50.0, device=device)  # (A,) = (99,)
    te = torch.arange(5.0, 201.0, 1.0, device=device)  # (B,) = (196,)
    return tr, te


def phantom(n: int, device: torch.device) -> torch.Tensor:
    """One-hot label map L_{xys} of a cartoon axial brain slice: shape (n, n, S), zero outside the head.

    Layers from outside in: scalp fat, a muscle band, grey-matter cortex, white matter, two CSF ventricles.
    """
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, n, device=device), torch.linspace(-1, 1, n, device=device), indexing="ij"
    )
    r = torch.sqrt((x / 0.80) ** 2 + (y / 0.95) ** 2)  # elliptical "radius" of the head
    label = torch.full((n, n), -1, dtype=torch.long, device=device)  # -1 = air
    label[r < 0.98] = 3  # fat (scalp)
    label[r < 0.90] = 4  # muscle / tissue under the scalp
    label[r < 0.84] = 1  # grey matter
    angle = torch.atan2(y, x)
    label[r < 0.62 + 0.08 * torch.cos(9 * angle)] = 0  # white matter with a wiggly (gyral) boundary
    for cx in (-0.17, 0.17):  # two lateral ventricles filled with CSF
        ventricle = ((x - cx) / 0.10) ** 2 + ((y + 0.05) / 0.32) ** 2 < 1
        label[ventricle] = 2
    label[(x / 0.05) ** 2 + ((y - 0.45) / 0.18) ** 2 < 1] = 2  # a CSF-filled fissure
    onehot = torch.zeros(n, n, len(TISSUES), device=device)
    inside = label >= 0
    onehot[inside, label[inside]] = 1.0
    return onehot
