"""Build the concrete tensors: the example matrices, the grid lines, the unit circle and the unit square."""

import math

import torch

c, s = math.cos(math.radians(60)), math.sin(math.radians(60))
# name: (matrix A, caption). Every matrix is shown as a path from the identity to A.
MATRICES = {
    "stretch": ([[2.0, 1.0], [0.5, 1.5]], "eigenvalues 2.5 and 1: two lines stay put"),
    "shear": ([[1.0, 1.0], [0.0, 1.0]], "shear: area kept, one eigen-line"),
    "rotation": ([[c, -s], [s, c]], "rotation by 60°: no real eigenvectors"),
    "reflection": ([[-0.5, s], [s, 0.5]], "reflection in the 60° line: det = −1"),
    "flip": ([[0.5, 1.2], [1.0, 0.2]], "det < 0: orientation flips"),
}
TOUR = ["stretch", "shear", "rotation", "reflection", "flip"]


def matrices(choice: str, custom: list[float], device: torch.device) -> tuple[torch.Tensor, list[str]]:
    """The matrices to show, A_{sij} of shape (S, 2, 2), and a caption for each."""
    if choice == "tour":
        names = TOUR
    elif choice == "custom":
        a = torch.tensor(custom, dtype=torch.float64, device=device).reshape(1, 2, 2)
        return a, [f"custom A = [[{custom[0]:g}, {custom[1]:g}], [{custom[2]:g}, {custom[3]:g}]]"]
    else:
        names = [choice]
    a = torch.tensor([MATRICES[n][0] for n in names], dtype=torch.float64, device=device)
    return a, [MATRICES[n][1] for n in names]


def grid_lines(n_lines: int, half: float, device: torch.device, samples: int = 41) -> torch.Tensor:
    """Points along n_lines vertical and n_lines horizontal lines covering [−half, half]²: (2·n_lines, samples, 2)."""
    ticks = torch.linspace(-half, half, n_lines, dtype=torch.float64, device=device)  # where the lines sit
    along = torch.linspace(-half, half, samples, dtype=torch.float64, device=device)  # points on each line
    vertical = torch.stack([ticks[:, None].expand(-1, samples), along[None, :].expand(n_lines, -1)], dim=-1)
    horizontal = vertical.flip(-1)  # swap x and y
    return torch.cat([vertical, horizontal])


def unit_circle(n: int, device: torch.device) -> torch.Tensor:
    """Points (cos θ, sin θ) on the unit circle: (n, 2)."""
    theta = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64, device=device)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)


def unit_square(device: torch.device) -> torch.Tensor:
    """Corners of the unit square, counter-clockwise and closed: (5, 2). Its image has area |det A|."""
    return torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.float64, device=device)


def levi_civita(device: torch.device) -> torch.Tensor:
    """ε_{ij}: ε_12 = 1, ε_21 = −1, 0 on the diagonal: (2, 2)."""
    return torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64, device=device)
