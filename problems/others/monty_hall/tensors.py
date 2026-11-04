"""Build the concrete tensors: millions of games, one row per number of doors, played all at once."""

import torch


def door_counts(max_doors: int) -> torch.Tensor:
    """N_b: the numbers of doors compared side by side, from the classic 3 up to max_doors: shape (B,)."""
    ladder = [3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    return torch.tensor([n for n in ladder if n <= max_doors])


def games(
    doors: torch.Tensor, n_games: int, host: str, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Car door c_{bg}, first pick p_{bg} and the other door the host leaves shut k_{bg}, all of shape (B, G).

    Row b plays G games with N_b doors. `floor(u · N)` turns a uniform u ∈ [0, 1) into a uniform door 0 … N − 1,
    and broadcasting N_b against (B, G) gives every row its own number of doors without a loop.
    """
    gen = torch.Generator().manual_seed(seed)
    n = doors.to(torch.float64)[:, None]  # (B, 1)
    u = torch.rand(3, len(doors), n_games, generator=gen, dtype=torch.float64)  # (3, B, G)
    car = (u[0] * n).long()  # (B, G) where the car is
    pick = (u[1] * n).long()  # (B, G) the contestant's first choice
    other = (pick + 1 + (u[2] * (n - 1)).long()) % doors[:, None]  # (B, G) a uniform door other than the pick
    if host == "knowing":
        # The host opens N − 2 goat doors. If the pick is a goat, the only door he can leave shut is the car;
        # if the pick is the car, every other door is a goat and he leaves a random one shut.
        kept = torch.where(pick != car, car, other)
    else:
        kept = other  # an ignorant host opens N − 2 random doors (other than the pick) and may reveal the car
    return car.to(device), pick.to(device), kept.to(device)
