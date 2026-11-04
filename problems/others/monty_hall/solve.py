"""Play every game, then follow the stay and switch win rates as the games accumulate."""

import torch

from . import tensors
from .equation import outcomes, running_rate


def play(doors: torch.Tensor, n_games: int, host: str, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Running stay/switch rates (B, G) and the fraction of valid games (B,) for every number of doors."""
    car, pick, kept = tensors.games(doors, n_games, host, seed, device)
    stay, switch, valid = outcomes(car, pick, kept, host)
    return {
        "stay": running_rate(stay, valid),
        "switch": running_rate(switch, valid),
        "valid": valid.double().mean(dim=-1),
        "picks": pick,
        "switch_wins": switch,
        "valid_games": valid,
    }


def final_rates(doors: torch.Tensor, n_games: int, host: str, seed: int, device: torch.device) -> torch.Tensor:
    """Win rates after all games: row 0 stay, row 1 switch, shape (2, B)."""
    res = play(doors, n_games, host, seed, device)
    return torch.stack([res["stay"][:, -1], res["switch"][:, -1]])


def rate_by_first_pick(picks: torch.Tensor, wins: torch.Tensor, valid: torch.Tensor, n_doors: int) -> torch.Tensor:
    """Switch win rate split by the first pick, over the valid games of one row: (G,) ×3 → (n_doors,). Flat."""
    onehot = torch.nn.functional.one_hot(picks, n_doors).double() * valid.double()[:, None]  # (G, N)
    return torch.einsum("gn,g->n", onehot, wins.double()) / onehot.sum(0)  # repeated g ⇒ summed
