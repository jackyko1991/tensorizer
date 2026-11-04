r"""The game → tensor translation.

There are N doors, one car and N − 1 goats. The contestant picks door p; the host, who knows where the car c is,
opens N − 2 of the other doors, all goats, and leaves one other door k shut. Stay or switch?

    \text{stay wins} = [p = c],\qquad \text{switch wins} = [k = c].

Since the host must leave the car shut whenever p ≠ c, switching wins exactly when the first pick was wrong:

    P(\text{stay}) = \frac1N,\qquad P(\text{switch}) = \frac{N-1}{N}      (1/3 and 2/3 for N = 3).

For G games the measured rate \hat p_n after n games is a prefix mean of 0/1 outcomes, a binomial proportion with
standard error \sqrt{p(1-p)/n}; the 2σ band holds the curve about 95 % of the time.

An ignorant host opens N − 2 random doors; the games in which he reveals the car are void (valid = 0). Given that
the car is still behind a shut door, stay and switch are equally likely to win: 1/2 each.
"""

import torch


def outcomes(
    car: torch.Tensor, pick: torch.Tensor, kept: torch.Tensor, host: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stay wins [p = c], switch wins [k = c] and valid games (car not revealed), all bool of shape (B, G)."""
    stay = pick == car
    switch = kept == car
    valid = stay | switch  # the car is behind one of the two shut doors (always true for a knowing host)
    return stay, switch, valid


def opened_doors(pick: torch.Tensor, kept: torch.Tensor, n_doors: int) -> torch.Tensor:
    """Door mask O_{bgd} = [d ≠ p_{bg}][d ≠ k_{bg}]: the N − 2 doors the host opens.   (B, G) → (B, G, N)"""
    d = torch.arange(n_doors, device=pick.device)  # (N,)
    return (d != pick[..., None]) & (d != kept[..., None])


def running_rate(wins: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    r"""\hat p_{bn} = \sum_{g \le n} w_{bg} / \sum_{g \le n} v_{bg}: win rate after every game.   (B, G) → (B, G)"""
    valid = torch.ones_like(wins) if valid is None else valid
    hits = torch.cumsum((wins & valid).double(), dim=-1)
    played = torch.cumsum(valid.double(), dim=-1)
    return hits / played.clamp(min=1)


def binomial_band(p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""2σ = 2\sqrt{p(1-p)/n}: the half-width of the 95 % band around the exact rate p after n games."""
    return 2 * torch.sqrt(p * (1 - p) / n)


def valid_fraction(doors: torch.Tensor, host: str) -> torch.Tensor:
    """Share of games that count: 1 for a knowing host, 2/N for an ignorant one (car still shut).   (B,) → (B,)"""
    return torch.ones(len(doors), dtype=torch.float64) if host == "knowing" else 2 / doors.double()


def exact_rates(doors: torch.Tensor, host: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Theory: (1/N, (N − 1)/N) for a knowing host, (1/2, 1/2) for an ignorant one.   (B,) → (B,), (B,)"""
    n = doors.double()
    if host == "knowing":
        return 1 / n, (n - 1) / n
    return torch.full_like(n, 0.5), torch.full_like(n, 0.5)
