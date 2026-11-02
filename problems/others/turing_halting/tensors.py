"""Build the concrete tensors: transition tables of whole batches of Turing machines, and blank tapes.

A machine with n states (A, B, C, …) and 2 symbols has a rule for every (state, symbol read) pair: the symbol to
write, the move (−1 left, +1 right) and the next state. States are numbered 0 … n−1 and the halt state H is n.
"""

import torch

# Champions in the bbchallenge notation: per state "write move next" for symbol 0 then 1, H = halt.
CHAMPIONS = {
    "BB(2)": "1RB1LB_1LA1RH",  # S(2) = 6 steps, Σ(2) = 4 ones
    "BB(3) steps": "1RB1RH_1LB0RC_1LC1LA",  # S(3) = 21 steps (leaves 5 ones)
    "BB(3) ones": "1RB1RH_0RC1RB_1LC1LA",  # Σ(3) = 6 ones (in 14 steps)
    "BB(4)": "1RB1LB_1LA0LC_1RH1LD_1RD0RA",  # S(4) = 107 steps, Σ(4) = 13 ones (Brady 1983)
}
LETTERS = "ABCDEFGH"


def parse(text: str) -> torch.Tensor:
    """'1RB1LB_1LA1RH' → transition table δ of shape (n, 2, 3): δ[q, r] = (write, move, next)."""
    states = text.split("_")
    n = len(states)
    rule = lambda s: [int(s[0]), 1 if s[1] == "R" else -1, n if s[2] in "HZ" else LETTERS.index(s[2])]  # noqa: E731
    return torch.tensor([[rule(row[0:3]), rule(row[3:6])] for row in states])


def to_text(table: torch.Tensor) -> str:
    """Inverse of parse, for printing: (n, 2, 3) → '1RB1LB_1LA1RH'."""
    n = table.shape[0]
    cell = lambda w, m, q: f"{w}{'R' if m > 0 else 'L'}{'H' if q == n else LETTERS[q]}"  # noqa: E731
    return "_".join("".join(cell(*table[q, r].tolist()) for r in range(2)) for q in range(n))


def rule_options(n: int, normalise: bool) -> torch.Tensor:
    """Every possible single rule (write, move, next) as rows of a (K, 3) tensor.

    Exhaustive: K = 2 writes × 2 moves × (n + 1) next states. Normalised: a halting rule is always 1RH, because
    what it writes and where it moves changes neither when the machine halts nor (except for one more 1) its score.
    """
    rows = [[w, m, q] for q in range(n) for w in (0, 1) for m in (-1, 1)]
    rows += [[1, 1, n]] if normalise else [[w, m, n] for w in (0, 1) for m in (-1, 1)]
    return torch.tensor(rows)


def all_machines(n: int, normalise: bool, device: torch.device) -> torch.Tensor:
    """Every n-state machine as one batch: (B, n, 2, 3).

    Exhaustive: B = (4(n+1))^{2n}, all 20 736 machines for n = 2. Normalised: the first rule is fixed to
    A0 → 1RB (a machine that starts by writing 1, by left–right symmetry moving right, into a new state; the
    other first moves either halt at once or loop forever) and halting rules are 1RH, so B = (4n+1)^{2n−1},
    371 293 machines for n = 3.
    """
    options = rule_options(n, normalise).to(device)  # (K, 3)
    k, free = len(options), 2 * n - (1 if normalise else 0)
    code = torch.arange(k**free, device=device)  # (B,) machine number written in base K
    digits = torch.stack([(code // k**i) % k for i in range(free)], dim=1)  # (B, free)
    table = options[digits]  # (B, free, 3)
    if normalise:
        first = torch.tensor([1, 1, 1], device=device).expand(len(code), 1, 3)  # A0 → 1RB
        table = torch.cat([first, table], dim=1)
    return table.reshape(-1, n, 2, 3)


def random_machines(n: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    """A random sample of normalised n-state machines (A0 → 1RB, halting rules 1RH): (count, n, 2, 3)."""
    options = rule_options(n, normalise=True)  # (K, 3)
    gen = torch.Generator().manual_seed(seed)
    digits = torch.randint(len(options), (count, 2 * n - 1), generator=gen)
    first = torch.tensor([1, 1, 1]).expand(count, 1, 3)
    return torch.cat([first, options[digits]], dim=1).reshape(count, n, 2, 3).to(device)


def blank(batch: int, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """All-zero tapes T (B, L) uint8, heads h (B,) in the middle cell, states s (B,) = 0 (state A)."""
    tape = torch.zeros(batch, length, dtype=torch.uint8, device=device)
    head = torch.full((batch,), length // 2, dtype=torch.long, device=device)
    state = torch.zeros(batch, dtype=torch.long, device=device)
    return tape, head, state
