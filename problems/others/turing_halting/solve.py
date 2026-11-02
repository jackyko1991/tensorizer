"""Run a whole batch of Turing machines for a fixed step budget and record when (and whether) each one halts."""

import torch

from problems._common.progress import steps

from .equation import step
from .tensors import blank


def run(table: torch.Tensor, budget: int, length: int | None = None) -> dict[str, torch.Tensor]:
    """Simulate every machine of table (B, n, 2, 3) for at most `budget` steps on a blank tape.

    The tape has L = 2·budget + 1 cells by default with the head in the middle, so no machine can reach its end;
    with a shorter tape a machine that would step off it is frozen and counted as undecided, like one still running.
    Returns halt_time (B,) (the step on which it entered H, −1 if undecided), ones (B,) (1s left on the tape),
    off_tape (B,) and the final tape (B, L).
    """
    n = table.shape[1]
    length = length or 2 * budget + 1
    tape, head, state = blank(len(table), length, table.device)
    halt_time = torch.full_like(state, -1)
    off = torch.zeros_like(state, dtype=torch.bool)
    for t in steps(budget, "steps"):
        active = (state < n) & ~off  # still running and still on the tape
        if not active.any():
            break
        tape, head, state = step(tape, head, state, table, active)
        halt_time = torch.where(active & (state == n), t + 1, halt_time)  # entered H on step t + 1
        off |= (head < 0) | (head >= length)
        head = head.clamp(0, length - 1)  # an off-tape machine is frozen at the edge
    ones = tape.sum(dim=1, dtype=torch.long)
    return {"halt_time": halt_time, "ones": ones, "off_tape": off & (halt_time < 0), "tape": tape}


def halted_fraction(halt_time: torch.Tensor, budget: int) -> torch.Tensor:
    """f(t) = fraction of machines that halted within t steps, t = 0 … budget: (budget + 1,)."""
    t = torch.arange(budget + 1, device=halt_time.device)
    halted = (halt_time[None, :] >= 0) & (halt_time[None, :] <= t[:, None])  # (budget + 1, B)
    return halted.double().mean(dim=1)


def champions(halt_time: torch.Tensor, ones: torch.Tensor) -> dict[int, int]:
    """For every step count s at which some machine halts: the index of the machine that halts at s with the most
    1s. The champion within a budget t is then the entry with the largest s ≤ t."""
    out = {}
    for s in halt_time[halt_time >= 0].unique().tolist():
        idx = torch.nonzero(halt_time == s)[:, 0]
        out[int(s)] = int(idx[ones[idx].argmax()])
    return out


def space_time(table: torch.Tensor, n_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Tape and head position after every step of one machine (n, 2, 3): rows (n_steps + 1, L), heads (n_steps + 1,).
    The tape is L = 2·n_steps + 1 cells with the head starting in the middle."""
    n, length = table.shape[0], 2 * n_steps + 1
    tape, head, state = blank(1, length, table.device)
    rows, heads = [tape[0].clone()], [int(head[0])]
    for _ in range(n_steps):
        tape, head, state = step(tape, head, state, table[None], state < n)
        rows.append(tape[0].clone())
        heads.append(int(head[0]))
    return torch.stack(rows), torch.tensor(heads)
