r"""Turing machine → tensor translation.

A Turing machine (Turing 1936) is a head on an infinite tape of 0s and 1s, in one of finitely many states q. Each
step it reads the symbol r under the head and looks up its rule δ(q, r) = (w, m, q'): write w, move m = ±1, go to
state q'. Reaching the halt state H stops it. For a batch of B machines b, each with its own table δ_b, one step is

    r_b = T_{b, h_b}                          (read: gather along the tape)
    (w_b, m_b, q'_b) = δ_{b, s_b, r_b}        (look up the rule: advanced indexing)
    T_{b, h_b} ← w_b,   h_b ← h_b + m_b,   s_b ← q'_b      (write: scatter; move; change state)

applied only where machine b is still running (s_b ≠ H). No index is summed: every machine is independent, so one
step of the whole batch is one gather, one look-up and one scatter, whatever B is.
"""

import torch


def read(tape: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    r"""r_b = T_{b, h_b}.   tape (B, L), head (B,) → (B,)"""
    return tape.gather(1, head[:, None])[:, 0]


def rule(table: torch.Tensor, state: torch.Tensor, symbol: torch.Tensor) -> torch.Tensor:
    r"""(w, m, q')_b = δ_{b, s_b, r_b}.   table (B, n, 2, 3), state (B,), symbol (B,) → (B, 3)"""
    b = torch.arange(len(table), device=table.device)
    q = state.clamp(max=table.shape[1] - 1)  # a halted machine (s = H = n) looks up any row; it is masked out
    return table[b, q, symbol.long()]


def step(
    tape: torch.Tensor, head: torch.Tensor, state: torch.Tensor, table: torch.Tensor, active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""One step of every machine where active (B,) is True; the others stay frozen.

    tape (B, L) uint8, head (B,), state (B,), table (B, n, 2, 3) → new (tape, head, state)
    """
    r = read(tape, head)  # (B,)
    w, m, q = rule(table, state, r).unbind(-1)  # each (B,)
    write = torch.where(active, w.to(tape.dtype), r)  # a frozen machine "writes" what is already there
    tape.scatter_(1, head[:, None], write[:, None])  # T_{b, h_b} ← w_b, in place: O(B) work, not O(B·L)
    head = head + torch.where(active, m, 0)  # h_b ← h_b + m_b
    state = torch.where(active, q, state)  # s_b ← q'_b
    return tape, head, state
