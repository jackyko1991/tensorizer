"""Run the machine for a batch of settings, and break a message by scoring every rotor order and start position."""

import torch

from problems._common.progress import steps

from . import tensors
from .equation import encipher, index_of_coincidence, keypress_matrix, letter_counts, rotor_matrix, step, unigram_score


def positions(start: torch.Tensor, notch: torch.Tensor, n: int) -> torch.Tensor:
    """Window letters at each of n key presses (the rotors step before the current flows): (B, 3) → (B, n, 3)."""
    pos = torch.empty((start.shape[0], n, 3), dtype=torch.long, device=start.device)
    p = start
    for t in range(n):
        p = step(p, notch)
        pos[:, t] = p
    return pos


def all_positions(notch: torch.Tensor, n: int) -> torch.Tensor:
    """Window letters for all 26³ starts (row order of tensors.all_starts): (B, n, 3).

    The left rotor never kicks anything, so the stepping is computed for the 26² (mid, right) starts only and the
    left start is added afterwards: 26 times less work than stepping every start.
    """
    starts = tensors.all_starts(notch.device)  # (B, 3)
    pos = positions(starts[: 26 * 26], notch, n).repeat(26, 1, 1)  # (B, n, 3), left start 0 everywhere
    pos[..., 0] = (pos[..., 0] + starts[:, None, 0]) % 26
    return pos


def run_machine(text: torch.Tensor, m: dict, start: torch.Tensor, rings: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Encrypt (or, the same thing, decrypt) one text with B start positions at once: (T,), (B, 3) → (B, T)."""
    return encipher(text.expand(start.shape[0], -1), (positions(start, m["notch"], len(text)) - rings) % 26, m, s)


def keypress_matrices(m: dict, start: torch.Tensor, rings: torch.Tensor, s: torch.Tensor, n: int) -> torch.Tensor:
    """The 26×26 matrix E_t of each of the first n key presses for one setting: start (3,) → (n, 26, 26)."""
    offsets = (positions(start[None], m["notch"], n)[0] - rings) % 26  # (n, 3)
    r = [
        rotor_matrix(tensors.matrix(m["w"][k]), tensors.shift(-offsets[:, k]), tensors.shift(offsets[:, k]))
        for k in range(3)
    ]
    return keypress_matrix(tensors.matrix(s), r[2], r[1], r[0], tensors.matrix(m["u"]))  # R₁ = right rotor


def score(y: torch.Tensor, method: str, log_p: torch.Tensor) -> torch.Tensor:
    """How English-like every candidate decryption looks: (B, T) → (B,)."""
    c = letter_counts(y)
    return index_of_coincidence(c) if method == "ioc" else unigram_score(c, log_p)


def break_enigma(cipher: torch.Tensor, orders: list, rings: torch.Tensor, method: str) -> dict:
    """Score all 26³ start positions for every rotor order (empty plugboard, known rings).

    Returns scores (O, 26, 26, 26) indexed [order, left, mid, right] and the best order and start.
    """
    dev = cipher.device
    starts = tensors.all_starts(dev)  # (B, 3), B = 17,576
    s, log_p = torch.arange(26, device=dev), tensors.english_log_probs(dev)
    text = cipher.expand(len(starts), -1)  # (B, T): the same ciphertext for every candidate
    scores = torch.empty(len(orders), len(starts), device=dev)
    for i in steps(len(orders), "rotor orders"):
        m = tensors.machine(orders[i], dev)
        offsets = (all_positions(m["notch"], len(cipher)) - rings) % 26  # (B, T, 3)
        scores[i] = score(encipher(text, offsets, m, s), method, log_p)  # decrypt all B candidates, then score
    best = int(scores.argmax())
    order_i, start_i = divmod(best, len(starts))
    return {"scores": scores.reshape(len(orders), 26, 26, 26), "order": order_i, "start": starts[start_i]}
