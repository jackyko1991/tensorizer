"""Build the concrete tensors: rotor wirings, the reflector, the plugboard, the messages and every start position.

Letters are indices 0–25 (A = 0). A wiring is stored as a permutation vector w of shape (26,): contact x is wired
to contact w[x]. The same permutation as a 26×26 matrix has M[w[x], x] = 1, so M @ e_x = e_{w[x]} for a one-hot e_x.
"""

import itertools
import string

import torch

from .equation import rotor_table

ALPHABET = string.ascii_uppercase

# Historical Enigma I wirings (entry contact A..Z → exit contact) and the window letter at which each rotor
# kicks its left neighbour on the next key press (the notch).
ROTORS = {
    "I": ("EKMFLGDQVZNTOWYHXUSPAIBRCJ", "Q"),
    "II": ("AJDKSIRUXBLHWTMCQGZNPYFVOE", "E"),
    "III": ("BDFHJLCPRTXVZNYEIWGAKMUSQO", "V"),
    "IV": ("ESOVPZJAYQUIRHXLNFTGKDCMWB", "J"),
    "V": ("VZBRGITYUPSDNHLXAWMJQOFECK", "Z"),
}
REFLECTOR_B = "YRUHQSLDPXNGOKMIEBFZCWVJAT"

# English letter frequencies in percent, A..Z (standard tables from large text corpora).
ENGLISH_FREQ = [
    8.167, 1.492, 2.782, 4.253, 12.702, 2.228, 2.015, 6.094, 6.966, 0.153, 0.772, 4.025, 2.406,
    6.749, 7.507, 1.929, 0.095, 5.987, 6.327, 9.056, 2.758, 0.978, 2.360, 0.150, 1.974, 0.074,
]  # fmt: skip

# Self-written plaintexts (no copyright); spaces and punctuation are dropped before encryption.
MESSAGES = {
    "weather": (
        "Weather report for the northern sector at six hundred hours. Visibility is good along the coast with "
        "light winds from the west and a calm sea. Clouds will build over the hills during the afternoon and rain "
        "is expected after dark. Supply ships should leave the harbour before noon and keep close to the shore "
        "until the storm has passed. Report any change in the barometer to headquarters at once."
    ),
    "lighthouse": (
        "The old lighthouse keeper climbed the spiral stairs every evening to light the lamp. From the top he "
        "could see the fishing boats returning with the tide and the gulls circling above the rocks. In winter "
        "the waves were so high that spray reached the windows of the lantern room, yet the light never failed. "
        "Sailors said that as long as the beam swept across the bay, nobody would be lost at sea."
    ),
    "permutations": (
        "A permutation is a square matrix with exactly one entry equal to one in every row and every column. "
        "Multiplying such matrices composes the permutations, and the transpose undoes them. The Enigma machine "
        "multiplied eight of these matrices for every key that was pressed, and because the rotors moved after "
        "each letter the product changed all the time. Breaking it means searching over the starting positions "
        "until the plaintext looks like English again."
    ),
}


def to_index(text: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """Keep only the letters of text and turn them into indices: (T,) long."""
    return torch.tensor([ALPHABET.index(c) for c in text.upper() if c in ALPHABET], dtype=torch.long, device=device)


def to_text(x: torch.Tensor) -> str:
    """Indices (T,) → string."""
    return "".join(ALPHABET[i] for i in x.tolist())


def perm(letters: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """A wiring string such as 'EKMF…' as a permutation vector w: (26,) long."""
    return to_index(letters, device)


def inverse(w: torch.Tensor) -> torch.Tensor:
    """w⁻¹ with w⁻¹[w[x]] = x: (26,) → (26,)."""
    inv = torch.empty_like(w)
    inv[w] = torch.arange(26, device=w.device)
    return inv


def matrix(w: torch.Tensor) -> torch.Tensor:
    """Permutation matrix M_{ij} = [i = w[j]], so that M e_x = e_{w[x]}: (..., 26) → (..., 26, 26)."""
    return torch.nn.functional.one_hot(w, 26).transpose(-1, -2).float()


def shift(o: torch.Tensor) -> torch.Tensor:
    """Cyclic shift ρ^o as a matrix (x → x + o): o (...,) → (..., 26, 26)."""
    x = torch.arange(26, device=o.device)
    return matrix((x + o[..., None]) % 26)


def plugboard(pairs: str, device: torch.device | str = "cpu") -> torch.Tensor:
    """Steckerbrett S as a permutation vector: 'AB CD' swaps A↔B and C↔D, every other letter is unplugged: (26,)."""
    s = torch.arange(26, device=device)
    for a, b in pairs.upper().split():
        i, j = ALPHABET.index(a), ALPHABET.index(b)
        s[i], s[j] = j, i
    return s


def machine(order: tuple[str, ...], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Wirings W, W⁻¹ (3, 26), their tables (3, 26, 26), notches (3,) and reflector U (26,) for (left, mid, right)."""
    w = torch.stack([perm(ROTORS[name][0], device) for name in order])  # (3, 26)
    w_inv = torch.stack([inverse(r) for r in w])  # (3, 26)
    return {
        "w": w,
        "w_inv": w_inv,
        "f": torch.stack([rotor_table(r) for r in w]),  # (3, 26, 26): F[k, o, x]
        "f_inv": torch.stack([rotor_table(r) for r in w_inv]),
        "notch": to_index("".join(ROTORS[name][1] for name in order), device),
        "u": perm(REFLECTOR_B, device),
    }


def rotor_orders(rotor_set: str) -> list[tuple[str, str, str]]:
    """Every ordered choice of 3 different rotors: 6 orders from I–III, 60 from I–V."""
    names = ["I", "II", "III"] if rotor_set == "I-III" else list(ROTORS)
    return list(itertools.permutations(names, 3))


def all_starts(device: torch.device | str = "cpu") -> torch.Tensor:
    """All 26³ = 17,576 window settings (left, mid, right), row b = 676·left + 26·mid + right: (B, 3)."""
    a = torch.arange(26, device=device)
    return torch.cartesian_prod(a, a, a)


def english_log_probs(device: torch.device | str = "cpu") -> torch.Tensor:
    """log p_k of the English unigram distribution: (26,)."""
    p = torch.tensor(ENGLISH_FREQ, device=device)
    return torch.log(p / p.sum())
