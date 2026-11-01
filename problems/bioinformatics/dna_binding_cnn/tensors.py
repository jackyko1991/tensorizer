"""Build the concrete tensors: a planted binding motif, random DNA with and without it, and the network weights."""

import torch

BASES = "ACGT"
MOTIFS = {  # consensus binding sites of three well-known transcription factors
    "nfkb": "GGGACTTTCC",  # NF-κB
    "ap1": "TGACTCA",  # AP-1 (Fos/Jun)
    "ebox": "CACGTG",  # E-box (Myc/Max)
}


def motif_pwm(consensus: str, strength: float, device: torch.device) -> torch.Tensor:
    """Position weight matrix π_{kb}, shape (W, 4): the consensus base with probability `strength`, the others share
    the rest. strength = 1 plants the exact word; smaller values give the degenerate sites real factors bind."""
    pwm = torch.full((len(consensus), 4), (1 - strength) / 3)
    for k, base in enumerate(consensus):
        pwm[k, BASES.index(base)] = strength
    return pwm.to(device)


def make_dataset(n_seq: int, length: int, pwm: torch.Tensor, seed: int, device: torch.device):
    """n_seq random sequences (uniform bases); a random half carry one motif instance, drawn from the PWM, at a
    random position. Returns the one-hot sequences x_{nlb} (N, ℓ, 4), the labels y_n (N,) and the motif starts."""
    gen = torch.Generator().manual_seed(seed)
    W = pwm.shape[0]
    seqs = torch.randint(4, (n_seq, length), generator=gen)
    labels = (torch.rand(n_seq, generator=gen) < 0.5).float()
    starts = torch.randint(length - W + 1, (n_seq,), generator=gen)
    site = torch.multinomial(pwm.cpu(), n_seq, replacement=True, generator=gen).T  # (N, W) bases drawn per position
    for n in torch.nonzero(labels).flatten().tolist():
        seqs[n, starts[n] : starts[n] + W] = site[n]
    x = torch.nn.functional.one_hot(seqs, 4).float()
    return x.to(device), labels.to(device), starts.to(device)


def init_weights(n_filters: int, width: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Trainable parameters: filters W_{fkb} (F, w, 4), filter biases c_f (F,), read-out v_f (F,) and v_0 ()."""
    gen = torch.Generator().manual_seed(seed)
    params = {
        "W": 0.1 * torch.randn(n_filters, width, 4, generator=gen),
        "c": torch.zeros(n_filters),
        "v": 0.1 * torch.randn(n_filters, generator=gen),
        "v0": torch.zeros(()),
    }
    return {k: p.to(device).requires_grad_(True) for k, p in params.items()}
