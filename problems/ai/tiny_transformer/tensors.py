"""Build the concrete tensors: the Shakespeare text as integer tokens, training batches and the GPT weights.

The text (data/shakespeare.txt) is Macbeth and Romeo and Juliet from the public-domain Complete Works in Project
Gutenberg eBook #100, with the eBook header and licence removed and curly quotes turned into ASCII.
Every distinct character is one token (V of them). The weights are plain tensors in a dict (no nn.Module,
no nn.Transformer) so every product in equation.py is an explicit einsum.
"""

import math
from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "shakespeare.txt"


def load_text() -> str:
    return DATA.read_text(encoding="ascii")


def encode_text(text: str, device: torch.device) -> tuple[list[str], torch.Tensor]:
    """The sorted character vocabulary (V,) and the whole text as token ids (len(text),) int64."""
    chars = sorted(set(text))
    index = {ch: i for i, ch in enumerate(chars)}
    return chars, torch.tensor([index[ch] for ch in text], dtype=torch.long, device=device)


def decode(tokens: torch.Tensor, chars: list[str]) -> str:
    return "".join(chars[i] for i in tokens.tolist())


def batch(data: torch.Tensor, n: int, ctx: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n random windows of the text: inputs x (n, T) and targets y (n, T), y = x shifted one character left."""
    start = torch.randint(len(data) - ctx - 1, (n,), generator=gen, device=data.device)
    idx = start[:, None] + torch.arange(ctx, device=data.device)  # (n, T) positions in the text
    return data[idx], data[idx + 1]


def init_weights(v: int, d: int, heads: int, layers: int, ctx: int, seed: int, device: torch.device) -> dict:
    """GPT weights. c = model channel (d), h = head, k = channel within a head (d / heads).

    tok (V, d) token embedding;  pos (T, d) learned position embedding
    per layer:  g1, b1 (d,) LayerNorm;  Wq, Wk, Wv (H, d, d/H) + biases (H, d/H);  Wo (H, d/H, d) + bo (d,)
                g2, b2 (d,) LayerNorm;  W1 (d, 4d) + c1 (4d,);  W2 (4d, d) + c2 (d,)
    gf, bf (d,) final LayerNorm;  head (d, V) + bias (V,) read-out to next-character logits
    The read-out starts tiny, so every logit is ≈ 0 and the first loss is ln V.
    """
    gen = torch.Generator().manual_seed(seed)
    dk = d // heads
    s = 1 / math.sqrt(d)

    def rand(*shape, scale):
        return (scale * torch.randn(*shape, generator=gen)).to(device).requires_grad_(True)

    def const(value, *shape):
        return torch.full(shape, float(value), device=device, requires_grad=True)

    w = {"tok": rand(v, d, scale=0.3), "pos": rand(ctx, d, scale=0.3)}
    w["layers"] = [
        {
            "g1": const(1, d),
            "b1": const(0, d),
            "Wq": rand(heads, d, dk, scale=s),
            "bq": const(0, heads, dk),
            "Wk": rand(heads, d, dk, scale=s),
            "bk": const(0, heads, dk),
            "Wv": rand(heads, d, dk, scale=s),
            "bv": const(0, heads, dk),
            "Wo": rand(heads, dk, d, scale=s / math.sqrt(2 * layers)),
            "bo": const(0, d),
            "g2": const(1, d),
            "b2": const(0, d),
            "W1": rand(d, 4 * d, scale=s),
            "c1": const(0, 4 * d),
            "W2": rand(4 * d, d, scale=0.5 * s / math.sqrt(2 * layers)),
            "c2": const(0, d),
        }  # fmt: skip
        for _ in range(layers)
    ]
    w.update({"gf": const(1, d), "bf": const(0, d), "head": rand(d, v, scale=1e-3), "bias": const(0, v)})
    return w


def parameters(w: dict) -> list[torch.Tensor]:
    """All trainable tensors, for the optimiser."""
    top = [w["tok"], w["pos"], w["gf"], w["bf"], w["head"], w["bias"]]
    return top + [t for layer in w["layers"] for t in layer.values()]


def parameter_count(v: int, d: int, layers: int, ctx: int) -> int:
    """V d (tokens) + T d (positions) + L (12 d² + 13 d) (blocks) + 2 d (final LayerNorm) + d V + V (read-out).

    Per block: Q, K, V, O are 4 d² weights + 4 d biases; the MLP d → 4d → d is 8 d² + 5 d; two LayerNorms 4 d.
    """
    return v * d + ctx * d + layers * (12 * d * d + 13 * d) + 2 * d + d * v + v
