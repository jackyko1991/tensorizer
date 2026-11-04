"""Build the concrete tensors: bit strings, the two text formats as token ids, and M stacked transformers.

Vocabulary (5 tokens): 0, 1, ":", "=", "_" (blank: padding, and the hidden scratchpad cells of experiment c).

    direct   b_1 … b_n = p                      length n + 2
    CoT      b_1 … b_n : s_1 … s_n = p          length 2n + 3,   s_k = b_1 ⊕ … ⊕ b_k,  p = s_n

Every sequence is padded with blanks to one context length T, so direct and CoT models of every n share one
tensor (M, N, T) and train in one batched run. The weights of M independent models are stacked along a leading
model axis m (no nn.Module, no vmap): every product in equation.py is an explicit einsum over that axis.
"""

import math

import torch

from . import equation

VOCAB = ["0", "1", ":", "=", "_"]
ZERO, ONE, COLON, EQ, BLANK = range(5)


def seq_len(n: int, cot: bool) -> int:
    return 2 * n + 3 if cot else n + 2


def make_batch(bits: torch.Tensor, n: torch.Tensor, cot: torch.Tensor, ctx: int):
    """bits (N, n_max) 0/1, n (N,) string lengths, cot (N,) bool format → tokens, targets, loss mask, each (N, T).

    Built by position arithmetic, no loop over rows: position j holds b_{j+1} for j < n, then the format's tokens.
    targets = tokens shifted one left (next-token prediction). The mask selects what is trained: the answer p
    (direct) or the scratchpad s_1 … s_n, "=" and p (CoT).
    """
    n_max = bits.shape[1]
    j = torch.arange(ctx, device=bits.device)[None, :]  # (1, T) position
    nn = n[:, None]  # (N, 1)
    s = equation.running_parity(bits)  # (N, n_max) prefix parities (cells past n are never read)
    p = s.gather(1, (nn - 1).clamp(min=0))  # (N, 1) answer = last running parity
    bit_at = bits.gather(1, j.clamp(max=n_max - 1).expand(len(n), -1))  # (N, T) b_{j+1}
    s_at = s.gather(1, (j - nn - 1).clamp(0, n_max - 1))  # (N, T) s_{j-n} on the scratchpad cells
    blank = torch.full_like(bit_at, BLANK)
    direct = torch.where(j == nn + 1, p, blank)
    direct = torch.where(j == nn, torch.full_like(blank, EQ), direct)
    chain = torch.where(j == 2 * nn + 2, p, blank)
    chain = torch.where(j == 2 * nn + 1, torch.full_like(blank, EQ), chain)
    chain = torch.where((j > nn) & (j <= 2 * nn), s_at, chain)
    chain = torch.where(j == nn, torch.full_like(blank, COLON), chain)
    tokens = torch.where(j < nn, bit_at, torch.where(cot[:, None], chain, direct))
    targets = torch.cat([tokens[:, 1:], torch.full_like(tokens[:, :1], BLANK)], dim=1)
    mask = torch.where(cot[:, None], (j >= nn) & (j <= 2 * nn + 1), j == nn)
    return tokens, targets, mask


def int_to_bits(ints: torch.Tensor, n_max: int) -> torch.Tensor:
    """Integers (...,) → their n_max lowest bits (..., n_max), least significant first."""
    return (ints[..., None] >> torch.arange(n_max, device=ints.device)) & 1


class StringSplit:
    """A fixed train/test split of the 2^n strings of every length n: a quarter (at most 1024) are test-only.

    Training draws never return a test string, so test accuracy measures generalisation, not memory.
    Below n = 8 there are too few strings to hold any out, and the test set is all 2^n strings.
    """

    def __init__(self, lengths: list[int], seed: int, device: torch.device, n_max: int = 0):
        gen = torch.Generator().manual_seed(seed)
        self.n_max = max(lengths + [n_max])
        self.offset = torch.zeros(self.n_max + 2, dtype=torch.long)
        self.offset[1:] = torch.cumsum(2 ** torch.arange(self.n_max + 1), 0)  # table of length n starts here
        is_test = torch.zeros(int(self.offset[-1]), dtype=torch.bool)
        self.test, self.held_out = {}, {}
        for n in range(1, self.n_max + 1):  # a few small tables, one per length
            self.held_out[n] = n in lengths and 2**n >= 256
            if self.held_out[n]:
                pick = torch.randperm(2**n, generator=gen)[: min(1024, 2**n // 4)]
                is_test[self.offset[n] + pick] = True
            else:  # too few strings to hold any out: test on all of them (this measures fitting, not generalising)
                pick = torch.arange(2**n)
            self.test[n] = pick.to(device)
        self.is_test = is_test.to(device)
        self.offset = self.offset.to(device)

    def sample_train(self, n: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        """Random training strings of lengths n (...,) → bits (..., n_max); test strings are drawn again."""
        u = torch.rand(n.shape, generator=gen, device=n.device)
        ints = (u * 2.0**n).long()
        bad = self.is_test[self.offset[n] + ints]
        while bad.any():  # rejection sampling: a few rounds at most (a quarter of the strings are excluded)
            u = torch.rand(n.shape, generator=gen, device=n.device)
            ints = torch.where(bad, (u * 2.0**n).long(), ints)
            bad = self.is_test[self.offset[n] + ints]
        return int_to_bits(ints, self.n_max) * (torch.arange(self.n_max, device=n.device) < n[..., None])

    def sample_test(self, n: int, size: int, gen: torch.Generator) -> torch.Tensor:
        """size test strings of length n (with repeats when there are fewer) → bits (size, n_max)."""
        pool = self.test[n]
        pick = torch.randint(len(pool), (size,), generator=gen, device=pool.device)
        return int_to_bits(pool[pick], self.n_max)


def init_weights(m: int, v: int, d: int, heads: int, layers: int, ctx: int, seed: int, device: torch.device) -> dict:
    """M stacked GPT models (leading axis m). c = channel (d), h = head, k = channel within a head (d / heads).

    tok (M, V, d), pos (M, T, d) learned embeddings;  per layer: g1, b1 (M, d) LayerNorm; Wq, Wk, Wv (M, H, d, d/H)
    + biases (M, H, d/H); Wo (M, H, d/H, d) + bo (M, d); g2, b2 (M, d); W1 (M, d, 4d) + c1; W2 (M, 4d, d) + c2;
    gf, bf (M, d) final LayerNorm; head (M, d, V) + bias (M, V). Each model gets its own random draw.
    """
    gen = torch.Generator().manual_seed(seed)
    dk = d // heads
    s = 1 / math.sqrt(d)

    def rand(*shape, scale):
        return (scale * torch.randn(m, *shape, generator=gen)).to(device).requires_grad_(True)

    def const(value, *shape):
        return torch.full((m, *shape), float(value), device=device, requires_grad=True)

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
    """All trainable tensors, for the optimiser (each has the model axis first)."""
    top = [w["tok"], w["pos"], w["gf"], w["bf"], w["head"], w["bias"]]
    return top + [t for layer in w["layers"] for t in layer.values()]


def select(w: dict, idx: torch.Tensor | list[int]) -> dict:
    """The sub-stack of models idx (detached), e.g. only the CoT models for an evaluation."""
    idx = torch.as_tensor(idx, device=w["tok"].device)
    out = {k: v.detach()[idx] for k, v in w.items() if k != "layers"}
    out["layers"] = [{k: v.detach()[idx] for k, v in layer.items()} for layer in w["layers"]]
    return out


def parameter_count(v: int, d: int, layers: int, ctx: int) -> int:
    """One model: V d (tokens) + T d (positions) + L (12 d² + 13 d) (blocks) + 2 d (final LayerNorm) + d V + V."""
    return v * d + ctx * d + layers * (12 * d * d + 13 * d) + 2 * d + d * v + v
