"""Checks written first (TDD): what any correct hand-written GPT must satisfy.

Run with `pytest problems/ai/tiny_transformer/test.py`. Each test is an exact property: an untrained model guesses
uniformly (loss ln V), the causal mask hides the future, attention rows are probability distributions, the einsum
attention equals PyTorch's fused kernel, the model can memorise one batch, and the weight count obeys its formula.
"""

import math

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
V, D, H, LAYERS, T = 40, 32, 4, 2, 16


def _model(seed=0):
    return tensors.init_weights(V, D, H, LAYERS, T, seed=seed, device=CPU)


def _tokens(n=6, seed=1):
    return torch.randint(V, (n, T), generator=torch.Generator().manual_seed(seed))


def test_text_round_trip_and_vocabulary():
    text = tensors.load_text()
    assert 50_000 < len(text) < 1_000_000
    chars, data = tensors.encode_text(text, CPU)
    assert tensors.decode(data[:500], chars) == text[:500]
    assert len(chars) == len(set(text))


def test_initial_loss_is_log_vocab():
    w = _model()
    x, y = _tokens(seed=1), _tokens(seed=2)
    logits, _ = equation.forward(x, w)
    loss = equation.cross_entropy(logits, y).item()
    assert loss == pytest.approx(math.log(V), abs=0.05)  # an untrained model knows nothing: uniform over V letters


def test_causal_mask_hides_the_future():
    w = _model()
    x = _tokens()
    t = 7
    x2 = x.clone()
    x2[:, t + 1 :] = torch.randint(V, x2[:, t + 1 :].shape, generator=torch.Generator().manual_seed(5))
    out1, _ = equation.forward(x, w)
    out2, _ = equation.forward(x2, w)
    assert torch.allclose(out1[:, : t + 1], out2[:, : t + 1], atol=1e-6)  # positions ≤ t never see tokens > t
    assert not torch.allclose(out1[:, t + 1 :], out2[:, t + 1 :], atol=1e-3)


def test_attention_rows_sum_to_one_and_are_causal():
    w = _model()
    _, maps = equation.forward(_tokens(), w)  # (N, layers, H, T, T)
    assert maps.shape == (6, LAYERS, H, T, T)
    assert torch.allclose(maps.sum(-1), torch.ones(6, LAYERS, H, T), atol=1e-6)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    assert maps[..., upper].abs().max() == 0  # no weight on future positions


def test_einsum_attention_equals_scaled_dot_product_attention():
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(5, H, T, 8, generator=g, dtype=torch.float64) for _ in range(3))
    out, A = equation.attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-12)
    assert A.shape == (5, H, T, T)


def test_parameter_count_matches_formula():
    for v, d, h, n_layer, ctx in [(V, D, H, LAYERS, T), (65, 64, 4, 2, 64), (10, 48, 3, 3, 20)]:
        w = tensors.init_weights(v, d, h, n_layer, ctx, seed=0, device=CPU)
        counted = sum(p.numel() for p in tensors.parameters(w))
        assert counted == tensors.parameter_count(v, d, n_layer, ctx)  # V d + T d + L(12 d² + 13 d) + 2 d + d V + V


def test_overfits_one_tiny_batch():
    torch.manual_seed(0)
    w = _model()
    x = _tokens(n=4, seed=7)
    y = torch.roll(x, -1, dims=1)
    losses = solve.overfit(x, y, w, n_steps=150, lr=1e-2)
    assert losses[0] > 3.0 and losses[-1] < 0.1  # memorised: loss far below ln V ≈ 3.7
