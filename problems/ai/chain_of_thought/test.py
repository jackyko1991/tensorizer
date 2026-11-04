"""Checks written first (TDD): what any correct chain-of-thought parity experiment must satisfy.

Run with `pytest problems/others/chain_of_thought/test.py`. The data checks are exact: the scratchpad targets are the
running XOR and the final answer is the last running parity. The model checks are exact too: the causal mask hides
the future, attention rows are probability distributions, the batched model holds exactly M copies of the weights
and the M models train independently. Learning checks: both formats memorise a tiny batch, and a trained CoT model
beats chance clearly on held-out strings (slow).
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")
V, D, H, LAYERS = len(tensors.VOCAB), 32, 2, 2


def _bits(n_rows, n, seed=0):
    return torch.randint(2, (n_rows, n), generator=torch.Generator().manual_seed(seed))


def _batch(n_rows, n, cot, seed=0, ctx=None):
    """Tokens, targets and loss mask (n_rows, T) for strings of n bits in one format."""
    ctx = ctx or tensors.seq_len(n, cot=True)
    nn = torch.full((n_rows,), n)
    return tensors.make_batch(_bits(n_rows, n, seed), nn, torch.full((n_rows,), cot), ctx)


def test_running_parity_is_running_xor():
    b = _bits(200, 13, seed=1)
    s = equation.running_parity(b)  # (N, n)
    ref = torch.zeros_like(b)
    acc = torch.zeros(200, dtype=b.dtype)
    for k in range(13):  # slow reference: XOR one bit at a time
        acc = acc ^ b[:, k]
        ref[:, k] = acc
    assert torch.equal(s, ref)
    assert torch.equal(s[:, -1], b.sum(-1) % 2)  # the last running parity is the parity of the string


def test_formats_hold_the_running_xor_and_the_answer():
    n = 7
    b = _bits(50, n, seed=2)
    s = equation.running_parity(b)
    ctx = tensors.seq_len(n, cot=True)
    tok, tgt, mask = tensors.make_batch(b, torch.full((50,), n), torch.ones(50, dtype=torch.bool), ctx)
    assert torch.equal(tok[:, :n], b) and (tok[:, n] == tensors.COLON).all()
    assert torch.equal(tok[:, n + 1 : 2 * n + 1], s)  # scratchpad tokens s_1 … s_n are the running XOR
    assert (tok[:, 2 * n + 1] == tensors.EQ).all()
    assert torch.equal(tok[:, 2 * n + 2], s[:, -1])  # oracle chain: the answer p equals the last running parity
    assert torch.equal(tgt[:, :-1], tok[:, 1:])  # next-token targets
    assert mask.sum(-1).eq(n + 2).all()  # trained on s_1 … s_n, "=" and p
    tok_d, tgt_d, mask_d = tensors.make_batch(b, torch.full((50,), n), torch.zeros(50, dtype=torch.bool), ctx)
    assert torch.equal(tok_d[:, :n], b) and (tok_d[:, n] == tensors.EQ).all()
    assert torch.equal(tok_d[:, n + 1], s[:, -1]) and (tok_d[:, n + 2 :] == tensors.BLANK).all()
    assert mask_d.sum(-1).eq(1).all() and torch.equal(tgt_d[mask_d], s[:, -1])  # only the answer is trained


def test_mixed_lengths_in_one_batch():
    b = _bits(4, 8, seed=3)
    n = torch.tensor([2, 5, 8, 3])
    tok, _, mask = tensors.make_batch(b, n, torch.tensor([True, True, False, False]), ctx=19)
    for i in range(4):  # each row is the single-length format of its own n, padded with blanks
        one, _, m1 = tensors.make_batch(b[i : i + 1, : n[i]], n[i : i + 1], torch.tensor([bool(i < 2)]), ctx=19)
        assert torch.equal(tok[i], one[0]) and torch.equal(mask[i], m1[0])


def test_causal_mask_hides_the_future():
    w = tensors.init_weights(2, V, D, H, LAYERS, ctx=19, seed=0, device=CPU)
    tok, _, _ = _batch(6, 8, cot=True, seed=4)
    x = torch.stack([tok, tok])  # (M, N, T)
    t = 9
    x2 = x.clone()
    x2[..., t + 1 :] = torch.randint(V, x2[..., t + 1 :].shape, generator=torch.Generator().manual_seed(5))
    out1, _ = equation.forward(x, w)
    out2, _ = equation.forward(x2, w)
    assert torch.allclose(out1[..., : t + 1, :], out2[..., : t + 1, :], atol=1e-6)  # positions ≤ t never see > t
    assert not torch.allclose(out1[..., t + 1 :, :], out2[..., t + 1 :, :], atol=1e-3)


def test_attention_rows_sum_to_one_and_are_causal():
    w = tensors.init_weights(2, V, D, H, LAYERS, ctx=19, seed=0, device=CPU)
    tok, _, _ = _batch(5, 8, cot=True)
    _, maps = equation.forward(torch.stack([tok, tok]), w)  # (M, N, layers, H, T, T)
    T = tok.shape[1]
    assert maps.shape == (2, 5, LAYERS, H, T, T)
    assert torch.allclose(maps.sum(-1), torch.ones(2, 5, LAYERS, H, T), atol=1e-6)
    assert maps[..., torch.triu(torch.ones(T, T, dtype=torch.bool), 1)].abs().max() == 0


def test_einsum_attention_equals_scaled_dot_product_attention():
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(2, 5, H, 11, 8, generator=g, dtype=torch.float64) for _ in range(3))
    out, _ = equation.attention(q, k, v)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(out, ref, atol=1e-12)


def test_batched_parameter_count_is_two_single_models():
    for d, layers, ctx in [(D, LAYERS, 19), (64, 2, 35), (48, 3, 11)]:
        single = sum(p.numel() for p in tensors.parameters(tensors.init_weights(1, V, d, H, layers, ctx, 0, CPU)))
        pair = sum(p.numel() for p in tensors.parameters(tensors.init_weights(2, V, d, H, layers, ctx, 0, CPU)))
        assert single == tensors.parameter_count(V, d, layers, ctx)
        assert pair == 2 * single  # direct and CoT: two full models stacked along the model axis


def test_stacked_models_are_independent():
    w = tensors.init_weights(2, V, D, H, LAYERS, ctx=19, seed=0, device=CPU)
    tok, tgt, mask = _batch(6, 8, cot=True)
    x, y, m = (torch.stack([a, a]) for a in (tok, tgt, mask))
    loss = equation.masked_loss(equation.forward(x, w)[0], y, m)  # (M,)
    loss[0].backward()
    assert all(p.grad[1].abs().max() == 0 for p in tensors.parameters(w))  # model 0's loss never touches model 1


def test_both_formats_overfit_a_tiny_batch():
    torch.manual_seed(0)
    n = 4
    w = tensors.init_weights(2, V, D, H, LAYERS, ctx=tensors.seq_len(n, cot=True), seed=0, device=CPU)
    b = _bits(8, n, seed=7)
    nn = torch.full((8,), n)
    direct = tensors.make_batch(b, nn, torch.zeros(8, dtype=torch.bool), tensors.seq_len(n, cot=True))
    cot = tensors.make_batch(b, nn, torch.ones(8, dtype=torch.bool), tensors.seq_len(n, cot=True))
    x, y, m = (torch.stack([a, c]) for a, c in zip(direct, cot))  # model 0 direct, model 1 CoT
    losses = solve.overfit(x, y, m, w, n_steps=200, lr=1e-2)  # (steps, M)
    assert (losses[0] > 0.5).all() and (losses[-1] < 0.05).all()  # both memorised from ≈ ln 5


@pytest.mark.slow
def test_trained_cot_model_beats_chance():
    torch.manual_seed(0)
    n = 6
    cfg = solve.Config(lengths=[n], seeds=1, steps=400, d=D, heads=H, layers=LAYERS, batch=32, lr=3e-3)
    res = solve.run_experiments(cfg, CPU, n_checkpoints=2, length_gen=False)
    acc = res["acc"]  # (lengths, formats, seeds) final greedy accuracy on held-out strings
    assert acc[0, 1, 0] > 0.9  # CoT: far above the 0.5 of guessing
    tf = res["oracle_answer_agrees"]  # with the true chain fed in, the answer copies s_n
    assert tf[0, 0] > 0.95
