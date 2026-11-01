"""Physics checks, written first (TDD): what any correct DNA-binding CNN code must satisfy.

Run with `pytest problems/bioinformatics/dna_binding_cnn/test.py`. Each test is an exact invariant: the convolution
equals conv1d, information content lies in [0, 2] bits, and the trained filter recovers the planted motif.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_convolution_matches_torch_conv1d():
    gen = torch.Generator().manual_seed(5)
    x = torch.nn.functional.one_hot(torch.randint(4, (3, 30), generator=gen), 4).float()
    W, c = torch.randn(5, 7, 4, generator=gen), torch.randn(5, generator=gen)
    ref = torch.nn.functional.conv1d(x.transpose(1, 2), W.transpose(1, 2), c)  # (N, F, P)
    assert torch.allclose(equation.conv(x, W, c), ref, atol=1e-5)


def test_motif_similarity_and_information():
    pwm = tensors.motif_pwm("GGGACTTTCC", 0.9, CPU)
    r, shift = equation.motif_similarity(pwm, pwm)
    assert r == pytest.approx(1.0) and shift == 0
    padded = torch.cat([torch.full((2, 4), 0.25), pwm, torch.full((3, 4), 0.25)])
    assert equation.motif_similarity(padded, pwm) == (pytest.approx(1.0), 2)
    assert torch.allclose(equation.information(torch.full((3, 4), 0.25)), torch.zeros(3), atol=1e-6)
    assert equation.information(torch.eye(4)).allclose(torch.full((4,), 2.0))


def test_planted_sites_are_in_the_positive_sequences():
    pwm = tensors.motif_pwm("TGACTCA", 1.0, CPU)
    x, y, starts = tensors.make_dataset(200, 40, pwm, 0, CPU)
    seqs = x.argmax(-1)
    word = torch.tensor(["ACGT".index(b) for b in "TGACTCA"])
    for n in torch.nonzero(y).flatten().tolist():
        assert torch.equal(seqs[n, starts[n] : starts[n] + 7], word)
    assert 0.4 < y.mean().item() < 0.6


@pytest.fixture(scope="module")
def trained_cnn():
    motif = tensors.motif_pwm("GGGACTTTCC", 0.95, CPU)
    x, y, _ = tensors.make_dataset(2500, 50, motif, 0, CPU)
    x_tr, y_tr, x_te, y_te = x[500:], y[500:], x[:500], y[:500]
    weights = tensors.init_weights(4, 12, 0, CPU)
    hist = solve.train(weights, x_tr, y_tr, x_te, y_te, 8, 0.01, 100, 0, n_snapshots=5)
    return motif, weights, hist, x_te, y_te


def test_cnn_reaches_high_accuracy(trained_cnn):
    _, weights, hist, x_te, y_te = trained_cnn
    assert hist["test_loss"][-1] < 0.5 * hist["test_loss"][0]
    assert equation.accuracy(equation.forward(x_te, weights), y_te) > 0.9


def test_best_filter_recovers_the_planted_motif(trained_cnn):
    motif, weights, _, x_te, y_te = trained_cnn
    with torch.no_grad():
        m = torch.relu(equation.conv(x_te, weights["W"], weights["c"])).amax(2)
        importance = weights["v"] * (m[y_te == 1].mean(0) - m[y_te == 0].mean(0))
        pwms = equation.filter_pwms(x_te[y_te == 1], weights["W"], weights["c"])
    r, _ = equation.motif_similarity(pwms[int(importance.argmax())], motif)
    assert r > 0.9
