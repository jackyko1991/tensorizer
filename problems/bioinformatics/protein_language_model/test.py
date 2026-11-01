"""Physics checks, written first (TDD): what any correct protein-language-model code must satisfy.

Run with `pytest problems/bioinformatics/protein_language_model/test.py`. Each test is an exact invariant: attention
rows are probability distributions, contacts join opposite lattice colours, and the masked language model beats the
independent-site baselines and finds contacts.
"""

import pytest
import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def _family(n_seq=1500, sweeps=60, coupling=1.2):
    coords = tensors.lattice_fold(5, 6, CPU)
    contacts = tensors.contact_map(coords)
    h, J = tensors.planted_potts(contacts, 8, coupling, 0.5, seed=0)
    seqs = solve.gibbs_sample(h, J, tensors.parity(coords), n_seq, sweeps, seed=0)
    return contacts, seqs


def test_lattice_fold_contacts_and_parity():
    coords = tensors.lattice_fold(5, 6, CPU)
    steps = (coords[1:] - coords[:-1]).abs().sum(-1)
    assert torch.all(steps == 1)  # a connected chain
    contacts = tensors.contact_map(coords)
    assert torch.equal(contacts, contacts.T)
    colour = tensors.parity(coords)
    i, j = torch.nonzero(contacts, as_tuple=True)
    assert torch.all(colour[i] != colour[j])  # contacts join opposite colours: checkerboard Gibbs is valid


def test_attention_rows_sum_to_one():
    w = tensors.init_weights(30, 8, 32, 4, 2, seed=0, device=CPU)
    tokens = torch.randint(9, (5, 30), generator=torch.Generator().manual_seed(0))
    logits, maps = equation.forward(tokens, w)
    assert logits.shape == (5, 30, 8) and maps.shape == (5, 2, 4, 30, 30)
    assert torch.allclose(maps.sum(-1), torch.ones(5, 2, 4, 30), atol=1e-5)
    assert torch.all(maps >= 0)


def test_apc_rows_and_columns_average_out():
    F = torch.rand(12, 12, generator=torch.Generator().manual_seed(3))
    F = F + F.T
    G = equation.apc(F)
    assert torch.allclose(G, G.T, atol=1e-6)
    assert G.mean().abs() < 1e-6  # the overall background is removed


@pytest.mark.slow  # long training/sampling run: CI only, skipped by the in-app TDD check
def test_masked_lm_beats_baselines_and_attention_finds_contacts():
    torch.manual_seed(0)
    contacts, seqs = _family(n_seq=2300)
    train, test = seqs[:2000], seqs[2000:]
    ce_uni, ce_prof = solve.baselines(train, test, 8)
    assert ce_prof < ce_uni
    w = tensors.init_weights(30, 8, 32, 4, 2, seed=0, device=CPU)
    hist = solve.train(train, test, contacts, w, 8, 500, 128, 3e-3, 0.15, n_snapshots=5)
    assert hist["test"][-1] < hist["test"][0]
    assert hist["test"][-1] < ce_uni  # below the unigram baseline
    assert hist["test"][-1] < ce_prof  # and even below the per-site profile: the model uses context
    rand = solve.random_precision(contacts)
    assert hist["precision"][-1] > 5 * rand  # contacts from attention far above chance
