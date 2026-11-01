"""Physics checks, written first (TDD): what any correct T1/T2 relaxation code must satisfy.

Run with `pytest problems/medical/mri_relaxation/test.py`. Each test is a physical law: magnetisation follows the
Bloch relaxation exponentials, inversion recovery nulls at T1 ln 2, and the spin-echo signal gives the textbook
T1/T2 contrast.
"""

import math

import pytest
import torch

from . import equation, tensors

CPU = torch.device("cpu")


def test_relaxation_curves_match_exponentials():
    T1, T2, PD = tensors.tissue_parameters(CPU)
    t = torch.linspace(0, 3000, 31)
    Mz = equation.longitudinal(t, T1, PD)
    Mxy = equation.transverse(t, T2, PD)
    for s in range(len(T1)):
        for k in (0, 7, 30):
            tk = t[k].item()
            assert Mz[s, k].item() == pytest.approx(
                PD[s].item() * (1 - math.exp(-tk / T1[s].item())), rel=1e-5, abs=1e-7
            )
            assert Mxy[s, k].item() == pytest.approx(PD[s].item() * math.exp(-tk / T2[s].item()), rel=1e-5, abs=1e-7)
    # after one T1 the magnetisation has recovered 1 - 1/e of the way
    one_T1 = equation.longitudinal(T1[:1], T1[:1], torch.ones(1))
    assert one_T1.item() == pytest.approx(1 - math.exp(-1), rel=1e-6)


def test_inversion_recovery_null_at_T1_ln2():
    T1, _, PD = tensors.tissue_parameters(CPU)
    TI = equation.null_point(T1)
    Mz = equation.inversion_recovery(TI, T1, PD)  # (S, S): row s evaluated at every tissue's TI
    assert torch.diagonal(Mz).abs().max() < 1e-5
    assert equation.inversion_recovery(torch.zeros(1), T1, PD)[:, 0].tolist() == pytest.approx((-PD).tolist())


def test_weighting_contrast():
    T1, T2, PD = tensors.tissue_parameters(CPU)
    names = tensors.TISSUES
    wm, csf = names.index("white matter"), names.index("CSF")
    TR = torch.tensor([500.0, 4500.0])
    TE = torch.tensor([15.0, 100.0])
    S = equation.spin_echo_signal(PD, T1, T2, TR, TE)  # (S, 2, 2)
    t1w, t2w = S[:, 0, 0], S[:, 1, 1]
    assert t1w[wm] > t1w[csf]  # T1-weighted: white matter brighter than CSF
    assert int(t2w.argmax()) == csf  # T2-weighted: CSF brightest
    # the einsum equals the product written out
    expected = PD * (1 - torch.exp(-TR[1] / T1)) * torch.exp(-TE[1] / T2)
    assert torch.allclose(t2w, expected)


def test_phantom_image_is_label_contraction():
    T1, T2, PD = tensors.tissue_parameters(CPU)
    L = tensors.phantom(32, CPU)
    assert torch.all(L.sum(-1) <= 1) and L.sum() > 0
    S = equation.spin_echo_signal(PD, T1, T2, torch.tensor([500.0]), torch.tensor([15.0]))
    img = equation.image(L, S)[..., 0, 0]
    label = L.argmax(-1)
    inside = L.sum(-1) > 0
    assert torch.allclose(img[inside], S[label[inside], 0, 0])
    assert torch.all(img[~inside] == 0)
