"""Coding Skills page, "For loop vs tensor": each loop/tensor pair in scripts/loop_benchmark.py computes the
same result."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import loop_benchmark as lb  # noqa: E402

TINY = {"dot": 50, "pairwise": 7, "matmul": 5, "pendulum": 9, "diffusion": 6, "mc_pi": 200}


def test_every_case_has_a_tiny_size():
    assert set(TINY) == set(lb.CASES)


@pytest.mark.parametrize("case", sorted(TINY))
def test_loop_and_tensor_agree(case):
    gen = torch.Generator().manual_seed(1)
    assert lb.agree(case, lb.make_inputs(case, TINY[case], gen))


def test_mc_pi_is_close_to_pi():
    gen = torch.Generator().manual_seed(2)
    xs, ys = lb.make_inputs("mc_pi", 100_000, gen)
    assert abs(float(lb.mc_pi_tensor(xs, ys)) - torch.pi) < 0.03


def test_diffusion_keeps_boundary_and_smooths():
    u = torch.zeros(5, 5, dtype=torch.float64)
    u[2, 2] = 1.0
    new = lb.diffusion_tensor(u)
    assert new[2, 2] == pytest.approx(0.2) and new[1, 2] == pytest.approx(0.2) and new.sum() == pytest.approx(1.0)
