"""Coding Skills page, "BLAS": scripts/blas_benchmark.py counts FLOPs right and its configurations agree."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import blas_benchmark as bb  # noqa: E402


def test_flop_count_is_2n_cubed():
    assert bb.flops(1) == 2 and bb.flops(64) == 2 * 64**3
    # a 2 x 2 product: 4 outputs, each 2 multiplies + 1 add = 12 operations; the convention counts 2n^3 = 16
    assert bb.flops(2) == 16
    assert bb.gflops(1000, 2.0) == pytest.approx(1.0)  # 2e9 operations in 2 s


def test_cpu_configurations_cover_dtypes_and_thread_counts():
    ids = [c["id"] for c in bb.configs(cuda=False, all_threads=4)]
    assert ids == ["cpu_float32_1t", "cpu_float32_all", "cpu_float64_1t", "cpu_float64_all"]
    assert all(c["device"] == "cpu" for c in bb.configs(cuda=False, all_threads=4))
    assert len(bb.configs(cuda=True, all_threads=4)) == 6


@pytest.mark.parametrize("cfg", bb.configs(cuda=False, all_threads=2), ids=lambda c: c["id"])
def test_cpu_configurations_agree_with_float64_reference(cfg):
    gen = torch.Generator().manual_seed(3)
    A, B = bb.make_inputs(17, gen)
    ok, err = bb.agree(cfg, A, B)
    assert ok, err
    assert err < (1e-5 if cfg["dtype"] == "float32" else 1e-12)


def test_gemm_matches_index_formula():
    gen = torch.Generator().manual_seed(4)
    A, B = bb.make_inputs(5, gen)
    C = bb.gemm(A, B)
    want = torch.tensor([[sum(A[i, k] * B[k, j] for k in range(5)) for j in range(5)] for i in range(5)])
    torch.testing.assert_close(C, want.to(C.dtype), rtol=1e-12, atol=1e-12)


def test_thread_count_is_restored():
    before = torch.get_num_threads()
    bb.run({"device": "cpu", "dtype": "float32", "threads": 1}, *bb.make_inputs(8, torch.Generator().manual_seed(5)))
    assert torch.get_num_threads() == before


def test_time_config_returns_positive_seconds():
    s = bb.time_config({"id": "t", "device": "cpu", "dtype": "float32", "threads": 1}, 16, torch.Generator())
    assert 0 < s < 1


def test_cpu_peak_needs_every_factor():
    assert bb.cpu_peak({"cores": None, "max_mhz": 3000}, "AVX2") is None
    assert bb.cpu_peak({"cores": 4, "max_mhz": 3000}, "DEFAULT") is None
    p = bb.cpu_peak({"cores": 4, "max_mhz": 3000}, "AVX2")
    assert p["float32"]["1t"] == pytest.approx(2 * 8 * 2 * 3.0)  # 96 GFLOP/s per core
    assert p["float32"]["all"] == pytest.approx(4 * 96) and p["float64"]["all"] == pytest.approx(2 * 96)
