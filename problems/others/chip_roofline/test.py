"""Checks, written first (TDD): the roofline formulas, and that the measurements are sane on any device.

Run with `pytest problems/others/chip_roofline/test.py`.
"""

import json

import pytest
import torch

from . import equation, solve, tensors


def test_matmul_counts():
    assert equation.matmul_flops(4) == 128  # 4³ multiply-adds, 2 FLOP each
    assert equation.matmul_bytes(4, 4) == 3 * 16 * 4
    assert equation.intensity(3000, 2) == pytest.approx(1000)  # 2n / (3b)


def test_roofline_is_the_smaller_of_compute_and_memory():
    peak, bw = 100e12, 1e12
    assert equation.ridge(peak, bw) == 100
    assert equation.roofline(peak, bw, 10) == 10e12  # memory-bound
    assert equation.roofline(peak, bw, 1000) == peak  # compute-bound


def test_copy_counts_read_and_write():
    assert equation.copy_bytes(1000, 4) == 8000


def test_mps_has_no_float64_and_tf32_is_nvidia_only():
    assert tensors.supported("fp64", torch.device("mps")) == (False, "Apple's GPU backend (MPS) has no float64")
    assert not tensors.supported("tf32", torch.device("cpu"))[0]
    assert tensors.supported("bf16", torch.device("cpu"))[0]  # measured (fast with AMX/SME, flagged slow otherwise)
    assert tensors.supported("fp32", torch.device("cpu"))[0]


def test_measurements_on_cpu_are_positive_and_the_mode_is_restored():
    dev = torch.device("cpu")
    before = torch.get_float32_matmul_precision()
    m = solve.measure_matmul("fp32", 128, dev, repeats=2)
    assert m["tflops"] > 0 and m["seconds"] > 0
    assert torch.get_float32_matmul_precision() == before
    bw = solve.measure_bandwidth(4, dev, repeats=2)
    assert bw["gbs"] > 0.1


def test_result_line_is_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TENSORIZER_FAST", "1")
    monkeypatch.setenv("TENSORIZER_DEVICE", "cpu")
    from . import main

    out = main.run({"matrix_size": 256, "copy_mb": 8, "repeats": 2}, tmp_path)
    assert out.exists()
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("CHIP_RESULT ")][-1]
    result = json.loads(line[len("CHIP_RESULT ") :])
    assert result["deviceType"] == "cpu" and result["bandwidth"]["gbs"] > 0
    assert {m["precision"] for m in result["matmul"]} == set(tensors.PRECISIONS)


def test_slow_cpu_formats_are_probed_not_run_at_full_size():
    m = solve.measure_matmul("bf16", 2048, torch.device("cpu"), repeats=2)
    assert m["n"] in (256, 2048)
    if m["n"] == 256:
        assert "slow" in m
