"""Main loop: random initial conditions → solver data → train the Fourier neural operator → test → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors

N_COARSE = 64  # training resolution
N_FINE = 256  # solver resolution; also the zero-shot super-resolution test grid
N_TEST = 200  # held-out initial conditions
N_IC_MODES = 5  # u_0 is a sum of Fourier modes k = 1 … 5
N_LAYERS = 4
BATCH = 50
SEED = 0


def make_data(n_train: int, nu: float, t_final: float, dev: torch.device) -> dict:
    """Solve Burgers for n_train + N_TEST random initial conditions on the fine grid, then subsample to N_COARSE.

    Coarse tensors are float32 (for the network); the fine test set is kept in float64 for the solver.
    """
    x_fine, dx_fine = tensors.grid(N_FINE, dev)  # (M,)
    coeffs = tensors.random_modes(n_train + N_TEST, N_IC_MODES, SEED, dev)  # (B, K, 2)
    u0_fine = tensors.initial_conditions(coeffs, x_fine)  # (B, M)
    uT_fine = solve.solve_burgers(u0_fine, nu, t_final, dx_fine)  # (B, M)
    s = N_FINE // N_COARSE  # the coarse grid is every 4th fine point
    return {
        "x": x_fine[::s].float(),  # (N,)
        "u0_train": u0_fine[:n_train, ::s].float(),  # (n_train, N)
        "uT_train": uT_fine[:n_train, ::s].float(),
        "u0_test": u0_fine[n_train:, ::s].float(),  # (N_TEST, N)
        "uT_test": uT_fine[n_train:, ::s].float(),
        "x_fine": x_fine,  # (M,) float64
        "dx_fine": dx_fine,
        "u0_test_fine": u0_fine[n_train:],  # (N_TEST, M) float64
        "uT_test_fine": uT_fine[n_train:],
    }


def timed(fn, dev: torch.device, repeats: int = 3) -> float:
    """Best wall-clock time of fn() in seconds."""
    best = float("inf")
    for _ in range(repeats):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            fn()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    nu, t_final, n_train = params["nu"], params["t_final"], params["n_train"]

    t0 = time.perf_counter()
    data = make_data(n_train, nu, t_final, dev)
    t_data = time.perf_counter() - t0
    # Super-resolution panel: the test case with the steepest front at time T, at full resolution.
    slope = data["uT_test_fine"].diff(dim=1).abs().amax(dim=1) / data["dx_fine"]  # max |u_x| per case
    data["fine_idx"] = int(slope.argmax())
    data["x_fine32"] = data["x_fine"].float()
    data["u0_fine"] = data["u0_test_fine"][[data["fine_idx"]]].float()  # (1, M)

    w = tensors.init_fno(params["width"], params["modes"], N_LAYERS, SEED, dev)
    t0 = time.perf_counter()
    hist = solve.train(w, data, params["epochs"], params["lr"], BATCH, SEED)
    t_train = time.perf_counter() - t0

    # Identity baseline "u(T) = u_0", and the zero-shot super-resolution error on the 256-point grid.
    u0_fine32, uT_fine32 = data["u0_test_fine"].float(), data["uT_test_fine"].float()
    with torch.no_grad():
        baseline = equation.relative_l2(data["u0_test"], data["uT_test"]).item()
        fine_err = equation.relative_l2(equation.fno(u0_fine32, data["x_fine32"], w), uT_fine32).item()

    # Timing: the solver vs one FNO forward pass, for all N_TEST test initial conditions at once.
    t_solver = timed(lambda: solve.solve_burgers(data["u0_test_fine"], nu, t_final, data["dx_fine"]), dev)
    t_fno = timed(lambda: equation.fno(data["u0_test"], data["x"], w), dev)
    t_fno_fine = timed(lambda: equation.fno(u0_fine32, data["x_fine32"], w), dev)

    print(f"Burgers u_t + u u_x = ν u_xx, ν = {nu}, T = {t_final}   ·   {n_train} training / {N_TEST} test "
          f"initial conditions, solved on {N_FINE} points in {t_data:.1f} s")  # fmt: skip
    print(f"FNO: width {params['width']}, {params['modes']} Fourier modes, {N_LAYERS} Fourier layers, "
          f"{tensors.count(w):,} real parameters, trained {params['epochs']} epochs in {t_train:.1f} s")  # fmt: skip
    print(f"relative L2 error ({N_COARSE} pts)   train {hist['train_err'][-1]:.2%}   test {hist['test_err'][-1]:.2%}"
          f"   (baseline 'u(T) = u_0': {baseline:.2%})")  # fmt: skip
    print(f"zero-shot super-resolution: the same weights on {N_FINE} points, test error {fine_err:.2%}")
    print(f"time for all {N_TEST} test cases:  solver ({N_FINE} pts) {t_solver * 1e3:.1f} ms   ·   "
          f"FNO ({N_COARSE} pts) {t_fno * 1e3:.1f} ms   ·   FNO ({N_FINE} pts) {t_fno_fine * 1e3:.1f} ms")  # fmt: skip

    return plot.animate(data, hist, baseline, fine_err, out_dir / "neural_operator.gif")


if __name__ == "__main__":
    cli(run, __file__)
