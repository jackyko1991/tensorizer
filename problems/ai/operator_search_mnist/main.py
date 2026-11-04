"""Main loop: for every training-set size n and seed, search a tensor DAG (DARTS supernet → discrete DAG → equation)
and train a CNN on the same n images per class; compare accuracy, data and time; write the tables and the animation."""

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve
from . import tensors as T

TARGETS = (0.7, 0.8)  # accuracies for "images per class needed"


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    torch.manual_seed(0)
    data = T.load(dev)
    sizes = (1, 10) if fast else T.SIZES
    seeds = 1 if fast else params["seeds"]
    cfg = {"search_steps": 15 if fast else params["search_steps"],
           "retrain_steps": 10 if fast else params["retrain_steps"],
           "cnn_steps": 20 if fast else params["cnn_steps"], "batch": params["batch"]}  # fmt: skip
    show = params["show_n"] if params["show_n"] in sizes else sizes[-1]
    print(f"MNIST subset: pool {T.N_POOL} images (100 per digit), test {len(data['y_test'])} (200 per digit), "
          f"14×14 pixels; device {dev}; sizes n = {list(sizes)} per class, {seeds} seed(s)")  # fmt: skip
    print(f"CNN: {T.cnn_param_count()} weights; supernet: 6 cell edges × 12 operators, pool 2, reduce 7, head 2; "
          f"search {cfg['search_steps']} steps, fine-tune {cfg['retrain_steps']}, CNN ≤ {cfg['cnn_steps']} steps, "
          f"mini-batch ≤ {cfg['batch']}")  # fmt: skip

    tr, va = T.small_set(data["y_pool"], 1, 0)  # warm-up (first-call overheads), so no method pays them in its timing
    warm = {"search_steps": 2, "retrain_steps": 2, "cnn_steps": 2, "batch": cfg["batch"]}
    solve.run_cnn(data, tr, va, warm, 0), solve.run_tensor(data, tr, va, warm, 0)
    rows, showcase, t_all = [], None, time.perf_counter()
    for n in sizes:
        for seed in range(seeds):
            tr, va = T.small_set(data["y_pool"], n, seed)
            c = solve.run_cnn(data, tr, va, cfg, seed)
            d = solve.run_tensor(data, tr, va, cfg, seed)
            b = solve.baselines(data, tr, va)
            a = d["arch"]
            arch_txt = " ".join(f"{e + 1}:{op}" for e, op in a["edges"]) + f" | {a['pool']} {a['reduce']} {a['head']}"
            rows += [
                {"n": n, "seed": seed, "method": "cnn", "acc": c["acc"], "seconds": c["seconds"], "params": c["params"],
                 "constants": 0, "arch": "conv8-pool-conv16-pool-linear"},
                {"n": n, "seed": seed, "method": "tensor_dag", "acc": d["acc"], "seconds": d["seconds"],
                 "params": d["params"], "constants": d["constants"], "arch": arch_txt, "equation": d["equation"]},
                {"n": n, "seed": seed, "method": "supernet", "acc": d["acc_supernet"], "seconds": "", "params": "",
                 "constants": "", "arch": "mixture before discretisation"},
                {"n": n, "seed": seed, "method": "nearest_class_mean", "acc": b["ncm"], "seconds": "", "params": 0,
                 "constants": 10 * 196, "arch": "closed form"},
                {"n": n, "seed": seed, "method": "pca_lda", "acc": b["pca_lda"], "seconds": "", "params": 0,
                 "constants": "", "arch": "closed form"},
            ]  # fmt: skip
            print(f"n = {n:3d} seed {seed}: CNN {c['acc']:.3f} ({c['seconds']:.1f} s) | tensor DAG {d['acc']:.3f} "
                  f"(supernet {d['acc_supernet']:.3f}, {d['seconds']:.1f} s, {d['params']} learned) | "
                  f"class means {b['ncm']:.3f}, PCA+LDA {b['pca_lda']:.3f} | {arch_txt}", flush=True)  # fmt: skip
            if n == show and seed == 0:
                showcase = {"n": n, **d, "cnn_pred": c["pred"]}

    # data needed and time at equal accuracy
    mean = {m: [np.mean([r["acc"] for r in rows if r["n"] == n and r["method"] == m]) for n in sizes]
            for m in ("cnn", "tensor_dag", "nearest_class_mean", "pca_lda")}  # fmt: skip
    secs = {m: [np.mean([r["seconds"] for r in rows if r["n"] == n and r["method"] == m]) for n in sizes]
            for m in ("cnn", "tensor_dag")}  # fmt: skip
    need = {(m, t): solve.n_to_reach(sizes, mean[m], t) for m in mean for t in TARGETS}
    for t in TARGETS:
        print(f"images per class to reach {t:.0%} test accuracy (mean over seeds, log-linear between sizes): " +
              ", ".join(f"{m} {'never' if math.isnan(need[m, t]) else f'{need[m, t]:.0f}'}" for m in mean))  # fmt: skip
    for k, n in enumerate(sizes):
        print(f"n = {n:3d}: mean accuracy CNN {mean['cnn'][k]:.3f} vs tensor DAG {mean['tensor_dag'][k]:.3f}; "
              f"mean time {secs['cnn'][k]:.1f} s vs {secs['tensor_dag'][k]:.1f} s")  # fmt: skip

    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    with open(res / "stages.csv", "w", newline="", encoding="utf-8") as f:
        f.write(f"# operator_search_mnist: one row per (n images per class, seed, method); test set 2000 images; "
                f"device {dev}; seconds = wall time of training (search + fine-tune for the tensor DAG)\n")  # fmt: skip
        w = csv.writer(f)
        w.writerow(["n_per_class", "seed", "method", "test_accuracy", "train_seconds", "learned_parameters",
                    "data_constants", "architecture"])  # fmt: skip
        for r in rows:
            w.writerow([r["n"], r["seed"], r["method"], f"{r['acc']:.4f}",
                        r["seconds"] if r["seconds"] == "" else f"{r['seconds']:.3f}", r["params"], r["constants"],
                        r["arch"]])  # fmt: skip
        f.write("# images per class needed to reach a mean test accuracy (log-linear interpolation; nan = never)\n")
        for (m, t), v in need.items():
            w.writerow(["needed", f"target {t:.2f}", m, "", "", "", "", f"{v:.2f}"])
    with open(res / "equation.tex", "w", encoding="utf-8") as f:
        f.write("% Classification equations found by the DARTS search on MNIST (14x14), one per (n, seed).\n")
        f.write(
            "% S_x, S_y Sobel, \\Delta Laplacian, G Gabor, K_e learned 3x3 filter of edge e, * cross-correlation,\n"
        )
        f.write(
            "% S_\\lambda soft threshold, z = (x, z_1, z_2, z_3) stacked; m, s, mu_k, U from the training images.\n"
        )
        for r in [r for r in rows if r["method"] == "tensor_dag"]:
            f.write(f"% n = {r['n']} per class, seed {r['seed']}: test accuracy {r['acc']:.3f}\n")
            f.write("\\begin{aligned}\n" + " \\\\\n".join(r["equation"]) + "\n\\end{aligned}\n\n")
    print(f"the equation of the DAG found with n = {show} (seed 0):")
    for line in showcase["equation"]:
        print("   ", line)
    print(f"study {time.perf_counter() - t_all:.1f} s on {dev}; animation next{' (fast mode)' if fast else ''}")
    return plot.animate(rows, sizes, showcase, data, need, TARGETS, out_dir / "operator_search_mnist.gif")


if __name__ == "__main__":
    cli(run, __file__)
