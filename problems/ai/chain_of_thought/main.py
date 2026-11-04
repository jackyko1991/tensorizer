"""Main loop: parity strings → direct and chain-of-thought formats → M stacked transformers trained at once →
accuracy against n, length generalisation, hidden scratchpad steps and attention maps → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def _mean_pm(x: torch.Tensor) -> str:
    """'mean ± half-range' over the seeds (last axis)."""
    return f"{x.mean():.2f} ± {(x.max() - x.min()) / 2:.2f}"


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    lengths = [int(v) for v in str(params["lengths"]).split(",")]
    cfg = solve.Config(
        lengths=[4, 6] if fast else lengths,
        seeds=1 if fast else params["seeds"],
        steps=20 if fast else params["steps"],
        d=params["d_model"],
        heads=params["heads"],
        layers=params["layers"],
        batch=params["batch"],
        lr=params["lr"],
        gen_lengths=[4, 8] if fast else [4, 8, 12, 16],
        test_size=32 if fast else 256,
    )
    L, S = len(cfg.lengths), cfg.seeds
    t0 = time.time()
    res = solve.run_experiments(cfg, dev, n_checkpoints=4 if fast else 20)
    print(f"{res['M']} transformers trained together ({L} lengths × direct/CoT × {S} seeds + {2 * S} for length "
          f"generalisation), {cfg.layers} layers, d = {cfg.d}, {cfg.heads} heads, {cfg.steps} steps of "
          f"{cfg.batch} strings: {time.time() - t0:.0f} s")  # fmt: skip
    print(f"weights per model: {res['params_per_model'][0]} (n = {cfg.lengths[0]}) … {res['params_per_model'][-1]}; "
          "the direct and the CoT model of each n have the same count")  # fmt: skip

    print("(a) answer accuracy on test strings (chance 0.50), mean ± half-range over seeds:")
    for li, n in enumerate(cfg.lengths):
        note = "held-out strings" if res["held_out"][li] else f"all {2**n} strings, seen in training"
        print(f"  n = {n:2d}: direct {_mean_pm(res['acc'][li, 0])}   CoT {_mean_pm(res['acc'][li, 1])}   ({note})")
    print(
        "    with the true chain fed in, the CoT answer copies s_n in a fraction "
        f"{res['oracle_answer_agrees'].min():.3f} – {res['oracle_answer_agrees'].max():.3f} of cases"
    )
    if "gen_acc" in res:
        print(f"(b) trained on 1 ≤ n ≤ {cfg.train_max}, tested on longer strings:")
        for li, n in enumerate(cfg.gen_lengths):
            g = res["gen_acc"][:, :, li]
            print(f"  n = {n:2d}: direct {_mean_pm(g[0])}   CoT {_mean_pm(g[1])}")
    print("(c) CoT answer accuracy when only the first k scratchpad steps are kept (mean over seeds):")
    for li, n in enumerate(cfg.lengths):
        row = res["hidden"][li, :, : n + 1].mean(0)
        print(f"  n = {n:2d}: k = 0 … {n}: " + " ".join(f"{a:.2f}" for a in row.tolist()))
    l_best, h_best = res["best_head"]
    ns = res["n_show"]
    uniform = sum(1 / (ns + k) for k in range(1, ns + 1)) / ns  # query n + k − 1 sees n + k positions
    print(f"(d) CoT model, n = {res['n_show']}: layer {l_best + 1} head {h_best + 1} puts "
          f"{res['on_b'][l_best, h_best]:.2f} of the attention of the query that writes s_k on b_k "
          f"(uniform over the past would give {uniform:.2f}) and "
          f"{res['on_prev'][l_best, h_best]:.2f} on s_(k-1) itself")  # fmt: skip
    per_k = res["on_b_k"].flatten(0, 1).max(0).values  # (n,) best head at each step
    print("    best head at each step k = 1 … n, weight on b_k: " + " ".join(f"{a:.2f}" for a in per_k.tolist()))
    tok = res["sample_tok"].tolist()[: 2 * res["n_show"] + 2]  # bits, ":", the model-written chain, "="
    sample = "".join(tensors.VOCAB[t] for t in tok) + str(res["sample_answer"])
    print(f"    sample: {sample}  (model answer {res['sample_answer']}, true parity {res['sample_parity']})")

    plot.experiments_png(res, out_dir / "experiments.png")
    plot.write_table(res, out_dir / "accuracy.csv")
    return plot.animate(res, out_dir / "chain_of_thought.gif", hold=1 if fast else 2)


if __name__ == "__main__":
    cli(run, __file__)
