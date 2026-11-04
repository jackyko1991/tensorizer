"""Main loop: MNIST digit + three-word text → (digit op number) mod 10, as a Snakemake-style workflow DAG run by a
tensor engine; the same computation as a fixed-router MoE layer (identical outputs); a learned router; split
training → end-to-end fine-tuning → distillation into one dense student → int8."""

import time
from pathlib import Path

from problems._common.runtime import cli, fast_mode

from . import plot, solve, tensors


def _pm(x) -> str:
    """'mean ± half-range' over the seeds."""
    return f"{float(x.mean()):.3f} ± {float(x.max() - x.min()) / 2:.3f}"


def run(params: dict, out_dir: Path) -> Path:
    fast = fast_mode()
    out_dir = Path(out_dir)
    cfg = tensors.Config(
        seeds=params["seeds"],
        split_steps=40 if fast else params["split_steps"],
        router_steps=20 if fast else params["router_steps"],
        global_steps=20 if fast else params["global_steps"],
        student_steps=40 if fast else params["student_steps"],
        student_hidden=params["student_hidden"],
        temperature=params["temperature"],
        balance_weight=params["balance_weight"],
        test_size=200 if fast else 1000,
        time_repeats=1 if fast else 5,
        seq_samples=10 if fast else 100,
    )
    t0 = time.time()
    res = solve.run_stages(cfg, n_checkpoints=4 if fast else 12)
    M = cfg.seeds
    print(f"{M} seeds on the model axis, {cfg.test_size} test images × one fixed set of texts, CPU: "
          f"{time.time() - t0:.0f} s")  # fmt: skip

    eqv = res["equivalence"]
    jobs = eqv["jobs"]["fuse"]
    print(f"(1) workflow DAG ≡ top-1 MoE with G fixed to the rule choice: fuse runs {jobs[0]} of {jobs[1]} possible "
          "jobs (the router sits in the job adjacency)")  # fmt: skip
    for dt in ("float32", "float64"):
        e = eqv[dt]
        print(f"    {dt}: max |logit difference| = {e['max_abs_diff']:.1e}, bitwise equal: {e['bitwise_equal']}, "
              f"same prediction {e['same_predictions']}/{e['total']}")  # fmt: skip
    rec, ident, load = res["router_recovery"], res["router_identity"], res["router_load"]
    print("    learned router (softmax gate on the text features + load balancing, top-1, 1 of 3 experts):")
    print(f"      gate alone, in front of the workflow's own fuse experts: accuracy {_pm(res['acc']['router_given'])}, "
          f"routes each operation to its own expert for {_pm(ident[:M])} of the test samples")  # fmt: skip
    print(f"      gate and fresh experts learned together: accuracy {_pm(res['acc']['router'])}, routing matches the "
          f"operations (best expert permutation) {_pm(rec[M:])}; 1/3 would be no relation")  # fmt: skip
    for m in range(M):
        print(
            f"      seed {m + 1}, fresh experts: loads "
            + " ".join(f"{v:.2f}" for v in load[M + m].tolist())
            + ", true operation × expert "
            + str([[round(v, 2) for v in r] for r in res["router_confusion"][M + m].tolist()])
        )
    sub = res["subtask_acc"]
    print(f"(2) split training: image expert {_pm(sub['image'])} (digits), text expert {_pm(sub['text'])} "
          f"(number), fuse experts {_pm(sub['fuse'])} (300-entry table); chained, before any joint training: "
          f"{_pm(res['acc']['split'])}")  # fmt: skip
    print(f"(3) end-to-end fine-tuning through the DAG: {_pm(res['acc']['global'])} (digit accuracy of the image "
          f"expert {_pm(res['digit_acc']['split'])} → {_pm(res['digit_acc']['global'])})")  # fmt: skip
    print(f"(4) distilled student (T = {cfg.temperature}): {_pm(res['acc']['student'])}, agrees with its teacher on "
          f"{_pm(res['agree_teacher'])}; int8: {_pm(res['acc']['int8'])}")  # fmt: skip
    print("    model      params   bytes    ms per 1000 samples (1 thread)")
    for k in ("split", "global", "student", "int8"):
        print(f"    {k:8s} {res['params'][k]:8d} {res['bytes'][k]:8d}    {_pm(res['time'][k] * 1e3)}")
    print(f"    same DAG as one MoE layer: {_pm(res['time']['moe'] * 1e3)} ms; one job per call (B = 1, no process "
          f"start-up): {_pm(res['time']['one_by_one'] * 1e3)} ms")  # fmt: skip

    plot.write_tables(res, out_dir)
    plot.stages_png(res, out_dir / "stages.png")
    return plot.animate(res, out_dir / "moe_dag_pipeline.gif", n_frames=12 if fast else 40)


if __name__ == "__main__":
    cli(run, __file__)
