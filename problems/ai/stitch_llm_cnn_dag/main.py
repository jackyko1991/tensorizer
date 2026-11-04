"""Main loop: parse the training rule graph → run it with the tensor DAG engine (each rule one batched call over the
models: seeds × joint-set sizes) → accuracy of every stage vs the monolith, sample efficiency, fixed vs learned
router, measured schedule → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation as eq
from . import plot, solve
from . import tensors as T


def _pm(x: torch.Tensor) -> str:
    """'mean ± half-range' over the seeds."""
    return f"{x.mean():.3f} ± {(x.max() - x.min()) / 2:.3f}"


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    f = 0.03 if fast else 1.0
    cfg = solve.Config(
        seeds=list(range(1 if fast else int(params["seeds"]))),
        sizes=[100, 3000] if fast else [100, 300, 1000, 3000],
        steps_cnn=max(2, int(params["steps_cnn"] * f)),
        steps_lm=max(2, int(params["steps_lm"] * f)),
        steps_con=max(2, int(params["steps_con"] * f)),
        steps_ft=max(2, int(params["steps_ft"] * f)),
        steps_router=max(2, int(params["steps_router"] * f)),
        lr_ft=params["lr_ft"],
        alpha=params["alpha"],
        n_test=100 if fast else 1000,
    )
    S, N = len(cfg.seeds), len(cfg.sizes)
    d = solve.build_dag(cfg)
    lev = dict(zip(d["names"], d["levels"].tolist()))
    print(f"training DAG ({T.TRAINING_GRAPH.name}): {len(d['names'])} rules, {len(d['A'])} jobs, "
          f"{int(d['A'].sum())} job edges; A nilpotent: {solve.dag.is_nilpotent(d['A'])}")  # fmt: skip
    for k in range(int(d["levels"].max()) + 1):
        print(f"  level {k}: " + " ∥ ".join(n for n, lv in lev.items() if lv == k))
    t0 = time.perf_counter()
    st, seconds, d = solve.run_engine(cfg, dev, d)
    total = time.perf_counter() - t0
    sch = solve.schedule(d, seconds)
    models = {"train_cnn": S, "train_lm": S, "evaluate": 1}  # models in each rule's one batched call
    models.update({n: S * N for n in d["names"] if n not in models})
    print(f"device {dev}; {S} seeds × {N} joint-set sizes {cfg.sizes}; batch {cfg.batch}")
    for n in d["names"]:
        print(f"  {n:16s} level {lev[n]}  {models[n]:2d} models in one batched call  {seconds[n]:6.1f} s")
    print(f"serial (one worker, level by level) {sch['serial']:.1f} s; with one worker per rule the max-plus "
          f"critical path is {' → '.join(sch['path'])} = {sch['makespan']:.1f} s")  # fmt: skip

    ev = st["evaluate"]
    mono_count = sum(T.count(st["train_monolith"][k]) for k in ("cnn", "con", "lm"))
    print(f"weights per model: CNN {T.count(st['train_cnn']['w'])}, connector "
          f"{T.count(st['train_connector']['con'])}, LM {T.count(st['train_lm']['w'])}; monolith {mono_count}"
          " (the same architecture)")  # fmt: skip
    print(f"budget: the DAG pipeline takes {cfg.steps_cnn} + {cfg.steps_lm} + {cfg.steps_con} + {cfg.steps_ft} "
          f"optimiser steps of {cfg.batch}; the monolith gets the same {cfg.steps_mono} steps of {cfg.batch}, "
          "all on the joint task")  # fmt: skip
    print(f"CNN alone, MNIST test digits: {_pm(ev['cnn_acc'])};  LM alone, all 300 text prompts: {_pm(ev['lm_acc'])};"
          f"  CNN's label written as text → LM: {_pm(ev['symbolic'])}")  # fmt: skip
    print("joint test accuracy (1000 images + prompts, chance 0.1), mean ± half-range over seeds:")
    print("  n     " + "".join(f"{k:>18s}" for k in plot.STAGE))
    for j, n in enumerate(cfg.sizes):
        print(f"  {n:<5d} " + "".join(f"{_pm(ev['acc'][k][:, j]):>18s}" for k in plot.STAGE))
    for k, label in (("router", "Switch top-1"), ("router_soft", "trained dense")):
        print(
            f"learned router ({label}), share of positions sent to the modality's expert: "
            + ", ".join(f"n={n}: {_pm(ev['route_acc'][k][:, j])}" for j, n in enumerate(cfg.sizes))
        )

    # fixed vs learned router on a few test examples (seed 0, largest n), and test completions
    m = N - 1  # model (seed 0, largest n)
    data, test, c = st["data"], ev["test"], st["train_connector"]
    rows = torch.arange(12, device=dev)
    bt = solve.joint_batch(data, rows[None], test["op"][rows][None], test["b"][rows][None], train=False)
    G_learned = {}
    with torch.no_grad():
        for k in ("router", "router_soft"):
            _, G, p = solve.moe_forward(bt, *(T.select(c[j], [m]) for j in ("cnn", "con", "lm")),
                                        T.select(st["train_" + k]["router"], [m]), data["blank"])  # fmt: skip
            G_learned[k] = p[0, :, :, 0].cpu().numpy()
    ids, is_img = eq.position_items(bt["inp"])
    G_fixed = eq.route_fixed(1, len(rows), is_img)[0, :, :, 0]
    ex_rows = torch.arange(4, device=dev)
    ft, mono = st["global_finetune"], st["train_monolith"]
    tok = data["table"]["tokens"][data["y_test"][ex_rows], test["op"][ex_rows], test["b"][ex_rows]]
    x = data["x_test"][ex_rows]
    dag_out = solve.greedy(*(T.select(ft[k], [m]) for k in ("cnn", "con", "lm")), x, tok)
    mono_out = solve.greedy(*(T.select(mono[k], [m]) for k in ("cnn", "con", "lm")), x, tok)
    examples = []
    for i in range(len(ex_rows)):
        s = T.decode(tok[i])
        examples.append({"img": x[i].cpu().numpy(), "prompt": s[T.K : s.index("=") + 1], "dag": dag_out[i],
                         "mono": mono_out[i], "truth": s[s.index("=") + 1 : s.index(".")]})  # fmt: skip
        print(f"  test image of a {T.WORDS[int(data['y_test'][i])]}, prompt '{examples[-1]['prompt']}' → DAG "
              f"'{dag_out[i]}', monolith '{mono_out[i]}' (true '{examples[-1]['truth']}')")  # fmt: skip
    print(f"total {total:.0f} s")

    res = {"names": d["names"], "levels": d["levels"].tolist(), "A_rule": d["A_rule"].tolist(), "seconds": seconds,
           "path": sch["path"], "finish": sch["finish"].tolist(), "makespan": sch["makespan"], "serial": sch["serial"],
           "models": models, "loss": {n: st[n]["loss"] for n in d["names"] if n != "evaluate"},
           "acc": ev["acc"], "route_acc": ev["route_acc"], "cnn_acc": ev["cnn_acc"], "lm_acc": ev["lm_acc"],
           "symbolic": ev["symbolic"], "sizes": cfg.sizes, "seeds": cfg.seeds, "G_fixed": G_fixed.cpu().numpy(),
           "G_learned": G_learned, "route_acc_show": {k: float(v[0, -1]) for k, v in ev["route_acc"].items()},
           "examples": examples}  # fmt: skip
    plot.write_tables(res, out_dir)
    plot.schedule_png(res, out_dir / "schedule.png")
    return plot.animate(res, out_dir / "stitch_llm_cnn_dag.gif", n_frames=4 if fast else 36, hold=2 if fast else 10)


if __name__ == "__main__":
    cli(run, __file__)
