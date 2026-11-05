"""Main loop: B games of Snake stepped as tensors; a CNN sees each board, a mixture-of-experts controller picks the
move. Imitation of a planning teacher (+ one DAgger round) trains four controllers at once on one model axis:
(a) dense mixture, (b) Switch top-1, (c) dense-trained then top-1, (d) a monolithic MLP with as many weights."""

import math
import time
from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve
from . import tensors as T


def run(params: dict, out_dir: Path) -> Path:
    fast = fast_mode()
    out_dir = Path(out_dir)
    cfg = T.Config(
        grid=params["grid"],
        seeds=params["seeds"],
        experts=params["experts"],
        steps=40 if fast else params["steps"],
        balance_weight=params["balance_weight"],
        dagger=bool(params["dagger"]),
        teacher_games=32 if fast else 192,
        teacher_steps=40 if fast else 300,
        dagger_games=8 if fast else 48,
        dagger_steps=20 if fast else 200,
        eval_games=16 if fast else params["eval_games"],
        eval_steps=40 if fast else 300,
        show_steps=40 if fast else 160,
    )
    dev = device()
    t0 = time.time()
    res = solve.run_all(cfg, dev)
    S, K = cfg.seeds, cfg.experts
    print(f"{cfg.grid}×{cfg.grid} board, {S} seeds per variant on one model axis ({3 * S} trained models), "
          f"{dev.type.upper()}: {time.time() - t0:.0f} s")  # fmt: skip
    print("    " + ", ".join(f"{k} {v:.1f} s" for k, v in res["time"].items()))
    n_teacher, n_all = res["pool"]
    print(f"imitation data: {n_teacher} teacher states" + (f" + {n_all - n_teacher} DAgger states (students' "
          "games, teacher's labels)" if cfg.dagger else ""))  # fmt: skip
    pr = res["params"]
    print(f"weights per model: CNN {pr['cnn']}, MoE controller (router + {K} experts) {pr['moe']}, monolithic MLP "
          f"{pr['mono']}")  # fmt: skip
    print(f"score on {cfg.eval_games} shared games (≤ {cfg.eval_steps} moves, ends on a wall, the body, or "
          f"{cfg.hunger_limit()} moves without food):")  # fmt: skip
    print("    policy               food/game   per seed        moves survived   agreement with teacher (train)")
    acc = res["acc"][-50:].mean(0)
    for v in plot.POLICIES:
        x = res["food"][v]
        seeds = " ".join(f"{float(s):5.2f}" for s in x.mean(-1)) if x.dim() > 1 else ""
        ag = ""
        if v in ("dense", "switch", "mono"):
            j = ("dense", "switch", "mono").index(v)
            ag = f"{float(acc[j * S : (j + 1) * S].mean()):.3f}"
        print(f"    {T.LABELS[v]:20s} {float(x.mean()):8.2f}   {seeds:15s} {float(res['steps'][v].mean()):8.0f}"
              f"         {ag}")  # fmt: skip
    sp = res["spec"]
    print("routing (routed models, seed by seed): expert loads; mutual information (bit) between the routed expert "
          f"and the state (max log2 {K} = {math.log2(K):.2f})")  # fmt: skip
    for r, v in enumerate(("dense", "switch", "dense_top1")):
        for s in range(S):
            m = r * S + s
            loads = " ".join(f"{float(x):.2f}" for x in sp["load"][m])
            mis = ", ".join(f"{k} {float(sp['mi'][k][m]):.2f}" for k in ("dist", "ahead", "length", "turn"))
            print(f"    {T.LABELS[v]:18s} seed {s + 1}: loads {loads}; MI: {mis}")
    print("    (dist = distance to food, ahead = free cells straight ahead, length = body length, turn = the move "
          "the controller makes)")  # fmt: skip

    plot.write_tables(res, out_dir)
    plot.routing_png(res, out_dir / "routing.png")
    return plot.animate(res, out_dir / "snake_moe_control.gif", n_frames=6 if fast else 50)


if __name__ == "__main__":
    cli(run, __file__)
