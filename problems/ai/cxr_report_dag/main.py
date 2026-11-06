"""Main loop: parse the training rule graph → run it with the tensor DAG engine (each rule one batched call over the
seeds) → label F1 of the written reports for every stage vs the CNN alone and the monolith, the learned router,
caption metrics after the PMC stage, the measured schedule → plot."""

import time
from pathlib import Path

import numpy as np
import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation as eq
from . import plot, solve
from . import tensors as T

LAST: dict = {}  # the results of the latest run (read by the full-run test)


def _pm(x: torch.Tensor) -> str:
    """'mean ± half-range' over the seeds."""
    return f"{float(x.mean()):.3f} ± {float(x.max() - x.min()) / 2:.3f}"


def _names(y) -> str:
    return ", ".join(T.FINDINGS[f] for f in range(6) if y[f]) or "no finding"


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    f = 0.03 if fast else 1.0
    cfg = solve.Config(
        seeds=list(range(1 if fast else int(params["seeds"]))),
        steps_cnn=max(2, int(params["steps_cnn"] * f)),
        steps_lm=max(2, int(params["steps_lm"] * f)),
        steps_con=max(2, int(params["steps_con"] * f)),
        steps_ft=max(2, int(params["steps_ft"] * f)),
        steps_router=max(2, int(params["steps_router"] * f)),
        steps_cap=max(2, int(params["steps_cap"] * f)),
        lr_ft=params["lr_ft"],
        lr_cap=params["lr_cap"],
        n_test=16 if fast else None,
    )
    S = len(cfg.seeds)
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
    data, pmc = st["data"], st["pmc"]
    print(f"device {dev}; {S} seeds in every batched call; NIH {len(data['y_train'])} train / "
          f"{len(data['y_test'])} test images, PMC {len(pmc['cap_train'])} train / {len(pmc['cap_test'])} test "
          "captions")  # fmt: skip
    for n in d["names"]:
        print(f"  {n:16s} level {lev[n]}  {seconds[n]:6.1f} s")
    print(f"serial (one worker, level by level) {sch['serial']:.1f} s; with one worker per rule the max-plus "
          f"critical path is {' → '.join(sch['path'])} = {sch['makespan']:.1f} s")  # fmt: skip
    mono_count = sum(T.count(st["train_monolith"][k]) for k in ("cnn", "con", "lm"))
    print(f"weights per model: CNN {T.count(st['train_cnn']['w'])}, connector {T.count(st['train_connector']['con'])}"
          f", LM {T.count(st['train_lm']['w'])}; monolith {mono_count} (the same architecture)")  # fmt: skip
    print(f"budget: the DAG pipeline takes {cfg.steps_cnn} CNN + {cfg.steps_lm} LM + {cfg.steps_con} connector + "
          f"{cfg.steps_ft} fine-tune steps of {cfg.batch}; the monolith gets the same multiply-adds: {cfg.steps_mono}"
          " joint steps")  # fmt: skip

    ev = st["evaluate"]
    f1_micro = {"cnn": ev["cnn_micro"], **{k: r["micro"] for k, r in ev["rep"].items()}}
    f1_macro = {"cnn": ev["cnn_macro"], **{k: r["macro"] for k, r in ev["rep"].items()}}
    print(f"CNN alone: mean AUROC over the six findings {_pm(ev['auc'].mean(1))} (chance 0.5)")
    print(f"NIH test, labels parsed back from the written report (micro F1; random guessing at the training "
          f"prevalence {ev['chance']:.3f}), mean ± half-range over seeds:")  # fmt: skip
    print(f"  {'stage':16s} {'micro F1':>16s} {'macro F1':>16s} {'well-formed':>16s} {'exact template':>16s}")
    for k in plot.STAGE:
        r = ev["rep"].get(k)
        extra = f"{_pm(r['wellformed']):>16s} {_pm(r['exact']):>16s}" if r else f"{'—':>16s} {'—':>16s}"
        print(f"  {k:16s} {_pm(f1_micro[k]):>16s} {_pm(f1_macro[k]):>16s} {extra}")
    print(f"learned router (trained dense, used top-1): share of positions sent to the modality's expert "
          f"{_pm(ev['route_acc'])}")  # fmt: skip
    for key, label in (("caponly", "caption alone"), ("cap", "caption + citing sentences (the stage-2 target)")):
        w = np.array([len(c.split()) for c in pmc[f"{key}_train"] + pmc[f"{key}_test"]])
        print(f"PMC text, {label}: words median {np.median(w):.0f}, IQR {np.percentile(w, 25):.0f}–"
              f"{np.percentile(w, 75):.0f}, range {w.min()}–{w.max()}")  # fmt: skip
    print(f"  ({pmc['cut']} texts whose first sentence alone exceeds {T.MAX_CAP} characters were cut at a word)")
    print("PMC test texts (stage 2):")
    print(f"  {'model':18s} {'BLEU-1':>14s} {'ROUGE-L':>14s} {'kw recall':>14s} {'kw precision':>14s} "
          f"{'nats/letter':>14s}")  # fmt: skip
    for k, c in ev["cap"].items():
        print(f"  {k:18s} " + " ".join(f"{_pm(c[m]):>14s}" for m in ("bleu1", "rouge_l", "kw_recall",
                                                                     "kw_precision", "nats")))  # fmt: skip
    print(f"  (constant = the training caption closest to all others: '{ev['constant']}')")

    # examples: the first NIH test image of four label groups (whatever the model writes), the first PMC test figures
    with np.load(T.DATA / "nih_cxr_subset.npz") as z:
        raw = z["images"][z["split"] == 1]
    with np.load(T.DATA / "pmc_cxr_subset.npz") as z:
        raw_p = z["images"][z["split"] == 1]
    y_test = data["y_test"][: len(ev["txt"]["finetune"][0])].int().tolist()
    examples = []
    for want in ((0,) * 6, (1, 0, 0, 0, 0, 0), (0, 1, 0, 0, 0, 0), (0, 0, 0, 0, 0, 1)):
        hits = [i for i, y in enumerate(y_test) if tuple(y) == want] or [len(examples)]
        i = hits[0]
        dag_txt = ev["txt"]["finetune"][0][i]
        examples.append({"img": raw[i], "q": ev["q"][0, i].numpy(), "dag": dag_txt or "(nothing)",
                         "mono": ev["txt"]["monolith"][0][i] or "(nothing)", "pred": T.parse_report(dag_txt),
                         "truth": tuple(y_test[i]), "truth_short": _names(y_test[i])})  # fmt: skip
        print(f"  NIH test image {i} ({_names(y_test[i])}): DAG '{dag_txt}' | monolith "
              f"'{ev['txt']['monolith'][0][i]}'")  # fmt: skip
    ct = ev["cap_txt"]
    cap_examples = [{"img": raw_p[i], "truth": pmc["cap_test"][i], "gen": ct["caption"][0][i] or "(nothing)"}
                    for i in range(min(2, len(ct["caption"][0])))]  # fmt: skip
    for i, ex in enumerate(cap_examples):
        print(f"  PMC test figure {i}: true '{ex['truth']}' | model '{ex['gen']}'")
    print(f"total {total:.0f} s")

    tok = eq.position_items(st["table"]["tokens"][:12, :-1][None])[1]
    G_fixed = eq.route_fixed(1, 12, tok)[0, :, :, 0].cpu().numpy()
    res = {"names": d["names"], "levels": d["levels"].tolist(), "A_rule": d["A_rule"].tolist(), "seconds": seconds,
           "path": sch["path"], "finish": sch["finish"].tolist(), "makespan": sch["makespan"], "serial": sch["serial"],
           "loss": {n: st[n]["loss"] for n in d["names"] if n != "evaluate"}, "seeds": cfg.seeds,
           "f1_micro": f1_micro, "f1_macro": f1_macro, "wellformed": {k: r["wellformed"] for k, r in ev["rep"].items()},
           "exact": {k: r["exact"] for k, r in ev["rep"].items()}, "auc": ev["auc"], "route_acc": ev["route_acc"],
           "route_show": float(ev["route_acc"][0]), "chance": ev["chance"], "cap": ev["cap"], "G_fixed": G_fixed,
           "G_learned": ev["route_p"][0, :12].numpy(), "examples": examples, "cap_examples": cap_examples,
           "cap_rows": [(pmc["cap_test"][i], ct["finetune"][0][i], ct["caption"][0][i], ct["caption_shuffled"][0][i])
                        for i in range(len(ct["caption"][0]))],
           "rep_rows": [(_names(y_test[i]), " ".join(f"{v:.2f}" for v in ev["q"][0, i].tolist()),
                         ev["txt"]["finetune"][0][i], ev["txt"]["monolith"][0][i])
                        for i in range(len(y_test))]}  # fmt: skip
    LAST.clear()
    LAST.update(res)
    plot.write_tables(res, out_dir)
    plot.schedule_png(res, out_dir / "schedule.png")
    return plot.animate(res, out_dir / "cxr_report_dag.gif", n_frames=3 if fast else 14, hold=2 if fast else 18)


if __name__ == "__main__":
    cli(run, __file__)
