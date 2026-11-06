"""Main loop: UMAP on an MNIST subset and on a Swiss roll, written as tensor operations; the same layout with
autograd forces; PCA as the linear baseline; quality metrics against the bundled umap-learn reference; tables and the
animation of the layout epochs."""

import csv
import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation as E
from . import plot, solve
from . import tensors as T

K_METRIC = 10  # neighbourhood size of trustworthiness, continuity, recall and the k-NN classifier


def metrics(x, y, labels=None, flat=None) -> dict:
    """All quality numbers of one 2D layout y of data x (labels: digits; flat: true flat coordinates)."""
    m = {"trust": E.trustworthiness(x, y, K_METRIC), "cont": E.continuity(x, y, K_METRIC),
         "recall": E.knn_recall(x, y, K_METRIC)}  # fmt: skip
    if labels is not None:
        m["acc"] = E.knn_accuracy(y, labels, K_METRIC)
    if flat is not None:
        m["trust_flat"] = E.trustworthiness(flat, y, K_METRIC)
        m["recall_flat"] = E.knn_recall(flat, y, K_METRIC)
    return m


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    k, md, seed = int(params["n_neighbors"]), float(params["min_dist"]), int(params["seed"])
    epochs = 40 if fast else int(params["n_epochs"])
    x, lab = T.mnist(dev, n=400 if fast else int(params["n_mnist"]))
    xs, flat, t_roll = T.swiss_roll(dev, n=400 if fast else T.N_ROLL)
    print(f"MNIST: {len(x)} images (balanced), X {tuple(x.shape)}; Swiss roll: X {tuple(xs.shape)}, true flat "
          f"coordinates U {tuple(flat.shape)}; device {dev}; k = {k}, min_dist = {md}, {epochs} epochs, "
          f"{params['neg_rate']} negatives per edge sample")  # fmt: skip
    a, b = E.fit_ab(md)
    print(f"curve fit: q = 1/(1 + a d^(2b)) with a = {a:.4f}, b = {b:.4f}")

    runs, rows = {}, []
    snaps = 5 if fast else plot.FRAMES
    for name, data, mode in (("mnist", x, "forces"), ("mnist", x, "autograd"), ("swiss_roll", xs, "forces")):
        prep = runs.get((name, "forces"))  # the autograd run reuses the graph and the spectral start
        o = solve.umap(data, k, md, epochs, seed, mode, snaps=snaps if mode == "forces" else 0,
                       neg_rate=int(params["neg_rate"]), prep=prep)  # fmt: skip
        runs[name, mode] = o
        s = o["seconds"]
        n_fail = int((o["sigma"] <= 1e-3 * o["dist"][:, 1:].mean(1) * 1.0001).sum())
        print(f"{name:10s} {mode:8s}: kNN + fuzzy set {s['graph']:.2f} s ({o['edges']} directed edges, σ at its "
              f"floor for {n_fail} points), spectral start {s['spectral']:.2f} s, layout {s['layout']:.2f} s, "
              f"total {s['total']:.2f} s")  # fmt: skip

    ref = T.reference()
    full = not fast and len(x) == 2500  # the reference was computed on the full bundled subset
    for name, data, extra in (("mnist", x, {"labels": lab}), ("swiss_roll", xs, {"flat": flat})):
        cands = [("PCA (SVD)", solve.pca(data), 0.0)]
        for mode in ("forces", "autograd"):
            if (name, mode) in runs:
                o = runs[name, mode]
                cands.append((f"tensor UMAP ({mode})", o["Y"], o["seconds"]["total"]))
        if ref is not None and (full or name == "swiss_roll") and not fast:
            warm = [float(r["seconds"]) for r in ref["rows"] if r["dataset"] == name and r["run"] == "1"]
            cands.append(("umap-learn (offline)", torch.from_numpy(ref[name]).to(dev), warm[0]))
        base = next((c[1] for c in cands if c[0].startswith("umap-learn")), None)
        for method, y, secs in cands:
            m = metrics(data, y, **extra)
            m["procrustes"] = solve.procrustes(y, base)[1] if base is not None else float("nan")
            rows.append({"dataset": name, "method": method, "seconds": secs, **m})
            txt = ", ".join(f"{key} {v:.3f}" for key, v in m.items())
            print(f"{name:10s} {method:22s}: {txt}" + (f", {secs:.1f} s" if secs else ""))

    res = out_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    keys = ["trust", "cont", "recall", "acc", "trust_flat", "recall_flat", "procrustes"]
    with open(res / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        f.write(f"# umap_tensor: quality of each 2D layout, neighbourhood size k = {K_METRIC}. trust = "
                "trustworthiness, cont = continuity, recall = shared k-NN fraction, acc = leave-one-out 10-NN digit "
                "accuracy, *_flat = against the true flat Swiss-roll coordinates, procrustes = disparity to the "
                "umap-learn layout after the best rotation/reflection/scale (0 = same shape); seconds = wall time "
                f"(umap-learn: warm run, offline, single thread); device {dev}\n")  # fmt: skip
        w = csv.writer(f)
        w.writerow(["dataset", "method", "seconds", *keys])
        for r in rows:
            w.writerow([r["dataset"], r["method"], f"{r['seconds']:.3f}",
                        *(f"{r[k]:.4f}" if k in r else "" for k in keys)])  # fmt: skip
    o = runs["mnist", "forces"]
    with open(res / "mnist_embedding.csv", "w", newline="", encoding="utf-8") as f:
        f.write(
            "# umap_tensor: final tensor-UMAP layout (hand-written forces) of the MNIST subset; y1, y2 in "
            "arbitrary units; rho, sigma = the fuzzy-set parameters of each point (pixel-space units)\n"
        )
        w = csv.writer(f)
        w.writerow(["index", "digit", "y1", "y2", "rho", "sigma"])
        for i, (yy, d, r_, s_) in enumerate(zip(o["Y"].tolist(), lab.tolist(), o["rho"].tolist(),
                                                o["sigma"].tolist())):  # fmt: skip
            w.writerow([i, d, f"{yy[0]:.4f}", f"{yy[1]:.4f}", f"{r_:.4f}", f"{s_:.4f}"])

    t0 = time.perf_counter()
    plot.result_pngs(runs, rows, x, lab, xs, t_roll, ref if full else None, res)
    out = plot.animate(runs, lab, t_roll, epochs, out_dir / "umap_tensor.gif")
    print(f"figures {time.perf_counter() - t0:.1f} s")
    return out


if __name__ == "__main__":
    cli(run, __file__)
