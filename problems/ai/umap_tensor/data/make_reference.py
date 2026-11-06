"""One-off, OFFLINE: the umap-learn reference embedding bundled with this problem (never run by the problem itself).

umap-learn and scikit-learn are not dependencies of this project, so this ran once in a throwaway environment outside
the repository:

    pixi exec --spec python=3.12 --spec umap-learn --spec scikit-learn python -I make_reference.py \
        <problem>/data/mnist_subset.npz <out_dir>

It writes umap_learn_reference.npz (the 2D coordinates) and umap_learn_reference.csv (versions, a and b, wall times,
and scikit-learn's own trustworthiness and 10-NN accuracy, which the tests use to cross-check the tensor metrics).
The Swiss roll is regenerated here with the same numpy code as tensors.swiss_roll_numpy (numpy's PCG64 stream is
stable across numpy versions); its checksum is stored so the tests can verify both sides saw the same points.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

K = 10  # neighbourhood size of the quality metrics


def swiss_roll_numpy(n=1500, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    t = 1.5 * np.pi * (1 + 2 * rng.random(n))
    h = 21.0 * rng.random(n)
    x = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1) + noise * rng.standard_normal((n, 3))
    return x.astype(np.float32), t.astype(np.float32), h.astype(np.float32)


def main(subset: Path, out: Path) -> None:
    import sklearn
    import umap
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier
    from umap.umap_ import find_ab_params

    with np.load(subset) as d:
        xm = d["images"].reshape(len(d["images"]), -1).astype(np.float32) / 255.0
        ym = d["labels"].astype(np.int64)
    xs, ts, _ = swiss_roll_numpy()
    a, b = find_ab_params(1.0, 0.1)
    out.mkdir(parents=True, exist_ok=True)
    emb, rows = {}, []
    for name, x in (("mnist", xm), ("swiss_roll", xs)):
        for run in range(2):  # run 0 includes numba's just-in-time compilation; run 1 is the warm time
            t0 = time.perf_counter()
            y = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit_transform(x)
            secs = time.perf_counter() - t0
            rows.append({"dataset": name, "method": "umap-learn", "run": run, "seconds": f"{secs:.3f}"})
        emb[name] = y.astype(np.float32)
        p = PCA(n_components=2).fit_transform(x).astype(np.float32)
        emb[name + "_pca"] = p
        for method, z in (("umap-learn", y), ("sklearn-pca", p)):
            r = {"dataset": name, "method": method, "run": "final", "seconds": "",
                 "sk_trustworthiness": f"{trustworthiness(x, z, n_neighbors=K):.5f}"}  # fmt: skip
            if name == "mnist":
                acc = cross_val_score(KNeighborsClassifier(K), z, ym, cv=10).mean()  # 10-fold, close to LOO
                r["sk_knn_accuracy_10fold"] = f"{acc:.5f}"
            rows.append(r)
    np.savez_compressed(out / "umap_learn_reference.npz", mnist=emb["mnist"], swiss_roll=emb["swiss_roll"],
                        mnist_pca=emb["mnist_pca"], swiss_roll_pca=emb["swiss_roll_pca"],
                        swiss_checksum=np.float64(xs.astype(np.float64).sum()))  # fmt: skip
    keys = ["dataset", "method", "run", "seconds", "sk_trustworthiness", "sk_knn_accuracy_10fold"]
    with open(out / "umap_learn_reference.csv", "w", newline="", encoding="utf-8") as f:
        f.write("# umap-learn reference for problems/ai/umap_tensor, computed offline by data/make_reference.py "
                "(generated; licence: as the problem). Source of the method: McInnes, Healy & Melville 2018, "
                "arXiv:1802.03426; umap-learn is BSD-3-Clause.\n")  # fmt: skip
        f.write(f"# umap-learn {umap.__version__}, scikit-learn {sklearn.__version__}, numpy {np.__version__}, "
                f"python {sys.version.split()[0]}, {sys.platform}; UMAP(n_neighbors=15, min_dist=0.1, "
                f"random_state=42) (random_state makes it single-threaded and repeatable)\n")  # fmt: skip
        f.write(f"# find_ab_params(spread=1, min_dist=0.1): a = {a:.6f}, b = {b:.6f}\n")
        f.write(f"# metrics with k = {K}; seconds = wall time of fit_transform (run 0 includes numba JIT)\n")
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(open(out / "umap_learn_reference.csv", encoding="utf-8").read())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
