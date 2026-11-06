"""Migrating CPU-only ML to tensors: scikit-learn next to the same model written as PyTorch tensor code.

Feeds the "Migrating CPU-only ML to tensors" tab of the Coding Skills page (frontend/src/content/cpuml_bench.json).
scikit-learn is not a dependency of this project, so the measurement runs in two halves:

    # 1. scikit-learn half, in a throwaway environment (numpy + scikit-learn, no torch):
    pixi exec --spec scikit-learn --spec numpy python scripts/cpuml_benchmark.py --sklearn-only
    # 2. tensor half, in the project environment (CPU, and GPU with the CUDA build of torch); reads the
    #    scikit-learn results of step 1 from .cache/, checks agreement and writes the JSON for the page:
    pixi run -e cuda python scripts/cpuml_benchmark.py

Each case is a pair of functions, `<id>_sklearn` (the scikit-learn estimator as you would call it) and
`<id>_tensor` (the same model as tensor operations: device-agnostic, float32). The page shows the source of
both, taken from this file, so the code on the page is the code that was timed.

Agreement is checked on real data (2500 MNIST digits bundled with problems/ai/umap_tensor, 2000 for training
and 500 for testing): accuracy, objective value, explained variance, predictions and coefficients, with the
objective of every model evaluated by one numpy formula for both results. Timing uses synthetic data of growing
size (the same numpy-generated arrays in both halves): warm-up call, then the median of repeats, GPU calls
bracketed by torch.cuda.synchronize() with the inputs already on the device; the whole set is measured
`--passes` times and the spread of the speed-ups is kept. The two halves run one after the other, not
interleaved, so the CPU load of both is recorded.
"""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch  # the tensor half; the scikit-learn half runs without it
except ImportError:
    torch = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "frontend" / "src" / "content" / "cpuml_bench.json"
CACHE = ROOT / ".cache"
SK_JSON = CACHE / "cpuml_sklearn.json"
SK_NPZ = CACHE / "cpuml_sklearn.npz"
MNIST = ROOT / "problems" / "ai" / "umap_tensor" / "data" / "mnist_subset.npz"
N_CLASSES = 10

# Model settings shared by both halves (the regularisers must be the same objective on both sides).
PCA_K = 50  # components kept on MNIST; 16 on the synthetic data
SVM_C = 0.01  # LinearSVC: 1/2 |w|^2 + C sum max(0, 1 - y w.x)^2
KSVM_C = 1.0  # SVC(kernel="rbf")
LOGREG_C = 0.1  # LogisticRegression: 1/2 |W|^2 + C sum cross-entropy  (C = 1/lambda)
KNN_K = 10
KMEANS_K = 10


# --- 1. PCA ------------------------------------------------------------------------------------------------


def pca_sklearn(X, k):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=k, svd_solver="full").fit(X)
    return pca.components_, pca.explained_variance_ratio_


def pca_tensor(X, k):
    Xc = X - X.mean(dim=0)  # centring: (N, D) - (D,) broadcasts over the rows
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)  # Xc = U diag(S) Vh
    var = S**2 / (X.shape[0] - 1)  # variance along each principal axis
    return Vh[:k], var[:k] / var.sum()  # components (k, D), explained variance ratio (k,)


# --- 2. linear SVM ------------------------------------------------------------------------------------------


def linsvm_sklearn(X, y, C):
    from sklearn.svm import LinearSVC

    svm = LinearSVC(C=C, loss="squared_hinge", tol=1e-5, max_iter=20_000).fit(X, y)  # one-vs-rest
    return np.vstack([svm.coef_.T, svm.intercept_])  # (D + 1, K)


def linsvm_tensor(X, y, C):
    Xb = torch.cat([X, torch.ones_like(X[:, :1])], dim=1)  # (N, D+1): liblinear penalises the bias as a weight
    Y = 2.0 * (y[:, None] == torch.arange(N_CLASSES, device=X.device)) - 1  # (N, K): +1 own class, -1 rest
    W = torch.zeros(Xb.shape[1], N_CLASSES, device=X.device, requires_grad=True)  # all K one-vs-rest SVMs
    opt = torch.optim.LBFGS([W], max_iter=2000, tolerance_grad=1e-7, tolerance_change=0, history_size=20,
                            line_search_fn="strong_wolfe")  # fmt: skip

    def objective():
        opt.zero_grad()
        hinge = (1 - Y * (Xb @ W)).clamp(min=0)  # (N, K)
        loss = (0.5 * (W**2).sum() + C * (hinge**2).sum()) / (C * len(X))  # scaled: same minimiser
        loss.backward()
        return loss

    opt.step(objective)
    return W.detach()


# --- 3. kernel SVM ------------------------------------------------------------------------------------------


def ksvm_sklearn(X, y, C, gamma):
    from sklearn.svm import SVC

    svm = SVC(C=C, kernel="rbf", gamma=gamma).fit(X, y)  # SMO on the dual (libsvm)
    alpha = np.zeros(len(X))
    alpha[svm.support_] = np.abs(svm.dual_coef_[0])  # dual_coef_ = y_i alpha_i on the support vectors
    return alpha, float(svm.intercept_[0])


def ksvm_tensor(X, y, C, gamma, iters=3000):
    sq = (X**2).sum(dim=1)
    K = torch.exp(-gamma * (sq[:, None] + sq[None, :] - 2 * X @ X.T).clamp(min=0))  # (N, N) by broadcasting
    Q = K * (y[:, None] * y[None, :])  # y_i y_j K_ij
    v = torch.ones_like(y)
    for _ in range(30):  # power iteration: largest eigenvalue L of Q sets the step 1/L
        v = Q @ v
        v = v / v.norm()
    L = v @ Q @ v
    a = z = torch.zeros_like(y)
    t, last = 1.0, None
    for it in range(iters):  # accelerated projected gradient ascent on the SVC dual (FISTA)
        a_new = project(z + (1 - Q @ z) / L, y, C)  # gradient of sum(a) - a.Q.a/2, then back into the box
        t_new = (1 + (1 + 4 * t * t) ** 0.5) / 2
        z = a_new + (t - 1) / t_new * (a_new - a)
        a, t = a_new, t_new
        if it % 50 == 49:
            dual = a.sum() - 0.5 * a @ Q @ a
            if last is not None and dual - last <= 1e-6 * dual.abs():
                break
            last = dual
    f = K @ (a * y)
    free = (a > 1e-3 * C) & (a < (1 - 1e-3) * C)  # margin support vectors: y_i (f_i + b) = 1
    b = (y - f)[free].mean() if free.any() else (y - f)[a > 0].mean()
    return a, b


def project(v, y, C, grid=32, rounds=5):
    """Closest point to v with 0 <= a <= C and sum(y a) = 0: a = clip(v - mu y, 0, C), mu found by a grid search."""
    lo = -(v.abs().max() + C)
    hi = -lo
    for _ in range(rounds):  # sum(y clip(v - mu y)) falls as mu grows: keep the grid cell where it crosses 0
        mu = torch.linspace(0, 1, grid + 1, device=v.device) * (hi - lo) + lo  # (G+1,)
        h = (y * (v[None, :] - mu[:, None] * y).clamp(0, C)).sum(dim=1)  # (G+1,) all candidates at once
        i = torch.searchsorted(-h, torch.zeros(1, device=v.device)).clamp(1, grid)
        lo, hi = mu[i - 1], mu[i]
    return (v - 0.5 * (lo + hi) * y).clamp(0, C)


# --- 4. logistic regression ---------------------------------------------------------------------------------


def logreg_sklearn(X, y, C):
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=C, tol=1e-6, max_iter=5000).fit(X, y)  # multinomial, lbfgs, L2 penalty
    return np.vstack([lr.coef_.T, lr.intercept_])  # (D + 1, K)


def logreg_tensor(X, y, C):
    W = torch.zeros(X.shape[1], N_CLASSES, device=X.device, requires_grad=True)
    b = torch.zeros(N_CLASSES, device=X.device, requires_grad=True)  # bias: not penalised, as in sklearn
    opt = torch.optim.LBFGS([W, b], max_iter=2000, tolerance_grad=1e-7, tolerance_change=0, history_size=20,
                            line_search_fn="strong_wolfe")  # fmt: skip

    def objective():
        opt.zero_grad()
        ce = torch.nn.functional.cross_entropy(X @ W + b, y, reduction="sum")  # softmax + log-loss
        loss = (0.5 * (W**2).sum() + C * ce) / (C * len(X))  # lambda = 1/C; scaled: same minimiser
        loss.backward()  # autograd: dJ/dW = X^T (softmax - onehot) + W / C, up to the scale
        return loss

    opt.step(objective)
    return torch.cat([W, b[None]]).detach()


# --- 5. k nearest neighbours --------------------------------------------------------------------------------


def knn_sklearn(Xtr, ytr, Xte, k):
    from sklearn.neighbors import KNeighborsClassifier

    return KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr).predict(Xte)


def knn_tensor(Xtr, ytr, Xte, k):
    d = torch.cdist(Xte, Xtr)  # (M, N) every test-train distance
    idx = d.topk(k, largest=False).indices  # (M, k) the k nearest
    votes = torch.nn.functional.one_hot(ytr[idx], N_CLASSES).sum(dim=1)  # (M, K) votes per class
    return votes.argmax(dim=1)  # ties: the smallest label, as in sklearn


# --- 6. k-means ---------------------------------------------------------------------------------------------


def kmeans_sklearn(X, centers):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=len(centers), init=centers, n_init=1, max_iter=300, tol=0, algorithm="lloyd").fit(X)
    return km.cluster_centers_, km.labels_, km.n_iter_


def kmeans_tensor(X, centers, max_iter=300):
    labels = torch.cdist(X, centers).argmin(dim=1)  # (N,) nearest centre of every point
    n_iter = 0
    while n_iter < max_iter:  # Lloyd: move every centre to its cluster mean, then reassign every point
        sums = torch.zeros_like(centers).index_add_(0, labels, X)  # (k, D) sum per cluster
        counts = torch.bincount(labels, minlength=len(centers)).clamp(min=1)
        centers = sums / counts[:, None]
        new = torch.cdist(X, centers).argmin(dim=1)
        n_iter += 1
        if torch.equal(new, labels):
            break  # no point changed cluster: converged
        labels = new
    return centers, labels, n_iter


# --- data ---------------------------------------------------------------------------------------------------


def mnist() -> dict[str, np.ndarray]:
    """2500 MNIST digits as float32 in [0, 1], split 2000 / 500 by a fixed permutation."""
    d = np.load(MNIST)
    X = d["images"].reshape(len(d["images"]), -1).astype(np.float32) / 255
    y = d["labels"].astype(np.int64)
    p = np.random.default_rng(0).permutation(len(X))
    tr, te = p[:2000], p[2000:]
    return {"Xtr": X[tr], "ytr": y[tr], "Xte": X[te], "yte": y[te]}


def rbf_gamma(X: np.ndarray) -> float:
    """sklearn's gamma="scale": 1 / (D var(X))."""
    return float(1.0 / (X.shape[1] * X.var()))


def blobs(n: int, d: int, seed: int, spread: float = 0.35) -> tuple[np.ndarray, np.ndarray]:
    """n points in d dimensions around 10 Gaussian centres (classes overlap a little)."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, spread, (N_CLASSES, d))
    y = rng.integers(0, N_CLASSES, n)
    return (centres[y] + rng.normal(0, 1, (n, d))).astype(np.float32), y.astype(np.int64)


def synthetic(case: str, n: int) -> tuple:
    """Timing inputs for `case` at size n, as numpy arrays (identical in both halves)."""
    if case == "pca":
        rng = np.random.default_rng(n)
        X = rng.normal(0, 1, (n, 16)) @ rng.normal(0, 1, (16, 128)) + 0.1 * rng.normal(0, 1, (n, 128))
        return (X.astype(np.float32), 16)
    if case == "linsvm":
        X, y = blobs(n, 64, n)
        return (X, y, SVM_C)
    if case == "ksvm":
        X, y = blobs(n, 32, n)
        yb = np.where(y % 2 == 0, 1.0, -1.0).astype(np.float32)
        return (X, yb, KSVM_C, rbf_gamma(X))
    if case == "logreg":
        X, y = blobs(n, 64, n)
        return (X, y, LOGREG_C)
    if case == "knn":
        X, y = blobs(n + 1000, 64, n)
        return (X[1000:], y[1000:], X[:1000], KNN_K)  # 1000 queries against n training points
    if case == "kmeans":
        X, _ = blobs(n, 32, n, spread=1.0)
        return (X, X[:KMEANS_K].copy())
    raise KeyError(case)


CASES: dict[str, dict] = {
    "pca": {"sizes": [2_000, 20_000, 200_000], "unit": "N"},
    "linsvm": {"sizes": [2_000, 20_000, 100_000], "unit": "N"},
    "ksvm": {"sizes": [1_000, 2_000, 5_000], "unit": "N"},
    "logreg": {"sizes": [2_000, 20_000, 100_000], "unit": "N"},
    "knn": {"sizes": [2_000, 20_000, 200_000], "unit": "N"},
    "kmeans": {"sizes": [2_000, 20_000, 200_000], "unit": "N"},
}
EXTRA_CODE = {"ksvm": ["project"]}  # helpers shown under the tensor code


def mnist_inputs(case: str, m: dict) -> tuple:
    if case == "pca":
        return (m["Xtr"], PCA_K)
    if case in ("linsvm", "logreg"):
        return (m["Xtr"], m["ytr"], SVM_C if case == "linsvm" else LOGREG_C)
    if case == "ksvm":
        return (m["Xtr"], np.where(m["ytr"] % 2 == 0, 1.0, -1.0).astype(np.float32), KSVM_C, rbf_gamma(m["Xtr"]))
    if case == "knn":
        return (m["Xtr"], m["ytr"], m["Xte"], KNN_K)
    if case == "kmeans":
        return (m["Xtr"], m["Xtr"][:KMEANS_K].copy())
    raise KeyError(case)


# --- objectives and predictions (numpy: one formula for both results) --------------------------------------


def with_bias(X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1), X.dtype)]).astype(np.float64)


def linsvm_objective(W: np.ndarray, X: np.ndarray, y: np.ndarray, C: float) -> float:
    Y = np.where(y[:, None] == np.arange(N_CLASSES), 1.0, -1.0)
    hinge = np.clip(1 - Y * (with_bias(X) @ W), 0, None)
    return float(0.5 * (W**2).sum() + C * (hinge**2).sum())


def logreg_objective(Wb: np.ndarray, X: np.ndarray, y: np.ndarray, C: float) -> float:
    Z = with_bias(X) @ Wb
    Z = Z - Z.max(axis=1, keepdims=True)
    logp = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    return float(0.5 * (Wb[:-1] ** 2).sum() - C * logp[np.arange(len(y)), y].sum())


def rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    A, B = A.astype(np.float64), B.astype(np.float64)
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.clip(d2, 0, None))


def ksvm_dual(alpha: np.ndarray, X: np.ndarray, y: np.ndarray, gamma: float) -> float:
    ay = alpha * y
    return float(alpha.sum() - 0.5 * ay @ rbf_kernel(X, X, gamma) @ ay)


def ksvm_predict(alpha, b, Xtr, ytr, Xte, gamma) -> np.ndarray:
    return np.sign(rbf_kernel(Xte, Xtr, gamma) @ (alpha * ytr) + b)


def kmeans_inertia(centers: np.ndarray, labels: np.ndarray, X: np.ndarray) -> float:
    return float(((X.astype(np.float64) - centers[labels]) ** 2).sum())


# --- timing -------------------------------------------------------------------------------------------------


def timeit(fn, args, sync=None, budget=1.5, max_reps=7) -> float:
    """Median seconds per call: one warm-up, then repeats until `budget` seconds or `max_reps`."""
    fn(*args)
    if sync:
        sync()
    times: list[float] = []
    while len(times) < max_reps and (len(times) < 3 or sum(times) < budget):
        t0 = time.perf_counter()
        fn(*args)
        if sync:
            sync()
        times.append(time.perf_counter() - t0)
        if times[-1] > budget:
            break
    return statistics.median(times)


def fns(case: str, kind: str):
    return getattr(sys.modules[__name__], f"{case}_{kind}")


# --- machine description ------------------------------------------------------------------------------------


def cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout.strip()
            if out:
                return out
        except Exception:
            pass
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def cpu_busy_percent(seconds: float = 1.0) -> float | None:
    """System-wide CPU use over `seconds` (Windows: GetSystemTimes; Linux: /proc/stat)."""
    try:
        if sys.platform == "win32":
            FT = ctypes.c_ulonglong

            def sample():
                idle, kern, user = FT(), FT(), FT()
                ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user))
                return idle.value, kern.value + user.value

            i0, t0 = sample()
            time.sleep(seconds)
            i1, t1 = sample()
            return round(100 * (1 - (i1 - i0) / max(t1 - t0, 1)), 1)
        if Path("/proc/stat").is_file():

            def sample():
                v = [int(x) for x in Path("/proc/stat").read_text().split("\n")[0].split()[1:]]
                return v[3] + v[4], sum(v)

            i0, t0 = sample()
            time.sleep(seconds)
            i1, t1 = sample()
            return round(100 * (1 - (i1 - i0) / max(t1 - t0, 1)), 1)
    except Exception:
        pass
    return None


# --- half 1: scikit-learn -----------------------------------------------------------------------------------


def run_sklearn(passes: int) -> None:
    import scipy
    import sklearn
    from threadpoolctl import threadpool_info

    m = mnist()
    outs: dict[str, np.ndarray] = {}
    mnist_s: dict[str, float] = {}
    for case in CASES:
        fn = fns(case, "sklearn")
        args = mnist_inputs(case, m)
        mnist_s[case] = timeit(fn, args)
        res = fn(*args)
        res = res if isinstance(res, tuple) else (res,)
        for i, r in enumerate(res):
            outs[f"{case}_{i}"] = np.asarray(r)
        print(f"  mnist {case:7s} {mnist_s[case]:8.3f} s", flush=True)

    load_before = cpu_busy_percent()
    timings: dict[str, list[list[float]]] = {c: [] for c in CASES}
    for p in range(passes):
        print(f"pass {p + 1}/{passes}")
        for case, spec in CASES.items():
            row = []
            for n in spec["sizes"]:
                row.append(timeit(fns(case, "sklearn"), synthetic(case, n)))
                print(f"  {case:7s} N={n:<7d} {row[-1]:9.3e} s", flush=True)
            timings[case].append(row)
    load_after = cpu_busy_percent()

    blas = [
        f"{i.get('internal_api')} {i.get('version') or ''} ({i.get('num_threads')} threads)".replace("  ", " ")
        for i in threadpool_info()
        if i.get("user_api") == "blas"
    ]
    CACHE.mkdir(exist_ok=True)
    np.savez(SK_NPZ, **outs)
    SK_JSON.write_text(
        json.dumps(
            {
                "sklearn": sklearn.__version__,
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "python": platform.python_version(),
                "blas": ", ".join(blas),
                "date": dt.date.today().isoformat(),
                "cpu_load_percent": [load_before, load_after],
                "mnist_s": mnist_s,
                "timings": timings,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"wrote {SK_JSON} and {SK_NPZ}")


# --- half 2: tensors ----------------------------------------------------------------------------------------


def to_torch(args: tuple, device: str) -> tuple:
    return tuple(torch.as_tensor(a, device=device) if isinstance(a, np.ndarray) else a for a in args)


def np_out(res) -> tuple:
    res = res if isinstance(res, tuple) else (res,)
    return tuple(r.detach().cpu().numpy() if torch.is_tensor(r) else np.asarray(r) for r in res)


def agreement(case: str, m: dict, tensor_out: tuple, sk: tuple | None) -> dict[str, float | None]:
    """Metrics of one tensor result on MNIST; with scikit-learn's result `sk`, also how close the two are."""
    if case == "pca":
        V, evr = tensor_out
        r: dict[str, float | None] = {"evr_sum": float(evr.sum()), "evr1": float(evr[0])}
        if sk:
            r["evr_maxdiff"] = float(np.abs(evr - sk[1]).max())
            r["min_cos10"] = float(np.abs((V[:10] * sk[0][:10]).sum(1)).min())
        return r
    if case in ("linsvm", "logreg"):
        obj = linsvm_objective if case == "linsvm" else logreg_objective
        C = SVM_C if case == "linsvm" else LOGREG_C
        W = tensor_out[0].astype(np.float64)
        pred = (with_bias(m["Xte"]) @ W).argmax(1)
        r = {"test_acc": float((pred == m["yte"]).mean()), "objective": obj(W, m["Xtr"], m["ytr"], C)}
        if sk:
            Ws = sk[0].astype(np.float64)
            r["pred_match"] = float((pred == (with_bias(m["Xte"]) @ Ws).argmax(1)).mean())
            r["coef_rel_diff"] = float(np.abs(W[:-1] - Ws[:-1]).max() / np.abs(Ws[:-1]).max())
        return r
    if case == "ksvm":
        alpha, b = tensor_out[0].astype(np.float64), float(tensor_out[1])
        ytr = np.where(m["ytr"] % 2 == 0, 1.0, -1.0)
        yte = np.where(m["yte"] % 2 == 0, 1.0, -1.0)
        g = rbf_gamma(m["Xtr"])
        pred = ksvm_predict(alpha, b, m["Xtr"], ytr, m["Xte"], g)
        r = {
            "test_acc": float((pred == yte).mean()),
            "dual": ksvm_dual(alpha, m["Xtr"], ytr, g),
            "n_sv": float((alpha > 1e-4 * KSVM_C).sum()),
        }
        if sk:
            r["pred_match"] = float((pred == ksvm_predict(sk[0], float(sk[1]), m["Xtr"], ytr, m["Xte"], g)).mean())
        return r
    if case == "knn":
        pred = tensor_out[0]
        r = {"test_acc": float((pred == m["yte"]).mean())}
        if sk:
            r["pred_match"] = float((pred == sk[0]).mean())
        return r
    if case == "kmeans":
        centers, labels, n_iter = tensor_out
        r = {"inertia": kmeans_inertia(centers, labels, m["Xtr"]), "iters": float(n_iter)}
        if sk:
            r["label_match"] = float((labels == sk[1]).mean())
        return r
    raise KeyError(case)


def sklearn_metrics(case: str, m: dict, sk: tuple) -> dict[str, float]:
    """The same metrics for scikit-learn's own result."""
    if case == "pca":
        return {"evr_sum": float(sk[1].sum()), "evr1": float(sk[1][0])}
    if case == "ksvm":
        r = agreement(case, m, (sk[0], sk[1]), None)
        return r
    if case == "kmeans":
        return {"inertia": kmeans_inertia(sk[0], sk[1], m["Xtr"]), "iters": float(sk[2])}
    return agreement(case, m, sk, None)


def run_tensor(passes: int, use_cuda: bool, note: str) -> None:
    cuda = use_cuda and torch.cuda.is_available()
    sync = torch.cuda.synchronize if cuda else None
    devices = ["cpu"] + (["cuda"] if cuda else [])
    sk_info = json.loads(SK_JSON.read_text(encoding="utf-8")) if SK_JSON.is_file() else None
    sk_npz = np.load(SK_NPZ) if SK_NPZ.is_file() else None
    if sk_info is None:
        print("no scikit-learn results in .cache/: run the --sklearn-only half first for the comparison")

    m = mnist()
    agree: dict[str, dict] = {}
    for case in CASES:
        sk = None
        if sk_npz is not None:
            sk = tuple(sk_npz[f"{case}_{i}"] for i in range(3) if f"{case}_{i}" in sk_npz)
        rows: dict[str, dict] = {}
        if sk:
            rows["sklearn"] = sklearn_metrics(case, m, sk) | {"seconds": sk_info["mnist_s"][case]}
        for dev in devices:
            args = to_torch(mnist_inputs(case, m), dev)
            fn = fns(case, "tensor")
            secs = timeit(fn, args, sync=sync if dev == "cuda" else None)
            rows[dev] = agreement(case, m, np_out(fn(*args)), sk) | {"seconds": secs}
        agree[case] = rows
        print(f"  mnist {case:7s} " + "  ".join(f"{k}: {v}" for k, v in rows.items()), flush=True)

    load_before = cpu_busy_percent()
    t0 = time.perf_counter()
    times: dict[str, dict[str, list[list[float]]]] = {c: {d: [] for d in devices} for c in CASES}
    for p in range(passes):
        print(f"pass {p + 1}/{passes}")
        for case, spec in CASES.items():
            for dev in devices:
                row = []
                for n in spec["sizes"]:
                    args = to_torch(synthetic(case, n), dev)
                    row.append(timeit(fns(case, "tensor"), args, sync=sync if dev == "cuda" else None))
                    print(f"  {case:7s} {dev:4s} N={n:<7d} {row[-1]:9.3e} s", flush=True)
                    del args
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                times[case][dev].append(row)
    elapsed = time.perf_counter() - t0
    load_after = cpu_busy_percent()

    cases: dict[str, list[dict]] = {}
    for case, spec in CASES.items():
        out_rows = []
        for i, n in enumerate(spec["sizes"]):
            row: dict = {"n": n}
            sk_t = [p[i] for p in sk_info["timings"][case]] if sk_info else []
            row["sklearn_s"] = statistics.median(sk_t) if sk_t else None
            for dev, key in (("cpu", "cpu"), ("cuda", "gpu")):
                t = [p[i] for p in times[case].get(dev, [])]
                row[f"tensor_{key}_s"] = statistics.median(t) if t else None
                if t and sk_t:
                    row[f"speedup_{key}"] = statistics.median(sk_t) / statistics.median(t)
                    row[f"speedup_{key}_range"] = [min(sk_t) / max(t), max(sk_t) / min(t)]
                else:
                    row[f"speedup_{key}"] = None
                    row[f"speedup_{key}_range"] = None
            out_rows.append(row)
        cases[case] = out_rows

    def code(case: str) -> dict[str, str]:
        tensor = inspect.getsource(fns(case, "tensor")).rstrip()
        for h in EXTRA_CODE.get(case, []):
            tensor += "\n\n\n" + inspect.getsource(getattr(sys.modules[__name__], h)).rstrip()
        return {"sklearn": inspect.getsource(fns(case, "sklearn")).rstrip(), "tensor": tensor}

    out = {
        "machine": {
            "cpu": cpu_name(),
            "threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if cuda else None,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "date": dt.date.today().isoformat(),
            "passes": passes,
            "seconds": round(elapsed, 1),
            "cpu_load_percent": [load_before, load_after],
            "note": note,
        },
        "sklearn": {k: v for k, v in (sk_info or {}).items() if k not in ("timings", "mnist_s")} or None,
        "settings": {
            "pca_k": PCA_K,
            "svm_C": SVM_C,
            "ksvm_C": KSVM_C,
            "logreg_C": LOGREG_C,
            "knn_k": KNN_K,
            "kmeans_k": KMEANS_K,
            "mnist": [len(m["Xtr"]), len(m["Xte"])],
        },
        "units": {c: CASES[c]["unit"] for c in CASES},
        "code": {c: code(c) for c in CASES},
        "agreement": agree,
        "cases": cases,
        "umap": UMAP_RESULTS,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({elapsed:.0f} s)")


# The UMAP case study is problems/ai/umap_tensor; its bundled run (meta.yaml notes, data/umap_learn_reference.csv):
# MNIST, 2500 digits, k = 10; wall time on CPU with one thread (umap-learn: second call, after numba compilation).
UMAP_RESULTS = {
    "rows": {
        "forces": {"trust": 0.957, "cont": 0.963, "recall": 0.39, "acc": 0.887, "seconds": 11},
        "autograd": {"trust": 0.960, "cont": 0.963, "recall": 0.40, "acc": 0.898, "seconds": 15},
        "umap_learn": {"trust": 0.963, "cont": 0.963, "recall": 0.40, "acc": 0.896, "seconds": 14},
        "pca": {"trust": 0.740, "cont": 0.913, "recall": 0.07, "acc": 0.414, "seconds": None},
    },
    "umap_learn_first_call_s": 28,
    "procrustes": 0.05,
    "versions": "umap-learn 0.5.12, scikit-learn 1.9.1",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sklearn-only", action="store_true", help="run the scikit-learn half (no torch needed)")
    ap.add_argument("--passes", type=int, default=3, help="measure everything this many times (spread)")
    ap.add_argument("--no-cuda", action="store_true", help="skip the GPU even if one is available")
    ap.add_argument("--note", default="", help="free text stored in machine.note")
    args = ap.parse_args()
    if args.sklearn_only:
        run_sklearn(args.passes)
    else:
        run_tensor(args.passes, not args.no_cuda, args.note)


if __name__ == "__main__":
    main()
