r"""The operator library as tensor operations, each with the formula the equation writer prints, and the typed
supernet: a DAG whose every edge is a softmax(α)-weighted mixture of all the operators of its family.

Typing rule (what makes every mixture well defined: all operators of one family map the same input shape to the same
output shape, so their outputs can be added):

    cell family     map → map      (B, 1, 14, 14) → (B, 1, 14, 14)   0, identity, x², |x|, ReLU, tanh, S_λ, K∗, Sobel-x,
                                                                       Sobel-y, Laplacian, Gabor
    pool family     maps → maps    (B, 4, 14, 14) → (B, 4, 7, 7)     avgpool₂, maxpool₂
    reduce family   maps → vector  (B, 4, 7, 7) → (B, D_r)           vec, rowsum, colsum, |F|₄, ⟨·, μ_k⟩, Uᵀvec, M_pq
    head family     vector → logits (B, D_r) → (B, 10)               linear W f̂ + b, prototypes −e^τ‖f̂ − p_k‖²/D

A reduction's output length D_r differs between reductions, so each reduction carries its own heads and the mixture
is taken over the logits: ℓ = Σ_r a_r Σ_h b_h head_{r,h}(f̂_r). The cell is a DARTS cell: nodes z_1, z_2, z_3, each
the sum of the mixed edges from every earlier node (x = z_0 is the image); its output is the stack z = (x, z_1, z_2,
z_3): DARTS stacks only z_1 … z_3; the image itself is kept as a fourth channel (a skip connection), so the
pixels always reach the read-out.

    mixed edge     ō_e(z) = Σ_o softmax(α_e)_o o(z)          (masked softmax: α_eo = −∞ outside the edge's family)
    node           z_i = Σ_{j<i} ō_{(j,i)}(z_j)
    standardise    f̂ = (f − m) / s                            (m, s: feature mean and std over the training images)
    loss           L = −(1/B) Σ_b log softmax(ℓ_b)_{y_b}
"""

import functools
import math

import torch
import torch.nn.functional as F

from . import tensors as T

CELL_OPS = ("zero", "identity", "square", "abs", "relu", "tanh", "shrink", "conv3", "sobel_x", "sobel_y", "laplace",
            "gabor")  # fmt: skip
POOL_OPS = ("avgpool", "maxpool")
RED_OPS = ("vec", "rowsum", "colsum", "fft", "template", "pca", "moments")
HEAD_OPS = ("linear", "proto")
OPS = CELL_OPS + POOL_OPS + RED_OPS + HEAD_OPS  # the 23 columns of α
FIXED = ("sobel_x", "sobel_y", "laplace", "gabor")
N_NODES = 3
CHANNELS = N_NODES + 1  # the cell output (x, z_1, z_2, z_3)
CELL_EDGES = tuple((j, i) for i in range(1, N_NODES + 1) for j in range(i))  # (0,1) (0,2) (1,2) (0,3) (1,3) (2,3)
EDGES = tuple(f"z{j}→z{i}" for j, i in CELL_EDGES) + ("pool", "reduce", "head")  # the 9 rows of α
SHORT = {"zero": "0", "identity": "id", "square": "x²", "abs": "|x|", "relu": "ReLU", "tanh": "tanh",
         "shrink": "soft thresh.",
         "conv3": "K∗ (3×3)", "sobel_x": "Sobel-x", "sobel_y": "Sobel-y", "laplace": "Laplacian", "gabor": "Gabor",
         "avgpool": "avg-pool 2", "maxpool": "max-pool 2", "vec": "vec", "rowsum": "row sums", "colsum": "col sums",
         "fft": "|FFT|", "template": "⟨z, μ_k⟩", "pca": "PCA", "moments": "moments", "linear": "linear",
         "proto": "prototypes"}  # fmt: skip

# The formula of every operator, written into the equation: {a} is the argument, {e} the edge number (filters K_e).
LATEX = {
    "zero": "0",
    "identity": "{a}",
    "square": "({a})^{{2}}",
    "abs": "|{a}|",
    "relu": r"\mathrm{{ReLU}}({a})",
    "tanh": r"\tanh({a})",
    "shrink": r"S_{{\lambda}}({a})",
    "conv3": r"K_{{{e}}} \ast {a}",
    "sobel_x": r"S_{{x}} \ast {a}",
    "sobel_y": r"S_{{y}} \ast {a}",
    "laplace": r"\Delta \ast {a}",
    "gabor": r"G \ast {a}",
    "avgpool": r"\mathrm{{avgpool}}_{{2}}({a})",
    "maxpool": r"\mathrm{{maxpool}}_{{2}}({a})",
    "vec": r"\mathrm{{vec}}({a})",
    "rowsum": r"\mathrm{{rowsum}}({a})",
    "colsum": r"\mathrm{{colsum}}({a})",
    "fft": r"|\mathcal{{F}}({a})|_{{4}}",
    "template": r"\langle {a}, \mu_{{k}} \rangle",
    "pca": r"U^{{\top}} \mathrm{{vec}}({a})",
    "moments": r"M_{{pq}}({a})",
}


def family_mask() -> torch.Tensor:
    """M (9, 23) bool: M_eo = operator o may sit on edge e (the typing rule)."""
    M = torch.zeros(len(EDGES), len(OPS), dtype=torch.bool)
    for e, name in enumerate(EDGES):
        fam = {"pool": POOL_OPS, "reduce": RED_OPS, "head": HEAD_OPS}.get(name, CELL_OPS)
        M[e, [OPS.index(o) for o in fam]] = True
    return M


def mixture_weights(alpha: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """softmax over each edge's own family: w_eo = exp α_eo / Σ_{o' ∈ family(e)} exp α_eo', 0 outside it. (9, 23)."""
    return torch.softmax(alpha.masked_fill(~mask, float("-inf")), dim=1)


# ---------------------------------------------------------------- cell operators: map (B, 1, H, W) → map
def cell_op(name: str, z: torch.Tensor, K: torch.Tensor | None = None) -> torch.Tensor:
    """One cell operator applied to z (B, 1, H, W); K (3, 3) is the edge's learned filter (conv3 only).
    ∗ is the 2D cross-correlation with zero padding (what CNN layers compute)."""
    if name == "zero":
        return torch.zeros_like(z)
    if name == "identity":
        return z
    if name == "square":
        return z * z
    if name == "abs":
        return z.abs()
    if name == "relu":
        return z.clamp(min=0)
    if name == "tanh":
        return torch.tanh(z)
    if name == "shrink":  # soft threshold S_λ(z) = sign(z) max(|z| − λ, 0)
        return torch.sign(z) * (z.abs() - T.LAMBDA).clamp(min=0)
    if name == "conv3":
        return F.conv2d(z, K[None, None], padding=1)
    k = _bank(z.device, z.dtype)[FIXED.index(name)]
    return F.conv2d(z, k[None, None], padding=2)


@functools.lru_cache(maxsize=8)
def _bank(dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return T.fixed_kernels().to(dev, dtype)


@functools.lru_cache(maxsize=8)
def _moments(h: int, dev: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return T.moment_basis(h).to(dev, dtype)


def _pattern(w: torch.Tensor) -> list[list[str]]:
    """Per entry of w: "0" (skip), "1" (use as is) or "w" (multiply); read once on the CPU (no per-op GPU sync)."""
    wc = w.detach().cpu()
    return [["0" if v == 0 else "1" if v == 1 else "w" for v in row] for row in wc.tolist()]


def mixed_edge(z: torch.Tensor, w: torch.Tensor, K: torch.Tensor, pat: list[str] | None = None) -> torch.Tensor:
    """ō(z) = Σ_o w_o o(z) over the cell family; w (23,) one row of the mixture weights. Operators with w_o = 0 are not
    evaluated (a one-hot row is then exactly that one operator)."""
    pat = pat if pat is not None else _pattern(w[None])[0]
    out = None
    for i, name in enumerate(CELL_OPS):
        if pat[i] != "0" and name != "zero":
            term = cell_op(name, z, K) if pat[i] == "1" else w[i] * cell_op(name, z, K)
            out = term if out is None else out + term
    return torch.zeros_like(z) if out is None else out


ELEMENTWISE = ("identity", "square", "abs", "relu", "tanh", "shrink")


def cell(x: torch.Tensor, w: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """z_0 = x; z_i = Σ_{j<i} ō_(j,i)(z_j); returns the stack (x, z_1, z_2, z_3), (B, 4, H, W). w (9, 23), K (6, 3, 3).

    A discrete DAG (every weight 0 or 1) is evaluated edge by edge with cell_op, exactly as the written equation is.
    A mixture is batched per source node j: ONE convolution with the stacked kernels gives all four fixed filters and
    the learned 3×3 filters (zero-padded to 5×5) of all edges leaving j; with the six element-wise operators that is
    the stack S (B, 10, HW) of parameter-free outputs, and ō_e = Σ_o w_eo S_o + w_e,conv K_e ∗ z_j for all those
    edges in one einsum."""
    pat = _pattern(w)
    if all(p in ("0", "1") for row in pat[: len(CELL_EDGES)] for p in row):
        z = [x]
        for i in range(1, N_NODES + 1):
            node = None
            for e, (j, ii) in enumerate(CELL_EDGES):
                op = next((CELL_OPS[o] for o in range(1, len(CELL_OPS)) if pat[e][o] == "1"), None)
                if ii == i and op is not None:
                    t = cell_op(op, z[j], K[e])
                    node = t if node is None else node + t
            z.append(torch.zeros_like(x) if node is None else node)
        return torch.cat(z, dim=1)

    B, _, H, W_ = x.shape
    bank = _bank(x.device, x.dtype)  # (4, 5, 5)
    cols = [OPS.index(n) for n in ELEMENTWISE + FIXED]
    ic = OPS.index("conv3")
    z, contrib = [x], {}
    for j in range(N_NODES):
        out_e = [e for e, (jj, _) in enumerate(CELL_EDGES) if jj == j]
        Kp = F.pad(K[out_e], (1, 1, 1, 1))  # (E_j, 5, 5): 3×3 filters centred in 5×5
        R = F.conv2d(z[j], torch.cat([bank, Kp])[:, None], padding=2).flatten(2)  # (B, 4 + E_j, HW)
        zf = z[j].flatten(1)[:, None]  # (B, 1, HW)
        S = torch.cat([cell_op(n, zf) for n in ELEMENTWISE] + [R[:, :4]], 1)  # (B, 10, HW)
        mix = torch.einsum("bol,eo->bel", S, w[out_e][:, cols]) + R[:, 4:] * w[out_e, ic][None, :, None]
        for k, e in enumerate(out_e):
            contrib[e] = mix[:, k].view(B, 1, H, W_)
        z.append(sum(contrib[e] for e, (_, i) in enumerate(CELL_EDGES) if i == j + 1))
    return torch.cat(z, dim=1)


# ---------------------------------------------------------------- pool family: (B, C, H, W) → (B, C, H/2, W/2)
def pool_op(name: str, z: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(z, 2) if name == "avgpool" else F.max_pool2d(z, 2)


# ---------------------------------------------------------------- reduce family: (B, C, h, h) → (B, D_r)
def reduce_op(name: str, Z: torch.Tensor, stats: dict | None = None) -> torch.Tensor:
    """vec: all pixels; rowsum_ch = Σ_w Z_chw; colsum_cw = Σ_h Z_chw; |F(Z)|_cuv = |Σ_hw Z_chw e^{−2πi(uh+vw)/h}|/h for
    u, v < 4; template ⟨Z_c, μ_kc⟩ = Σ_hw Z_chw μ_kchw (μ_k the class means of Z over the training images);
    PCA (Uᵀ vec Z)_cq = Σ_p U_cpq Z_cp (U_c the top principal axes of channel c over the training images);
    moments M_pq(Z)_c = mean_hw u^p v^q Z_chw for (p, q) ∈ {00, 10, 01, 20, 02, 11}, u, v ∈ [−1, 1]."""
    if name == "vec":
        return Z.flatten(1)
    if name == "rowsum":
        return Z.sum(3).flatten(1)
    if name == "colsum":
        return Z.sum(2).flatten(1)
    if name == "fft":
        return torch.fft.fft2(Z, norm="ortho").abs()[..., : T.FFT_KEEP, : T.FFT_KEEP].flatten(1)
    if name == "template":
        return torch.einsum("bchw,kchw->bck", Z, stats["mu"]).flatten(1)
    if name == "pca":
        return torch.einsum("bcp,cpq->bcq", Z.flatten(2), stats["U"]).flatten(1)
    if name == "moments":
        return torch.einsum("bchw,phw->bcp", Z, _moments(Z.shape[-1], Z.device, Z.dtype)).flatten(1) / Z[0, 0].numel()
    raise ValueError(name)


def reduce_dims(channels: int = CHANNELS, h: int = 7) -> dict:
    """D_r, the feature length of each reduction for C channels of h × h."""
    return {"vec": channels * h * h, "rowsum": channels * h, "colsum": channels * h,
            "fft": channels * T.FFT_KEEP**2, "template": channels * T.CLASSES, "pca": channels * T.Q_PCA,
            "moments": channels * 6}  # fmt: skip


def class_means(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """μ_k = mean of the rows of X with label k, as one product with the one-hot labels: (N, ...) → (10, ...)."""
    Y = F.one_hot(y, T.CLASSES).to(X.dtype)  # (N, 10)
    mu = Y.T @ X.flatten(1) / Y.sum(0)[:, None].clamp(min=1)
    return mu.view(T.CLASSES, *X.shape[1:])


def data_statistics(Z: torch.Tensor, y: torch.Tensor, reds: list[str]) -> dict:
    """The data-derived constants (no gradient) from the pooled maps Z (N, C, h, h) of the training images: class
    means μ (10, C, h, h), principal axes U (C, h², q) with a fixed sign (largest entry positive), the feature mean m_r
    (D_r,) and ONE scale per channel (the RMS deviation over that channel's features, repeated over its block: a
    per-feature std from a few images would blow up pixels that happen to be blank), and the class means
    c_r (10, D_r) of f̂_r = (f_r − m_r)/s_r."""
    Z = Z.detach()
    st = {"mu": class_means(Z, y)}
    Zc = Z.flatten(2) - Z.flatten(2).mean(0)  # (N, C, P)
    _, _, Vh = torch.linalg.svd(Zc.transpose(0, 1), full_matrices=False)  # (C, min(N,P), P)
    U = Vh[:, : T.Q_PCA].transpose(1, 2)  # (C, P, q')
    if U.shape[2] < T.Q_PCA:  # fewer images than components: pad with zero axes
        U = torch.cat([U, U.new_zeros(U.shape[0], U.shape[1], T.Q_PCA - U.shape[2])], 2)
    sign = torch.sign(torch.gather(U, 1, U.abs().argmax(1, keepdim=True)))
    st["U"] = U * torch.where(sign == 0, torch.ones_like(sign), sign)
    for r in reds:
        f = reduce_op(r, Z, st)
        m = f.mean(0)
        C = Z.shape[1]
        s = ((f - m) ** 2).view(len(f), C, -1).mean((0, 2)).sqrt().clamp(min=1e-6)  # (C,) one scale per channel
        s = s.repeat_interleave(f.shape[1] // C)  # (D,) constant on each channel's block of features
        st[f"m_{r}"], st[f"s_{r}"] = m, s
        st[f"c_{r}"] = class_means((f - m) / s, y)
    return st


def init_heads(th: dict, st: dict, reds: list[str], scale: float = 3.0) -> None:
    """Start every head at the nearest class mean in its feature space (the parameter-free baseline): prototypes
    p_k = c_k, e^τ = scale; linear W_k = 2 scale c_k / D, b_k = −scale ‖c_k‖² / D (the same decisions and logits up to
    a k-independent term). Training then learns a correction to it."""
    with torch.no_grad():
        for r in reds:
            c = st[f"c_{r}"]
            D = c.shape[1]
            th[f"P_{r}"].copy_(c)
            th[f"tau_{r}"].fill_(math.log(scale))
            th[f"W_{r}"].copy_(2 * scale * c / D)
            th[f"b_{r}"].copy_(-scale * (c**2).sum(1) / D)


# ---------------------------------------------------------------- head family: f̂ (B, D) → logits (B, 10)
def head_op(name: str, fh: torch.Tensor, th: dict, r: str) -> torch.Tensor:
    """linear: ℓ_k = Σ_j W_kj f̂_j + b_k;  proto: ℓ_k = −e^τ ‖f̂ − p_k‖² / D (argmax ℓ = nearest prototype)."""
    if name == "linear":
        return torch.einsum("bj,kj->bk", fh, th[f"W_{r}"]) + th[f"b_{r}"]
    d2 = ((fh[:, None, :] - th[f"P_{r}"][None]) ** 2).sum(-1)
    return -torch.exp(th[f"tau_{r}"]) * d2 / fh.shape[1]


def pooled(x: torch.Tensor, w: torch.Tensor, th: dict) -> torch.Tensor:
    """The mixed cell and the mixed pool: (B, 1, 14, 14) → (B, 4, 7, 7)."""
    z = cell(x, w, th["K"])
    e = EDGES.index("pool")
    pat = _pattern(w[e : e + 1])[0]
    out = None
    for name in POOL_OPS:
        o = OPS.index(name)
        if pat[o] != "0":
            t = pool_op(name, z) if pat[o] == "1" else w[e, o] * pool_op(name, z)
            out = t if out is None else out + t
    return out


def logits(Z: torch.Tensor, w: torch.Tensor, th: dict, st: dict) -> torch.Tensor:
    """ℓ = Σ_r a_r Σ_h b_h head_{r,h}(f̂_r), f̂_r = (reduce_r(Z) − m_r)/s_r; reductions with a_r = 0 are skipped.
    The mixture is batched over the reductions: the f̂_r are concatenated (B, ΣD_r), each block scaled by its a_r,
    so all linear heads are one product and all prototype heads one weighted sum of squared differences."""
    er, eh = EDGES.index("reduce"), EDGES.index("head")
    pat = _pattern(w)
    reds = [r for r in RED_OPS if pat[er][OPS.index(r)] != "0"]
    heads = [h for h in HEAD_OPS if pat[eh][OPS.index(h)] != "0"]
    fh = [(reduce_op(r, Z, st) - st[f"m_{r}"]) / st[f"s_{r}"] for r in reds]
    if len(reds) == len(heads) == 1 and pat[er][OPS.index(reds[0])] == pat[eh][OPS.index(heads[0])] == "1":
        return head_op(heads[0], fh[0], th, reds[0])  # one discrete DAG
    D = torch.tensor([f.shape[1] for f in fh], device=Z.device)
    a = w[er, [OPS.index(r) for r in reds]]  # (R,)
    f_all = torch.cat(fh, 1)  # (B, ΣD)
    out = 0.0
    if "linear" in heads:
        W_all = torch.cat([th[f"W_{r}"] for r in reds], 1)  # (10, ΣD)
        b_all = torch.stack([th[f"b_{r}"] for r in reds])  # (R, 10)
        lin = torch.einsum("bj,kj->bk", f_all * a.repeat_interleave(D), W_all) + a @ b_all
        out = out + w[eh, OPS.index("linear")] * lin
    if "proto" in heads:
        P_all = torch.cat([th[f"P_{r}"] for r in reds], 1)
        tau = torch.stack([th[f"tau_{r}"] for r in reds])
        coef = (a * torch.exp(tau) / D).repeat_interleave(D)  # (ΣD,) a_r e^{τ_r} / D_r on block r
        d2 = (f_all[:, None, :] - P_all[None]) ** 2  # (B, 10, ΣD)
        out = out - w[eh, OPS.index("proto")] * torch.einsum("bkj,j->bk", d2, coef)
    return out


def forward(x: torch.Tensor, w: torch.Tensor, th: dict, st: dict) -> torch.Tensor:
    """The whole supernet (or, with one-hot rows of w, one discrete DAG): images (B, 1, 14, 14) → logits (B, 10)."""
    return logits(pooled(x, w, th), w, th, st)


# ---------------------------------------------------------------- the CNN and the parameter-free baselines
def cnn(x: torch.Tensor, p: dict) -> torch.Tensor:
    """conv 3×3 → ReLU → max-pool 2 → conv 3×3 → ReLU → max-pool 2 → linear: (B, 1, 14, 14) → (B, 10)."""
    h = F.max_pool2d(F.relu(F.conv2d(x, p["W1"], p["b1"], padding=1)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, p["W2"], p["b2"], padding=1)), 2)
    return h.flatten(1) @ p["W3"].T + p["b3"]


def nearest_class_mean(x_train: torch.Tensor, y_train: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """ŷ = argmin_k ‖x − μ_k‖², μ_k the mean training image of digit k (no learned parameter)."""
    X, Q = x_train.flatten(1), x.flatten(1)
    mu = class_means(X, y_train)  # (10, 196)
    return torch.cdist(Q, mu).argmin(1)


def pca_lda(x_train: torch.Tensor, y_train: torch.Tensor, x: torch.Tensor, q: int = 40, shrink: float = 0.1):
    """Closed form: PCA to q ≤ N − 1 axes, then LDA with a shrunk within-class covariance
    Σ = (1 − γ) S_w + γ tr(S_w)/q I (γ = 0.1; with one image per class S_w = 0 → Σ = I, nearest class mean in PCA
    space). δ_k(x) = μ_kᵀ Σ⁻¹ x − ½ μ_kᵀ Σ⁻¹ μ_k, equal priors."""
    X = x_train.flatten(1).double()
    mean = X.mean(0)
    _, _, Vh = torch.linalg.svd(X - mean, full_matrices=False)
    P = Vh[: min(q, len(X) - 1)].T  # (784, q)
    Y, Qx = (X - mean) @ P, (x.flatten(1).double() - mean) @ P
    mu = class_means(Y, y_train)
    R = Y - mu[y_train]
    Sw = R.T @ R / len(Y)
    tr = float(torch.trace(Sw)) / len(Sw)
    Sig = (1 - shrink) * Sw + (shrink * tr if tr > 0 else 1.0) * torch.eye(len(Sw), dtype=Sw.dtype, device=Sw.device)
    A = torch.linalg.solve(Sig, mu.T)  # Σ⁻¹ μ_k as columns (q, 10)
    return (Qx @ A - 0.5 * (mu * A.T).sum(1)).argmax(1)
