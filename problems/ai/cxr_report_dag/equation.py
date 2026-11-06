r"""A chest X-ray CNN and a tiny report LM stitched by one DAG edge, the same edge as a mixture-of-experts layer, and
the metrics (label F1 of generated reports, AUROC, BLEU-1 and ROUGE-L of captions).

Indices: m = model on the stacked axis (seed), n = example, t = input position, c = LM channel (d), f = finding
(six), h = hidden channel, v = vocabulary letter, x = expert (img, txt).

    q_{mnf} = \sigma(\mathrm{CNN}_m(\mathrm{image}_{mn}))_f                      (M, N, 6) finding probabilities
    c_{mntc} = \mathrm{GELU}(q_{mnf} W^1_{mfh} + S_{mth}) W^2_{mhc} + b^2_{mc}   (f, h summed) connector, slot t < K
    X^0_{mntc} = c_{mntc} for t < K,   \mathrm{Tok}_{m, t_{mnt}, c} for t ≥ K       the DAG edge: a soft label header
    z_{mntv} = \mathrm{LM}_m(X^0 + P)                                          next-letter logits

As an MoE layer at the LM input: X^0_{mntc} = G_{mntx} Y^x_{mntc} (x summed) with the fixed router
G_{mnt,img} = [t < K], or a learned router p = softmax(r W^R + b^R) trained dense and used top-1.

The LM blocks come from problems/ai/chain_of_thought; the token lookup, connector, LM read-out, routers, dispatch and
masked cross-entropy are imported from problems/ai/stitch_llm_cnn_dag (the same functions, this problem's shapes).
"""

import torch
import torch.nn.functional as F

from problems.ai.chain_of_thought.equation import layer_norm, mlp
from problems.ai.stitch_llm_cnn_dag.equation import (  # noqa: F401
    balance_loss,
    connector,
    dispatch,
    embed,
    lm_logits,
    masked_ce,
    route_fixed,
    route_learned,
)

from . import tensors as T


# ---------------------------------------------------------------- the CNN (one grouped convolution per layer)
def cnn_features(x: torch.Tensor, w: dict) -> torch.Tensor:
    """Images x (M, N, 64, 64) → features (M, N, E); the model axis becomes conv groups, so models never mix."""
    m, n = x.shape[:2]
    z = x.transpose(0, 1)  # (N, M, 64, 64): one input channel per model
    z = F.max_pool2d(F.relu(F.conv2d(z, w["c1"].flatten(0, 1), w["b1"].flatten(), groups=m)), 2)  # (N, 8M, 30, 30)
    z = F.max_pool2d(F.relu(F.conv2d(z, w["c2"].flatten(0, 1), w["b2"].flatten(), groups=m)), 2)  # (N, 16M, 14, 14)
    z = F.max_pool2d(F.relu(F.conv2d(z, w["c3"].flatten(0, 1), w["b3"].flatten(), groups=m)), 2)  # (N, 16M, 6, 6)
    z = z.reshape(n, m, 576).transpose(0, 1)  # (M, N, 576)
    return F.relu(torch.einsum("mnk,mke->mne", z, w["fc"]) + w["bf"][:, None])


def cnn_logits(h: torch.Tensor, w: dict) -> torch.Tensor:
    """Finding logits (M, N, 6) from the features h (M, N, E)."""
    return torch.einsum("mne,mef->mnf", h, w["head"]) + w["bh"][:, None]


def cnn_output(x: torch.Tensor, w: dict) -> torch.Tensor:
    """q = σ(CNN logits): six independent finding probabilities, what the CNN passes along the DAG edge."""
    return torch.sigmoid(cnn_logits(cnn_features(x, w), w))


def bce(logits: torch.Tensor, y: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    """Multi-label loss per model (M,): binary cross-entropy, positives weighted by (negatives / positives)."""
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none").mean((1, 2))


# ---------------------------------------------------------------- the DAG-wired model
def soft_prefix(x: torch.Tensor, w_cnn: dict, w_con: dict) -> torch.Tensor:
    """CNN → connector: the K soft tokens of each image.  x (M, N, 64, 64) → (M, N, K, d)."""
    m, n = x.shape[:2]
    q = cnn_output(x, w_cnn)  # (M, N, 6)
    qk = q[:, :, None].expand(m, n, T.K, q.shape[-1]).reshape(m, n * T.K, -1)
    slots = torch.arange(T.K, device=x.device).repeat(n)[None].expand(m, -1)  # (M, N·K) slot of every row
    return connector(qk, slots, w_con).reshape(m, n, T.K, -1)


def stitched_inputs(x: torch.Tensor, tokens: torch.Tensor, w_cnn: dict, w_con: dict, w_lm: dict) -> torch.Tensor:
    """The DAG edge CNN → connector → LM: K soft tokens from the image replace the label header.
    x (M, N, 64, 64), tokens (M, N, T) → X0 (M, N, T, d)."""
    return torch.cat([soft_prefix(x, w_cnn, w_con), embed(tokens[:, :, T.K :], w_lm)], dim=2)


def stitched_logits(x, tokens, w_cnn, w_con, w_lm) -> torch.Tensor:
    return lm_logits(stitched_inputs(x, tokens, w_cnn, w_con, w_lm), w_lm)


# ---------------------------------------------------------------- the same edge as an MoE layer
def position_items(tokens: torch.Tensor):
    """At each input position an id (the IMG placeholder at the K header slots, else the letter) and whether it
    shows the image.  → ids (M, N, T), is_img (T,) bool."""
    is_img = torch.arange(tokens.shape[2], device=tokens.device) < T.K
    return torch.where(is_img, torch.full_like(tokens, T.STOI[T.IMG]), tokens), is_img


def router_features(x: torch.Tensor, ids: torch.Tensor, is_img: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
    """r_{mnt} = 8×8 average-pooled image of the position (a blank page at text positions) ⊕ one-hot id."""
    m, n, t = ids.shape
    pooled = F.avg_pool2d(x.reshape(m * n, 1, T.IMG_SIZE, T.IMG_SIZE), 8).reshape(m, n, 1, 64)
    pooled_blank = F.avg_pool2d(blank[None, None], 8).reshape(1, 1, 1, 64)
    img = torch.where(is_img[None, None, :, None], pooled, pooled_blank)
    return torch.cat([img, F.one_hot(ids, len(T.VOCAB)).to(x.dtype)], dim=-1)


def moe_inputs(x, ids, G, w_cnn, w_con, w_lm, blank, q_img=None) -> torch.Tensor:
    """X0_{mntc} = G_{mntx} Y^x_{mntc}: dispatch every position to its expert, run each expert batched on its own
    group (CNN once per image and once on a blank page, connector per routed position; token lookup), scatter back,
    combine. q_img: the CNN outputs if already computed (greedy decoding reuses them at every letter)."""
    m, n, t = ids.shape
    flat_ids = ids.reshape(m, n * t)
    q_img = cnn_output(x, w_cnn) if q_img is None else q_img  # (M, N, 6)
    q_blank = cnn_output(blank.expand(m, 1, T.IMG_SIZE, T.IMG_SIZE), w_cnn)  # (M, 1, 6)
    is_img = torch.arange(t, device=x.device) < T.K
    Y = []
    for x_e in range(2):
        idx, valid = dispatch(G[..., x_e].reshape(m, n * t) > 0)
        if x_e == 0:  # E_img = CNN + connector at slot t of example n (a text position sees a blank page)
            ex, slot = idx // t, idx % t
            q = torch.where(is_img[slot][..., None], q_img.gather(1, ex[..., None].expand(-1, -1, 6)), q_blank)
            out = connector(q, slot.clamp(max=T.K - 1), w_con)
        else:  # E_txt = the LM's own token embedding
            model = torch.arange(m, device=x.device)[:, None]
            out = w_lm["tok"][model, flat_ids.gather(1, idx)]
        out = out * valid[..., None]
        Y.append(torch.zeros(m, n * t, out.shape[-1], dtype=out.dtype, device=x.device).scatter_add(
            1, idx[..., None].expand_as(out), out).reshape(m, n, t, -1))  # fmt: skip
    return torch.einsum("mntx,xmntc->mntc", G, torch.stack(Y))


# ---------------------------------------------------------------- greedy decoding with a key/value cache
def lm_step(X_new: torch.Tensor, w: dict, cache: list, pos: int) -> torch.Tensor:
    r"""One new input vector per sequence through the LM, attending to the cached keys and values of the earlier
    positions (the same numbers lm_logits computes for that position, without recomputing the past).
    X_new (M, N, 1, d), cache[layer] = (K, V) each (M, N, H, pos, d_k) or None → logits (M, N, 1, V).
    S_{ij} = Q_{ik}K_{jk}/\sqrt{d_k} over j ≤ pos (k summed): the causal mask is automatic, the future is not there."""
    X = X_new + w["pos"][:, None, pos : pos + 1]
    for li, p in enumerate(w["layers"]):
        h = layer_norm(X, p["g1"], p["b1"])
        q = torch.einsum("mnic,mhck->mnhik", h, p["Wq"]) + p["bq"][:, None, :, None, :]
        k = torch.einsum("mnic,mhck->mnhik", h, p["Wk"]) + p["bk"][:, None, :, None, :]
        v = torch.einsum("mnic,mhck->mnhik", h, p["Wv"]) + p["bv"][:, None, :, None, :]
        if cache[li] is not None:
            k, v = torch.cat([cache[li][0], k], 3), torch.cat([cache[li][1], v], 3)
        cache[li] = (k, v)
        A = torch.softmax(torch.einsum("mnhik,mnhjk->mnhij", q, k) / q.shape[-1] ** 0.5, -1)
        mix = torch.einsum("mnhij,mnhjk->mnhik", A, v)
        X = X + torch.einsum("mnhik,mhkc->mnic", mix, p["Wo"]) + p["bo"][:, None, None]
        X = X + mlp(layer_norm(X, p["g2"], p["b2"]), p)
    return torch.einsum("mntc,mcv->mntv", layer_norm(X, w["gf"], w["bf"]), w["head"]) + w["bias"][:, None, None]


@torch.no_grad()
def greedy_cached(X_prompt: torch.Tensor, w: dict, n_new: int) -> torch.Tensor:
    """Feed the prompt vectors (M, N, P, d) one by one, then write n_new letters, each the argmax given what is
    written so far; padding after EOS; stop when every row has written EOS.  → ids (M, N, n_new)."""
    cache = [None] * len(w["layers"])
    for t in range(X_prompt.shape[2]):
        z = lm_step(X_prompt[:, :, t : t + 1], w, cache, t)
    m, n = X_prompt.shape[:2]
    out = torch.full((m, n, n_new), T.STOI[T.PAD], dtype=torch.long, device=X_prompt.device)
    done = torch.zeros(m, n, dtype=torch.bool, device=X_prompt.device)
    for k in range(n_new):
        nxt = z[:, :, 0].argmax(-1)
        out[..., k] = torch.where(done, torch.full_like(nxt, T.STOI[T.PAD]), nxt)
        done |= nxt == T.STOI[T.EOS]
        if bool(done.all()) or k == n_new - 1:
            break
        z = lm_step(embed(nxt[..., None], w), w, cache, X_prompt.shape[2] + k)
    return out


# ---------------------------------------------------------------- metrics
def f1(pred: torch.Tensor, true: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Label-level F1 per model: pred (M, N, F) bool, true (N, F) bool → micro F1 (M,), macro F1 (M,).
    tp_{mf} = Σ_n pred·true (n summed); micro pools the counts over f, macro averages the per-finding F1."""
    p, y = pred.float(), true.float()[None]
    tp = torch.einsum("mnf,mnf->mf", p, y.expand_as(p))
    fp, fn = p.sum(1) - tp, y.sum(1) - tp
    per = 2 * tp / (2 * tp + fp + fn).clamp(min=1e-9)
    micro = 2 * tp.sum(1) / (2 * tp.sum(1) + fp.sum(1) + fn.sum(1)).clamp(min=1e-9)
    return micro, per.mean(1)


def auroc(score: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Area under the ROC curve = P(score of a random positive > score of a random negative), ties count ½."""
    pos, neg = score[y > 0], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return torch.tensor(float("nan"))
    d = pos[:, None] - neg[None, :]
    return ((d > 0).float() + 0.5 * (d == 0).float()).mean()


def bleu1(hyp: str, ref: str) -> float:
    """Clipped unigram precision × brevity penalty (word level, one reference)."""
    h, r = hyp.split(), ref.split()
    if not h:
        return 0.0
    match = sum(min(h.count(w), r.count(w)) for w in set(h))
    bp = 1.0 if len(h) > len(r) else torch.exp(torch.tensor(1 - len(r) / len(h))).item()
    return bp * match / len(h)


def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 = 2·LCS / (|hyp| + |ref|) on words; the LCS by the usual dynamic programme over the two strings."""
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    row = [0] * (len(r) + 1)
    for a in h:
        prev = 0
        for j, b in enumerate(r, 1):
            prev, row[j] = row[j], prev + 1 if a == b else max(row[j], row[j - 1])
    return 2 * row[-1] / (len(h) + len(r))
