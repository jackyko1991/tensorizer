r"""Two separately trained networks stitched by one DAG edge, and the same edge written as a mixture-of-experts layer.

Indices: m = model on the stacked axis (seed, joint-set size), n = example, t = input position, c = LM channel (d),
e = digit class of the CNN output, f = hidden channel, v = vocabulary letter, x = expert (img, txt).

    q_{mne} = \mathrm{softmax}_e\,\mathrm{CNN}_m(\mathrm{image}_{mn})          (M, N, 10) digit probabilities
    c_{mntc} = \mathrm{GELU}(q_{mne} W^1_{mef} + S_{mtf}) W^2_{mfc} + b^2_{mc}     (e, f summed) connector, slot t
    X^0_{mntc} = c_{mntc} for t < K,   \mathrm{Tok}_{m, t_{mnt}, c} for t ≥ K       the DAG edge: soft prefix
    z_{mntv} = \mathrm{LM}_m(X^0 + P)                                          next-letter logits

As an MoE layer at the LM input (x = img, txt; G the router, one-hot):

    Y^{img}_{mntc} = c_{mntc},  Y^{txt}_{mntc} = \mathrm{Tok}_{m, t_{mnt}, c}       each expert run on its group only
    X^0_{mntc} = G_{mntx} Y^x_{mntc}                                           (x summed) the combine
    fixed router:   G_{mnt,img} = [t < K]        (known from the data type, like a workflow rule chosen by wildcard)
    learned router: p = \mathrm{softmax}(r_{mnt} W^R + b^R),  G = p ⊙ \mathrm{onehot}(\arg\max p)   (Switch top-1)
                    or trained with the dense G = p (both experts everywhere), then used top-1
    balance loss:   \mathcal L_{bal} = X \sum_x f_x P_x,   f_x = share of positions sent to x, P_x = mean p_x

The LM blocks (pre-norm attention and MLP with a model axis) are imported from problems/ai/chain_of_thought.
"""

import torch
import torch.nn.functional as F

from problems.ai.chain_of_thought.equation import layer_norm, mlp, self_attention

from . import tensors as T


# ---------------------------------------------------------------- the CNN (one grouped convolution per layer)
def cnn_features(x: torch.Tensor, w: dict) -> torch.Tensor:
    """Images x (M, N, 28, 28), model m sees its own images → h (M, N, E). The model axis becomes conv groups:
    channel block m of every layer only ever meets model m's filters, so the M networks never mix."""
    m, n = x.shape[:2]
    z = x.transpose(0, 1)  # (N, M, 28, 28): one input channel per model
    z = F.max_pool2d(F.relu(F.conv2d(z, w["c1"].flatten(0, 1), w["b1"].flatten(), groups=m)), 2)  # (N, 6M, 12, 12)
    z = F.max_pool2d(F.relu(F.conv2d(z, w["c2"].flatten(0, 1), w["b2"].flatten(), groups=m)), 2)  # (N, 12M, 4, 4)
    z = z.reshape(n, m, 192).transpose(0, 1)  # (M, N, 192)
    return F.relu(torch.einsum("mnk,mke->mne", z, w["fc"]) + w["bf"][:, None])


def cnn_logits(h: torch.Tensor, w: dict) -> torch.Tensor:
    """Digit logits (M, N, 10) from the features h (M, N, E)."""
    return torch.einsum("mne,mek->mnk", h, w["head"]) + w["bh"][:, None]


def cnn_output(x: torch.Tensor, w: dict) -> torch.Tensor:
    """q = softmax(CNN logits): the 10 digit probabilities, what the CNN passes along the DAG edge.  → (M, N, 10)"""
    return torch.softmax(cnn_logits(cnn_features(x, w), w), -1)


# ---------------------------------------------------------------- the LM on input vectors
def lm_logits(X0: torch.Tensor, w: dict) -> torch.Tensor:
    """Input vectors X0 (M, N, T, d) (token embeddings or soft tokens) → next-letter logits (M, N, T, V)."""
    X = X0 + w["pos"][:, None, : X0.shape[2]]
    for p in w["layers"]:
        X = X + self_attention(layer_norm(X, p["g1"], p["b1"]), p)[0]
        X = X + mlp(layer_norm(X, p["g2"], p["b2"]), p)
    return torch.einsum("mntc,mcv->mntv", layer_norm(X, w["gf"], w["bf"]), w["head"]) + w["bias"][:, None, None]


def embed(tokens: torch.Tensor, w: dict) -> torch.Tensor:
    """Tok_{m, t_{mnt}, c}: model m looks up its own embedding table.  tokens (M, N, T) → (M, N, T, d)."""
    model = torch.arange(tokens.shape[0], device=tokens.device)[:, None, None]
    return w["tok"][model, tokens]


# ---------------------------------------------------------------- the connector and the DAG-wired model
def connector(q: torch.Tensor, slots: torch.Tensor, w: dict) -> torch.Tensor:
    """c_t(q) = GELU(q W1 + S_t) W2 + b2.  q (M, P, 10) CNN outputs, slots (M, P) positions t → (M, P, d)."""
    model = torch.arange(q.shape[0], device=q.device)[:, None]
    pre = torch.einsum("mpe,mef->mpf", q, w["W1"]) + w["S"][model, slots]
    return torch.einsum("mpf,mfc->mpc", F.gelu(pre), w["W2"]) + w["b2"][:, None]


def stitched_inputs(x: torch.Tensor, tokens: torch.Tensor, w_cnn: dict, w_con: dict, w_lm: dict) -> torch.Tensor:
    """The DAG edge CNN → connector → LM: K soft tokens from the image replace the operand word.
    x (M, N, 28, 28), tokens (M, N, T) → X0 (M, N, T, d)."""
    m, n, t = tokens.shape
    h = cnn_output(x, w_cnn)  # (M, N, 10)
    hk = h[:, :, None].expand(m, n, T.K, h.shape[-1]).reshape(m, n * T.K, -1)
    slots = torch.arange(T.K, device=x.device).repeat(n)[None].expand(m, -1)  # (M, N·K) slot of every row
    prefix = connector(hk, slots, w_con).reshape(m, n, T.K, -1)
    return torch.cat([prefix, embed(tokens[:, :, T.K :], w_lm)], dim=2)


def stitched_logits(x, tokens, w_cnn, w_con, w_lm) -> torch.Tensor:
    return lm_logits(stitched_inputs(x, tokens, w_cnn, w_con, w_lm), w_lm)


# ---------------------------------------------------------------- the same edge as an MoE layer
def position_items(tokens: torch.Tensor):
    """What sits at each input position: an id (IMG placeholder at the K image positions, else the letter) and an
    image (the digit at image positions, a blank page at text positions).  → ids (M, N, T), is_img (T,) bool."""
    t = tokens.shape[2]
    is_img = torch.arange(t, device=tokens.device) < T.K
    ids = torch.where(is_img, torch.full_like(tokens, T.STOI[T.IMG]), tokens)
    return ids, is_img


def router_features(x: torch.Tensor, ids: torch.Tensor, is_img: torch.Tensor, blank: torch.Tensor) -> torch.Tensor:
    """r_{mnt} = 7×7 average-pooled image of the position ⊕ one-hot id.  → (M, N, T, 49 + V)."""
    m, n, t = ids.shape
    pooled = F.avg_pool2d(x.reshape(m * n, 1, 28, 28), 4).reshape(m, n, 1, 49)
    pooled_blank = F.avg_pool2d(blank[None, None], 4).reshape(1, 1, 1, 49)
    img = torch.where(is_img[None, None, :, None], pooled, pooled_blank)  # (M, N, T, 49)
    return torch.cat([img, F.one_hot(ids, len(T.VOCAB)).to(x.dtype)], dim=-1)


def route_fixed(m: int, n: int, is_img: torch.Tensor) -> torch.Tensor:
    """G (M, N, T, 2) one-hot by modality: expert 0 = img at the K prefix positions, expert 1 = txt elsewhere."""
    g = torch.stack([is_img, ~is_img], -1).to(torch.get_default_dtype())
    return g[None, None].expand(m, n, -1, -1)


def route_learned(r: torch.Tensor, w: dict, soft: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Switch top-1: p = softmax(r W + b), G = p ⊙ onehot(argmax p).  r (M, N, T, F) → G, p (M, N, T, 2).
    soft=True: G = p, a dense mixture (both experts run everywhere), used only while training the soft router."""
    p = torch.softmax(torch.einsum("mntf,mfx->mntx", r, w["W"]) + w["b"][:, None, None], -1)
    return (p if soft else p * F.one_hot(p.argmax(-1), 2).to(p.dtype)), p


def balance_loss(G: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """Switch load-balancing loss X Σ_x f_x P_x per model (M,): 1 when the split is even, X·max when one expert
    takes all."""
    f = (G > 0).to(p.dtype).mean((1, 2))  # (M, X) share of positions dispatched to each expert
    return p.shape[-1] * (f * p.mean((1, 2))).sum(-1)


def dispatch(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The positions of one expert's group, per model, packed to the front: mask (M, S) → idx (M, P), valid (M, P).
    P = the largest group over the models; shorter groups are padded (valid = False)."""
    cnt = mask.sum(1)
    P = int(cnt.max()) if mask.numel() else 0
    idx = torch.argsort((~mask).to(torch.uint8), dim=1, stable=True)[:, :P]
    return idx, torch.arange(P, device=mask.device)[None] < cnt[:, None]


def moe_inputs(x, ids, G, w_cnn, w_con, w_lm, blank) -> torch.Tensor:
    """X0_{mntc} = G_{mntx} Y^x_{mntc}: dispatch every position to its expert, run each expert batched on its own
    group (CNN features once per image, connector per routed position; token lookup), scatter back, combine."""
    m, n, t = ids.shape
    flat_ids = ids.reshape(m, n * t)
    h_img = cnn_output(x, w_cnn)  # (M, N, 10): the CNN runs once per image ...
    h_blank = cnn_output(blank.expand(m, 1, 28, 28), w_cnn)  # (M, 1, 10) ... and once on the blank page
    is_img = torch.arange(t, device=x.device) < T.K
    Y = []
    for x_e in range(2):  # the two experts
        idx, valid = dispatch(G[..., x_e].reshape(m, n * t) > 0)
        if x_e == 0:  # E_img = CNN + connector at position t of example n
            ex, slot = idx // t, idx % t
            h = torch.where(is_img[slot][..., None], h_img.gather(1, ex[..., None].expand(-1, -1, h_img.shape[-1])),
                            h_blank)  # fmt: skip
            out = connector(h, slot, w_con)
        else:  # E_txt = the LM's own token embedding
            model = torch.arange(m, device=x.device)[:, None]
            out = w_lm["tok"][model, flat_ids.gather(1, idx)]
        out = out * valid[..., None]
        Y.append(torch.zeros(m, n * t, out.shape[-1], dtype=out.dtype, device=x.device).scatter_add(
            1, idx[..., None].expand_as(out), out).reshape(m, n, t, -1))  # fmt: skip
    return torch.einsum("mntx,xmntc->mntc", G, torch.stack(Y))


# ---------------------------------------------------------------- losses and accuracy
def masked_ce(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-model mean −log p of the masked targets (nats).  logits (M, N, T, V), targets, mask (M, N, T) → (M,)."""
    nll = -torch.gather(torch.log_softmax(logits, -1), -1, targets[..., None])[..., 0]
    return (nll * mask).sum((1, 2)) / mask.sum((1, 2))


def exact(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Every masked letter is the argmax given the true prefix ⇔ greedy decoding writes exactly these letters.
    → (M, N) bool."""
    return ((logits.argmax(-1) == targets) | ~mask).all(-1)
