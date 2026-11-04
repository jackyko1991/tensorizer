r"""The laws as tensor operations: the stages of the pipeline, the top-1 mixture-of-experts layer, and the losses.

Indices: m = model (seed), n = sample, e = expert (= operation), c = slot in an expert's buffer, l, j = word
position, i, h, k, v = channels. Repeated indices are summed unless they appear in the output.

    image expert     z^img_{mnv} = W2_{mhv} relu(W1_{mih} x_{ni} + b1_{mh}) + b2_{mv}                   (i, h summed)
    text expert      X_{mnlc} = T_{m,t_{nl},c} + P_{mlc},  A_{mnlj} = softmax_j(Q_{mnlk} K_{mnjk}/√d)
                     h_{mnc} = mean_l (X + A V W^O)_{mnlc},   z^txt_{mnv} = h_{mnc} W^N_{mcv} + b^N_{mv}
    features         u_{mn} = [softmax z^img_{mn}, softmax z^txt_{mn}]   (20 numbers)
    router           G_{mne} ∈ {0, 1},  Σ_e G_{mne} = 1      (fixed: the parsed operation; learned: top-1 of p_{mne})
    dispatch         D_{mnec} = G_{mne} [c = rank of n among the samples routed to e]
    experts          U_{meci} = D_{mnec} u_{mni},   Y_{mech} = f_e(U_{mec})     (all experts in one batched product)
    combine          o_{mnh} = D_{mnec} g_{mne} Y_{mech}     g = G (fixed router) or G ⊙ p (Switch)
    classify         z_{mnv} = W^C_{mhv} relu(o_{mnh}) + b^C_{mv}
    load balance     ℒ_aux = E Σ_e f_e P_e,  f_e = ⟨G_{ne}⟩_n,  P_e = ⟨p_{ne}⟩_n          (Fedus et al. 2021)
    distillation     ℒ_KD = T² KL(softmax(z^t/T) ‖ softmax(z^s/T))                        (Hinton et al. 2015)
    int8             Ŵ_{io} = s_o round(W_{io}/s_o),  s_o = max_i |W_{io}| / 127              (per output channel)

With G fixed to the workflow's rule choice and g = G, every sum over e and c has exactly one nonzero term, so the
layer computes what the DAG computes: each sample through the one fuse rule its text selects.
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- stages
def image_logits(img: dict, x: torch.Tensor) -> torch.Tensor:
    """Image expert, an MLP: x (M, N, 784) pixels (each model its own batch) → digit logits (M, N, 10)."""
    hid = torch.relu(torch.einsum("mni,mih->mnh", x, img["W1"]) + img["b1"][:, None])
    return torch.einsum("mnh,mhv->mnv", hid, img["W2"]) + img["b2"][:, None]


def text_features(txt: dict, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Text expert, one bidirectional attention layer: tokens (M, N, 3) → pooled features h (M, N, d) and number
    logits (M, N, 10). Attention rows are probability distributions over the three words."""
    m = torch.arange(tok.shape[0], device=tok.device)[:, None, None]
    X = txt["emb"][m, tok] + txt["pos"][:, None]  # (M, N, L, d): model m looks words up in its own table
    Q, K, Vv = (torch.einsum("mnlc,mck->mnlk", X, txt[k]) for k in ("Wq", "Wk", "Wv"))
    A = torch.softmax(torch.einsum("mnlk,mnjk->mnlj", Q, K) / math.sqrt(X.shape[-1]), dim=-1)
    X = X + torch.einsum("mnlj,mnjk,mkc->mnlc", A, Vv, txt["Wo"])  # j, k summed
    h = X.mean(2)
    return h, torch.einsum("mnc,mcv->mnv", h, txt["Wn"]) + txt["bn"][:, None]


def features(z_img: torch.Tensor, z_txt: torch.Tensor) -> torch.Tensor:
    """u (M, N, 20): the digit and the number as probability vectors, the interface between the stages."""
    return torch.cat([torch.softmax(z_img, -1), torch.softmax(z_txt, -1)], -1)


def expert(u: torch.Tensor, W1, b1, W2, b2) -> torch.Tensor:
    """ONE fuse expert on its own group: u (M, n_e, 20), W1 (M, 20, H) … → (M, n_e, H)."""
    hid = torch.relu(torch.einsum("mni,mih->mnh", u, W1) + b1[:, None])
    return torch.einsum("mnh,mhk->mnk", hid, W2) + b2[:, None]


def experts_stacked(U: torch.Tensor, fuse: dict) -> torch.Tensor:
    """ALL fuse experts at once on their buffers: U (M, E, C, 20) → Y (M, E, C, H)."""
    hid = torch.relu(torch.einsum("meci,meih->mech", U, fuse["W1"]) + fuse["b1"][:, :, None])
    return torch.einsum("mech,mehk->meck", hid, fuse["W2"]) + fuse["b2"][:, :, None]


def classify(cls: dict, o: torch.Tensor) -> torch.Tensor:
    """Shared head: o (M, N, H) → answer logits (M, N, 10)."""
    return torch.einsum("mnh,mhv->mnv", torch.relu(o), cls["W"]) + cls["b"][:, None]


# ---------------------------------------------------------------- the mixture-of-experts layer
def gate_probs(gate: dict, h: torch.Tensor) -> torch.Tensor:
    """Learned router: p (M, N, E) = softmax(h W^G + b^G) from the text features."""
    return torch.softmax(torch.einsum("mnc,mce->mne", h, gate["W"]) + gate["b"][:, None], -1)


def top1(p: torch.Tensor) -> torch.Tensor:
    """G (…, E): one-hot of the most probable expert of each row (top-1 routing, as in Switch)."""
    return F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)


def dispatch(G: torch.Tensor, capacity: int) -> torch.Tensor:
    """D (M, N, E, C): sample n goes to slot c of expert e, its rank among the samples routed to e (a cumulative
    sum, as in GShard/Switch). Samples beyond the capacity would be dropped; here C ≥ the largest group."""
    rank = (torch.cumsum(G, dim=1) - 1) * G  # (M, N, E)
    slot = F.one_hot(rank.long().clamp(0, capacity - 1), capacity).to(G.dtype)
    return G[..., None] * slot * (rank < capacity)[..., None]


def combine(D: torch.Tensor, g: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """o_{mnh} = D_{mnec} g_{mne} Y_{mech}: each sample takes its expert's output, scaled by its gate value."""
    return torch.einsum("mnec,mne,mech->mnh", D, g, Y)


def load_balance_loss(p: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """ℒ_aux = E Σ_e f_e P_e per model (M,): 1 for a uniform router with balanced groups, E when it collapses."""
    return p.shape[-1] * (G.mean(1) * p.mean(1)).sum(-1)


# ---------------------------------------------------------------- losses, student, quantisation
def cross_entropy(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean −log p of the true class per model: z (M, N, V), y (M, N) → (M,)."""
    return -torch.log_softmax(z, -1).gather(-1, y[..., None])[..., 0].mean(1)


def kd_loss(z_s: torch.Tensor, z_t: torch.Tensor, T: float) -> torch.Tensor:
    """T² KL(softmax(z_t/T) ‖ softmax(z_s/T)) averaged over samples, per model (M,) (Hinton et al. 2015)."""
    log_t, log_s = torch.log_softmax(z_t / T, -1), torch.log_softmax(z_s / T, -1)
    return T * T * (log_t.exp() * (log_t - log_s)).sum(-1).mean(1)


def student_logits(st: dict, x: torch.Tensor, tok: torch.Tensor, n_vocab: int) -> torch.Tensor:
    """Dense student: [x, one-hot words] (M, N, 784 + 3·V) → logits (M, N, 10), no routing, no stages."""
    inp = torch.cat([x, F.one_hot(tok, n_vocab).flatten(2).to(x.dtype)], -1)
    h1 = torch.relu(torch.einsum("mni,mih->mnh", inp, st["W1"]) + st["b1"][:, None])
    h2 = torch.relu(torch.einsum("mnh,mhk->mnk", h1, st["W2"]) + st["b2"][:, None])
    return torch.einsum("mnk,mkv->mnv", h2, st["W3"]) + st["b3"][:, None]


def fake_quant(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric int8 per output channel: W (M, I, O) → Ŵ = s·round(W/s) and s (M, 1, O), s = max|W|/127."""
    s = W.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127
    return (W / s).round().clamp(-127, 127) * s, s
