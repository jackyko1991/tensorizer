"""Track finding as an integer linear program, written as tensor operations (shapes in comments).

Hits h on layer l; candidate segments e = (a → b) join hits on adjacent layers; triplets t = (e₁, e₂) share the
middle hit. With binary x_t (triplet chosen) and y_e (segment chosen) the ILP is

    max  Σ_t s_t x_t − λ Σ_e y_e
    s.t. Σ_{e out of h} y_e ≤ 1,   Σ_{e into h} y_e ≤ 1   (each hit on at most one track: set packing)
         Σ_{t: e₁(t)=e} x_t ≤ y_e,  Σ_{t: e₂(t)=e} x_t ≤ y_e  (continuity: a triplet needs both of its segments,
                                                            and a segment continues into one triplet per side)
         x, y ∈ {0, 1}             (LP relaxation: 0 ≤ x, y ≤ 1)

i.e. max cᵀz s.t. A z ≤ b, 0 ≤ z ≤ 1 with z = (x, y). The matrix A is built sparse; its products are summed over
the variable index. Also here: the TrackML score, tracking efficiency and fake rate.
"""

import torch

from .tensors import Cuts


def wrap(angle: torch.Tensor) -> torch.Tensor:
    """Angle difference mapped to (−π, π]."""
    return torch.remainder(angle + torch.pi, 2 * torch.pi) - torch.pi


def segments(xyz: torch.Tensor, layer: torch.Tensor, cuts: Cuts) -> torch.Tensor:
    """Candidate segments (E, 2): (inner hit, outer hit) on adjacent layers that a track with p_T ≥ p_T,min from
    the beam line could join. Δφ ≤ asin(r_b/2R) − asin(r_a/2R) + tol and |z₀| ≤ z₀,max for every pair at once."""
    r = xyz[:, :2].norm(dim=-1)  # (N,)
    phi = torch.atan2(xyz[:, 1], xyz[:, 0])
    big_r = cuts.radius_min()
    out = []
    for lay in range(int(layer.max()) if layer.numel() else 0):  # 3 adjacent layer pairs
        ia = (layer == lay).nonzero().squeeze(1)  # (Na,)
        ib = (layer == lay + 1).nonzero().squeeze(1)  # (Nb,)
        ra, rb = r[ia][:, None], r[ib][None, :]  # (Na, 1), (1, Nb)
        dphi = wrap(phi[ib][None, :] - phi[ia][:, None]).abs()  # (Na, Nb)
        dphi_max = torch.asin((rb / (2 * big_r)).clamp(max=1)) - torch.asin((ra / (2 * big_r)).clamp(max=1))
        z0 = xyz[ia, 2][:, None] - ra * (xyz[ib, 2][None, :] - xyz[ia, 2][:, None]) / (rb - ra)  # (Na, Nb) mm
        ok = (dphi <= dphi_max + cuts.dphi_tol) & (z0.abs() <= cuts.z0_max)
        pa, pb = ok.nonzero().unbind(1)
        out.append(torch.stack([ia[pa], ib[pb]], 1))
    return torch.cat(out) if out else torch.zeros(0, 2, dtype=torch.long, device=xyz.device)


def doublet_curvature(xyz: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """Signed curvature κ (1/mm) of the circle through the beam line (0, 0) and both hits of each segment: (E,).
    κ = 2 (a × b) / (|a| |b| |b − a|) in the transverse plane."""
    a, b = xyz[seg[:, 0], :2], xyz[seg[:, 1], :2]  # (E, 2)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return 2 * cross / (a.norm(dim=-1) * b.norm(dim=-1) * (b - a).norm(dim=-1))


def polar_angle(xyz: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """Polar angle θ of each segment in the r–z plane: (E,)."""
    a, b = xyz[seg[:, 0]], xyz[seg[:, 1]]
    return torch.atan2(b[:, :2].norm(dim=-1) - a[:, :2].norm(dim=-1), b[:, 2] - a[:, 2])


def z0(xyz: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    """Where each segment, extended in r–z, crosses the beam line: (E,) mm."""
    a, b = xyz[seg[:, 0]], xyz[seg[:, 1]]
    ra, rb = a[:, :2].norm(dim=-1), b[:, :2].norm(dim=-1)
    return a[:, 2] - ra * (b[:, 2] - a[:, 2]) / (rb - ra)


def join(seg: torch.Tensor, n_hits: int) -> torch.Tensor:
    """All pairs (e₁, e₂) with outer hit of e₁ = inner hit of e₂: (T, 2) segment indices, no Python loop."""
    order = torch.argsort(seg[:, 0])  # segments sorted by inner hit
    count = torch.bincount(seg[:, 0], minlength=n_hits)  # (N,) segments leaving each hit
    start = torch.cumsum(count, 0) - count
    k = count[seg[:, 1]]  # (E,) continuations of each segment
    e1 = torch.repeat_interleave(torch.arange(seg.shape[0], device=seg.device), k)  # (T,)
    within = torch.arange(e1.numel(), device=seg.device) - torch.repeat_interleave(torch.cumsum(k, 0) - k, k)
    e2 = order[start[seg[e1, 1]] + within]
    return torch.stack([e1, e2], 1)


def triplet_score(xyz: torch.Tensor, seg: torch.Tensor, trip: torch.Tensor, cuts: Cuts) -> torch.Tensor:
    """s_t = 1 − (Δκ/σ_κ)² − (Δθ/σ_θ)²: a helix has the same curvature and polar angle on both halves. (T,)"""
    kappa, theta = doublet_curvature(xyz, seg), polar_angle(xyz, seg)  # (E,)
    dk = kappa[trip[:, 1]] - kappa[trip[:, 0]]
    dt = theta[trip[:, 1]] - theta[trip[:, 0]]
    return 1 - (dk / cuts.sigma_kappa) ** 2 - (dt / cuts.sigma_theta) ** 2


def triplets(xyz: torch.Tensor, seg: torch.Tensor, cuts: Cuts) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate triplets (T, 2) with positive score s_t (T,)."""
    trip = join(seg, xyz.shape[0])
    score = triplet_score(xyz, seg, trip, cuts)
    keep = score > 0
    return trip[keep], score[keep]


def candidates(xyz: torch.Tensor, layer: torch.Tensor, cuts: Cuts):
    """Segments (E, 2), triplets (T, 2) and scores (T,), keeping only the segments that belong to some triplet."""
    seg = segments(xyz, layer, cuts)
    trip, score = triplets(xyz, seg, cuts)
    used = torch.zeros(seg.shape[0], dtype=torch.bool, device=seg.device)
    used[trip.reshape(-1)] = True
    new_index = torch.cumsum(used.long(), 0) - 1  # old segment index → new
    return seg[used], new_index[trip], score


def ilp(seg: torch.Tensor, trip: torch.Tensor, score: torch.Tensor, n_hits: int, lam: float):
    """Sparse A (M, V), b (M,), c (V,) of  max cᵀz  s.t.  A z ≤ b,  0 ≤ z ≤ 1,  z = (x (T,), y (E,)).

    Rows: N out-degree rows, N in-degree rows, then for every segment e the continuity rows
    Σ_{t: e₁(t) = e} x_t − y_e ≤ 0 and Σ_{t: e₂(t) = e} x_t − y_e ≤ 0 (a chosen segment continues into at most one
    chosen triplet on each side; this implies x_t ≤ y_e and gives a much tighter LP): M ≤ 2N + 2E
    (empty rows dropped)."""
    n_t, n_e, dev = trip.shape[0], seg.shape[0], seg.device
    e, t = torch.arange(n_e, device=dev), torch.arange(n_t, device=dev)
    two_n = 2 * n_hits
    rows = torch.cat(
        [seg[:, 0], n_hits + seg[:, 1], two_n + trip[:, 0], two_n + n_e + trip[:, 1], two_n + e, two_n + n_e + e]
    )
    cols = torch.cat([n_t + e, n_t + e, t, t, n_t + e, n_t + e])
    vals = torch.cat([torch.ones(2 * n_e + 2 * n_t), -torch.ones(2 * n_e)]).to(dev, torch.float64)
    m, v = two_n + 2 * n_e, n_t + n_e
    b = torch.cat([torch.ones(two_n), torch.zeros(2 * n_e)]).to(dev, torch.float64)
    keep = torch.zeros(m, dtype=torch.bool, device=dev)
    keep[rows] = True  # drop empty rows (hits without any candidate segment)
    b, rows = b[keep], (torch.cumsum(keep.long(), 0) - 1)[rows]
    a = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (int(keep.sum()), v)).coalesce()
    c = torch.cat([score.double(), torch.full((n_e,), -lam, dtype=torch.float64, device=dev)])
    return a, b, c


def objective(x: torch.Tensor, y: torch.Tensor, score: torch.Tensor, lam: float) -> float:
    """Σ s_t x_t − λ Σ y_e."""
    return (score.double() @ x.double() - lam * y.double().sum()).item()


def tracks_from_segments(chosen: torch.Tensor, n_hits: int) -> torch.Tensor:
    """Track label per hit (N,) from the chosen segments (K, 2), −1 for unused hits. Every hit has at most one
    segment in and one out, so tracks are chains; each hit gets the id of its chain's first hit (pointer jumping)."""
    dev = chosen.device
    parent = torch.arange(n_hits, device=dev)
    parent[chosen[:, 1]] = chosen[:, 0]  # each hit points to its predecessor
    for _ in range(max(1, n_hits.bit_length())):  # log₂ N doublings reach the chain root
        parent = parent[parent]
    label = torch.full((n_hits,), -1, dtype=torch.long, device=dev)
    used = torch.zeros(n_hits, dtype=torch.bool, device=dev)
    used[chosen.reshape(-1)] = True
    label[used] = parent[used]
    return label


def _pairs(label: torch.Tensor, particle: torch.Tensor):
    """Unique (track, particle) pairs with their hit counts; −1 labels become one singleton track per hit."""
    n = label.numel()
    lab = torch.where(label >= 0, label, n + torch.arange(n, device=label.device))
    key = lab * (int(particle.max()) + 1) + particle
    uniq, inv, cnt = torch.unique(key, return_inverse=True, return_counts=True)
    return lab, uniq, inv, cnt, int(particle.max()) + 1


def trackml_score(label: torch.Tensor, particle: torch.Tensor, weight: torch.Tensor) -> float:
    """Official TrackML score: a (track, particle) pair counts when more than half of the track's hits belong to
    the particle AND more than half of the particle's hits are on the track; the score is the summed hit weight of
    those intersections, normalised so that the truth scores 1. Unassigned hits (−1) are singleton tracks."""
    lab, uniq, inv, cnt, n_p = _pairs(label, particle)
    track_of, part_of = uniq // n_p, uniq % n_p
    track_size = torch.bincount(lab)[track_of]
    part_size = torch.bincount(particle, minlength=n_p)[part_of]
    good = (2 * cnt > track_size) & (2 * cnt > part_size) & (part_of > 0)
    w = torch.zeros(uniq.numel(), dtype=weight.dtype, device=weight.device).index_add_(0, inv, weight)
    return (w[good].sum() / weight.sum()).item()


def reconstructable(particle, layer, pt, pt_min: float) -> torch.Tensor:
    """Particles (P,) with p_T ≥ p_T,min and hits on at least 3 distinct layers (index 0 = noise is never)."""
    n_p = int(particle.max()) + 1
    layers_hit = torch.bincount(torch.unique(particle * 16 + layer) // 16, minlength=n_p)  # distinct layers
    pt_p = torch.zeros(n_p, dtype=pt.dtype, device=pt.device).scatter_reduce(0, particle, pt, "amax")
    reco = (layers_hit >= 3) & (pt_p >= pt_min)
    reco[0] = False
    return reco


def efficiency_fake(label, particle, layer, pt, pt_min: float) -> tuple[float, float]:
    """Efficiency: the fraction of reconstructable particles (p_T ≥ p_T,min, hits on ≥ 3 layers) with a track that
    holds ≥ 3 of their hits. Fake rate: the fraction of tracks (≥ 3 hits) with no particle owning ≥ 3 of its hits."""
    n_p = int(particle.max()) + 1
    reco = reconstructable(particle, layer, pt, pt_min)
    lab, uniq, inv, cnt, _ = _pairs(label, particle)
    track_of, part_of = uniq // n_p, uniq % n_p
    matched = (cnt >= 3) & (part_of > 0)
    found = torch.zeros(n_p, dtype=torch.bool, device=label.device)
    found[part_of[matched]] = True
    eff = (found & reco).sum().item() / max(1, reco.sum().item())
    sizes = torch.bincount(lab)
    tracks = (sizes >= 3).nonzero().squeeze(1)
    good_track = torch.zeros(sizes.numel(), dtype=torch.bool, device=label.device)
    good_track[track_of[matched]] = True
    fake = (~good_track[tracks]).sum().item() / max(1, tracks.numel())
    return eff, fake
