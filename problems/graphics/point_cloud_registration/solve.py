"""The registration pipeline: VoxelGrid → StatisticalOutlierRemoval → NormalEstimation → FPFH → feature matching →
batched RANSAC (Kabsch per hypothesis) → batched ICP (point-to-point and point-to-plane) from several starts."""

import torch

from problems._common.progress import steps

from . import equation, tensors


def preprocess(X, eye, voxel: float, k_sor: int = 20, alpha: float = 1.0, features: bool = True) -> dict:
    """One scan → downsampled, cleaned points with normals, curvature and (optionally) FPFH descriptors."""
    down, _ = equation.voxel_downsample(X, voxel)
    keep = equation.statistical_outlier_mask(down, k_sor, alpha)
    P = down[keep]
    N, curv = equation.estimate_normals(P, k=15, viewpoint=eye)
    return {"down": down, "P": P, "N": N, "curv": curv, "F": equation.fpfh(P, N, k=25) if features else None}


def match_features(Fs: torch.Tensor, Ft: torch.Tensor):
    """Mutual nearest neighbours in descriptor space: source i ↔ target j when each is the other's best match."""
    _, st = equation.knn(Fs, Ft, 1)  # (Ns, 1)
    _, ts = equation.knn(Ft, Fs, 1)  # (Nt, 1)
    i = torch.arange(len(Fs), device=Fs.device)
    mutual = ts[st[:, 0], 0] == i
    return i[mutual], st[mutual, 0]


def ransac(P: torch.Tensor, Q: torch.Tensor, n_hypotheses: int, eps: float, gen: torch.Generator, chunk: int = 1000):
    """RANSAC over correspondences p_m ↔ q_m (M, 3) each, many hypotheses at once.

    Each hypothesis draws 3 correspondences, is rejected unless the triangle's edge lengths agree within 10 % (PCL's
    CorrespondenceRejectorPoly: a rigid motion preserves lengths), and is fitted by Kabsch; its score is the number
    of correspondences it maps within eps. The best one is refitted on all its inliers.
    Returns R (3, 3), t (3,), inlier mask (M,).
    """
    M = len(P)
    best_score, best = -1, None
    for s in range(0, n_hypotheses, chunk):
        b = min(chunk, n_hypotheses - s)
        idx = torch.randint(0, M, (b, 3), generator=gen)  # (b, 3) sampled correspondences
        p, q = P[idx], Q[idx]  # (b, 3, 3)
        lp = (p - p.roll(1, dims=1)).norm(dim=-1)  # (b, 3) triangle edge lengths
        lq = (q - q.roll(1, dims=1)).norm(dim=-1)
        ok = ((lp / lq.clamp(min=1e-12)).clamp(max=10) - 1).abs().amax(1) < 0.1
        ok &= lp.amin(1) > 1e-6
        R, t = equation.kabsch(p, q)  # (b, 3, 3), (b, 3): one SVD per hypothesis, batched
        resid = (equation.transform(R, t, P.expand(b, M, 3)) - Q).norm(dim=-1)  # (b, M)
        score = torch.where(ok, (resid < eps).sum(1), -1)
        k = int(score.argmax())
        if score[k] > best_score:
            best_score, best = int(score[k]), (R[k], t[k])
    R, t = best
    for _ in range(3):  # refit on the inliers (a few rounds of least squares)
        inl = (equation.transform(R, t, P) - Q).norm(dim=-1) < eps
        R, t = equation.kabsch(P[inl], Q[inl])
    inl = (equation.transform(R, t, P) - Q).norm(dim=-1) < eps
    return R, t, inl


def icp(src, target, normals, R0, t0, iterations: int, method: str, max_dist: float):
    """Iterative closest point, batched over B starting motions.

    src (B, N, 3) or (N, 3); target (M, 3) with normals (M, 3); R0 (B, 3, 3), t0 (B, 3).
    Each iteration: move src, find each point's nearest target point (pairs further than max_dist are ignored), then
      point_to_point: Kabsch on the pairs (Besl & McKay 1992) gives the whole motion again;
      point_to_plane: one linearised least-squares step on (R x + t - q)·n (Chen & Medioni 1992).
    Returns R, t and the history {"R", "t", "rmse"} (entry 0 = start).
    """
    B = len(R0)
    src = src.expand(B, -1, -1)
    R, t = R0.clone(), t0.clone()
    hist = {"R": [R.clone()], "t": [t.clone()], "rmse": []}
    eye6 = torch.eye(6, dtype=src.dtype, device=src.device)
    for _ in steps(iterations, f"ICP {method}"):
        x = equation.transform(R, t, src)  # (B, N, 3)
        d, j = equation.knn(x.reshape(-1, 3), target, 1)
        d, j = d.reshape(B, -1), j.reshape(B, -1)  # (B, N)
        w = (d < max_dist).to(src.dtype)
        hist["rmse"].append(((w * d**2).sum(1) / w.sum(1).clamp(min=1)).sqrt())
        q = target[j]  # (B, N, 3) matched target points
        if method == "point_to_point":
            R, t = equation.kabsch(src, q, w)
        else:
            H, g = equation.point_to_plane_system(x, q, normals[j], w)
            xi = torch.linalg.solve(H + 1e-9 * eye6, -g)  # (B, 6) = (ω, δt)
            dR = equation.rodrigues(xi[:, :3])
            R, t = equation.compose(dR, xi[:, 3:], R, t)  # apply the small motion after the current one
        hist["R"].append(R.clone())
        hist["t"].append(t.clone())
    x = equation.transform(R, t, src)
    d = equation.knn(x.reshape(-1, 3), target, 1)[0].reshape(B, -1)
    w = (d < max_dist).to(src.dtype)
    hist["rmse"].append(((w * d**2).sum(1) / w.sum(1).clamp(min=1)).sqrt())
    hist["rmse"] = torch.stack(hist["rmse"])  # (iterations + 1, B)
    return R, t, hist


def perturbations(R, t, n: int, angle_deg: float, trans: float, gen: torch.Generator):
    """n starting motions: the given one first, then random small rotations and shifts around it."""
    omega = torch.randn(n, 3, generator=gen, dtype=R.dtype)
    omega = omega / omega.norm(dim=1, keepdim=True) * torch.deg2rad(torch.tensor(angle_deg, dtype=R.dtype))
    dt = trans * torch.randn(n, 3, generator=gen, dtype=R.dtype)
    omega[0], dt[0] = 0, 0
    return equation.compose(equation.rodrigues(omega), dt, R.expand(n, 3, 3), t.expand(n, 3))


def register(params: dict, seed: int = 0) -> dict:
    """The whole pipeline on the two Spot scans; errors against the known ground truth."""
    gen = torch.Generator().manual_seed(seed + 100)
    scans = tensors.make_scans(params, seed)
    voxel = params["voxel"]
    tgt = preprocess(scans["A"], scans["eye_a"], voxel)
    src = preprocess(scans["B"], scans["eye_b"], voxel)
    i_s, i_t = match_features(src["F"], tgt["F"])
    R0, t0, inl = ransac(src["P"][i_s], tgt["P"][i_t], params["hypotheses"], 2.5 * voxel, gen)
    Rs, ts = perturbations(R0, t0, params["starts"], 3.0, 0.3 * voxel, gen)
    fine = preprocess(scans["A"], scans["eye_a"], voxel / 2, features=False)  # a denser target for ICP
    icp_out = {}
    for method in ["point_to_point", "point_to_plane"]:
        R, t, hist = icp(src["P"], fine["P"], fine["N"], Rs, ts, params["icp_iterations"], method, voxel)
        icp_out[method] = {"R": R, "t": t, "hist": hist}
    best = icp_out["point_to_plane"]
    b = int(best["hist"]["rmse"][-1].argmin())  # the start that ended with the smallest RMSE
    R_true, t_true = equation.invert(scans["R_gt"], scans["t_gt"])  # the motion that undoes the unknown one
    rot_err = float(equation.rotation_error_deg(best["R"][b], R_true))
    trans_err = float((best["t"][b] - t_true).norm())
    return {
        **scans, "src": src, "tgt": tgt, "fine": fine, "match": (i_s, i_t), "ransac": (R0, t0, inl), "starts": (Rs, ts),
        "icp": icp_out, "best": b, "R_true": R_true, "t_true": t_true, "rot_err": rot_err, "trans_err": trans_err,
        "ransac_rot_err": float(equation.rotation_error_deg(R0, R_true)),
        "ransac_trans_err": float((t0 - t_true).norm()), "rmse": float(best["hist"]["rmse"][-1, b]), "size": 1.0,
    }  # fmt: skip
