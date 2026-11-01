"""The reader: binarise → find the finders → homography → sample modules → format → unmask → Reed–Solomon → text.

Image-wide steps (thresholding, matched filtering, sampling, Reed–Solomon) run on the whole batch of photos at once;
picking peaks and fitting each photo's homography is a short loop over photos.
"""

import itertools

import torch
import torch.nn.functional as F

from . import equation as eq
from . import tensors

FINDER = ((1.5, 2.5, 3.5), (1, -1, 1))  # 1:1:3:1:1 dark-light-dark-light-dark, band edges in modules from centre
SCALES = [2.5 * 1.15**k for k in range(11)]  # 2.5 … 10 pixels per module tried by the finder search


# --- 1. binarise -----------------------------------------------------------------------------------------------------


def binarise(img: torch.Tensor, window: int = 31, offset: float = 0.03) -> torch.Tensor:
    """Adaptive threshold: dark where the (lightly smoothed) pixel is below its local box mean minus an offset.
    img (B, H, W) → (B, H, W) bool. The local mean follows uneven lighting, a global threshold would not."""
    smooth = eq.box_mean(img, 3)
    return smooth < eq.box_mean(img, window) - offset


# --- 2. finder patterns ----------------------------------------------------------------------------------------------


def finder_response(dark: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """R_{bxy} = min(max_s r^h_s, max_s r^v_s): the 1:1:3:1:1 profile must fit along the row and along the column
    through (x, y), each at its best scale s (pixels per module; perspective makes them differ). Along any line
    through its centre a finder keeps these ratios, whatever the rotation. dark (B, H, W) → R in [−1, 1]
    (1 = perfect match) and the mean of the two best scales, both (B, H, W)."""
    d = dark.float() * 2 - 1  # dark +1, light −1
    best_h, best_v = torch.full_like(d, -1.0), torch.full_like(d, -1.0)
    scale_h, scale_v = torch.zeros_like(d), torch.zeros_like(d)
    for s in SCALES:
        rh, rv = eq.line_responses(d, eq.profile_kernel(s, *FINDER))
        scale_h, scale_v = torch.where(rh > best_h, s, scale_h), torch.where(rv > best_v, s, scale_v)
        best_h, best_v = torch.maximum(best_h, rh), torch.maximum(best_v, rv)
    return torch.minimum(best_h, best_v), (scale_h + scale_v) / 2


def peaks(R, pooled, scale, dark, radius: int = 7, count: int = 8, floor: float = 0.55) -> torch.Tensor:
    """Local maxima of one response map R (H, W) (where R equals its max-pooled version `pooled`) above floor →
    (k, 4) rows (x, y, response, scale), where the response also includes the two diagonals (diagonal_response).
    Each position is refined to the centroid of the plateau ≥ 90 % of its peak (a finder centre is a 3 × 3-module
    plateau, so the argmax alone would sit on its edge)."""
    ys, xs = torch.nonzero((R == pooled) & (R > floor), as_tuple=True)
    vals = R[ys, xs]
    order = vals.argsort(descending=True)[: 4 * count]
    out = []
    h, w = R.shape
    for i in order.tolist():
        x, y, v = xs[i].item(), ys[i].item(), vals[i].item()
        if any((x - o[0]) ** 2 + (y - o[1]) ** 2 < radius**2 for o in out):
            continue  # plateau ties: keep one maximum per blob
        y0, y1, x0, x1 = max(0, y - radius), min(h, y + radius + 1), max(0, x - radius), min(w, x + radius + 1)
        patch = R[y0:y1, x0:x1]
        weight = (patch >= 0.9 * v).float()
        gy, gx = torch.meshgrid(torch.arange(y0, y1), torch.arange(x0, x1), indexing="ij")
        cx, cy = float((gx * weight).sum() / weight.sum()), float((gy * weight).sum() / weight.sum())
        out.append([cx, cy, min(v, diagonal_response(dark, cx, cy)), float(scale[y, x])])
        if len(out) == count:
            break
    return torch.tensor(out).reshape(-1, 4)


def diagonal_response(dark: torch.Tensor, x: float, y: float) -> float:
    """The 1:1:3:1:1 match along both diagonals through (x, y), best scale each, the smaller of the two. Finders
    pass in every direction; the odd cross of data modules that fools the row/column test seldom does."""
    h, w = dark.shape
    kernels = [eq.profile_kernel(s, *FINDER) for s in SCALES]
    half = (len(kernels[-1]) - 1) // 2
    K = torch.stack([F.pad(k, (half - len(k) // 2,) * 2) for k in kernels])  # (S, 2·half + 1), zero-padded
    t = torch.arange(-half, half + 1) / 2**0.5  # one-pixel steps along a diagonal
    best = []
    for dy in (1, -1):
        u, v = (x + t).round().long(), (y + dy * t).round().long()
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = torch.where(inside, dark[v.clamp(0, h - 1), u.clamp(0, w - 1)].float() * 2 - 1, -1.0)
        best.append(float((K @ d).max()))
    return min(best)


def pick_finders(cands: torch.Tensor) -> tuple[torch.Tensor | None, float]:
    """Choose the three strongest candidates that form a right isosceles triangle, ordered top-left (the right-angle
    corner), top-right, bottom-left by the sign of the cross product. The finders of a version 1–3 symbol are
    14–22 modules apart and match at similar scales. cands (k, 4) rows (x, y, response, scale) → (3, 2) or None,
    and the mean scale (pixels per module)."""
    best, best_score, module = None, -1e9, 0.0
    for tri in itertools.combinations(range(len(cands)), 3):
        p, resp, scales = cands[list(tri), :2], cands[list(tri), 2], cands[list(tri), 3]
        if scales.max() > 1.8 * scales.min():
            continue
        for corner in range(3):  # perspective can make any vertex look like the right angle: try all three
            tl, p1, p2 = p[corner], *p[[k for k in range(3) if k != corner]]
            e1, e2 = p1 - tl, p2 - tl
            if e1[0] * e2[1] - e1[1] * e2[0] < 0:  # image y points down: TL→TR × TL→BL > 0
                p1, p2, e1, e2 = p2, p1, e2, e1
            l1, l2 = e1.norm(), e2.norm()
            if min(l1, l2) < 10 * scales.mean() or max(l1, l2) > 28 * scales.mean():
                continue
            error = float((e1 @ e2).abs() / (l1 * l2) + (l1 - l2).abs() / torch.maximum(l1, l2))  # 0: right, equal
            if error > 0.6:
                continue
            score = float(resp.mean()) - 0.25 * error
            if score > best_score:
                best_score, best, module = score, torch.stack([tl, p1, p2]), float(scales.mean())
    return best, module


# --- 3. homography ---------------------------------------------------------------------------------------------------


def find_alignment(dark: torch.Tensor, guess: torch.Tensor, ex: torch.Tensor, ey: torch.Tensor):
    """2D matched filter for the 5 × 5-module alignment pattern (dark ring, light ring, dark centre), its template
    drawn through the local affine frame (ex, ey = one module along a row / a column, in pixels), searched ±5
    modules around the predicted centre. dark (H, W) → centre (x, y) and the peak correlation (1 = perfect)."""
    h, w = dark.shape
    module = float(torch.maximum(ex.norm(), ey.norm()))
    half, r = int(2.7 * module) + 1, int(5 * module) + 1
    pad = r + half
    x0, y0 = int(guess[0]) - pad, int(guess[1]) - pad
    if not (-pad <= x0 < w and -pad <= y0 < h):
        return guess, 0.0
    t = torch.arange(-half, half + 1.0)
    dv, du = torch.meshgrid(t, t, indexing="ij")
    XY = torch.linalg.solve(torch.stack([ex, ey], 1), torch.stack([du.reshape(-1), dv.reshape(-1)]))  # code offsets
    ring = XY.abs().max(0).values.reshape(du.shape)  # Chebyshev distance from the centre, in modules
    band = torch.where(ring < 0.5, 1.0, torch.where(ring < 1.5, -1.0, torch.where(ring < 2.5, 1.0, 0.0)))
    band = torch.where(((ring - 0.5).abs() < 0.2) | ((ring - 1.5).abs() < 0.2), 0.0, band)  # blur-tolerant edges
    kernel = band / band.abs().sum()
    padded = F.pad(dark.float() * 2 - 1, (pad, pad, pad, pad), value=-1.0)  # outside the photo counts as light
    patch = padded[y0 + pad : y0 + 3 * pad + 1, x0 + pad : x0 + 3 * pad + 1]
    resp = F.conv2d(patch[None, None], kernel[None, None])[0, 0]  # (2r + 1, 2r + 1)
    yy, xx = torch.meshgrid(torch.arange(2 * r + 1.0), torch.arange(2 * r + 1.0), indexing="ij")
    resp = resp - 0.03 * ((xx - r) ** 2 + (yy - r) ** 2).sqrt() / module  # prefer the nearer of two equal hits
    score = float(resp.max())
    weight = (resp >= score - 0.03).float()
    centre = torch.stack([(xx * weight).sum(), (yy * weight).sum()]) / weight.sum()
    return centre + torch.tensor([x0 + half, y0 + half]), score


def code_points(n: int, version: int) -> torch.Tensor:
    """Code-plane positions (module units) of TL, TR, BL finder centres and of the 4th point (4, 2)."""
    fourth = [n - 3.5, n - 3.5] if version == 1 else [n - 6.5, n - 6.5]
    return torch.tensor([[3.5, 3.5], [n - 3.5, 3.5], [3.5, n - 3.5], fourth])


def homography(dark: torch.Tensor, finders: torch.Tensor, version: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Code plane → image H (3, 3) from the 3 finder centres and a 4th point: the alignment pattern (version ≥ 2),
    found near where the affine map of the finders predicts it, or the parallelogram corner (version 1, as ZXing
    does). Returns H and the four image points."""
    n = tensors.size(version)
    tl, tr, bl = finders
    if version == 1:
        fourth = tr + bl - tl
    else:
        t = (n - 10) / (n - 7)  # affine prediction of the alignment centre at (n − 6.5, n − 6.5)
        guess = tl + t * (tr - tl) + t * (bl - tl)
        fourth, score = find_alignment(dark, guess, (tr - tl) / (n - 7), (bl - tl) / (n - 7))
        if score < 0.3:
            fourth = guess
    img_pts = torch.cat([finders, fourth[None]])
    return eq.dlt(code_points(n, version)[None], img_pts[None])[0], img_pts


def function_template(version: int, quiet: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Known modules of every symbol of this version: finders + separators, timing, alignment, the dark module and a
    ring of `quiet` light quiet-zone modules. → code-plane centres (M, 2) and target values (M,) (+1 dark, −1 light)."""
    val, fun = tensors.function_patterns(version)
    n = tensors.size(version)
    fmt = torch.zeros(n, n, dtype=torch.bool)
    for pos in tensors.format_positions(n):
        fmt[pos[:, 0], pos[:, 1]] = True
    fixed = fun & ~fmt
    fixed[n - 8, 8] = True
    big = torch.zeros(n + 2 * quiet, n + 2 * quiet, dtype=torch.bool)
    big[quiet:-quiet, quiet:-quiet] = fixed
    big_val = torch.zeros_like(big, dtype=torch.long)
    big_val[quiet:-quiet, quiet:-quiet] = val
    ring = torch.ones_like(big)
    ring[quiet:-quiet, quiet:-quiet] = False
    keep = big | ring
    i, j = torch.nonzero(keep, as_tuple=True)
    centres = torch.stack([j, i], -1).float() + 0.5 - quiet  # (X, Y) of module centres
    return centres, (big_val[i, j] * 2 - 1).float()


def refine_homography(dark: torch.Tensor, H: torch.Tensor, version: int, steps: int = 30):
    """Fine-tune H (B, 3, 3) so the known modules land on matching pixels: maximise the correlation
    C = mean_m t_m · d̃(H x_m) between template values t_m (±1) and the smoothed ±1 photo d̃, by Adam on a
    projective correction H = H₀ S⁻¹ (I + Δ) S (S scales the code plane to unit size). Gradients flow through the
    bilinear grid_sample. Coarse to fine: heavy smoothing first (wide basin), then light (sharp optimum). This is
    direct image alignment (Lucas–Kanade style), batched over photos. → refined H (B, 3, 3) and the final
    correlation C (B,) on the sharpest image (≈ 1 when every known module matches, ≈ 0 for a wrong version)."""
    n = tensors.size(version)
    centres, target = function_template(version, quiet=2)
    offs = torch.tensor([[-0.2, -0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2]])
    pts = (centres[:, None] + offs).reshape(-1, 2).to(H.device)  # 4 samples per module
    target = target.repeat_interleave(4).to(H.device)
    h, w = dark.shape[1:]
    S = torch.diag(torch.tensor([1 / n, 1 / n, 1.0], device=H.device))
    eye = torch.eye(3, device=H.device)
    delta = torch.zeros(H.shape[0], 3, 3, device=H.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=3e-3)
    for width in (9, 5, 3):
        soft = eq.box_mean(dark.float() * 2 - 1, width)
        for _ in range(steps):
            Hd = H @ torch.linalg.inv(S) @ (eye + delta) @ S
            uv = eq.apply_homography(Hd, pts.expand(H.shape[0], -1, -1))
            grid = torch.stack([(2 * uv[..., 0] + 1) / w - 1, (2 * uv[..., 1] + 1) / h - 1], -1)[:, None]
            val = F.grid_sample(soft[:, None], grid, mode="bilinear", align_corners=False, padding_mode="border")
            corr = (val[:, 0, 0] * target).mean(1)
            loss = -corr.sum()
            opt.zero_grad()
            loss.backward()
            delta.grad[:, 2, 2] = 0  # the overall scale of H is free: keep it fixed
            opt.step()
    with torch.no_grad():
        Hd = H @ torch.linalg.inv(S) @ (eye + delta) @ S
        return Hd / Hd[:, 2:, 2:], corr.detach()


# --- 4. sample the module grid ---------------------------------------------------------------------------------------


def code_to_grid(H: torch.Tensor, coords: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """grid_sample coordinates in [−1, 1] of the code-plane points (X, Y) on the product grid coords × coords."""
    h, w = size
    Y, X = torch.meshgrid(coords, coords, indexing="ij")
    pts = torch.stack([X, Y], -1).reshape(1, -1, 2).expand(H.shape[0], -1, -1).to(H.device)
    uv = eq.apply_homography(H, pts)  # (B, P, 2) pixel coordinates
    return torch.stack([(2 * uv[..., 0] + 1) / w - 1, (2 * uv[..., 1] + 1) / h - 1], -1)[:, None]


def sample(img: torch.Tensor, H: torch.Tensor, n: int, per_module: int = 3) -> torch.Tensor:
    """Module votes: grid_sample img (B, H, W) at per_module² points in the central 60 % of each module of the code
    plane mapped into the photo by H (B, 3, 3), averaged per module → (B, n, n)."""
    offs = (torch.arange(per_module) + 0.5) / per_module * 0.6 + 0.2  # away from the module edges
    coords = (torch.arange(n)[:, None] + offs[None]).reshape(-1)  # (n·p,)
    grid = code_to_grid(H, coords, img.shape[1:])
    vals = F.grid_sample(img[:, None], grid, mode="bilinear", align_corners=False, padding_mode="border")
    return vals.reshape(-1, n, per_module, n, per_module).mean((2, 4))


def rectify(img: torch.Tensor, H: torch.Tensor, n: int, px: int = 6) -> torch.Tensor:
    """The photo resampled onto the code plane, quiet zone included, px pixels per module. → (B, m·px, m·px)"""
    coords = (torch.arange((n + 2 * tensors.QUIET) * px) + 0.5) / px - tensors.QUIET
    grid = code_to_grid(H, coords, img.shape[1:])
    out = F.grid_sample(img[:, None], grid, mode="bilinear", align_corners=False, padding_mode="border")
    return out.reshape(img.shape[0], len(coords), len(coords))


# --- 5. format, unmask, codewords ------------------------------------------------------------------------------------


def read_format(modules: torch.Tensor) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Read both format copies, take the nearest BCH codeword. modules (B, n, n) → ECC level, mask, bit errors."""
    first, second = tensors.format_positions(modules.shape[-1])
    copies = torch.stack([modules[:, first[:, 0], first[:, 1]], modules[:, second[:, 0], second[:, 1]]], 1)
    index, errors = eq.format_decode(copies.reshape(-1, 15))  # (2B,)
    index, errors = index.view(-1, 2), errors.view(-1, 2)
    pick = errors.argmin(1, keepdim=True)
    index, errors = index.gather(1, pick)[:, 0], errors.gather(1, pick)[:, 0]
    from_bits = {b: e for e, b in tensors.ECL_BITS.items()}
    return [from_bits[i >> 3] for i in index.tolist()], index & 7, errors


def read_codewords(modules: torch.Tensor, version: int, masks: torch.Tensor) -> torch.Tensor:
    """XOR the mask back off (it is an involution), read the data modules in placement order, pack 8 bits per byte.
    modules (B, n, n), masks (B,) → codeword stream (B, total)."""
    n = tensors.size(version)
    _, fun = tensors.function_patterns(version)
    pos = tensors.data_positions(fun).to(modules.device)
    pats = torch.stack([eq.mask_pattern(k, n, modules.device) for k in range(8)])[masks].long()  # (B, n, n)
    bits = (modules ^ pats)[:, pos[:, 0], pos[:, 1]][:, : 8 * tensors.TOTAL_CODEWORDS[version]]
    weights = 2 ** torch.arange(7, -1, -1, device=modules.device)
    return (bits.reshape(bits.shape[0], -1, 8) * weights).sum(-1)


def deinterleave(stream: torch.Tensor, version: int, ecl: str) -> torch.Tensor:
    """Undo the column-by-column interleaving. stream (B, total) → blocks (B, nb, k + T)."""
    lay = tensors.layout(version, ecl)
    nb, k, t = lay["blocks"], lay["k"], lay["n_ec"]
    data = stream[:, : nb * k].reshape(-1, k, nb).transpose(1, 2)
    ecc = stream[:, nb * k :].reshape(-1, t, nb).transpose(1, 2)
    return torch.cat([data, ecc], 2)


# --- 6. Reed–Solomon decoding ----------------------------------------------------------------------------------------


def berlekamp_massey(S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shortest LFSR Λ(x) = 1 + Λ₁x + … generating the syndromes, for a batch S (B, T) → Λ (B, T + 1), degree L (B,).
    Its roots are the inverse error locators X_k^{-1}."""
    B, T = S.shape
    dev = S.device
    C = torch.zeros(B, T + 1, dtype=torch.long, device=dev)
    C[:, 0] = 1
    prev, L = C.clone(), torch.zeros(B, dtype=torch.long, device=dev)
    m, b = torch.ones(B, dtype=torch.long, device=dev), torch.ones(B, dtype=torch.long, device=dev)
    j = torch.arange(T + 1, device=dev)
    for step in range(T):
        d = eq.gf_sum(eq.gf_mul(C[:, : step + 1], S[:, : step + 1].flip(1)))  # discrepancy ⊕_i Λ_i S_{step−i}
        src = j[None] - m[:, None]
        shifted = torch.where(src >= 0, prev.gather(1, src.clamp(min=0)), 0)  # x^m · B(x)
        updated = C ^ eq.gf_mul(eq.gf_div(d, b)[:, None], shifted)
        change, grow = d != 0, (d != 0) & (2 * L <= step)
        prev = torch.where(grow[:, None], C, prev)
        b = torch.where(grow, d, b)
        L = torch.where(grow, step + 1 - L, L)
        m = torch.where(grow, 1, m + 1)
        C = torch.where(change[:, None], updated, C)
    return C, L


def rs_decode(c: torch.Tensor, n_ec: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correct up to T/2 byte errors in each block of c (B, N): syndromes → Berlekamp–Massey → Chien search →
    Forney. Returns corrected blocks (B, N), errors fixed (B,), ok (B,) (False = more errors than it can fix)."""
    B, N = c.shape
    dev = c.device
    S = eq.syndromes(c, n_ec)  # (B, T)
    lam, L = berlekamp_massey(S)
    power = N - 1 - torch.arange(N, device=dev)  # position i ↔ locator X_i = α^{N−1−i}
    X, X_inv = eq.gf_pow_alpha(power), eq.gf_pow_alpha(-power)
    at_roots = eq.gf_poly_eval(lam, X_inv) == 0  # Chien search: (B, N) error positions
    # Forney: e_i = X_i · Ω(X_i⁻¹) / Λ'(X_i⁻¹), Ω = S(x) Λ(x) mod x^T, Λ' keeps the odd terms (characteristic 2).
    k = torch.arange(n_ec, device=dev)
    idx = k[:, None] - k[None]  # (T, T): Ω_k = ⊕_{i ≤ k} S_i Λ_{k−i}
    lam_k = torch.where(idx >= 0, lam[:, idx.clamp(min=0)], 0)  # (B, T, T)
    omega = eq.gf_sum(eq.gf_mul(S[:, None, :], lam_k), dim=-1)  # (B, T)
    deriv = torch.where(torch.arange(n_ec, device=dev) % 2 == 0, lam[:, 1:], 0)  # coefficient of x^k in Λ'
    num, den = eq.gf_poly_eval(omega, X_inv), eq.gf_poly_eval(deriv, X_inv)  # (B, N)
    magnitude = eq.gf_mul(X[None].expand(B, -1), eq.gf_div(num, torch.where(den == 0, 1, den)))
    ok = (at_roots.sum(1) == L) & (2 * L <= n_ec)
    fixed = c ^ torch.where(at_roots & ok[:, None], magnitude, 0)
    ok = ok & (eq.syndromes(fixed, n_ec) == 0).all(1)
    return torch.where(ok[:, None], fixed, c), torch.where(ok, L, 0), ok


def parse_bytes(data: torch.Tensor) -> list[str | None]:
    """Byte-mode payload: 4-bit mode (0100), 8-bit length, then bytes shifted by half a byte. data (B, n) → texts."""
    hi, lo = data >> 4, data & 15
    out = []
    for b in range(data.shape[0]):
        length = int(lo[b, 0] << 4 | hi[b, 1])
        if int(hi[b, 0]) != 4 or length > data.shape[1] - 2:
            out.append(None)
            continue
        payload = (lo[b, 1 : 1 + length] << 4) | hi[b, 2 : 2 + length]
        out.append(bytes(payload.tolist()).decode("utf-8", errors="replace"))
    return out


# --- the whole reader ------------------------------------------------------------------------------------------------


def decode(img: torch.Tensor, keep: bool = False) -> list[dict]:
    """Decode a batch of photos (B, H, W). Each result has 'text' (None on failure) and diagnostics; keep=True adds
    the intermediate images used by the animation."""
    dark = binarise(img)
    R, scale = finder_response(dark)
    pooled = F.max_pool2d(R[:, None], (1, 15), stride=1, padding=(0, 7))  # 15 × 15 max filter, separably
    pooled = F.max_pool2d(pooled, (15, 1), stride=1, padding=(7, 0))[:, 0]
    results = [{"text": None, "stage": "finders"} for _ in range(img.shape[0])]
    found = []
    for b in range(len(results)):
        finders, module = pick_finders(peaks(R[b].cpu(), pooled[b].cpu(), scale[b].cpu(), dark[b].cpu()))
        if finders is not None:
            found.append((b, finders))
            span = 0.5 * ((finders[1] - finders[0]).norm() + (finders[2] - finders[0]).norm())
            results[b]["size_estimate"] = float(span) / module + 7  # n ≈ finder distance / module size + 7
    if keep:
        for b, r in enumerate(results):
            r["dark"] = dark[b].cpu()
    if not found:
        return results
    # Try each version: fit H from the finders (+ alignment), refine it against the known modules, and keep the
    # version whose template fits best, with a weak prior from the size estimate n ≈ finder distance / module + 7.
    fits = {b: [] for b, _ in found}  # per photo: (score, version, fields) for each version tried
    for v in (1, 2, 3):
        H0, pts = zip(*[homography(dark[b].cpu(), f, v) for b, f in found])
        H, corr = refine_homography(dark[[b for b, _ in found]], torch.stack(H0).to(img.device), v)
        for (b, f), h0, h, p, c in zip(found, H0, H, pts, corr.tolist()):
            score = c - 0.05 * abs(results[b]["size_estimate"] - tensors.size(v)) / 4
            fits[b].append((score, v, {"version": v, "fit": c, "H0": h0, "H": h, "points": p, "finders": f}))
    for rank in (0, 1):  # best-fitting version first; if Reed–Solomon then fails, retry with the runner-up
        todo = [b for b in fits if rank == 0 or results[b]["text"] is None]
        for b in todo:
            results[b].update({"stage": "format", **sorted(fits[b], key=lambda x: -x[0])[rank][2]})
        for v in (1, 2, 3):
            group = [b for b in todo if results[b]["version"] == v]
            if group:
                decode_group(img, dark, group, v, results, keep)
    return results


def decode_group(img, dark, group: list[int], version: int, results: list[dict], keep: bool):
    """Steps 4–6 for photos that share a version: sample modules, format, unmask, Reed–Solomon, text."""
    n = tensors.size(version)
    H = torch.stack([results[b]["H"] for b in group])
    votes = sample(dark[group].float(), H, n)
    modules = (votes > 0.5).long()  # (G, n, n)
    levels, masks, fmt_err = read_format(modules)
    stream = read_codewords(modules, version, masks)
    for level in set(levels):
        sel = [i for i, e in enumerate(levels) if e == level]
        lay = tensors.layout(version, level)
        blocks = deinterleave(stream[sel], version, level)  # (g, nb, N)
        fixed, n_err, ok = rs_decode(blocks.reshape(-1, blocks.shape[-1]), lay["n_ec"])
        fixed, n_err, ok = fixed.view(blocks.shape), n_err.view(len(sel), -1), ok.view(len(sel), -1)
        data = fixed[..., : lay["k"]].reshape(len(sel), -1)
        texts = parse_bytes(data)
        for i, s in enumerate(sel):
            r = results[group[s]]
            r.update(
                {
                    "stage": "decoded" if ok[i].all() and texts[i] is not None else "reed-solomon",
                    "ecl": level,
                    "mask": int(masks[s]),
                    "format_errors": int(fmt_err[s]),
                    "byte_errors": n_err[i].tolist(),
                    "rs_ok": ok[i].tolist(),
                    "modules": modules[s].cpu(),
                    "received": blocks[i].cpu(),
                    "corrected": fixed[i].cpu(),
                    "text": texts[i] if ok[i].all() else None,
                }
            )
            if keep:
                r["rectified"] = rectify(img[group[s]][None], H[s][None], n)[0].cpu()
