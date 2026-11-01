r"""QR code → tensor translation: GF(256) arithmetic, Reed–Solomon and BCH codes, masks, homographies and filters.

GF(256). A byte a = (a₇ … a₀) is the polynomial Σ a_k x^k over GF(2), reduced modulo the primitive polynomial
p(x) = x⁸ + x⁴ + x³ + x² + 1 (0x11D). Addition is XOR. α = x = 2 has order 255, so every non-zero byte is α^l with
l = log a, and multiplication becomes two table look-ups (tensor gathers):

    a ⊗ b = exp[(log a + log b) mod 255],      a ⊘ b = exp[(log a − log b) mod 255].

A GF "matrix product" y_j = ⊕_i x_i ⊗ M_ij is an elementwise gf_mul followed by an XOR reduction over i (gf_sum): the
GF(256) analogue of einsum('i,ij->j') with the summed index i.

Reed–Solomon, QR flavour. A block of N bytes is the polynomial c(x) = Σ_i c_i x^{N−1−i} (first byte = highest power).
It is a codeword iff it is divisible by g(x) = Π_{j=0}^{T−1} (x − α^j), T = number of ECC bytes, i.e. iff all syndromes

    S_j = c(α^j) = ⊕_i c_i ⊗ V_ij,     V_ij = α^{j(N−1−i)}            (i summed with XOR)

vanish. Systematic encoding appends the remainder of d(x)·x^T mod g(x), which is linear in the k data bytes:

    e_j = ⊕_i d_i ⊗ G_ij,     row i of G = x^{T+k−1−i} mod g(x).

Format information: 5 bits (ECC level, mask) → BCH(15, 5) code with generator 0x537, XOR-ed with 0x5412. Any two of
its 32 codewords differ in at least 7 bits, so the nearest codeword corrects up to 3 bit errors.
"""

import functools

import torch
import torch.nn.functional as F

PRIMITIVE = 0x11D  # x^8 + x^4 + x^3 + x^2 + 1
FORMAT_GENERATOR = 0x537  # x^10 + x^8 + x^5 + x^4 + x^2 + x + 1
FORMAT_XOR = 0x5412


def gf_tables() -> tuple[torch.Tensor, torch.Tensor]:
    """exp (510,) with exp[l] = α^l (stored twice so exp[l1 + l2] needs no mod) and log (256,) with log[α^l] = l."""
    exp, log = torch.zeros(510, dtype=torch.long), torch.zeros(256, dtype=torch.long)
    x = 1
    for power in range(255):
        exp[power], log[x] = x, power
        x <<= 1  # multiply by α = x
        if x & 0x100:
            x ^= PRIMITIVE  # reduce modulo p(x)
    exp[255:] = exp[:255]
    return exp, log


EXP, LOG = gf_tables()


def gf_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a ⊗ b elementwise (broadcasting): exp[log a + log b], and 0 if either factor is 0."""
    exp, log = EXP.to(a.device), LOG.to(a.device)
    return torch.where((a == 0) | (b == 0), 0, exp[log[a] + log[b]])


def gf_div(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a ⊘ b elementwise for b ≠ 0: exp[(log a − log b) mod 255]."""
    exp, log = EXP.to(a.device), LOG.to(a.device)
    return torch.where(a == 0, 0, exp[(log[a] - log[b]) % 255])


def gf_pow_alpha(power: torch.Tensor) -> torch.Tensor:
    """α^power for integer tensors (negative powers allowed)."""
    return EXP.to(power.device)[power % 255]


def gf_sum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """XOR-reduce along dim: split bytes into 8 bits, add each bit plane modulo 2, reassemble."""
    k = torch.arange(8, device=x.device)
    bits = (x.movedim(dim, -1)[..., None] >> k) & 1  # (..., N, 8)
    return ((bits.sum(-2) % 2) << k).sum(-1)


def gf_matmul(x: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    r"""y_{bj} = ⊕_i x_{bi} ⊗ M_{ij}   (i summed with XOR).   x (B, I), M (I, J) → (B, J)"""
    return gf_sum(gf_mul(x[:, :, None], M[None]), dim=1)


def gf_poly_eval(P: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""P_b(x_n) = ⊕_d P_{bd} ⊗ x_n^d for ascending coefficients P (B, D) at non-zero points x (N,) → (B, N)."""
    d = torch.arange(P.shape[1], device=P.device)
    powers = gf_pow_alpha(LOG.to(x.device)[x][:, None] * d)  # (N, D): x_n^d
    return gf_sum(gf_mul(P[:, None, :], powers[None]), dim=-1)


# --- Reed–Solomon over GF(256) ---------------------------------------------------------------------------------------


def rs_generator(n_ec: int) -> torch.Tensor:
    """g(x) = Π_{j<T} (x − α^j) as descending coefficients (T + 1,), g[0] = 1."""
    g = torch.ones(1, dtype=torch.long)
    for j in range(n_ec):
        zero = torch.zeros(1, dtype=torch.long)
        g = torch.cat([g, zero]) ^ torch.cat([zero, gf_mul(g, gf_pow_alpha(torch.tensor(j)))])  # g·x + g·α^j
    return g


def rs_generator_matrix(k: int, n_ec: int) -> torch.Tensor:
    """G (k, T): row i holds x^{T+k−1−i} mod g(x), descending, so ECC bytes = ⊕_i d_i ⊗ G_i (systematic encoding)."""
    g = rs_generator(n_ec)
    r = g[1:].clone()  # x^T ≡ g(x) − x^T = g[1:]   (minus = plus in GF(2^8))
    rows = [r]
    for _ in range(k - 1):  # x·r mod g: shift left and fold the overflowing coefficient back in
        lead, r = r[0], torch.cat([r[1:], torch.zeros(1, dtype=torch.long)])
        r = r ^ gf_mul(lead, g[1:])
        rows.append(r)
    return torch.stack(rows[::-1])  # row i ↔ power T + k − 1 − i


def rs_encode(data: torch.Tensor, n_ec: int) -> torch.Tensor:
    r"""e_{bj} = ⊕_i d_{bi} ⊗ G_{ij}.   data (B, k) bytes → ECC bytes (B, T)."""
    return gf_matmul(data, rs_generator_matrix(data.shape[1], n_ec).to(data.device))


def syndrome_matrix(n: int, n_ec: int, device=None) -> torch.Tensor:
    """V (N, T) with V_ij = α^{j (N−1−i)}: column j evaluates the block polynomial at α^j."""
    i, j = torch.arange(n, device=device)[:, None], torch.arange(n_ec, device=device)[None]
    return gf_pow_alpha(j * (n - 1 - i))


def syndromes(c: torch.Tensor, n_ec: int) -> torch.Tensor:
    r"""S_{bj} = ⊕_i c_{bi} ⊗ α^{j(N−1−i)}.   c (B, N) → (B, T); all zero ⇔ c is a codeword."""
    return gf_matmul(c, syndrome_matrix(c.shape[1], n_ec, c.device))


# --- format information: BCH(15, 5) ---------------------------------------------------------------------------------


def format_code(ecl_bits: int, mask: int) -> int:
    """15-bit format word: data = (ECC level bits, mask) · x^10 + remainder mod 0x537, XOR 0x5412."""
    data = (ecl_bits << 3) | mask
    rem = data << 10
    for bit in range(14, 9, -1):  # long division by the degree-10 generator
        if rem >> bit & 1:
            rem ^= FORMAT_GENERATOR << (bit - 10)
    return ((data << 10) | rem) ^ FORMAT_XOR


def format_table() -> torch.Tensor:
    """All 32 format codewords as bits (32, 15), bit i = coefficient of x^i; row index = (ECC level bits) · 8 + mask."""
    words = torch.tensor([format_code(d >> 3, d & 7) for d in range(32)])
    return (words[:, None] >> torch.arange(15)) & 1


def format_decode(bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest valid codeword by Hamming distance. bits (B, 15) → index (B,) = ecl·8 + mask, bit errors (B,)."""
    dist = (bits[:, None, :] != format_table().to(bits.device)[None]).sum(-1)  # (B, 32)
    errors, index = dist.min(dim=1)
    return index, errors


# --- masks and the penalty score -------------------------------------------------------------------------------------


def mask_pattern(k: int, n: int, device=None) -> torch.Tensor:
    """Mask k of the standard as an (n, n) bool grid; i = row, j = column. Flipping data modules where it is True."""
    i, j = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    rule = [
        (i + j) % 2 == 0,
        i % 2 == 0,
        j % 3 == 0,
        (i + j) % 3 == 0,
        (i // 2 + j // 3) % 2 == 0,
        (i * j) % 2 + (i * j) % 3 == 0,
        ((i * j) % 2 + (i * j) % 3) % 2 == 0,
        ((i + j) % 2 + (i * j) % 3) % 2 == 0,
    ]
    return rule[k]


def penalty(grid: torch.Tensor) -> torch.Tensor:
    """Standard mask penalty N1 + N2 + N3 + N4 for a batch of symbols grid (K, n, n) of 0/1 → (K,).

    N1: each run of ≥ 5 equal modules in a row/column scores 3 + (length − 5) = (windows of 5 in it) + 2 per run.
    N2: 3 per 2 × 2 block of one colour. N3: 40 per finder-like 1:1:3:1:1 pattern with 4 light modules on one side
    (counted by window matching, outside the symbol counts as light). N4: 10 per 5 % the dark share is off 50 %."""
    total = torch.zeros(grid.shape[0], dtype=torch.long, device=grid.device)
    pattern = torch.tensor([1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0], device=grid.device)
    for g in (grid, grid.transpose(1, 2)):  # rows, then columns
        win = g.unfold(2, 5, 1)  # (K, n, n−4, 5)
        same = (win == win[..., :1]).all(-1)
        starts = torch.ones_like(same)
        starts[..., 1:] = g[..., :-5] != g[..., 1:-4]  # run begins at window start
        total += same.sum((1, 2)) + 2 * (same & starts).sum((1, 2))
        padded = F.pad(g, (4, 4))
        win11 = padded.unfold(2, 11, 1)  # (K, n, n−2, 11)
        hits = (win11 == pattern).all(-1) | (win11 == pattern.flip(0)).all(-1)
        total += 40 * hits.sum((1, 2))
    blocks = (grid[:, :-1, :-1] == grid[:, 1:, :-1]) & (grid[:, :-1, :-1] == grid[:, :-1, 1:])
    total += 3 * (blocks & (grid[:, :-1, :-1] == grid[:, 1:, 1:])).sum((1, 2))
    dark = grid.float().mean((1, 2))
    total += 10 * ((100 * dark - 50).abs() // 5).long()
    return total


# --- projective geometry ---------------------------------------------------------------------------------------------


def apply_homography(H: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    r"""x'_{bni} = H_{bij} x_{bnj} / (H_{b3j} x_{bnj})   (j summed).   H (B, 3, 3), x (B, N, 2) → (B, N, 2)"""
    xh = torch.cat([x, torch.ones_like(x[..., :1])], -1)
    y = torch.einsum("bij,bnj->bni", H, xh)
    return y[..., :2] / y[..., 2:]


def dlt(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    r"""Direct linear transform: h = argmin_{|h|=1} |A h| by SVD, two rows of A per match from x' × (H x) = 0.

    src, dst (B, N ≥ 4, 2) → H (B, 3, 3) with H₃₃ = 1. Points are first scaled to unit size (Hartley normalisation)."""

    def normaliser(x):
        centre, scale = x.mean(1), 2**0.5 / (x - x.mean(1, keepdim=True)).norm(dim=-1).mean(1)
        T = torch.zeros(x.shape[0], 3, 3, dtype=x.dtype, device=x.device)
        T[:, 0, 0] = T[:, 1, 1] = scale
        T[:, :2, 2] = -scale[:, None] * centre
        T[:, 2, 2] = 1
        return T

    T1, T2 = normaliser(src), normaliser(dst)
    a, b = apply_homography(T1, src), apply_homography(T2, dst)
    u, v, u2, v2 = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    one, zero = torch.ones_like(u), torch.zeros_like(u)
    rows_u = torch.stack([-u, -v, -one, zero, zero, zero, u2 * u, u2 * v, u2], -1)  # (B, N, 9)
    rows_v = torch.stack([zero, zero, zero, -u, -v, -one, v2 * u, v2 * v, v2], -1)
    A = torch.cat([rows_u, rows_v], 1)  # (B, 2N, 9)
    H = torch.linalg.svd(A)[2][:, -1].reshape(-1, 3, 3)  # right singular vector of the smallest singular value
    H = torch.linalg.solve(T2, H @ T1)
    return H / H[:, 2:, 2:]


# --- image filters ---------------------------------------------------------------------------------------------------


def box_mean(img: torch.Tensor, k: int) -> torch.Tensor:
    r"""Local mean ar I_{bxy} = (1/k²) Σ_{|u|,|v| ≤ k/2} I_{b,x+u,y+v}: a k × k box filter, done as two 1D
    conv2d passes (a row box, then a column box).  (B, H, W) → same"""
    kernel = torch.full((1, 1, 1, k), 1.0 / k, device=img.device)
    x = F.conv2d(F.pad(img[:, None], (k // 2, k // 2, 0, 0), mode="replicate"), kernel)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), kernel.transpose(2, 3))[:, 0]


@functools.cache
def profile_kernel(scale: float, edges: tuple, signs: tuple, slack: float = 0.25) -> torch.Tensor:
    """1D matched filter for a run-length profile: tap t gets the sign of the band |t|/scale falls in, or 0 within
    `slack` modules of a band edge (so blur and a slightly wrong scale cost little); normalised so a perfect match
    gives 1. Finder 1:1:3:1:1 (dark, light, dark, light, dark): edges (1.5, 2.5, 3.5), signs (+1, −1, +1) (|t| in
    modules from the centre). Cached per scale."""
    half = int(edges[-1] * scale)
    u = torch.arange(-half, half + 1, dtype=torch.float32).abs() / scale
    band = torch.bucketize(u, torch.tensor(edges[:-1]), right=True).clamp(max=len(signs) - 1)
    near_edge = (u[:, None] - torch.tensor(edges)).abs().min(1).values < slack
    k = torch.where(near_edge, 0.0, torch.tensor(signs, dtype=torch.float32)[band])
    return k / k.abs().sum()


def line_responses(d: torch.Tensor, kernel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Correlate the ±1 image d (B, H, W) with a 1D profile along rows and along columns:
    r^h_{bxy} = Σ_t K_t d_{b,x,y+t},  r^v_{bxy} = Σ_t K_t d_{b,x+t,y}.  Outside the image counts as light (−1)."""
    half = (len(kernel) - 1) // 2
    padded = F.pad(d[:, None], (half,) * 4, value=-1.0)
    kh = kernel.to(d.device).view(1, 1, 1, -1)
    rh = F.conv2d(padded[..., half:-half, :], kh)[:, 0]
    rv = F.conv2d(padded[..., half:-half], kh.transpose(2, 3))[:, 0]
    return rh, rv
