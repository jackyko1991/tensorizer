"""Build the concrete tensors: the QR symbol (function patterns, codewords, placement, mask) and synthetic photos of it.

Grid convention: module (i, j) is row i, column j, with 1 = dark. In the code plane the module occupies
X ∈ [j, j + 1), Y ∈ [i, i + 1) (module units); the quiet zone is the 4-module light border around it.
Pixel convention: pixel (column u, row v) has its centre at image coordinates (u, v).
"""

import math

import torch
import torch.nn.functional as F

from . import equation as eq

# ISO/IEC 18004 tables for versions 1–3 (one or two equal blocks, so interleaving is a transpose).
TOTAL_CODEWORDS = {1: 26, 2: 44, 3: 70}
ECC_PER_BLOCK = {"L": (7, 10, 15), "M": (10, 16, 26), "Q": (13, 22, 18), "H": (17, 28, 22)}
BLOCKS = {"L": (1, 1, 1), "M": (1, 1, 1), "Q": (1, 1, 2), "H": (1, 1, 2)}
ECL_BITS = {"L": 1, "M": 0, "Q": 3, "H": 2}  # the 2 bits stored in the format information
LEVELS = ["L", "M", "Q", "H"]
QUIET = 4


def size(version: int) -> int:
    return 17 + 4 * version


def layout(version: int, ecl: str) -> dict:
    """Block structure: number of blocks, data bytes k and ECC bytes T per block, byte-mode capacity."""
    nb, n_ec = BLOCKS[ecl][version - 1], ECC_PER_BLOCK[ecl][version - 1]
    k = TOTAL_CODEWORDS[version] // nb - n_ec
    return {"blocks": nb, "k": k, "n_ec": n_ec, "capacity": (8 * nb * k - 12) // 8}  # 4 mode + 8 count bits


def smallest_version(message: bytes, ecl: str, at_least: int = 1) -> int | None:
    fits = [v for v in range(max(1, at_least), 4) if len(message) <= layout(v, ecl)["capacity"]]
    return fits[0] if fits else None


def format_positions(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(row, col) of format bit i (bit 0 = x^0) for the copy around the top-left finder and the split second copy."""
    first = [(i, 8) for i in range(6)] + [(7, 8), (8, 8), (8, 7)] + [(8, 14 - i) for i in range(9, 15)]
    second = [(8, n - 1 - i) for i in range(8)] + [(n - 15 + i, 8) for i in range(8, 15)]
    return torch.tensor(first), torch.tensor(second)


def function_patterns(version: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed modules: values (n, n) and a mask (n, n) of modules that are not data (finders + separators, timing,
    alignment, format areas, the dark module). Format bits are left light here and drawn by draw_format."""
    n = size(version)
    val = torch.zeros(n, n, dtype=torch.long)
    fun = torch.zeros(n, n, dtype=torch.bool)
    val[6, :] = val[:, 6] = (torch.arange(n) % 2 == 0).long()  # timing patterns
    fun[6, :] = fun[:, 6] = True
    d = torch.arange(-4, 5)
    ring = torch.maximum(d.abs()[:, None], d.abs()[None])  # Chebyshev distance from the centre
    for r, c in [(3, 3), (3, n - 4), (n - 4, 3)]:  # finder 7 × 7 plus its light separator
        rows, cols = r + d, c + d
        ok_r, ok_c = (rows >= 0) & (rows < n), (cols >= 0) & (cols < n)
        sub = ((ring != 2) & (ring != 4)).long()[ok_r][:, ok_c]
        val[rows[ok_r][:, None], cols[ok_c][None]] = sub
        fun[rows[ok_r][:, None], cols[ok_c][None]] = True
    if version >= 2:  # one alignment pattern centred at (n − 7, n − 7)
        a = n - 7
        val[a - 2 : a + 3, a - 2 : a + 3] = (ring[2:7, 2:7] != 1).long()
        fun[a - 2 : a + 3, a - 2 : a + 3] = True
    for pos in format_positions(n):
        fun[pos[:, 0], pos[:, 1]] = True
    val[n - 8, 8], fun[n - 8, 8] = 1, True  # the dark module
    return val, fun


def data_positions(fun: torch.Tensor) -> torch.Tensor:
    """(row, col) of every data module in placement order (M, 2): two-module-wide columns from the right, zig-zagging
    up and down, skipping the vertical timing column 6."""
    n = fun.shape[0]
    order = []
    right = n - 1
    while right >= 1:
        if right == 6:
            right = 5
        upward = ((right + 1) & 2) == 0
        for step in range(n):
            row = n - 1 - step if upward else step
            for col in (right, right - 1):
                if not fun[row, col]:
                    order.append((row, col))
        right -= 2
    return torch.tensor(order)


def draw_format(grid: torch.Tensor, ecl: str, mask: int) -> torch.Tensor:
    """Write both copies of the 15 format bits into grid (..., n, n)."""
    word = eq.format_code(ECL_BITS[ecl], mask)
    bits = (torch.tensor(word) >> torch.arange(15)) & 1
    grid = grid.clone()
    for pos in format_positions(grid.shape[-1]):
        grid[..., pos[:, 0], pos[:, 1]] = bits
    return grid


def data_codewords(message: bytes, version: int, ecl: str) -> torch.Tensor:
    """Byte-mode bit stream: mode 0100, 8-bit length, the bytes, terminator, then pad bytes 0xEC 0x11 → (nb·k,)."""
    lay = layout(version, ecl)
    n_data = lay["blocks"] * lay["k"]
    bits = [0, 1, 0, 0] + [(len(message) >> s) & 1 for s in range(7, -1, -1)]
    bits += [(byte >> s) & 1 for byte in message for s in range(7, -1, -1)]
    bits += [0] * min(4, 8 * n_data - len(bits))  # terminator
    bits += [0] * (-len(bits) % 8)
    words = [int("".join(map(str, bits[i : i + 8])), 2) for i in range(0, len(bits), 8)]
    words += [0xEC, 0x11] * ((n_data - len(words)) // 2 + 1)
    return torch.tensor(words[:n_data])


def codewords(message: bytes, version: int, ecl: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Data split into blocks D (nb, k), ECC bytes E (nb, T) by Reed–Solomon, interleaved column by column."""
    lay = layout(version, ecl)
    D = data_codewords(message, version, ecl).reshape(lay["blocks"], lay["k"])
    E = eq.rs_encode(D, lay["n_ec"])
    return torch.cat([D.T.reshape(-1), E.T.reshape(-1)]), torch.cat([D, E], 1)


def symbol(message: bytes, version: int, ecl: str, mask: int | None = None) -> dict:
    """The finished QR symbol. mask=None → try all 8 masks at once and keep the lowest penalty."""
    val, fun = function_patterns(version)
    pos = data_positions(fun)
    stream, blocks = codewords(message, version, ecl)
    bits = ((stream[:, None] >> torch.arange(7, -1, -1)) & 1).reshape(-1)  # MSB first
    raw = val.clone()
    raw[pos[: len(bits), 0], pos[: len(bits), 1]] = bits  # remainder bits stay 0
    n = size(version)
    masks = torch.stack([eq.mask_pattern(k, n) for k in range(8)]) & ~fun  # data modules only
    candidates = torch.stack([draw_format(raw ^ masks[k].long(), ecl, k) for k in range(8)])  # (8, n, n)
    scores = eq.penalty(candidates)
    chosen = int(scores.argmin()) if mask is None else mask
    return {
        "grid": candidates[chosen],
        "unmasked": raw,
        "function": fun,
        "positions": pos,
        "mask": chosen,
        "penalties": scores,
        "blocks": blocks,
        "version": version,
        "ecl": ecl,
    }


# --- synthetic photos ------------------------------------------------------------------------------------------------


def homographies(n: int, count: int, image: int, perspective: float, gen: torch.Generator) -> torch.Tensor:
    """Random code-plane → image homographies (B, 3, 3): rotate by any angle, scale the symbol to about 60 % of the
    frame, then push each corner by up to perspective × side in a random direction and solve the DLT."""
    corners = torch.tensor([[0.0, 0.0], [n, 0.0], [n, n], [0.0, n]])
    theta = 2 * math.pi * torch.rand(count, generator=gen)
    side = image * 0.6 * (0.9 + 0.15 * torch.rand(count, generator=gen))
    rot = torch.stack([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)], -1).view(-1, 2, 2)
    centred = (corners - n / 2) / n  # (4, 2)
    shift = image / 2 + 0.04 * image * (2 * torch.rand(count, 1, 2, generator=gen) - 1)
    img_corners = shift + side[:, None, None] * torch.einsum("bij,cj->bci", rot, centred)
    img_corners = img_corners + perspective * side[:, None, None] * (2 * torch.rand(count, 4, 2, generator=gen) - 1)
    return eq.dlt(corners.expand(count, 4, 2), img_corners)


def render(grid: torch.Tensor, H: torch.Tensor, image: int, occlusion: float, noise: float, gen: torch.Generator):
    """Photos (B, image, image) in [0, 1] of the symbol under homographies H (B, 3, 3), with a textured table, an
    occluding blob of area occlusion × n² placed over the symbol, uneven lighting, blur and pixel noise."""
    n, B = grid.shape[0], H.shape[0]
    ss = 2  # supersampling
    s = (torch.arange(ss * image) + 0.5) / ss - 0.5
    vv, uu = torch.meshgrid(s, s, indexing="ij")
    pix = torch.stack([uu, vv], -1).reshape(1, -1, 2).expand(B, -1, -1)
    XY = eq.apply_homography(torch.linalg.inv(H), pix)  # (B, P, 2) code-plane point seen by every sub-pixel
    X, Y = XY[..., 0], XY[..., 1]
    col, row = X.floor().long(), Y.floor().long()
    inside = (col >= 0) & (col < n) & (row >= 0) & (row < n)
    module = grid[row.clamp(0, n - 1), col.clamp(0, n - 1)] * inside
    card = (X >= -QUIET) & (X < n + QUIET) & (Y >= -QUIET) & (Y < n + QUIET)
    table = (
        0.42
        + 0.06 * torch.sin(0.09 * pix[..., 0] + 0.5 * torch.sin(0.05 * pix[..., 1]))
        + 0.04 * torch.sin(0.4 * pix[..., 1])
    )
    img = torch.where(card, 0.92 - 0.8 * module, table)
    # Occluding blob: an ellipse over the data region (a thumb or a sticker), dark or light.
    r = torch.exp(0.3 * (2 * torch.rand(B, generator=gen) - 1))  # aspect ratio
    a = n * (occlusion / math.pi) ** 0.5
    centre = n * (0.35 + 0.3 * torch.rand(B, 2, generator=gen))
    phi = math.pi * torch.rand(B, generator=gen)
    dx, dy = X - centre[:, :1], Y - centre[:, 1:]
    p = dx * torch.cos(phi)[:, None] + dy * torch.sin(phi)[:, None]
    q = -dx * torch.sin(phi)[:, None] + dy * torch.cos(phi)[:, None]
    blob = (p / (a * r[:, None])) ** 2 + (q * r[:, None] / a) ** 2 < 1
    shade = torch.where(torch.rand(B, generator=gen) < 0.5, 0.18, 0.8)[:, None]
    if occlusion > 0:
        img = torch.where(blob, shade, img)
    img = img.reshape(B, image, ss, image, ss).mean((2, 4))
    # Uneven lighting: a random linear gradient times a vignette.
    t = torch.linspace(-1, 1, image)
    gy, gx = torch.meshgrid(t, t, indexing="ij")
    ang = 2 * math.pi * torch.rand(B, 1, 1, generator=gen)
    light = (0.8 + 0.3 * (gx * torch.cos(ang) + gy * torch.sin(ang))) * (1 - 0.18 * (gx**2 + gy**2))
    img = img * light
    taps = torch.exp(-0.5 * (torch.arange(-2, 3.0) / 0.9) ** 2)  # Gaussian blur σ = 0.9 px, separable
    taps = taps / taps.sum()
    img = F.conv2d(F.pad(img[:, None], (2, 2, 0, 0), mode="replicate"), taps.view(1, 1, 1, 5))
    img = F.conv2d(F.pad(img, (0, 0, 2, 2), mode="replicate"), taps.view(1, 1, 5, 1))[:, 0]
    img = img + noise * torch.randn(img.shape, generator=gen)
    return img.clamp(0, 1)


def photos(grid, count, image, perspective, occlusion, noise, gen, chunk: int = 8):
    """Render count photos in chunks (memory). Returns images (B, image, image) and the true homographies."""
    H = homographies(grid.shape[0], count, image, perspective, gen)
    imgs = [render(grid, H[i : i + chunk], image, occlusion, noise, gen) for i in range(0, count, chunk)]
    return torch.cat(imgs), H


def codeword_map(version: int, ecl: str) -> tuple[torch.Tensor, torch.Tensor]:
    """For every module, the (block, byte position) its bit belongs to after de-interleaving; −1 for modules that
    carry no codeword (function patterns, remainder bits). → two (n, n) long tensors."""
    lay = layout(version, ecl)
    nb, k = lay["blocks"], lay["k"]
    _, fun = function_patterns(version)
    pos = data_positions(fun)[: 8 * TOTAL_CODEWORDS[version]]
    byte = torch.arange(len(pos)) // 8  # index in the interleaved stream
    is_ecc = byte >= nb * k
    rel = torch.where(is_ecc, byte - nb * k, byte)
    n = size(version)
    block, place = torch.full((n, n), -1), torch.full((n, n), -1)
    block[pos[:, 0], pos[:, 1]] = rel % nb
    place[pos[:, 0], pos[:, 1]] = torch.where(is_ecc, k + rel // nb, rel // nb)
    return block, place
