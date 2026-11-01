r"""Image filtering → tensor translation.

A linear filter replaces every pixel by a weighted sum of its neighbours, with the same weights K everywhere:

    O_{ij} = \sum_{a,b} K_{ab}\, I_{i+a-r,\, j+b-r}          (a, b run over the (2r+1)×(2r+1) kernel window)

(this is cross-correlation, which deep-learning libraries call convolution; for the symmetric kernels used here
the two are the same). Unfolding copies the window around every pixel p into a column P_{kp}, k = (a, b), so the
filter is a single contraction O_p = K_k P_{kp}. The convolution theorem says the same filter is a product in
Fourier space, Ô_{uv} = Ĥ_{uv} Î_{uv}, where the frequency response Ĥ shows which spatial frequencies survive.

Non-linear filters reuse the unfolded windows: the median filter takes the median of every column P_{:p}.
"""

import torch
import torch.nn.functional as F


def pad(img: torch.Tensor, rh: int, rw: int, mode: str) -> torch.Tensor:
    """(H, W) → (1, 1, H + 2rh, W + 2rw), extending the border by reflection or wrapping around (circular)."""
    return F.pad(img[None, None], (rw, rw, rh, rh), mode=mode)


def filter2d(img: torch.Tensor, kernel: torch.Tensor, mode: str = "reflect") -> torch.Tensor:
    r"""O_{ij} = \sum_{ab} K_{ab} I_{i+a-r, j+b-r} with PyTorch's conv2d.   (H, W), (kh, kw) → (H, W)"""
    x = pad(img, kernel.shape[0] // 2, kernel.shape[1] // 2, mode)
    return F.conv2d(x, kernel[None, None])[0, 0]


def filter2d_unfold(img: torch.Tensor, kernel: torch.Tensor, mode: str = "reflect") -> torch.Tensor:
    r"""O_p = K_k P_{kp}   (k summed over the window): the same filter as one einsum.   → (H, W)"""
    kh, kw = kernel.shape
    patches = F.unfold(pad(img, kh // 2, kw // 2, mode), (kh, kw))[0]  # (kh·kw, H·W)
    return torch.einsum("k,kp->p", kernel.flatten(), patches).reshape(img.shape)


def separable(img: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    r"""G_{ab} = g_a g_b ⇒ filter the rows with g, then the columns: 2(2r+1) instead of (2r+1)² weights per pixel."""
    rows = filter2d(img, g[None, :])  # (1, k) kernel: along j
    return filter2d(rows, g[:, None])  # (k, 1) kernel: along i


def gradient(img: torch.Tensor, kx: torch.Tensor, ky: torch.Tensor):
    r"""Sobel gradients G_x, G_y, magnitude \sqrt{G_x² + G_y²} and orientation atan2(G_y, G_x).   all (H, W)"""
    gx, gy = filter2d(img, kx), filter2d(img, ky)
    return gx, gy, torch.sqrt(gx**2 + gy**2), torch.atan2(gy, gx)


def unsharp(img: torch.Tensor, g: torch.Tensor, amount: float) -> torch.Tensor:
    r"""Unsharp mask O = I + α (I − G * I): add back the detail that blurring removes."""
    return img + amount * (img - separable(img, g))


def median(img: torch.Tensor, size: int) -> torch.Tensor:
    """Median of every size × size window (non-linear, so not a convolution).   (H, W) → (H, W)"""
    patches = F.unfold(pad(img, size // 2, size // 2, "reflect"), size)[0]  # (size², H·W)
    return patches.median(dim=0).values.reshape(img.shape)


def embed(kernel: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Place the flipped kernel on an H × W grid with its centre at pixel (0, 0), wrapping around."""
    out = torch.zeros(shape, device=kernel.device)
    kh, kw = kernel.shape
    out[:kh, :kw] = kernel.flip(0, 1)  # flip: the FFT product is a true convolution, conv2d is a correlation
    return out.roll((-(kh // 2), -(kw // 2)), dims=(0, 1))


def filter_fft(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    r"""Convolution theorem: O = F⁻¹[Ĥ · Î], equal to filter2d with circular padding.   → (H, W)"""
    h_hat = torch.fft.fft2(embed(kernel, img.shape))
    return torch.fft.ifft2(h_hat * torch.fft.fft2(img)).real


def frequency_response(kernel: torch.Tensor, n: int) -> torch.Tensor:
    """Ĥ(k_x, k_y = 0) for k_x = 0 … π (n/2 + 1 values): the factor each horizontal frequency is multiplied by.
    A symmetric kernel has a real response; a negative value flips the sign of that frequency."""
    return torch.fft.rfft2(embed(kernel, (n, n)))[0].real
