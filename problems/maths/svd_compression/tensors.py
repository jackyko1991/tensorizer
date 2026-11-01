"""Build the concrete tensors: a procedurally drawn colour test image (nothing is downloaded) and its grey version."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

LUMA = [0.299, 0.587, 0.114]  # weights w_c of red, green, blue in the brightness


def test_image(size: int, device: torch.device) -> torch.Tensor:
    """A synthetic scene drawn with matplotlib: sky gradient, sun, hills, a house, stripes and text.

    Returns A_{cij} in [0, 1] with shape (3, size, size): channel c (red, green, blue), row i, column j.
    """
    fig = plt.figure(figsize=(1, 1), dpi=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    t = np.linspace(0, 1, 64)[:, None]
    sky = np.stack([0.55 + 0.4 * t, 0.75 + 0.2 * t, np.ones_like(t)], axis=-1)  # (64, 1, 3), light at the bottom
    ax.imshow(sky[::-1], extent=[0, 1, 0.35, 1], aspect="auto")
    ax.add_patch(Circle((0.78, 0.8), 0.09, color="#f59e0b"))
    ax.add_patch(Polygon([[0, 0.35], [0.3, 0.62], [0.62, 0.35]], color="#16a34a"))
    ax.add_patch(Polygon([[0.35, 0.35], [0.72, 0.55], [1, 0.35]], color="#65a30d"))
    ax.add_patch(Rectangle((0, 0), 1, 0.35, color="#a16207"))
    for k in range(6):  # a striped field: sharp edges need many singular values
        ax.add_patch(Rectangle((0, 0.05 * k), 1, 0.025, color="#ca8a04"))
    ax.add_patch(Rectangle((0.1, 0.22), 0.22, 0.2, color="#dc2626"))  # house
    ax.add_patch(Polygon([[0.07, 0.42], [0.21, 0.52], [0.35, 0.42]], color="#7f1d1d"))
    ax.add_patch(Rectangle((0.18, 0.22), 0.06, 0.1, color="#1e3a8a"))
    # The figure is 1 inch = 72 points tall, so font sizes are fractions of the image height.
    ax.text(0.5, 0.1, "A = UΣVᵀ", fontsize=10, ha="center", va="center", color="white", weight="bold")
    ax.text(0.05, 0.86, "SVD", fontsize=13, va="center", color="#1e40af", weight="bold", style="italic")
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())[:size, :size, :3]
    plt.close(fig)
    return torch.tensor(rgba / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1).contiguous()


def grey(rgb: torch.Tensor) -> torch.Tensor:
    """Brightness A_{ij} = w_c A_{cij} (c summed): (3, H, W) → (H, W)."""
    w = torch.tensor(LUMA, dtype=rgb.dtype, device=rgb.device)
    return torch.einsum("c,cij->ij", w, rgb)
