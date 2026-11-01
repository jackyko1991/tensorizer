"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import apply, det


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    custom = [params["a11"], params["a12"], params["a21"], params["a22"]]
    a, captions = tensors.matrices(params["matrix"], custom, dev)  # (S, 2, 2)
    eps = tensors.levi_civita(dev)
    lines = tensors.grid_lines(params["grid_lines"], 4.0, dev)  # (L, 41, 2), covering [−4, 4]²
    circle = tensors.unit_circle(200, dev)  # (200, 2)
    square = tensors.unit_square(dev)  # (5, 2)

    per = 70 // len(a)  # at most 70 frames in all
    moving = per * 3 // 4  # 3/4 of them walk from I to A, the rest rest on A
    m, which, t = solve.path(a, frames=moving, hold=per - moving)  # (T, 2, 2)
    y_lines, y_circle, y_square = apply(m, lines), apply(m, circle), apply(m, square)  # (T, …, 2)
    print(f"{len(a)} matrices, {m.shape[0]} frames: one einsum maps {lines.shape[0] * lines.shape[1]} grid points")

    u, sigma = solve.singular(m)  # (T, 2, 2), (T, 2)
    d = det(m, eps)  # (T,)
    axes = solve.circle_axes(y_circle)  # (T, 2)
    eig = []
    for s, (a_s, caption) in enumerate(zip(a, captions)):
        lam, vec = solve.real_eigenvectors(a_s)
        eig.append(vec)
        end = (which == s).nonzero()[-1].item()  # last frame of this matrix: M = A
        rows = ", ".join(f"[{r[0]:+.2f} {r[1]:+.2f}]" for r in a_s.tolist())
        print(f"A = [{rows}]  {caption}")
        print(f"   det A = {d[end]:+.3f} (area ×{abs(d[end]):.3f}{', mirrored' if d[end] < 0 else ''})")
        print(
            f"   singular values {sigma[end, 0]:.3f}, {sigma[end, 1]:.3f};  circle image axes "
            f"{axes[end, 0]:.3f}, {axes[end, 1]:.3f}"
        )
        if lam.numel():
            v_moved = a_s @ vec  # A v = λ v: the image lies on the same line
            print(
                f"   eigenvalues {', '.join(f'{v:+.3f}' for v in lam.tolist())};  "
                f"|A v − λ v| = {(v_moved - vec * lam).abs().max():.1e}"
            )
        else:
            print("   no real eigenvalues: no direction stays on its own line")
    return plot.animate(
        m, which, t, y_lines, y_circle, y_square, u, sigma, d, eig, captions, out_dir / "linear_maps_2d.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
