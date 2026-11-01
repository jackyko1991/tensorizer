"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    f, h = params["f"], params["h"]
    s = tensors.object_distances(params["s_start"], params["s_end"], f)
    slope_in = tensors.principal_slopes(h, s, f)

    result = solve.solve(s, h, f, slope_in)

    for k in range(0, len(s), 20):
        kind = "real, inverted" if result["s_img"][k] > 0 else "virtual, upright"
        print(f"s = {s[k]:5.1f} cm  ->  s' = {result['s_img'][k]:7.1f} cm,  m = {result['m'][k]:6.2f}  ({kind})")
    return plot.animate(s, h, f, result, out_dir / "thin_lens_image.gif")


if __name__ == "__main__":
    cli(run, __file__)
