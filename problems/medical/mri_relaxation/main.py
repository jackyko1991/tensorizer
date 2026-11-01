"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import null_point


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    T1, T2, PD = tensors.tissue_parameters(dev)
    t_long = tensors.time_axis(5000.0, 400, dev)  # ms, for T1 recovery
    t_short = tensors.time_axis(300.0, 300, dev)  # ms, for T2 decay
    TR, TE = tensors.protocol_grids(dev)
    labels = tensors.phantom(params["size"], dev)

    protocols = {  # weighting → (TR, TE) in ms
        "T1-weighted": (params["tr_short"], params["te_short"]),
        "PD-weighted": (params["tr_long"], params["te_short"]),
        "T2-weighted": (params["tr_long"], params["te_long"]),
    }
    path = solve.sweep(TR, TE, list(protocols.values()), params["n_frames"])
    Mz, Mxy = solve.relaxation_curves(T1, T2, PD, t_long, t_short)
    S, frames = solve.solve(T1, T2, PD, TR, TE, labels, path)

    print("tissue         T1 (ms)  T2 (ms)   PD   null TI = T1 ln2 (ms)")
    for name, a, b, p, ti in zip(tensors.TISSUES, T1.tolist(), T2.tolist(), PD.tolist(), null_point(T1).tolist()):
        print(f"{name:13s} {a:8.0f} {b:8.0f} {p:5.2f} {ti:10.0f}")
    for weighting, (tr, te) in protocols.items():
        s = S[:, solve.nearest(TR, tr), solve.nearest(TE, te)]
        ranking = " > ".join(tensors.TISSUES[i] for i in s.argsort(descending=True).tolist())
        print(f"{weighting} (TR {tr:.0f}, TE {te:.0f} ms): {ranking}")
    return plot.animate(t_long, t_short, Mz, Mxy, TR, TE, path, frames, out_dir / "mri_relaxation.gif")


if __name__ == "__main__":
    cli(run, __file__)
