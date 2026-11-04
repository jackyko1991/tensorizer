"""Main loop: two partial scans of Spot → VoxelGrid → outlier removal → normals → FPFH → RANSAC → ICP → errors."""

import time
from pathlib import Path

from problems._common.runtime import cli, fast_mode

from . import plot, solve
from .equation import rotation_error_deg

SEED = 0


def run(params: dict, out_dir: Path) -> Path:
    params = dict(params)
    if fast_mode():
        params.update(n_points=12000, hypotheses=1000, icp_iterations=10, starts=3)
    t0 = time.perf_counter()
    out = solve.register(params, SEED)
    elapsed = time.perf_counter() - t0
    tgt, src = out["tgt"], out["src"]
    print(
        f"scan A: {out['n_a']} visible points + {len(out['A']) - out['n_a']} outliers; "
        f"scan B: {out['n_b']} + {len(out['B']) - out['n_b']} (Spot, object size 1, noise σ = {params['noise']:g})"
    )
    print(
        f"VoxelGrid ({params['voxel']:g}): A {len(out['A'])} → {len(tgt['down'])}, B {len(out['B'])} → "
        f"{len(src['down'])};  StatisticalOutlierRemoval: A → {len(tgt['P'])}, B → {len(src['P'])}"
    )
    print(f"normals: median curvature λ0/Σλ = {tgt['curv'].median():.4f};  FPFH descriptors: {tuple(tgt['F'].shape)}")
    R0, t0_, inl = out["ransac"]
    print(
        f"FPFH mutual matches: {len(out['match'][0])};  RANSAC ({params['hypotheses']} hypotheses, batched Kabsch): "
        f"{int(inl.sum())} inliers, rotation error {out['ransac_rot_err']:.2f}°, translation error "
        f"{out['ransac_trans_err']:.4f}"
    )
    for method, res in out["icp"].items():
        rmse = res["hist"]["rmse"]
        best = int(rmse[-1].argmin())
        err = rotation_error_deg(res["R"][best], out["R_true"])
        print(
            f"ICP {method:>14} ({params['starts']} starts × {params['icp_iterations']} iterations): best RMSE "
            f"{rmse[0, best]:.4f} → {rmse[-1, best]:.4f}, rotation error {err:.2f}°, translation error "
            f"{(res['t'][best] - out['t_true']).norm():.4f}"
        )
    print(
        f"result (point-to-plane, best start): rotation error {out['rot_err']:.3f}° (test < 2°), translation error "
        f"{out['trans_err']:.4f} = {100 * out['trans_err'] / out['size']:.2f}% of the object size (test < 2%), "
        f"RMSE {out['rmse']:.4f};  pipeline {elapsed:.1f} s"
    )
    return plot.animate(out, out_dir / "point_cloud_registration.gif")


if __name__ == "__main__":
    cli(run, __file__)
