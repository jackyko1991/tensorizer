"""Animate the reader on each photo (photo → binarised → finders + homography → rectified → module errors →
Reed–Solomon fixes and text) next to the success rate against occlusion for the four ECC levels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

from . import equation as eq
from . import tensors

STAGES = ["photo", "adaptive threshold", "finders + homography", "rectified", "module errors", "Reed–Solomon"]
LEVEL_COLOURS = {"L": "#9ca3af", "M": "#2563eb", "Q": "#16a34a", "H": "#dc2626"}
GREY = plt.get_cmap("gray", 16)  # few grey levels keep the GIF small


def animate(img, results, sym, block, place, message, occs, rates, versions, occlusion, path: Path) -> Path:
    n = sym["grid"].shape[0]
    truth = sym["grid"].numpy()
    events = [(b, s) for b in range(len(results)) for s in range(len(STAGES))]

    fig = plt.figure(figsize=(9.6, 5.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1.15], wspace=0.25, hspace=0.35)
    ax_img, ax_txt, ax_rate = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])

    x = 100 * np.array(occs)
    for level, rate in rates.items():
        ax_rate.plot(x, 100 * rate.numpy(), "o-", ms=4, color=LEVEL_COLOURS[level],
                     label=f"{level} (v{versions[level]})")  # fmt: skip
    ax_rate.axvline(100 * occlusion, color="#f59e0b", ls="--", lw=1.2, label="these photos")
    ax_rate.set_xlabel("occluded share of the symbol (%)")
    ax_rate.set_ylabel("photos decoded (%)")
    ax_rate.set_ylim(-5, 105)
    ax_rate.set_title("success rate per ECC level", fontsize=10)
    ax_rate.legend(fontsize=7, loc="lower left", ncol=2)
    ax_txt.axis("off")
    text = ax_txt.text(0, 1, "", family="monospace", fontsize=8.2, va="top", transform=ax_txt.transAxes)

    def update(f: int):
        b, stage = events[f]
        r = results[b]
        ax_img.clear()
        ax_img.set_axis_off()
        reached = 5 if "modules" in r else (2 if "finders" in r else 1)  # last stage this photo got to
        shown = min(stage, reached)
        if shown == 0:
            ax_img.imshow(img[b].numpy(), cmap=GREY, vmin=0, vmax=1)
        elif shown == 1:
            ax_img.imshow(r["dark"].numpy(), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        elif shown == 2:
            ax_img.imshow(img[b].numpy(), cmap=GREY, vmin=0, vmax=1)
            if "finders" in r:
                corners = torch.tensor([[0.0, 0.0], [n, 0.0], [n, n], [0.0, n], [0.0, 0.0]])
                v = r["version"]
                corners = corners * tensors.size(v) / n
                outline = eq.apply_homography(r["H"][None].cpu(), corners[None])[0].numpy()
                ax_img.plot(outline[:, 0], outline[:, 1], color="#2563eb", lw=1.8)
                f_ = r["finders"].numpy()
                ax_img.scatter(f_[:, 0], f_[:, 1], s=140, facecolors="none", edgecolors="#f59e0b", linewidths=2)
                for p, label in zip(f_, ["TL", "TR", "BL"]):
                    ax_img.text(p[0] + 9, p[1] - 9, label, color="#b45309", fontsize=9, weight="bold")
                fourth = r["points"][3].numpy()
                ax_img.scatter([fourth[0]], [fourth[1]], s=60, marker="s", facecolors="none", edgecolors="#dc2626")
        elif shown == 3:
            nv = tensors.size(r["version"])
            ax_img.imshow(r["rectified"].numpy(), cmap=GREY, vmin=0, vmax=1, extent=(-4, nv + 4, nv + 4, -4))
            ax_img.plot([0, nv, nv, 0, 0], [0, 0, nv, nv, 0], color="#2563eb", lw=1)
        else:
            mods = r["modules"].numpy()
            rgb = np.where(mods[..., None] == 1, 0.1, 0.95) * np.ones(3)
            if stage == 4 and mods.shape == truth.shape:
                wrong = mods != truth
                rgb[wrong] = [0.86, 0.15, 0.15]
            elif stage == 5 and mods.shape == truth.shape and r["ecl"] == sym["ecl"]:
                fixed = (r["received"] != r["corrected"]).numpy()  # (nb, N) bytes Reed–Solomon changed
                hit = (block >= 0).numpy() & fixed[block.clamp(min=0).numpy(), place.clamp(min=0).numpy()]
                rgb[hit] = np.where(mods[hit][:, None] == 1, [0.7, 0.35, 0.0], [0.98, 0.75, 0.35])
            ax_img.imshow(rgb, interpolation="nearest")
        names = {1: "binarised by a local box mean", 2: "1:1:3:1:1 finders, DLT outline"}
        title = names.get(shown, STAGES[shown])
        if shown == 4 and r["modules"].shape == truth.shape:
            title = f"sampled modules: {int((r['modules'].numpy() != truth).sum())} wrong (red)"
        if shown == 5:
            title = "bytes fixed by Reed–Solomon (orange)"
        ax_img.set_title(f"photo {b + 1}/{len(results)} · {title}", fontsize=10)

        lines = [f"message : '{message}'", f"symbol  : version {sym['version']}-{sym['ecl']}, mask {sym['mask']}", ""]
        lines.append(f"photo {b + 1}, step {stage + 1}/6: {STAGES[stage]}")
        if "finders" in r and stage >= 2:
            lines.append(f"version from the template fit: {r['version']} (fit {r['fit']:.2f})")
        if "byte_errors" in r and stage >= 3:
            lines.append(f"format: level {r['ecl']}, mask {r['mask']}, {r['format_errors']} bit(s) fixed (BCH)")
        if "byte_errors" in r and stage >= 4:
            t = tensors.layout(r["version"], r["ecl"])["n_ec"] // 2
            fixed = ", ".join(f"{e}/{t}" if ok else f">{t}" for e, ok in zip(r["byte_errors"], r["rs_ok"]))
            lines.append(f"RS byte errors fixed per block: {fixed}")
        if stage == 5 or stage > reached:
            if r["text"] is not None:
                verdict = "correct" if r["text"] == message else "WRONG"
                lines += ["", f"decoded : '{r['text']}'", f"          {verdict}"]
            else:
                lines += ["", f"failed at: {r['stage']}"]
        text.set_text("\n".join(lines))

    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.1)
    return save_gif(fig, update, frame_indices(len(events), 120), path, fps=2)
