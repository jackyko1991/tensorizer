"""Branching simulation: every particle is a row of a few tensors; splits replace rows by two new ones."""

import torch

from problems._common.progress import steps

from .equation import can_split, children, interaction_probability


def shower(
    state: dict[str, torch.Tensor],
    depths: torch.Tensor,
    steps_per_x0: int,
    e_crit: float,
    stochastic: bool,
    spread: float = 0.5,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Follow the cascade through the depth grid.

    Returns counts (T, 2) of [photons, e±] in flight, energy in flight (T,), deposited energy (T,), and the
    track segments (M, 5) = (z_start, x_start, z_end, x_end, kind) of every particle for drawing.
    """
    gen = torch.Generator(device=depths.device).manual_seed(seed)
    dt = 1.0 / steps_per_x0
    deposited = torch.zeros((), dtype=torch.float64, device=depths.device)
    counts, in_flight, dep, tracks = [], [], [], []

    def record():
        kind = state["kind"]
        counts.append(torch.stack([(kind == 0).sum(), (kind == 1).sum()]))
        in_flight.append(state["energy"].sum())
        dep.append(deposited.clone())

    record()
    for step in steps(len(depths) - 1, "depth steps"):
        z = depths[step + 1]
        state["x"] = state["x"] + state["angle"] * dt  # fly one step forward
        if stochastic:
            u = torch.rand(state["energy"].shape, generator=gen, device=depths.device, dtype=torch.float64)
            hit = u < interaction_probability(state["kind"], dt)
        else:
            hit = (step + 1 - state["born"]) >= steps_per_x0  # exactly one X0 after birth
        split = hit & can_split(state["energy"], e_crit)
        absorbed = hit & ~split

        tracks.append(_segments(state, hit, z))  # every particle that interacted ends its track here
        deposited = deposited + state["energy"][absorbed].sum()  # too soft to multiply: energy stays here

        energy, kind = children(state["energy"][split], state["kind"][split])  # (2S,) each
        n_split = int(split.sum())
        if stochastic:
            direction = torch.randn(2 * n_split, generator=gen, device=depths.device, dtype=torch.float64)
        else:
            direction = torch.cat([torch.ones(n_split), -torch.ones(n_split)]).to(depths)  # symmetric tree
        new = {
            "energy": energy,
            "kind": kind,
            "x": state["x"][split].repeat(2),
            "angle": state["angle"][split].repeat(2) + spread * e_crit / energy * direction,
            "z0": z.expand(2 * n_split),
            "x0": state["x"][split].repeat(2),
            "born": torch.full((2 * n_split,), step + 1, device=depths.device),
        }
        keep = ~hit
        state = {key: torch.cat([state[key][keep], new[key]]) for key in state}
        record()

    everyone = torch.ones_like(state["energy"], dtype=torch.bool)
    tracks.append(_segments(state, everyone, depths[-1]))  # particles still flying at the end
    return {
        "counts": torch.stack(counts),
        "in_flight": torch.stack(in_flight),
        "deposited": torch.stack(dep),
        "tracks": torch.cat(tracks),
    }


def _segments(state: dict[str, torch.Tensor], mask: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """(z_start, x_start, z_end, x_end, kind) of the masked particles: shape (m, 5)."""
    m = int(mask.sum())
    return torch.stack(
        [state["z0"][mask], state["x0"][mask], z.expand(m), state["x"][mask], state["kind"][mask].to(z.dtype)], dim=1
    )
