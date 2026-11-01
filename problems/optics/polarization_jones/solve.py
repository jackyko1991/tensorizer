"""Send the light through the wave plate and the analyzer at every angle; record fields and intensities."""

import torch

from .equation import apply, chain, field_trace, intensity, polarizer


def solve(e_in: torch.Tensor, w: torch.Tensor, angles: torch.Tensor, wt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a dict with

    ellipse_in (T, 2), ellipse_plate (T, 2): field traces before and after the plate,
    e_out (A, 2) complex: field after the analyzer at every analyzer angle,
    intensity (A,): transmitted intensity with the plate,  malus (A,): the same without it.
    """
    e_plate = apply(w, e_in)  # (2,) light between plate and analyzer
    e_out, i_out, i_malus = [], [], []
    for theta in angles.tolist():
        p = polarizer(theta)
        m = chain(w, p)  # plate first, then analyzer:  M = P W
        e_out.append(apply(m, e_in))
        i_out.append(intensity(e_out[-1]))
        i_malus.append(intensity(apply(p, e_in)))  # analyzer alone
    return {
        "e_plate": e_plate,
        "ellipse_in": field_trace(e_in, wt),
        "ellipse_plate": field_trace(e_plate, wt),
        "e_out": torch.stack(e_out),
        "intensity": torch.stack(i_out),
        "malus": torch.stack(i_malus),
    }


def describe(e: torch.Tensor) -> tuple[float, float, float]:
    """(|Ex|, |Ey|, phase of Ey relative to Ex in degrees) of a Jones vector."""
    phase = torch.rad2deg(torch.angle(e[1]) - torch.angle(e[0]))
    phase = (phase + 180) % 360 - 180  # wrap into [−180°, 180°)
    return e[0].abs().item(), e[1].abs().item(), phase.item()
