"""Build the concrete tensors: the real TrackML hits, the φ wedge that sets the problem size, toy events and cuts.

data/trackml_event1000_barrel.csv is one simulated proton–proton event of the TrackML Particle Tracking Challenge
(CERN / Kaggle 2018, ⟨μ⟩ = 200 pile-up collisions at HL-LHC conditions), training event 1000, cut to the four
pixel-barrel layers (volume 8) and |η| < 1: 11 519 hits from 2 452 particles plus 6 % noise hits. Source: Zenodo
record 4730167 (doi:10.5281/zenodo.4730167), licence CC BY 4.0. The other CSVs are the offline solver benchmark and
the cited TrackML state of the art.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).parent / "data"
DATA = HERE / "trackml_event1000_barrel.csv"
PARTICLES = HERE / "trackml_particles.csv"
BENCHMARK = HERE / "solver_benchmark.csv"
SOTA = HERE / "trackml_sota.csv"
DTYPE = torch.float64
B_FIELD = 2.0  # T, solenoid field of the TrackML detector
LAYER_R = (32.3, 72.1, 116.1, 172.1)  # mm, mean radius of the four pixel-barrel layers (volume 8)
N_LAYERS = len(LAYER_R)


@dataclass
class Cuts:
    """Geometric cuts for candidate segments, the triplet score widths and the cost of an unused segment."""

    pt_min: float = 1.0  # GeV: tracks softer than this curl too much to pass the Δφ cut
    z0_max: float = 150.0  # mm: the segment must point back to the luminous region on the beam line
    dphi_tol: float = 0.004  # rad: extra Δφ allowed for multiple scattering and module tilt
    sigma_kappa: float = 1.2e-4  # 1/mm: width of the curvature mismatch Δκ between the two halves of a triplet
    sigma_theta: float = 0.006  # rad: width of the polar-angle mismatch Δθ
    lam: float = 0.1  # cost of a selected segment (only segments that carry a good triplet pay for themselves)

    def radius_min(self) -> float:
        """Smallest helix radius allowed, in mm: R = p_T / (0.3 B) m with p_T in GeV and B in T."""
        return self.pt_min / (0.3 * B_FIELD) * 1000.0


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and ln[0] != "#"]
    return lines[0].split(","), [ln.split(",") for ln in lines[1:]]


def load_table(path: Path) -> list[dict]:
    """A small CSV (lines starting with # are comments) as a list of dicts, numbers converted where possible."""
    header, rows = _read_csv(path)

    def num(v: str):
        try:
            return float(v)
        except ValueError:
            return v

    return [{h: num(v) for h, v in zip(header, row)} for row in rows]


def load_event(path: Path = DATA, particles: Path = PARTICLES, device=None) -> dict:
    """Read the hits: xyz (N, 3) mm, layer (N,), particle (N,) (0 = noise), weight (N,), pt (N,) GeV (0 for noise)."""
    header, rows = _read_csv(path)
    t = torch.tensor([[float(v) for v in r] for r in rows], dtype=DTYPE)
    col = {h: t[:, i] for i, h in enumerate(header)}
    p_header, p_rows = _read_csv(particles)
    p = torch.tensor([[float(v) for v in r] for r in p_rows], dtype=DTYPE)
    pt_of = torch.zeros(int(p[:, 0].max()) + 1, dtype=DTYPE)
    pt_of[p[:, 0].long()] = p[:, p_header.index("pt")]
    particle = col["particle"].long()
    event = {
        "xyz": torch.stack([col["x"], col["y"], col["z"]], -1),
        "layer": col["layer"].long(),
        "particle": particle,
        "weight": col["weight"],
        "pt": pt_of[particle],
    }
    return {k: v.to(device) for k, v in event.items()}


def wedge(event: dict, dphi_deg: float, phi0_deg: float = 0.0) -> dict:
    """Keep the hits with azimuth φ in [φ₀, φ₀ + Δφ): the problem size grows linearly with Δφ (360° = all hits)."""
    phi = torch.rad2deg(torch.atan2(event["xyz"][:, 1], event["xyz"][:, 0]))  # (N,) in (−180°, 180°]
    keep = torch.remainder(phi - phi0_deg, 360.0) < dphi_deg
    return {k: v[keep] for k, v in event.items()}


def helix_hits(pt, phi0, eta, z0, charge, radii) -> torch.Tensor:
    """Where a helix from the beam line (x = y = 0, z = z0) crosses cylinders of radius r: (P, L, 3) in mm.

    In a solenoid B ẑ the transverse path is a circle of radius R = p_T/(0.3 B) through the origin; at radius r it
    has turned by asin(r/2R), and the arc length s = 2R asin(r/2R) sets z = z0 + s sinh η.
    """
    big_r = (pt / (0.3 * B_FIELD) * 1000.0)[:, None]  # (P, 1) mm
    r = radii[None, :]  # (1, L)
    turn = torch.asin(r / (2 * big_r))  # (P, L)
    phi = phi0[:, None] - charge[:, None] * turn
    z = z0[:, None] + 2 * big_r * turn * torch.sinh(eta)[:, None]
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], -1)


def toy_event(n_particles: int, noise_fraction: float = 0.0, seed: int = 0, device=None) -> dict:
    """A TrackML-like toy: helices with p_T in 1.5–10 GeV, evenly spread in φ (well separated), on the 4 barrel
    layers, plus uniform noise hits. Same keys as load_event."""
    gen = torch.Generator().manual_seed(seed)
    radii = torch.tensor(LAYER_R, dtype=DTYPE)
    pt = 1.5 + 8.5 * torch.rand(n_particles, generator=gen, dtype=DTYPE)
    phi0 = (torch.arange(n_particles, dtype=DTYPE) + 0.3 * torch.rand(n_particles, generator=gen, dtype=DTYPE)) * (
        2 * math.pi / n_particles
    )
    eta = 1.6 * torch.rand(n_particles, generator=gen, dtype=DTYPE) - 0.8
    z0 = 40.0 * torch.randn(n_particles, generator=gen, dtype=DTYPE).clamp(-2.5, 2.5)
    charge = torch.where(torch.rand(n_particles, generator=gen) < 0.5, -1.0, 1.0).to(DTYPE)
    xyz = helix_hits(pt, phi0, eta, z0, charge, radii).reshape(-1, 3)  # (P·L, 3)
    layer = torch.arange(N_LAYERS).repeat(n_particles)
    particle = torch.arange(1, n_particles + 1).repeat_interleave(N_LAYERS)
    n_noise = int(round(noise_fraction * xyz.shape[0]))
    if n_noise:
        nl = torch.randint(0, N_LAYERS, (n_noise,), generator=gen)
        ph = 2 * math.pi * torch.rand(n_noise, generator=gen, dtype=DTYPE)
        zz = radii[nl] * torch.sinh(1.6 * torch.rand(n_noise, generator=gen, dtype=DTYPE) - 0.8)
        noise = torch.stack([radii[nl] * torch.cos(ph), radii[nl] * torch.sin(ph), zz], -1)
        xyz, layer = torch.cat([xyz, noise]), torch.cat([layer, nl])
        particle = torch.cat([particle, torch.zeros(n_noise, dtype=torch.long)])
    pts = torch.cat([torch.zeros(1, dtype=DTYPE), pt])[particle]
    weight = (particle > 0).to(DTYPE)
    event = {"xyz": xyz, "layer": layer, "particle": particle, "weight": weight / weight.sum(), "pt": pts}
    return {k: v.to(device) for k, v in event.items()}
