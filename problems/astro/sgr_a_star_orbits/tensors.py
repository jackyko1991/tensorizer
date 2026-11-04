"""Build the concrete tensors: the S2 data table, the parameter packing, random starting orbits and synthetic data.

data/s2.csv holds the S2 measurements used by Gillessen et al. 2017 (ApJ 837, 30), table 5 from VizieR/CDS
(J/ApJ/837/30): 145 astrometric epochs (offsets from Sgr A* in RA and Dec, mas, 1992–2016, NTT and VLT) and 44
radial velocities (km/s, 2000–2016, VLT and Keck). Everything is float64.

Parameter vector p (B, 8) = [a (mas), e, i (°), Ω (°), ω (°), t_p (yr), M (10⁶ M☉), R0 (kpc)]. The fit works on an
unconstrained copy θ = [ln a, logit e, i, Ω, ω, t_p, ln M, ln R0] (angles in degrees), so a > 0, 0 < e < 1, M, R0 > 0.
"""

from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "s2.csv"
DTYPE = torch.float64
NAMES = ("a", "e", "inc", "Omega", "omega", "tp", "M", "R0")


def load_s2(path: Path = DATA) -> dict:
    """Astrometry {t_a, ra, dec, ra_err, dec_err} (Na,) and radial velocities {t_v, rv, rv_err} (Nv,)."""
    lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and ln[0] != "#"]
    header = lines[0].split(",")
    rows = [dict(zip(header, ln.split(","))) for ln in lines[1:]]
    astro = [r for r in rows if r["kind"] == "a"]
    vel = [r for r in rows if r["kind"] == "rv"]

    def col(rs, key):
        return torch.tensor([float(r[key]) for r in rs], dtype=DTYPE)

    data = {
        "t_a": col(astro, "epoch"), "ra": col(astro, "ra_mas"), "ra_err": col(astro, "ra_err"),
        "dec": col(astro, "dec_mas"), "dec_err": col(astro, "dec_err"),
        "t_v": col(vel, "epoch"), "rv": col(vel, "rv_kms"), "rv_err": col(vel, "rv_err"),
    }  # fmt: skip
    data["n_res"] = 2 * len(astro) + len(vel)
    return data


def pack(values: dict) -> torch.Tensor:
    """Physical parameters from a dict with keys NAMES: (8,)."""
    return torch.tensor([float(values[k]) for k in NAMES], dtype=DTYPE)


def unconstrained(p: torch.Tensor) -> torch.Tensor:
    """θ from p: (B, 8) → (B, 8)."""
    e = p[:, 1]
    return torch.stack([p[:, 0].log(), torch.log(e / (1 - e)), p[:, 2], p[:, 3], p[:, 4], p[:, 5],
                        p[:, 6].log(), p[:, 7].log()], -1)  # fmt: skip


def constrained(theta: torch.Tensor) -> torch.Tensor:
    """p from θ: (B, 8) → (B, 8)."""
    return torch.stack([theta[:, 0].exp(), torch.sigmoid(theta[:, 1]), theta[:, 2], theta[:, 3], theta[:, 4],
                        theta[:, 5], theta[:, 6].exp(), theta[:, 7].exp()], -1)  # fmt: skip


def random_starts(n: int, gen: torch.Generator) -> torch.Tensor:
    """n broad random starts: a 60–250 mas, e 0.3–0.95, any orientation, t_p 1995–2012, M 2–8, R0 6–11 kpc: (n, 8)."""
    u = torch.rand(n, 8, generator=gen, dtype=DTYPE)
    lo = torch.tensor([60.0, 0.3, 0.0, 0.0, 0.0, 1995.0, 2.0, 6.0], dtype=DTYPE)
    hi = torch.tensor([250.0, 0.95, 180.0, 360.0, 360.0, 2012.0, 8.0, 11.0], dtype=DTYPE)
    return lo + (hi - lo) * u


def synthetic_s2(data: dict, p: torch.Tensor, gen: torch.Generator) -> dict:
    """The same epochs and error bars, with values from the orbit p (8,) plus Gaussian noise of those sizes."""
    from .equation import observables  # local import keeps this file about tensors

    ra, dec, _ = observables(p[None], data["t_a"])
    _, _, rv = observables(p[None], data["t_v"])
    out = dict(data)

    def noisy(x, err):
        return x[0] + err * torch.randn(err.shape, generator=gen, dtype=DTYPE)

    out["ra"], out["dec"] = noisy(ra, data["ra_err"]), noisy(dec, data["dec_err"])
    out["rv"] = noisy(rv, data["rv_err"])
    return out


def wrap_degrees(x: torch.Tensor) -> torch.Tensor:
    """Angles into [0, 360)."""
    return torch.remainder(x, 360.0)
