"""Build the concrete tensors: the LIGO strain, frequency grids, the chirp-mass bank and synthetic coloured noise.

data/gw150914_strain.csv holds 16 s of strain around GW150914 from both LIGO detectors (H1 Hanford, L1
Livingston), from the GWOSC release (GWTC-1-confident, v3, 4 kHz files, CC BY 4.0), low-passed at 900 Hz and
decimated to 2048 samples per second; values are strain × 10⁻²¹ (header lines start with #). Everything is float64.
"""

from pathlib import Path

import torch

DATA = Path(__file__).parent / "data" / "gw150914_strain.csv"
DTYPE = torch.float64
DETECTORS = ("H1", "L1")
GPS_EVENT = 1126259462.4  # GWOSC event time of GW150914


def load_strain(path: Path = DATA) -> dict:
    """Strain of both detectors: {'x': (2, N) strain, 'fs': Hz, 'gps_start': s, 't': (N,) s from the file start}."""
    header, rows = {}, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            if "=" in line:
                key, value = line[1:].split("=")
                header[key.strip()] = float(value)
        elif line and line[0] not in "hH":  # skip the column-name row
            rows.append([float(v) for v in line.split(",")])
    x = torch.tensor(rows, dtype=DTYPE).T * 1e-21  # (2, N)
    fs = header["sample_rate"]
    t = torch.arange(x.shape[1], dtype=DTYPE) / fs
    return {"x": x, "fs": fs, "gps_start": header["gps_start"], "t": t}


def chirp_mass_grid(lo: float, hi: float, n: int) -> torch.Tensor:
    """Template bank: n chirp masses from lo to hi (detector frame, M☉): (K,)."""
    return torch.linspace(lo, hi, n, dtype=DTYPE)


def tukey(n: int, alpha: float = 0.1) -> torch.Tensor:
    """Tukey window: flat in the middle, cosine tapers over a fraction alpha at both ends: (n,)."""
    w = torch.ones(n, dtype=DTYPE)
    r = max(1, int(alpha * n / 2))
    ramp = 0.5 * (1 - torch.cos(torch.pi * torch.arange(r, dtype=DTYPE) / r))
    w[:r], w[-r:] = ramp, ramp.flip(0)
    return w


def model_psd(f: torch.Tensor) -> torch.Tensor:
    """Advanced-LIGO design PSD (zero-detuned high power, Ajith 2011 fit), x = f/215 Hz, frozen below 10 Hz: (F,).

    Only used for synthetic tests; the real analysis estimates the PSD from the data with Welch's method.
    """
    x = f.clamp(min=10.0) / 215.0
    return 1e-49 * (x ** (-4.14) - 5 / x**2 + 111 * (1 - x**2 + x**4 / 2) / (1 + x**2 / 2))


def coloured_noise(psd: torch.Tensor, n: int, fs: float, count: int, gen: torch.Generator) -> torch.Tensor:
    """Gaussian noise with one-sided PSD psd on the rfft grid of n samples: (count, n).

    A DFT coefficient of such noise has E|X_k|² = n fs S(f_k)/2, so each complex coefficient gets independent real
    and imaginary parts of variance n fs S/4 (DC and Nyquist are real).
    """
    sigma = (n * fs * psd / 4).sqrt()
    re = torch.randn(count, len(psd), generator=gen, dtype=DTYPE)
    im = torch.randn(count, len(psd), generator=gen, dtype=DTYPE)
    im[:, 0] = 0.0
    im[:, -1] = 0.0
    spec = sigma * torch.complex(re, im)
    spec[:, 0] *= 2**0.5
    spec[:, -1] *= 2**0.5
    return torch.fft.irfft(spec, n=n)
