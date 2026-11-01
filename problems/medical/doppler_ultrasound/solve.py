"""Simulate the Doppler signal of all scatterers and turn it into a spectrogram with a short-time FFT."""

import torch

from .equation import doppler_shift, received_signal, velocity_field


def simulate(
    v_max: torch.Tensor, scat: dict[str, torch.Tensor], R: float, f0: float, theta: float, prf: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (velocities v_{tn} (T, N), received IQ signal s_t (T,) complex), sampled once per pulse."""
    v = velocity_field(v_max, scat["r"], R)  # (T, N)
    f_D = doppler_shift(v, f0, theta)  # (T, N) true (unaliased) shift of every scatterer
    s = received_signal(scat["amp"], scat["phase"], f_D, 1.0 / prf)  # (T,)
    return v, s


def spectrogram(
    s: torch.Tensor, prf: float, n_fft: int = 128, hop: int = 32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Short-time Fourier transform of the complex signal.

    Returns (window times (K,), frequencies (F,) from -PRF/2 to +PRF/2, power (F, K)).
    A complex signal has a two-sided spectrum: positive frequencies = flow towards the transducer.
    """
    window = torch.hann_window(n_fft, device=s.device)
    S = torch.stft(s, n_fft, hop_length=hop, window=window, center=True, return_complex=True, onesided=False)
    power = torch.fft.fftshift(S.abs() ** 2, dim=0) / window.pow(2).sum()  # (F, K), zero frequency in the middle
    freqs = torch.fft.fftshift(torch.fft.fftfreq(n_fft, d=1.0 / prf, device=s.device))  # (F,)
    times = torch.arange(power.shape[1], device=s.device) * hop / prf  # (K,) window centres
    return times, freqs, power


def mean_frequency(freqs: torch.Tensor, power: torch.Tensor) -> torch.Tensor:
    """Power-weighted mean frequency of each spectrum (the 'mean velocity' trace of a scanner): (K,)."""
    return torch.einsum("f,fk->k", freqs, power) / power.sum(0)
