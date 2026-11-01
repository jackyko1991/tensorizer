r"""Physics → tensor translation.

A neural radiance field (NeRF) is a function F_θ: x ↦ (σ, c) that gives every point x in space a density σ ≥ 0
(how much light it blocks per unit length) and a colour c. A camera pixel sees the light collected along its ray
x(t) = o + t d. Indices: r = ray, s = sample along the ray, i = space axis, k = frequency, h = hidden unit.

    encoding    γ(x)_{rsik} = (sin(2^k π x_{rsi}), cos(2^k π x_{rsi})),   k = 0 … L-1
    network     (σ_{rs}, c_{rs·}) = MLP(γ(x_{rs}))      layers h_{n+1} = ReLU(A_{hg} h_g + b_h), g summed
    opacity     α_{rs} = 1 - exp(-σ_{rs} δ_{rs}),   δ_{rs} = t_{r,s+1} - t_{rs}
    transmit.   T_{rs} = exp(-Σ_{s' < s} σ_{rs'} δ_{rs'})                 light that survives up to sample s
    colour      C_r = Σ_s T_{rs} α_{rs} c_{rs} + (1 - Σ_s T_{rs} α_{rs}) c_bg      s summed: w_{rs} = T_{rs} α_{rs}

The last line is the quadrature of the volume rendering integral C = ∫ T(t) σ(t) c(t) dt, exact when σ and c are
constant on each interval. Everything is differentiable, so the network is trained from photos alone.
"""

import math

import torch
import torch.nn.functional as F


def positional_encoding(x: torch.Tensor, n_freq: int) -> torch.Tensor:
    """γ(x) = [x, sin(2^k π x), cos(2^k π x)]_{k < L}: (..., 3) → (..., 3 + 6L). With L = 0 it returns x itself.

    The outer product x_{…i} 2^k π is an einsum with no summed index ('…i,k->…ik').
    """
    if n_freq == 0:
        return x
    freq = math.pi * 2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)  # (L,)
    xk = torch.einsum("...i,k->...ik", x, freq).flatten(-2)  # (..., 3L)
    return torch.cat([x, torch.sin(xk), torch.cos(xk)], dim=-1)


def mlp(h: torch.Tensor, layers: list[dict]) -> torch.Tensor:
    """h_{n+1} = ReLU(A_{og} h_{…g} + b_o)   (g summed), no ReLU on the output.   (..., D) → (..., 4)."""
    for n, layer in enumerate(layers):
        h = torch.einsum("og,...g->...o", layer["A"], h) + layer["b"]
        if n < len(layers) - 1:
            h = F.relu(h)
    return h


def radiance_field(x: torch.Tensor, layers: list[dict], n_freq: int, scale: float):
    """F_θ(x) → density σ (...,) ≥ 0 via softplus, colour c (..., 3) ∈ (0, 1) via sigmoid."""
    out = mlp(positional_encoding(x / scale, n_freq), layers)
    return F.softplus(out[..., 0] - 1.0), torch.sigmoid(out[..., 1:])


def stratified_samples(n_rays: int, n_samples: int, near: float, far: float, gen, device) -> torch.Tensor:
    """t_{rs}: one uniform random point in each of S equal bins of [near, far], per ray.   → (R, S).

    gen = None gives the bin midpoints (deterministic, used for evaluation).
    """
    edges = torch.linspace(near, far, n_samples + 1, device=device)
    width = edges[1] - edges[0]
    u = 0.5 * torch.ones(n_rays, n_samples) if gen is None else torch.rand(n_rays, n_samples, generator=gen)
    return edges[:-1] + width * u.to(device)


def render_weights(sigma: torch.Tensor, t: torch.Tensor, far: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Quadrature weights w_{rs} = T_{rs} (1 - e^{-σ_{rs} δ_{rs}}) and transmittance T_{rs}.   (R, S), (R, S).

    δ_{rs} = t_{r,s+1} - t_{rs}; the last interval runs to `far`.
    """
    delta = torch.cat([t[:, 1:] - t[:, :-1], far - t[:, -1:]], dim=1)  # (R, S)
    tau = sigma * delta  # optical thickness of each interval
    T = torch.exp(-torch.cumsum(tau, dim=1) + tau)  # exclusive sum: Σ_{s' < s}
    return T * (1 - torch.exp(-tau)), T


def volume_render(sigma, colour, t, far, background):
    """C_r = Σ_s w_{rs} c_{rsc} + (1 - Σ_s w_{rs}) b_c   (s summed), and the expected depth D_r = Σ_s w_{rs} t_{rs}.

    sigma (R, S), colour (R, S, 3), t (R, S) → colour (R, 3), depth (R,), weights (R, S).
    """
    w, _ = render_weights(sigma, t, far)
    acc = w.sum(dim=1)  # (R,) total opacity of the ray
    rgb = torch.einsum("rs,rsc->rc", w, colour) + (1 - acc)[:, None] * background
    depth = torch.einsum("rs,rs->r", w, t) + (1 - acc) * far  # an empty ray counts as 'at the far plane'
    return rgb, depth, w


def render_rays(o, d, layers, n_freq, scale, n_samples, near, far, background, gen=None):
    """Whole pipeline for a batch of rays o, d (R, 3): samples → field → quadrature → colour (R, 3), depth (R,)."""
    t = stratified_samples(o.shape[0], n_samples, near, far, gen, o.device)  # (R, S)
    x = o[:, None, :] + t[..., None] * d[:, None, :]  # (R, S, 3) sample points x_{rsi}
    sigma, colour = radiance_field(x, layers, n_freq, scale)
    rgb, depth, w = volume_render(sigma, colour, t, far, background)
    return rgb, depth, w, t


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB for images in [0, 1]: -10 log₁₀ MSE."""
    return -10 * torch.log10(((pred - target) ** 2).mean())
