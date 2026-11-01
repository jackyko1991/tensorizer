r"""Physics → tensor translation.

3D Gaussian splatting represents a scene by M anisotropic Gaussians. Gaussian m has a centre μ_m, a covariance
Σ_m, a colour c_m and an opacity o_m. Indices: v = camera (view), m = Gaussian, p = pixel, i, j, k, l = 3D axes,
a, b = 2D image axes.

    covariance     Σ_{mik}  = R_{mij} s_{mj}² R_{mkj}               j summed   (Σ = R S Sᵀ Rᵀ, always SPD)
    to camera      x_{vmi}  = W_{vij} μ_{mj} + t_{vi}               j summed
    projection     u_{vma}  = f x_{vma} / x_{vm3} + c               perspective divide
    2D covariance  Σ'_{vmab} = J_{vmai} W_{vij} Σ_{mjk} W_{vlk} J_{vmbl}   i, j, k, l summed   (Σ' = J W Σ Wᵀ Jᵀ)
    opacity        α_{vmp}  = o_m exp(-½ d_{vmpa} (Σ'⁻¹)_{vmab} d_{vmpb}),  d = pixel - u    a, b summed
                            = o_m exp(-½ Q_{vmk} φ_{pk})     the same, expanded: k summed over 6 monomials
    compositing    C_{vp}   = Σ_m c_m α_{vmp} T_{vmp},   T_{vmp} = Π_{n before m} (1 - α_{vnp})   m in depth order

J is the 2 × 3 Jacobian of the perspective projection: the projection is not linear, so the Gaussian is mapped
by its local linearisation at the centre (the 'EWA splatting' approximation).
"""

import torch

LOW_PASS = 0.3  # px²: added to Σ' so every splat covers at least about one pixel (anti-aliasing)
ALPHA_MAX = 0.99
NEAR = 0.2  # Gaussians closer to the camera than this are dropped


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternion (w, x, y, z) → rotation matrix.   q: (M, 4), normalised here → R: (M, 3, 3)."""
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def covariance_3d(quat: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Σ_{mik} = R_{mij} s_{mj}² R_{mkj}   (j summed): rotation × axis lengths², SPD by construction.   → (M, 3, 3)."""
    R = quat_to_rotmat(quat)
    s2 = torch.exp(2 * log_scale)  # (M, 3) squared axis lengths
    return torch.einsum("mij,mj,mkj->mik", R, s2, R)


def to_camera(mean: torch.Tensor, W: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_{vmi} = W_{vij} μ_{mj} + t_{vi}   (j summed): Gaussian centres in every camera frame.   → (V, M, 3)."""
    return torch.einsum("vij,mj->vmi", W, mean) + t[:, None, :]


def project(x_cam: torch.Tensor, f: float, c: float) -> torch.Tensor:
    """Pinhole projection u_a = f x_a / x_3 + c for a = 1, 2.   (..., 3) → (..., 2) pixel coordinates."""
    return f * x_cam[..., :2] / x_cam[..., 2:3] + c


def projection_jacobian(x_cam: torch.Tensor, f: float) -> torch.Tensor:
    """J_{ai} = ∂u_a / ∂x_i of the pinhole projection at x_cam.   (V, M, 3) → (V, M, 2, 3).

    J = [[f/z, 0, -f x/z²], [0, f/z, -f y/z²]]
    """
    x, y, z = x_cam.unbind(-1)
    zero = torch.zeros_like(z)
    return torch.stack(
        [torch.stack([f / z, zero, -f * x / z**2], -1), torch.stack([zero, f / z, -f * y / z**2], -1)], -2
    )


def covariance_2d(cov3: torch.Tensor, W: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """Σ'_{vmab} = J_{vmai} W_{vij} Σ_{mjk} W_{vlk} J_{vmbl}   (i, j, k, l summed) + low-pass.   → (V, M, 2, 2)."""
    JW = torch.einsum("vmai,vij->vmaj", J, W)  # (V, M, 2, 3): world → pixel linear map
    cov2 = torch.einsum("vmaj,mjk,vmbk->vmab", JW, cov3, JW)
    return cov2 + LOW_PASS * torch.eye(2, device=cov2.device)


def pixel_features(pix: torch.Tensor, c: float) -> torch.Tensor:
    """φ_{pk} = (u², 2uv, v², u, v, 1) of every pixel, with u, v measured from the image centre c.   → (P, 6)."""
    u, v = (pix - c).unbind(-1)
    return torch.stack([u * u, 2 * u * v, v * v, u, v, torch.ones_like(u)], dim=-1)


def splat_alpha(phi: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor, opacity: torch.Tensor, c: float):
    """α_{vmp} = o_{vm} exp(-½ d_a (Σ'⁻¹)_{ab} d_b)   (a, b summed), d = pixel - centre of the splat.

    The quadratic form is a polynomial in the pixel coordinates, so expanding it gives ONE matrix product
    q_{vmp} = Q_{vmk} φ_{pk} (k summed) with the Gaussian's coefficients Q = (A, B, C, -2(Aμ_u + Bμ_v),
    -2(Bμ_u + Cμ_v), Aμ_u² + 2Bμ_uμ_v + Cμ_v²), where Σ'⁻¹ = [[A, B], [B, C]]: much faster than forming d.
    phi (P, 6), mu2 (V, M, 2), cov2 (V, M, 2, 2), opacity (V, M) → (V, M, P), clamped to ALPHA_MAX.
    """
    sxx, sxy, syy = cov2[..., 0, 0], cov2[..., 0, 1], cov2[..., 1, 1]  # (V, M)
    det = sxx * syy - sxy**2
    A, B, C = syy / det, -sxy / det, sxx / det  # the 2 × 2 inverse ('conic'), written out
    mu, mv = (mu2 - c).unbind(-1)
    Q = torch.stack(
        [A, B, C, -2 * (A * mu + B * mv), -2 * (B * mu + C * mv), A * mu**2 + 2 * B * mu * mv + C * mv**2], -1
    )
    q = torch.einsum("vmk,pk->vmp", Q, phi)  # d_a (Σ'⁻¹)_{ab} d_b for every splat and pixel
    return (opacity[..., None] * torch.exp(-0.5 * q.clamp(min=0))).clamp(max=ALPHA_MAX)


def composite(alpha: torch.Tensor, colour: torch.Tensor, background: torch.Tensor):
    """Front-to-back alpha compositing C_{vpc} = Σ_m w_{vmp} c_{vmc} + T_final b_c,  w_{vmp} = α_{vmp} T_{vmp}.

    alpha (V, M, P) and colour (V, M, 3) must already be sorted near → far along m. Returns the image (V, P, 3),
    the weights w (V, M, P) and the transmittance T (V, M, P) in front of each Gaussian.
    """
    through = torch.cumprod(1 - alpha, dim=1)  # light that passes Gaussians 1 … m
    T = torch.cat([torch.ones_like(alpha[:, :1]), through[:, :-1]], dim=1)  # exclusive product: Gaussians before m
    w = alpha * T  # (V, M, P) contribution of each Gaussian to each pixel
    image = torch.einsum("vmp,vmc->vpc", w, colour) + through[:, -1, :, None] * background  # m summed
    return image, w, T


def render(g: dict, cams: dict, background: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """The whole differentiable rasteriser: Gaussians g, cameras cams → images (V, P, 3) and the 2D splats."""
    cov3 = covariance_3d(g["quat"], g["log_scale"])  # (M, 3, 3)
    x_cam = to_camera(g["mean"], cams["W"], cams["t"])  # (V, M, 3)
    visible = x_cam[..., 2] > NEAR  # (V, M)
    x_cam = torch.where(visible[..., None], x_cam, torch.tensor([0.0, 0.0, 1.0], device=x_cam.device))  # safe divide
    mu2 = project(x_cam, cams["f"], cams["c"])  # (V, M, 2)
    J = projection_jacobian(x_cam, cams["f"])  # (V, M, 2, 3)
    cov2 = covariance_2d(cov3, cams["W"], J)  # (V, M, 2, 2)
    opacity = torch.sigmoid(g["opacity_logit"])  # (M,)
    colour = torch.sigmoid(g["colour_logit"])  # (M, 3)

    # depth sort per camera on the small per-Gaussian tensors, before the big (V, M, P) ones are made
    order = x_cam[..., 2].argsort(dim=1)  # (V, M) nearest first
    v = torch.arange(order.shape[0], device=order.device)[:, None]
    phi = pixel_features(cams["pix"], cams["c"])  # (P, 6)
    alpha = splat_alpha(phi, mu2[v, order], cov2[v, order], (opacity * visible)[v, order], cams["c"])  # (V, M, P)
    image, _, _ = composite(alpha, colour[order], background)
    return image, {"mu2": mu2, "cov2": cov2, "visible": visible, "colour": colour, "opacity": opacity}


def photometric_loss(pred: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    """Mean L1 (|C - C*|) or L2 ((C - C*)²) difference over views, pixels and colour channels."""
    diff = pred - target
    return diff.abs().mean() if kind == "l1" else (diff**2).mean()


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio in dB for images in [0, 1]: -10 log₁₀ MSE."""
    return -10 * torch.log10(((pred - target) ** 2).mean())
