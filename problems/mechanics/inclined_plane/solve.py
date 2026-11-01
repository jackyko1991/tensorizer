"""Move every block down its ramp. The forces never change, so the acceleration is constant."""

import torch


def solve(a: torch.Tensor, s: torch.Tensor, ramp_length: float, t_max: float, n_times: int = 301) -> dict:
    """Return times (T,), distance travelled along the slope d (T, B) and speed v (T, B).

    The block starts at rest at the top. With constant acceleration a_{bi},
    the distance along the slope is d_b(t) = ½ a_∥b t², with a_∥b = a_{bi} s_{bi}.
    A block that reaches the bottom of the ramp stops there.
    """
    a_par = torch.einsum("bi,bi->b", a, s)  # (B,) acceleration along the slope
    times = torch.linspace(0.0, t_max, n_times, device=a.device)  # (T,)
    d = 0.5 * torch.einsum("t,b->tb", times**2, a_par)  # (T, B)
    v = torch.einsum("t,b->tb", times, a_par)
    at_bottom = d >= ramp_length
    d = torch.where(at_bottom, ramp_length, d)
    v = torch.where(at_bottom, 0.0, v)
    return {"times": times, "d": d, "v": v, "a_par": a_par}
