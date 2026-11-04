"""Run many stochastic epidemics at once, the mean-field ODE, and the vaccination strategies."""

import torch

from problems._common.integrators import integrate
from problems._common.progress import steps

from . import equation

S, I, R = 0, 1, 2  # noqa: E741  state codes, the epidemiologists' letters


def simulate(A, beta, gamma, dt, n_steps, infected0, vaccinated, gen, sis=False, record=False):
    """B stochastic SIR (or SIS) epidemics on the network A at once, realisation b with rate β_b.

    infected0 (B, N) bool: the first cases; vaccinated (B, N) bool or None: immune from the start (counted in R).
    Returns counts (T, B, 3) of S, I, R; ever (T, B) people infected so far (vaccinated excluded); and, if record,
    the state of every person in realisation 0, states (T, N) int.
    """
    b, n = infected0.shape
    vacc = torch.zeros_like(infected0) if vaccinated is None else vaccinated
    state = torch.full((b, n), S, dtype=torch.long, device=A.device)
    state[vacc] = R
    state[infected0 & ~vacc] = I
    p_rec = 1.0 - torch.exp(torch.tensor(-gamma * dt))
    ever = state == I
    counts, evers, states = [], [], []

    def log():
        counts.append(torch.stack([(state == c).sum(1) for c in (S, I, R)], dim=-1))  # (B, 3)
        evers.append(ever.sum(1))
        if record:
            states.append(state[0].clone())

    log()
    for _ in steps(n_steps, "epidemic steps"):
        if not (state == I).any():
            break  # every realisation has burned out
        infected = state == I
        catch = torch.rand(b, n, generator=gen, device=A.device) < equation.infection_prob(
            beta, dt, equation.pressure(A, infected)
        )
        recover = torch.rand(b, n, generator=gen, device=A.device) < p_rec
        new_inf = (state == S) & catch
        new_rec = infected & recover
        state = torch.where(new_inf, I, state)
        state = torch.where(new_rec, S if sis else R, state)
        ever = ever | new_inf
        log()
    while len(counts) < n_steps + 1:  # pad burned-out runs so every history has the same length
        log()
    out = {"counts": torch.stack(counts), "ever": torch.stack(evers)}
    if record:
        out["states"] = torch.stack(states)
    return out


def mean_field(A, beta, gamma, x0, dt, n_steps):
    """Individual-based mean field with RK4. x0 (B, N) initial infection probabilities.  → (T, 3, B, N) = (s, x, r)."""
    y0 = torch.stack([1.0 - x0, x0, torch.zeros_like(x0)])
    return integrate(lambda y, t: equation.mean_field_rhs(y, A, beta.to(x0.dtype), gamma), y0, dt, n_steps)


def vaccinate(A, fraction, strategy, b, gen):
    """(B, N) bool: who is vaccinated in each realisation. 'targeted' takes the highest-degree people (the hubs),
    'random' a fresh random set of the same size for every realisation."""
    n = A.shape[0]
    n_vacc = int(round(fraction * n))
    if strategy == "targeted":
        rank = torch.argsort(torch.argsort(A.sum(1), descending=True, stable=True))  # 0 = biggest hub
        return (rank < n_vacc)[None, :].expand(b, n).clone()
    rank = torch.argsort(torch.argsort(torch.rand(b, n, generator=gen), dim=1), dim=1).to(A.device)
    return rank < n_vacc


def remove(A, gone):
    """The network with the people in `gone` (N,) bool removed (their rows and columns zeroed)."""
    keep = (~gone).to(A.dtype)
    return A * keep[:, None] * keep[None, :]


def threshold_sweep(A, ratios, gamma, dt, n_steps, n_runs, gen):
    """Final size against β/γ: B = len(ratios) × n_runs realisations at once, one initial case each.

    Returns the mean fraction ever infected per ratio (R,), and the mean-field final size per ratio (R,).
    """
    n = A.shape[0]
    beta = (ratios * gamma).repeat_interleave(n_runs)  # (R · runs,)
    seeds = torch.randint(n, (beta.shape[0],), generator=gen)
    infected0 = torch.zeros(beta.shape[0], n, dtype=torch.bool, device=A.device)
    infected0[torch.arange(beta.shape[0]), seeds.to(A.device)] = True
    ever = simulate(A, beta, gamma, dt, n_steps, infected0, None, gen)["ever"][-1].float() / n
    sim = ever.view(len(ratios), n_runs).mean(1)
    mf = mean_field(A, ratios * gamma, gamma, torch.full((len(ratios), n), 1.0 / n, dtype=torch.float64), dt, n_steps)
    return sim, mf[-1, 2].mean(-1)


def vaccination_sweep(A, fractions, ratio, gamma, dt, n_steps, n_runs, gen):
    """Mean fraction infected (among everybody) against the vaccinated fraction, for both strategies.

    Also the mean-field herd-immunity test: β/γ · λ_max of the network left after vaccination (< 1 ⇒ no epidemic).
    """
    n = A.shape[0]
    out = {}
    for strategy in ("targeted", "random"):
        vacc = torch.cat([vaccinate(A, f, strategy, n_runs, gen) for f in fractions.tolist()])  # (F · runs, N)
        free = torch.argsort(torch.rand(vacc.shape, generator=gen).to(A.device) + vacc.float(), dim=1)[:, :3]
        infected0 = torch.zeros_like(vacc)
        infected0.scatter_(1, free, True)  # three unvaccinated first cases per run
        beta = torch.full((vacc.shape[0],), ratio * gamma, device=A.device)
        ever = simulate(A, beta, gamma, dt, n_steps, infected0, vacc, gen)["ever"][-1].float() / n
        lam = equation.lambda_max(torch.stack([remove(A, v) for v in vacc[::n_runs]]))  # (F,) one network each
        out[strategy] = (ever.view(len(fractions), n_runs).mean(1), ratio * lam)
    return out
