r"""The laws as tensor operations: a workflow DAG in linear algebra, and the vessel-network laws.

Workflow (jobs i, j; rules r, s; A_ij = 1 when job i reads a file that job j writes):

    job DAG          A = Σ_{s,r} (A_rule)_{sr} E_{sr} ⊗ M^{sr},  M^{sr} = ⊗_a M_a   (I, 𝟙 or 𝟙ᵀ per wildcard axis a)
    acyclic          A nilpotent ⇔ A^J = 0 ⇔ trace(A^k) = 0 for k = 1 … J (no closed walks)
    reachability     (I − A)^{-1} = I + A + A² + …  (a finite sum: it counts the paths j → i)
    levels           ℓ_i = max_j A_ij (ℓ_j + 1)  (boolean matrix–vector iteration)
    makespan         t_i = d_i + max_j A_ij t_j   (max-plus, i.e. tropical, matrix–vector iteration)
    rerun set        x = s ∨ (R s),  R = boolean closure of A, s = jobs that read a touched file

Vessel network (edges e = (u, v), nodes n, repeated indices summed):

    Poiseuille       G_e = π r_e⁴ / (8 μ L_e)
    Murray outlets   g_k = Q_tot r_k³ / (P_in Σ_m r_m³)
    Kirchhoff        K_nm p_m = b_n,  K = Bᵀ diag(G) B + diag(g)   (B the (E, N) incidence matrix)
    axial profile    u(ρ) = 2U(1 − ρ²),  U = |Q_e|/(π r_e²),  ρ = distance from the axis / r_e
    wall shear       τ_w = μ |∂u/∂n|_wall = 4μ|Q_e|/(π r_e³)
    soft threshold   s(x) = σ((I(x) − θ)/w),  differentiable in θ
"""

import math

import torch


# ---------------------------------------------------------------- the workflow DAG as tensors
def kron_block(axes_r: tuple, axes_s: tuple, sizes: dict) -> torch.Tensor:
    r"""Job-level block M^{sr} (n_s, n_r) between a producer rule r and a consumer rule s: a Kronecker product over the
    wildcard axes, I when both rules carry the axis (per-case edges A ⊗ I), 𝟙 (n_a, 1) when only the consumer does
    (a broadcast), 𝟙ᵀ (1, n_a) when only the producer does (a gather, e.g. into the cohort report)."""
    M = torch.ones(1, 1)
    for a, n in sizes.items():  # the canonical axis order: jobs are numbered row-major over these axes
        if a in axes_r and a in axes_s:
            M = torch.kron(M, torch.eye(n))
        elif a in axes_s:
            M = torch.kron(M, torch.ones(n, 1))
        elif a in axes_r:
            M = torch.kron(M, torch.ones(1, n))
    return M


def job_adjacency(A_rule: torch.Tensor, axes: list, sizes: dict) -> tuple[torch.Tensor, torch.Tensor]:
    r"""A = Σ_{s,r} (A_rule)_{sr} E_{sr} ⊗ M^{sr}: the (J, J) job adjacency and the offset of each rule's jobs."""
    n = torch.tensor([math.prod(sizes[a] for a in ax) for ax in axes])
    off = torch.cat([torch.zeros(1, dtype=torch.long), n.cumsum(0)])
    A = torch.zeros(int(off[-1]), int(off[-1]))
    for s, r in A_rule.nonzero().tolist():  # one block per rule edge (a handful), each a Kronecker product
        A[off[s] : off[s + 1], off[r] : off[r + 1]] = kron_block(axes[r], axes[s], sizes)
    return A, off


def walk_traces(A: torch.Tensor) -> torch.Tensor:
    r"""trace(A^k) for k = 1 … J: the number of closed walks of length k. All zero ⇔ no cycle ⇔ A nilpotent."""
    out, P = [], torch.eye(len(A), dtype=torch.float64)
    for _ in range(len(A)):
        P = P @ A.double()
        out.append(torch.trace(P))
    return torch.stack(out)


def is_nilpotent(A: torch.Tensor) -> bool:
    r"""A^J = 0 (checked by repeated squaring in the boolean semiring, so no overflow)."""
    P, k = A > 0, 1
    while k < len(A):
        P = (P.double() @ P.double()) > 0
        k *= 2
    return not bool(P.any())


def closure(A: torch.Tensor) -> torch.Tensor:
    r"""R = A + A² + … in the boolean semiring (R_ij: job i depends on job j, directly or not): repeated squaring
    R ← R ∨ R·R, about log₂(depth) products. (I − A)^{-1} = I + R as a 0/1 pattern."""
    R = A > 0
    while True:
        R2 = R | ((R.double() @ R.double()) > 0)
        if torch.equal(R2, R):
            return R
        R = R2


def levels(A: torch.Tensor) -> torch.Tensor:
    r"""ℓ_i = 0 for jobs with no inputs from other jobs, else 1 + max over their inputs: a job can start once every
    job of a lower level is done. Boolean iteration: done ← done ∨ (A·¬done = 0)."""
    done = torch.zeros(len(A), dtype=torch.bool)
    lev = torch.full((len(A),), -1, dtype=torch.long)
    k = 0
    while not bool(done.all()):
        ready = ~done & ((A.double() @ (~done).double()) == 0)  # all inputs done
        if not bool(ready.any()):
            raise ValueError("cycle: no job is ready")
        lev[ready], done = k, done | ready
        k += 1
    return lev


def finish_times(A: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    r"""Earliest finish times with unlimited cores: t_i = d_i + max_j A_ij t_j, a max-plus (tropical) fixed point
    reached after at most depth + 1 sweeps. max over an empty set = 0."""
    t = d.clone()
    for _ in range(len(A) + 1):
        start = torch.where(A > 0, t[None, :], torch.tensor(-math.inf, dtype=t.dtype)).max(1).values.clamp(min=0)
        t_new = d + start
        if torch.equal(t_new, t):
            return t
        t = t_new
    raise ValueError("no fixed point: the graph has a cycle")


def critical_path(A: torch.Tensor, t: torch.Tensor, d: torch.Tensor) -> list[int]:
    r"""The chain of jobs whose durations add up to the makespan: backtrack from argmax t through the input j that
    finishes exactly when job i starts (t_i − d_i)."""
    i = int(t.argmax())
    path = [i]
    while True:
        deps = A[i].nonzero().flatten()
        if len(deps) == 0:
            return path[::-1]
        i = int(deps[(t[deps] - (t[i] - d[i])).abs().argmin()])
        path.append(i)


def rerun_set(R: torch.Tensor, touched: torch.Tensor) -> torch.Tensor:
    r"""Jobs to rerun after the jobs in `touched` (bool (J,)) became outdated: x = s ∨ R s (boolean)."""
    return touched | ((R.double() @ touched.double()) > 0)


def lower_bound(d: torch.Tensor, cp: float, cores: int) -> float:
    r"""No schedule on c cores beats max(critical path, total work / c) (exact scheduling is NP-hard)."""
    return max(cp, float(d.sum()) / cores)


# ---------------------------------------------------------------- segmentation
def soft_threshold(image: torch.Tensor, theta: torch.Tensor, width: float) -> torch.Tensor:
    r"""s = σ((I − θ)/w) per voxel; image (B, X, Y, Z), theta (B,). Tends to the hard threshold I > θ as w → 0."""
    return torch.sigmoid((image - theta[:, None, None, None]) / width)


def dice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Dice = 2|a ∩ b| / (|a| + |b|) per case; a, b (B, X, Y, Z) bool → (B,)."""
    inter = (a & b).flatten(1).sum(1).double()
    return 2 * inter / (a.flatten(1).sum(1) + b.flatten(1).sum(1)).double().clamp(min=1)


# ---------------------------------------------------------------- centreline: min-plus eikonal on the voxel graph
def edge_cost(length: torch.Tensor, R_mid: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    r"""Travel time ℓ / R^p along a voxel edge: the front runs fastest along the middle of the vessel."""
    return length / R_mid.clamp(min=1e-9) ** power


def relax(T: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    r"""One min-plus (Bellman–Ford) sweep: T_v ← min(T_v, min_u T_u + w_uv) over the directed edges u → v."""
    return T.scatter_reduce(0, dst, T[src] + w, reduce="amin", include_self=True)


# ---------------------------------------------------------------- the vessel network
def poiseuille_conductance(r: torch.Tensor, L: torch.Tensor, mu: float) -> torch.Tensor:
    r"""G = πr⁴/(8μL) (m³ s⁻¹ Pa⁻¹): Hagen–Poiseuille flow Q = G Δp through a straight tube."""
    return math.pi * r**4 / (8 * mu * L.clamp(min=1e-9))


def poiseuille_speed(rho, U):
    r"""Hagen–Poiseuille axial speed u(ρ) = 2U(1 − ρ²) at the relative radius ρ = d/r ∈ [0, 1] (U the mean speed)."""
    return 2 * U * (1 - rho**2)


def wall_shear_stress(q, r, mu: float):
    r"""τ_w = μ |∂u/∂d| at the wall of a straight Poiseuille tube = 4μ|Q|/(πr³) (Pa), Q (m³/s) and r (m)."""
    return 4 * mu * abs(q) / (math.pi * r**3)


def murray_conductance(r_out: torch.Tensor, outlet: torch.Tensor, Q_tot: float, P: float) -> torch.Tensor:
    r"""Peripheral conductance of each outlet, g_k = Q_tot r_k³ / (P Σ_m r_m³): with a resistance-free network every
    outlet would take a share ∝ r³ (Murray's law) of Q_tot at the pressure P. r_out, outlet (B, N) → (B, N)."""
    r3 = torch.where(outlet, r_out, torch.zeros_like(r_out)) ** 3
    return Q_tot * r3 / (P * r3.sum(1, keepdim=True).clamp(min=1e-30))


def kirchhoff_matrix(G: torch.Tensor, edges: torch.Tensor, n_nodes: int) -> torch.Tensor:
    r"""K = Bᵀ diag(G) B, assembled from the edge list: K_uu += G, K_vv += G, K_uv −= G, K_vu −= G.
    G (B, E), edges (B, E, 2) → (B, N, N). (Q into node n) = −(K p)_n; mass balance is K p = b."""
    Bn, E = G.shape
    b = torch.arange(Bn)[:, None].expand(Bn, E)
    u, v = edges[..., 0], edges[..., 1]
    K = torch.zeros(Bn, n_nodes, n_nodes, dtype=G.dtype)
    for (i, j), sign in (((u, u), 1), ((v, v), 1), ((u, v), -1), ((v, u), -1)):
        K = K.index_put((b, i, j), sign * G, accumulate=True)
    return K
