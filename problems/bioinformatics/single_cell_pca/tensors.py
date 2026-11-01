"""Build the concrete tensors: a synthetic single-cell RNA-seq experiment with known cell types."""

import torch


def type_profiles(n_genes: int, n_types: int, n_markers: int, fold_change: float, seed: int, device: torch.device):
    """Mean expression μ_{kg} of gene g in cell type k, shape (K, G), and the marker genes of each type (K, M).

    Every gene has a baseline drawn from a log-normal (a few genes are very highly expressed, most are low);
    each type then switches on its own n_markers marker genes by a factor fold_change.
    """
    gen = torch.Generator().manual_seed(seed)
    base = torch.exp(torch.randn(n_genes, generator=gen) * 1.2 - 1.0)  # (G,) mean counts per cell at depth 1
    markers = torch.randperm(n_genes, generator=gen)[: n_types * n_markers].reshape(n_types, n_markers)  # disjoint sets
    mu = base.repeat(n_types, 1)  # (K, G)
    for k in range(n_types):
        mu[k, markers[k]] *= fold_change
    return mu.to(device), markers.to(device)


def sample_counts(n_cells: int, mu: torch.Tensor, dispersion: float, depth_spread: float, seed: int):
    """Draw a count matrix K_{cg}, shape (C, G), plus the true type of each cell (C,) and its size factor s_c (C,).

    Negative-binomial counts as a Gamma–Poisson mixture:  λ_{cg} ~ Gamma(shape 1/φ, mean s_c μ_{k(c) g}),
    K_{cg} ~ Poisson(λ_{cg}).  The variance is  m + φ m²  (overdispersed, as in real data).
    s_c ~ log-normal: cells are sequenced to different depths (library-size variation).
    """
    K, G = mu.shape
    dev = mu.device
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randint(K, (n_cells,), generator=gen)
    s = torch.exp(depth_spread * torch.randn(n_cells, generator=gen))  # (C,) size factors, median 1
    mean = s[:, None] * mu.cpu()[labels]  # (C, G)  m_{cg} = s_c μ_{k(c) g}
    with torch.random.fork_rng(devices=[]):  # the Gamma sampler has no generator argument: seed a private stream
        torch.manual_seed(seed + 1)
        shape = torch.full_like(mean, 1.0 / dispersion)
        lam = torch.distributions.Gamma(shape, shape / mean).sample()  # mean = shape / rate = m_cg
    counts = torch.poisson(lam, generator=gen)
    return counts.to(dev), labels.to(dev), s.to(dev)
