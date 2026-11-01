"""Build the concrete tensors: a small synthetic web graph (links, communities, dangling pages, drawing positions)."""

import math

import torch


def web_graph(n: int, n_dangling: int, seed: int, device: torch.device):
    """A random directed graph of n pages in 3 communities, with a few famous pages and n_dangling dead ends.

    Returns L_{ij} (n, n) with L_{ij} = 1 if page j links to page i (columns are sources), the community of
    every page (n,), and 2D positions for drawing (n, 2).
    """
    gen = torch.Generator().manual_seed(seed)
    community = torch.arange(n) % 3  # (n,)
    fame = 1.0 / (1.0 + torch.randperm(n, generator=gen).float()) ** 0.8  # a few pages are much more popular
    links = torch.zeros(n, n)
    dangling = torch.randperm(n, generator=gen)[:n_dangling]  # pages with no outgoing links
    for j in range(n):
        if j in dangling:
            continue
        weight = fame * torch.where(community == community[j], 4.0, 1.0)  # prefer famous pages nearby
        weight[j] = 0.0  # no self-links
        n_out = int(torch.randint(1, 5, (1,), generator=gen))
        targets = torch.multinomial(weight, n_out, replacement=False, generator=gen)
        links[targets, j] = 1.0

    angle = 2 * math.pi * community / 3 + 0.5  # communities sit around a circle
    centre = torch.stack([torch.cos(angle), torch.sin(angle)], dim=1)
    spread = 0.55 * torch.rand(n, generator=gen).sqrt()
    phi = 2 * math.pi * torch.rand(n, generator=gen)
    pos = centre + spread[:, None] * torch.stack([torch.cos(phi), torch.sin(phi)], dim=1)
    return links.to(device), community, pos
