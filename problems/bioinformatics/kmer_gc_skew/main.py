"""Main loop: parameters → synthetic genome → one-hot tensor → k-mers, GC skew, origin → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    L = int(params["length_kb"] * 1000)
    k, size = params["k"], params["window"]
    seq, origin, terminus = tensors.synthetic_genome(
        L, params["origin"], params["gc"], params["skew"], params["seed"], dev
    )
    X = tensors.one_hot(seq)  # (L, 4)
    is_gc, g_minus_c = tensors.gc_strand_weights(dev)  # (4,), (4,)

    counts = solve.count_kmers(X, tensors.place_values(k, dev))  # (4^k,)
    win = solve.windows(X, size, is_gc, g_minus_c)  # dict of (W,) tensors
    S, found_ori, found_ter = solve.find_origin(X, g_minus_c)  # (L,), int, int

    names = tensors.kmer_names(k)
    top = counts.argsort(descending=True)[:5].tolist()
    print(f"genome L = {L:,} bp (circular)   ·   GC content {X[:, 1:3].sum().item() / L:.3f}")
    print(f"{k}-mers: {counts.sum().item():,} counted = L - k + 1 = {L - k + 1:,}   ·   4^k = {4**k} words")
    print("most common: " + ", ".join(f"{names[m]} {counts[m].item()}" for m in top))
    print(f"windows of {size} bp: GC from {win['gc'].min().item():.3f} to {win['gc'].max().item():.3f}")
    for name, planted, found in [("origin", origin, found_ori), ("terminus", terminus, found_ter)]:
        off = solve.circular_distance(planted, found, L)
        print(f"planted {name:<8} {planted:>9,}   found {found:>9,}   off by {off:,} bp")
    return plot.animate(
        win, S, counts, names, L, (origin, terminus), (found_ori, found_ter), out_dir / "kmer_gc_skew.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
