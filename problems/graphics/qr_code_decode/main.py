"""Main loop: message → QR symbol → batch of synthetic photos → reader → success-rate sweep → animation."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation as eq
from . import plot, solve, tensors

IMAGE = 240  # photo size in pixels
MESSAGES = {
    "Reed-Solomon 1960": b"Reed-Solomon 1960",
    "Hello, tensors!": b"Hello, tensors!",
    "QR via PyTorch": b"QR via PyTorch",
    "x^8+x^4+x^3+x^2+1": b"x^8+x^4+x^3+x^2+1",
}
# ISO/IEC 18004 worked example: "01234567" as version 1-M (numeric mode), its 16 data and 10 ECC codewords.
ISO_DATA = [0x10, 0x20, 0x0C, 0x56, 0x61, 0x80] + [0xEC, 0x11] * 5
ISO_ECC = [0xA5, 0x24, 0xD4, 0xC1, 0xED, 0x36, 0xC7, 0x87, 0x2C, 0x55]


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    message, ecl = MESSAGES[params["message"]], params["ecc"]
    gen = torch.Generator().manual_seed(0)

    # 1. Encode, and check the pieces against known vectors.
    iso = eq.rs_encode(torch.tensor([ISO_DATA]), 10)[0].tolist()
    print(f"RS encoder on the ISO/IEC 18004 example '01234567' 1-M reproduces its 10 ECC bytes: {iso == ISO_ECC}")
    print(f"format word for level M, mask 0: {eq.format_code(0, 0):015b} (= 0x5412 mask, the standard value)")
    version = tensors.smallest_version(message, ecl, params["version"]) or 3
    if version != params["version"]:
        print(f"{len(message)} bytes do not fit version {params['version']}-{ecl}: using version {version}")
    sym = tensors.symbol(message, version, ecl)
    lay, n = tensors.layout(version, ecl), tensors.size(version)
    print(
        f"'{message.decode()}' ({len(message)} bytes) → version {version}-{ecl}: {n}×{n} modules, "
        f"{lay['blocks']} block(s) of {lay['k']} data + {lay['n_ec']} ECC bytes, corrects {lay['n_ec'] // 2} "
        f"byte errors per block"
    )
    print(f"mask penalties {sym['penalties'].tolist()} → mask {sym['mask']}")
    synd = eq.syndromes(sym["blocks"], lay["n_ec"])
    print(f"RS syndromes of the encoded blocks all zero: {bool((synd == 0).all())}")

    # 2. Photographs of it.
    count = 3 if fast else 5
    img, H_true = tensors.photos(sym["grid"], count, IMAGE, params["perspective"], params["occlusion"],
                                 params["noise"], gen)  # fmt: skip
    img = img.to(dev)

    # 3. Read them.
    tic = time.perf_counter()
    results = solve.decode(img, keep=True)
    print(f"decoded {count} photos ({IMAGE}×{IMAGE}, occlusion {params['occlusion']:.0%} of the symbol, "
          f"noise σ = {params['noise']}) in {time.perf_counter() - tic:.1f} s")  # fmt: skip
    for b, r in enumerate(results):
        if "modules" in r and r["version"] == version:
            r["module_errors"] = int((r["modules"] != sym["grid"]).sum())
        status = f"'{r['text']}'" if r["text"] is not None else f"failed ({r['stage']})"
        detail = ""
        if "byte_errors" in r:
            detail = (f"v{r['version']}-{r['ecl']} mask {r['mask']}, format bits fixed {r['format_errors']}, "
                      f"module errors {r.get('module_errors', '?')}, RS bytes fixed {r['byte_errors']}: ")  # fmt: skip
        print(f"  photo {b + 1}: {detail}{status}")
    ok = sum(r["text"] == message.decode() for r in results)
    print(f"{ok}/{count} photos read correctly")

    # 4. Success rate against occlusion, every ECC level, all photos of one level decoded as one batch.
    occs = [0.0, 0.16] if fast else [0.04 * i for i in range(8)]
    trials = 1 if fast else 5
    rates, versions = {}, {}
    tic = time.perf_counter()
    for level in tensors.LEVELS:
        v = tensors.smallest_version(message, level, params["version"]) or 3
        grid = tensors.symbol(message, v, level)["grid"]
        batch = torch.cat([tensors.photos(grid, trials, IMAGE, params["perspective"], o, params["noise"], gen)[0]
                           for o in occs])  # fmt: skip
        good = torch.tensor([r["text"] == message.decode() for r in solve.decode(batch.to(dev))])
        rates[level], versions[level] = good.view(len(occs), trials).float().mean(1), v
    print(f"success rate vs occlusion ({trials} photos per point, {time.perf_counter() - tic:.1f} s):")
    print("  occlusion " + " ".join(f"{o:5.0%}" for o in occs))
    for level in tensors.LEVELS:
        print(f"  {level} (v{versions[level]})    " + " ".join(f"{x:5.0%}" for x in rates[level].tolist()))

    block, place = tensors.codeword_map(version, ecl)
    return plot.animate(img.cpu(), results, sym, block, place, message.decode(), occs, rates, versions,
                        params["occlusion"], out_dir / "qr_code_decode.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
