"""Main loop: measure the memory bandwidth and the matmul rate in each precision, print them, draw the roofline.

The last printed line is `CHIP_RESULT {...}` (JSON): the Coding Skills ▸ Chips tab reads it to place this machine
on its roofline and in its comparison table.
"""

import json
from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    n = 256 if fast else int(params["matrix_size"])
    mb = 8 if fast else int(params["copy_mb"])
    repeats = 2 if fast else int(params["repeats"])

    result = {"device": solve.device_name(dev), "deviceType": dev.type, "n": n}
    result["bandwidth"] = solve.measure_bandwidth(mb, dev, repeats)
    print(f"device: {result['device']} ({dev.type})")
    print(f"memory copy of {mb} MB: {result['bandwidth']['gbs']:.1f} GB/s")
    result["matmul"] = []
    for name in tensors.PRECISIONS:
        m = solve.measure_matmul(name, n, dev, repeats)
        result["matmul"].append(m)
        if "skipped" in m:
            print(f"{name.upper():5s} skipped: {m['skipped']}")
        else:
            note = f"  ({m['slow']})" if "slow" in m else ""
            rate = f"{m['tflops']:.3f} TFLOP/s  ({m['seconds'] * 1e3:.2f} ms)"
            print(f"{name.upper():5s} {m['n']}x{m['n']} matmul: {rate}{note}")
    print("CHIP_RESULT " + json.dumps(result, separators=(",", ":")))
    return plot.roofline(result, out_dir / "chip_roofline.png")


if __name__ == "__main__":
    cli(run, __file__)
