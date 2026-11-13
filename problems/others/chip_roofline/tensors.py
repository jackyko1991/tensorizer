"""The operands: square matrices in each precision, and a large buffer for the bandwidth test."""

import torch

# name -> (dtype, matmul precision for float32 on NVIDIA: "highest" = true FP32, "high" = TF32 tensor cores)
PRECISIONS: dict[str, tuple[torch.dtype, str | None]] = {
    "fp64": (torch.float64, None),
    "fp32": (torch.float32, "highest"),
    "tf32": (torch.float32, "high"),
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
}


def supported(name: str, device: torch.device) -> tuple[bool, str]:
    """Whether a precision can run on the device, and why not."""
    dtype, mode = PRECISIONS[name]
    if device.type == "mps" and dtype == torch.float64:
        return False, "Apple's GPU backend (MPS) has no float64"
    if mode == "high" and device.type != "cuda":
        return False, "TF32 is an NVIDIA tensor-core mode"
    return True, ""


def operands(n: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Two random n×n matrices (values in [-1, 1), fixed seed)."""
    g = torch.Generator().manual_seed(0)
    a = (torch.rand(n, n, generator=g) * 2 - 1).to(device=device, dtype=dtype)
    b = (torch.rand(n, n, generator=g) * 2 - 1).to(device=device, dtype=dtype)
    return a, b


def buffers(megabytes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Source and destination float32 buffers of the given size, for the copy (bandwidth) test."""
    numel = megabytes * 1024 * 1024 // 4
    return torch.ones(numel, device=device), torch.empty(numel, device=device)
