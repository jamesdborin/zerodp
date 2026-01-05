"""Benchmark the throughput of the Triton mapped expert copy kernel.

This script sweeps the number of CPU offloaded experts from 8 to 120 (step 8)
for a combined tensor of shape [128, 2048, 1512] using FP8 tensors. It reports
the mean kernel time and the achieved bandwidth computed as the CPU tensor size
divided by the measured time.
"""

from __future__ import annotations

import argparse
from typing import Iterable

import torch

from cpuoffloader.triton_kernels import triton_mapped_expert_copy

TOTAL_EXPERTS = 128
EXPERT_SHAPE = (2048, 1512)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=
        "Benchmark the pagellm::triton_mapped_expert_copy custom Triton kernel."
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warmup kernel launches to issue before timing.",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=20,
        help="Number of timed kernel launches per configuration.",
    )
    parser.add_argument(
        "--min-offload",
        type=int,
        default=8,
        help="Minimum number of experts to offload to CPU (inclusive).",
    )
    parser.add_argument(
        "--max-offload",
        type=int,
        default=120,
        help="Maximum number of experts to offload to CPU (inclusive).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=8,
        help="Step size used when sweeping the offloaded expert count.",
    )
    parser.add_argument(
        "--fp8-format",
        choices=("e5m2", "e4m3fn"),
        default="e5m2",
        help="FP8 format to use for the benchmark tensors.",
    )
    return parser.parse_args()


def _resolve_fp8_dtype(format_name: str) -> torch.dtype:
    match format_name.lower():
        case "e5m2":
            dtype = getattr(torch, "float8_e5m2", None)
        case "e4m3fn":
            dtype = getattr(torch, "float8_e4m3fn", None)
        case _:
            dtype = None
    if dtype is None:
        raise RuntimeError(
            f"Requested FP8 format '{format_name}' is not available in this PyTorch build."
        )
    return dtype


def _allocate_tensors(num_cpu_experts: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    if num_cpu_experts < 0 or num_cpu_experts > TOTAL_EXPERTS:
        raise ValueError(
            f"num_cpu_experts must be in [0, {TOTAL_EXPERTS}], got {num_cpu_experts}."
        )
    num_gpu_experts = TOTAL_EXPERTS - num_cpu_experts
    cpu = torch.empty(
        (num_cpu_experts,) + EXPERT_SHAPE,
        dtype=dtype,
        device="cpu",
        pin_memory=True,
    )
    gpu = torch.empty((num_gpu_experts,) + EXPERT_SHAPE, dtype=dtype, device="cuda")
    combined = torch.empty(
        (TOTAL_EXPERTS,) + EXPERT_SHAPE, dtype=dtype, device="cuda", requires_grad=False
    )
    return cpu, gpu, combined


def _timed_kernel_launch(
    cpu_expert: torch.Tensor,
    gpu_expert: torch.Tensor,
    combined_expert: torch.Tensor,
    iterations: int,
) -> list[float]:
    times_ms: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        triton_mapped_expert_copy(cpu_expert, gpu_expert, combined_expert)
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _format_bandwidth(bytes_transferred: int, avg_ms: float) -> tuple[float, float]:
    seconds = avg_ms / 1e3
    bandwidth_bps = bytes_transferred / seconds if seconds > 0 else float("inf")
    bandwidth_gbps = bandwidth_bps / 1e9
    return bandwidth_bps, bandwidth_gbps


def _sweep_offload_counts(
    counts: Iterable[int], dtype: torch.dtype, warmup_iters: int, bench_iters: int
) -> None:
    for num_offloaded in counts:
        cpu, gpu, combined = _allocate_tensors(num_offloaded, dtype)
        for _ in range(warmup_iters):
            triton_mapped_expert_copy(cpu, gpu, combined)
        torch.cuda.synchronize()

        times_ms = _timed_kernel_launch(cpu, gpu, combined, bench_iters)
        avg_ms = sum(times_ms) / len(times_ms)
        min_ms = min(times_ms)
        max_ms = max(times_ms)
        bytes_transferred = cpu.numel() * cpu.element_size()
        bandwidth_bps, bandwidth_gbps = _format_bandwidth(bytes_transferred, avg_ms)

        print(
            f"offloaded_experts={num_offloaded:3d} | avg={avg_ms:8.4f} ms | "
            f"min={min_ms:8.4f} ms | max={max_ms:8.4f} ms | "
            f"bandwidth={bandwidth_bps:11.3e} B/s ({bandwidth_gbps:8.3f} GB/s) | "
            f"cpu_bytes={bytes_transferred}"
        )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled GPU is required to run this benchmark.")
    dtype = _resolve_fp8_dtype(args.fp8_format)
    if args.warmup_iters < 0:
        raise ValueError("warmup-iters must be >= 0.")
    if args.bench_iters <= 0:
        raise ValueError("bench-iters must be > 0.")
    if args.min_offload < 0 or args.max_offload > TOTAL_EXPERTS:
        raise ValueError(
            f"Offload sweep must stay within [0, {TOTAL_EXPERTS}] experts, "
            f"got range {args.min_offload}..{args.max_offload}."
        )
    if args.min_offload > args.max_offload:
        raise ValueError("min-offload must be <= max-offload.")
    if args.step <= 0:
        raise ValueError("Step size must be positive.")

    offload_counts = range(args.min_offload, args.max_offload + 1, args.step)
    _sweep_offload_counts(offload_counts, dtype, args.warmup_iters, args.bench_iters)


if __name__ == "__main__":
    main()
