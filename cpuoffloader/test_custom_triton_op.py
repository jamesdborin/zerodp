#!/usr/bin/env python

"""Sanity test for the custom Triton op using torch.compile."""

import torch

from cpuoffloader.triton_kernels import triton_mapped_expert_copy


def mapped_copy_entry(
    cpu_expert: torch.Tensor, gpu_expert: torch.Tensor
) -> torch.Tensor:
    combined_shape = (cpu_expert.size(0) + gpu_expert.size(0),) + cpu_expert.shape[1:]
    combined = torch.empty(
        combined_shape, device=gpu_expert.device, dtype=gpu_expert.dtype
    )
    return mapped_expert_copy(cpu_expert, gpu_expert, combined)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA device is required to run this test.")

    torch.manual_seed(0)

    num_cpu, num_gpu = 2, 3
    expert_shape = (4, 8)

    cpu_experts = torch.randn(
        (num_cpu,) + expert_shape, dtype=torch.float16
    ).pin_memory()
    gpu_experts = torch.randn(
        (num_gpu,) + expert_shape, dtype=torch.float16, device="cuda"
    )
    combined_expert = torch.randn(
        (num_cpu + num_gpu,) + expert_shape, dtype=torch.float16, device="cuda"
    )

    @torch.compile(fullgraph=True)
    def copy(a, b, c):
        triton_mapped_expert_copy(a, b, c)

    copy(cpu_experts, gpu_experts, combined_expert)
    torch.cuda.synchronize()

    expected = torch.cat([cpu_experts.to(device="cuda"), gpu_experts], dim=0)
    torch.testing.assert_close(combined_expert, expected)

    print("Custom Triton op with torch.compile executed successfully.")


if __name__ == "__main__":
    main()
