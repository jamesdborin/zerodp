import torch
import triton
import triton.language as tl

try:
    from torch.library import register_fake
except ImportError:  # PyTorch < 2.3
    pass  # type: ignore


@triton.jit
def _mapped_expert_copy_kernel(
    cpu_experts_ptr,
    gpu_experts_ptr,
    combined_experts_ptr,
    num_exp_cpu,
    numel_cpu,
    numel_gpu,
    expert_stride: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
):
    # Each pid processes a chunk of values
    expert_id = tl.program_id(0)
    chunk_id = tl.program_id(1)

    chunk_offset = chunk_id * CHUNK_SIZE
    num_iters = CHUNK_SIZE // BLOCK_SIZE

    if expert_id < num_exp_cpu:
        # Copy CPU expert
        for i in range(num_iters):
            offsets = expert_id * expert_stride + chunk_offset + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel_cpu
            vals = tl.load(cpu_experts_ptr + offsets, mask=mask)
            tl.store(combined_experts_ptr + offsets, vals, mask=mask)

    if expert_id >= num_exp_cpu:
        # Copy GPU expert
        gpu_expert_id = expert_id - num_exp_cpu
        for i in range(num_iters):
            offsets = gpu_expert_id * expert_stride + chunk_offset + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < numel_gpu
            vals = tl.load(gpu_experts_ptr + offsets, mask=mask)
            tl.store(
                combined_experts_ptr + num_exp_cpu * expert_stride + offsets,
                vals,
                mask=mask,
            )


@torch.library.custom_op("pagellm::triton_mapped_expert_copy", mutates_args=("combined_expert",))
def triton_mapped_expert_copy(
    cpu_expert: torch.Tensor, gpu_expert: torch.Tensor, combined_expert: torch.Tensor
) -> None:
    """
    Copy from a mapped CPU tensor to a CUDA tensor using a Triton kernel.
    """
    assert (
        cpu_expert.is_cpu and cpu_expert.is_pinned()
    ), "cpu_expert must be a pinned CPU tensor created via mapped_tensor"
    assert gpu_expert.is_cuda, "gpu_expert must be a CUDA tensor"
    assert combined_expert.is_cuda, "combined_expert must be a CUDA tensor"

    NUM_CPU_EXPERTS = cpu_expert.size(0)
    combined_expert[:NUM_CPU_EXPERTS].copy_(cpu_expert, non_blocking=True)
    combined_expert[NUM_CPU_EXPERTS:].copy_(gpu_expert)


# Custom op registration ---------------------------------------------------------------------------


@triton_mapped_expert_copy.register_fake
def _mapped_expert_copy_fake(
    cpu_expert: torch.Tensor, gpu_expert: torch.Tensor, combined_expert: torch.Tensor
) -> torch.Tensor:
    return None
