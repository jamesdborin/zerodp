# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 MoE model with CPU offloading support.

This module provides a variant of Qwen3MoE that offloads expert weights
to CPU mapped memory, copying them to GPU before each forward pass.
"""

import os
from itertools import islice

import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.distributed import (
    get_pp_group,
)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors


from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel, Qwen3MoeForCausalLM
from vllm.model_executor.layers.fused_moe import FusedMoE

from .triton_kernels import triton_mapped_expert_copy

logger = init_logger(__name__)

_OFFLOAD_ENV_VAR = "VLLM_CPU_OFFLOAD_NUM_EXPERTS"


def create_cpu_mapped_tensor(shape: list[int], dtype: torch.dtype) -> torch.Tensor:
    """
    Create a CPU tensor with mapped memory (GPU-accessible pinned memory).

    Args:
        shape: Tensor shape
        dtype: Tensor dtype

    Returns:
        A CPU tensor backed by mapped memory
    """
    return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)


def _get_requested_offload() -> int:
    value = os.getenv(_OFFLOAD_ENV_VAR)
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except ValueError:
        logger.warning(
            "Invalid value %r for %s; expected integer. Ignoring.",
            value,
            _OFFLOAD_ENV_VAR,
        )
        return 0


class OffloadFusedMoE(FusedMoE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_experts_to_offload = _get_requested_offload()
        self.num_experts_to_offload = max(0, self.num_experts_to_offload)
        self.num_experts_to_offload = min(
            self.num_experts_to_offload,
            self.local_num_experts,  # defined in base FusedMoE
        )

    def _offload_experts(self):
        """
        Offload experts to cpu after weight loading is complete.

        This removes the first num_experts_to_offload experts from w13_weight
        and stores them in pinned CPU memory for streaming during inference.
        w2_weight stays fully on GPU (no pruning/offloading for w2).

        It then creates a new tensor, w13_weight_gpu, to hold the remaining
        experts on GPU.

        This is because the fused moe method expects the combined weights to be
        in w13_weight during forward passes. So we just-in-time create a new tensor
        to hold the remaining experts on GPU and will assign it to w13_weight before
        each forward pass.

        Called by the Model class after load_weights.
        """
        if self.num_experts_to_offload <= 0:
            return

        # Prune w13_weight (gate_up_proj) ONLY: shape is [num_experts, ...]
        # w2_weight stays fully on GPU - no pruning needed
        if hasattr(self, "w13_weight") and self.w13_weight is not None:
            original_shape = self.w13_weight.data.shape

            # Extract the experts to offload and copy to pinned CPU memory
            self.w13_weight_offloaded = create_cpu_mapped_tensor(
                [self.num_experts_to_offload] + list(original_shape[1:]),
                self.w13_weight.data.dtype,
            )
            self.w13_weight_offloaded.copy_(self.w13_weight.data[: self.num_experts_to_offload].cpu())

            # Keep only the remaining experts on GPU
            self.w13_weight_gpu = self.w13_weight.data[self.num_experts_to_offload :].clone().contiguous()
            del self.w13_weight
            logger.debug(
                f"Pruned w13_weight: {original_shape} -> {self.w13_weight_gpu.data.shape} "
                f"(offloaded {self.num_experts_to_offload} experts to CPU pinned memory)"
            )

        # Note: w2_weight stays intact on GPU (all experts remain)
        if hasattr(self, "w2_weight") and self.w2_weight is not None:
            logger.debug(f"w2_weight remains on GPU with full shape: {self.w2_weight.data.shape}")

        logger.debug(
            f"Expert pruning complete: w13 has {self.local_num_experts - self.num_experts_to_offload} experts on GPU "
            f"+ {self.num_experts_to_offload} offloaded; w2 has all {self.local_num_experts} experts on GPU"
        )


@support_torch_compile
class Qwen3MoeModelOffload(Qwen3MoeModel):
    """Qwen3 MoE model with CPU offloading and ping-pong double buffering."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_experts_to_offload = _get_requested_offload()
        self.num_experts_to_offload = max(0, self.num_experts_to_offload)
        self.num_experts_to_offload = min(
            self.num_experts_to_offload,
            self.config.num_experts,
        )
        self.should_offload = self.num_experts_to_offload > 0
        self._copy_stream = torch.cuda.Stream() if self.should_offload else None
        self._buffers_initialized = False

    def initialize_buffers(self):
        """
        Initialize the ping-pong double buffers for MoE weight streaming.

        This should be called after weights are loaded and offloading is complete.
        Creates two GPU buffers sized to hold combined (CPU offloaded + GPU resident)
        w13 expert weights.
        """
        if self._buffers_initialized:
            return

        # Get the first MoE layer to determine buffer shapes/dtypes
        first_moe = self.layers[0].mlp.experts
        second_moe = self.layers[1].mlp.experts

        # Calculate combined shape: [num_offloaded + num_gpu_resident, ...]
        num_offloaded = first_moe.w13_weight_offloaded.shape[0]
        num_gpu_resident = first_moe.w13_weight_gpu.shape[0]
        total_experts = num_offloaded + num_gpu_resident

        w13_combined_shape = (total_experts,) + first_moe.w13_weight_offloaded.shape[1:]
        w13_dtype = first_moe.w13_weight_offloaded.dtype

        logger.info(
            f"Initializing ping-pong buffers for {len(self.layers)} MoE layers: "
            f"w13={w13_combined_shape} ({w13_dtype})"
        )

        # Create two buffers for ping-pong
        self.combined = [
            torch.empty(w13_combined_shape, dtype=w13_dtype, device="cuda", requires_grad=False),
            torch.empty(w13_combined_shape, dtype=w13_dtype, device="cuda", requires_grad=False),
        ]

        # Pre-load first layer's weights into combined[0]
        gpu_src = getattr(first_moe, "w13_weight_gpu", None)
        if gpu_src is None:
            gpu_src = first_moe.w13_weight.data
        triton_mapped_expert_copy(
            first_moe.w13_weight_offloaded,
            gpu_src,
            self.combined[0],
        )
        first_moe.w13_weight = self.combined[0]

        torch.cuda.synchronize()
        self._buffers_initialized = True
        logger.info("Ping-pong buffers initialized")

        # Assume you have two buffers: combined[0], combined[1] (ping-pong).

        # Create two event arrays once (in __init__):
        # copy_ready[i]: recorded on _copy_stream after buffer i has been filled.
        # compute_done[i]: recorded on the compute stream after buffer i is no longer needed (safe to overwrite).
        # Initialize compute_done[i] as “done” once so the first prefetch can proceed.
        #
        self.copy_done = [torch.cuda.Event(blocking=False), torch.cuda.Event(blocking=False)]
        self.compute_done = [torch.cuda.Event(blocking=False), torch.cuda.Event(blocking=False)]

        # mark buffers initially free (record on current/default stream once)
        for e in self.compute_done:
            e.record(torch.cuda.current_stream())

        # first buffer is ready to go (we copy on init)
        self.copy_done[0].record(self._copy_stream)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            # Collect auxiliary hidden states if specified
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_state = hidden_states + residual if residual is not None else hidden_states
                aux_hidden_states.append(aux_hidden_state)

            # on a second cuda stream, copy the next layer's offloaded experts
            # On the end layer copy the first layer's experts for next iteration
            # during warmup vllm runs the forward pass to assess activation memory usage
            # during this point the offloaded weights dont exist yet so we skip it.

            next_buf = (layer_idx + 1) % 2  # buffer being produced for next layer
            this_buf = layer_idx % 2  # buffer being consumed by this layer

            # 3) COPY STREAM: don't overwrite next_buf until compute is done with it
            if hasattr(self.layers[0].mlp.experts, "w13_weight_offloaded"):
                self.compute_done[next_buf].wait(self._copy_stream)

                next_layer_idx = (layer_idx + 1) % len(self.layers)
                next_moe = self.layers[next_layer_idx].mlp.experts
                gpu_src = getattr(next_moe, "w13_weight_gpu", None)
                cpu_src = next_moe.w13_weight_offloaded
                combined_buffer = self.combined[next_buf]

                NUM_CPU_EXPERTS = cpu_src.size(0)

                with torch.cuda.stream(self._copy_stream):
                    combined_buffer[NUM_CPU_EXPERTS:].copy_(gpu_src)
                    combined_buffer[:NUM_CPU_EXPERTS].copy_(cpu_src, non_blocking=True)

                next_moe.w13_weight = combined_buffer
                self.copy_done[next_buf].record(self._copy_stream)

            # 1) COMPUTE STREAM: don't use the this layer's buffer until it is done being copied into.
            self.copy_done[this_buf].wait(torch.cuda.current_stream())
            hidden_states, residual = layer(positions, hidden_states, residual)

            # 2) COMPUTE STREAM: once done consuming this_buf, mark it free for future copying
            self.compute_done[this_buf].record(torch.cuda.current_stream())

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)

        # Return auxiliary hidden states if collected
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        from vllm.model_executor.layers.fused_moe import FusedMoE

        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )


class Qwen3MoeForCausalLMOffload(Qwen3MoeForCausalLM):
    """
    Qwen3 MoE model with CPU offloading for expert weights.

    This model stores MoE expert weights in CPU mapped memory and copies
    them to GPU before each forward pass. This enables running larger
    MoE models by offloading expert weights to host memory.
    """

    def offload_experts_to_cpu(self):
        """
        Move all expert weights to CPU mapped memory.
        Call this after loading weights to enable CPU offloading.
        """
        logger.info("Offloading expert weights to CPU...")

        # self.moe_layers is populated by the base class in its __init__
        for i, moe_layer in enumerate(self.moe_layers):
            if moe_layer.num_experts_to_offload <= 0:
                continue
            moe_layer._offload_experts()
            logger.info(
                "Offloaded %s experts from MoE layer %s to CPU pinned memory",
                moe_layer.num_experts_to_offload,
                i,
            )
            logger.info("All expert weights offloaded to CPU")

    def offload_weights(self):
        logger.debug("Offloading weights to CPU and initializing buffers...")
        # move weights to cpu mapped memory
        self.offload_experts_to_cpu()
        logger.debug("Weights offloaded. Initializing buffers...")
        # initialize ping-pong buffers
        self.model.initialize_buffers()


from vllm.model_executor.model_loader.utils import process_weights_after_loading as orig


def process_and_offload_weights_after_loading(
    model: torch.nn.Module,
    model_config,
    target_device: torch.device,
) -> None:
    logger.debug("Processing and offloading weights after loading...")
    orig(model, model_config, target_device)
    logger.debug("Weight processing complete.")
    logger.debug("Offloading weights to CPU...")
    model.offload_weights()
