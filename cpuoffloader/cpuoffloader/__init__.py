import logging
logger = logging.getLogger(__name__)


def register() -> None:
    """
    vLLM general plugin entry point.
    Runs in every vLLM process before model initialization. :contentReference[oaicite:3]{index=3}
    """
    logger.info("[vllm-qwen3-offload] registering patches")

    # 1) Patch MoE layer class resolution so Qwen3MoE builds OffloadFusedMoE layers.
    from .offload_patch import apply_fused_moe_patch
    apply_fused_moe_patch()

    # 2) Override the model registry so the Qwen3MoE architecture resolves to your model.
    # ModelRegistry.register_model overwrites existing registrations (warns). :contentReference[oaicite:4]{index=4}
    # fqcn = "vllm_qwen3_offload.qwen3_moe_offload:Qwen3MoeForCausalLMOffload"
    
    # # The common arch string in vLLM is typically Qwen3MoeForCausalLM (per vLLM naming).
    # # We register a couple variants to be robust across minor naming differences.
    # for arch in ("Qwen3MoeForCausalLM", "Qwen3MoEForCausalLM"):
    #     ModelRegistry.register_model(arch, fqcn)
    #     logger.info("[vllm-qwen3-offload] ModelRegistry: %s -> %s", arch, fqcn)

    logger.info("[vllm-qwen3-offload] done")
