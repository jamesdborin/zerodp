import logging

logger = logging.getLogger(__name__)


_ALREADY_PATCHED = False


def apply_fused_moe_patch() -> None:
    global _ALREADY_PATCHED
    if _ALREADY_PATCHED:
        return

    # Import your replacement class
    from .qwen3_moe_offload import (
        OffloadFusedMoE,
        Qwen3MoeModelOffload,
        Qwen3MoeForCausalLMOffload,
        process_and_offload_weights_after_loading,
    )

    # Patch the canonical symbol used by most model code.
    import vllm.model_executor.layers.fused_moe as fused_moe_mod
    fused_moe_mod.FusedMoE = OffloadFusedMoE
    logger.info("[vllm-qwen3-offload] patched vllm.model_executor.layers.fused_moe.FusedMoE")

    import vllm.model_executor.models.qwen3_moe as qwen3_model_mod
    qwen3_model_mod.Qwen3MoeModel = Qwen3MoeModelOffload
    logger.info("[vllm-qwen3-offload] patched vllm.model_executor.models.qwen3_moe.Qwen3MoeModel")

    qwen3_model_mod.Qwen3MoeForCausalLM = Qwen3MoeForCausalLMOffload
    logger.info("[vllm-qwen3-offload] patched vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM")

    import vllm.model_executor.model_loader.utils as model_loader_utils_mod
    model_loader_utils_mod.process_weights_after_loading = process_and_offload_weights_after_loading
    logger.info("[vllm-qwen3-offload] patched vllm.model_executor.model_loader.utils.process_weights_after_loading")

    # Also patch any modules that imported FusedMoE into their local module namespace
    # (important if they did `from ...fused_moe import FusedMoE`).
    try:
        import vllm.model_executor.models.qwen3_moe as qwen3_moe_mod
        import vllm.model_executor.model_loader.utils as model_loader_utils_mod
        import vllm.model_executor.model_loader.base_loader as base_loader_mod
        setattr(qwen3_moe_mod, "FusedMoE", OffloadFusedMoE)
        setattr(qwen3_moe_mod, "Qwen3MoeModel", Qwen3MoeModelOffload)
        setattr(qwen3_moe_mod, "Qwen3MoeForCausalLM", Qwen3MoeForCausalLMOffload)
        setattr(model_loader_utils_mod, "process_weights_after_loading", process_and_offload_weights_after_loading)
        setattr(base_loader_mod, "process_weights_after_loading", process_and_offload_weights_after_loading)
        
        logger.info("[vllm-qwen3-offload] patched vllm.model_executor.models.qwen3_moe.FusedMoE")
        logger.info("[vllm-qwen3-offload] patched vllm.model_executor.models.qwen3_moe.Qwen3MoeModel")
        logger.info("[vllm-qwen3-offload] patched vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM")
        logger.info("[vllm-qwen3-offload] patched vllm.model_executor.model_loader.utils.process_weights_after_loading")

    except Exception:
        # If not imported yet, nothing to do; patch above is enough.
        pass

    _ALREADY_PATCHED = True
