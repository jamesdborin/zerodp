#!/usr/bin/env python3
"""
test_qwen3_offload_plugin_llmengine.py

Synchronous LLMEngine example (vLLM v0.5.5 style) so you can pdb into:
- plugin registration
- model construction
- load_weights() (your override)
- MoE offload + buffer init
- step() decode loop

Docs for LLMEngine/add_request/step (v0.5.5): :contentReference[oaicite:0]{index=0}
"""

import os
import pdb

# --- Ensure your general plugin loads BEFORE importing vllm ---
os.environ.setdefault("VLLM_PLUGINS", "cpuoffloader") # cpuoffloader
os.environ.setdefault("VLLM_CPU_OFFLOAD_NUM_EXPERTS", "8")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")

# Optional: makes debugging easier (no graph capture)
os.environ.setdefault("VLLM_USE_CUDA_GRAPH", "0")


def main():
    from vllm import EngineArgs, LLMEngine, SamplingParams  # v0 engine API :contentReference[oaicite:1]{index=1}

    model_name = os.environ.get("MODEL", "jamesdborin/Qwen3-30B-A3B-4layers")
    tp = int(os.environ.get("TP", "1"))


    engine_args = EngineArgs(
        model=model_name,
        tensor_parallel_size=tp,
        trust_remote_code=True,
        # keep synchronous engine happy:
        pipeline_parallel_size=1,
        # keep it small for quicker iteration:
        max_model_len=int(os.environ.get("MAX_LEN", "512")),
        enforce_eager=True,
    )

    engine = LLMEngine.from_engine_args(engine_args)  # :contentReference[oaicite:2]{index=2}

    prompt = "Write a one-sentence explanation of what a Mixture-of-Experts model is."
    sampling_params = SamplingParams(temperature=0.0, max_tokens=10)
    request_id = "debug-0"

    engine.add_request(request_id, prompt, sampling_params)  # :contentReference[oaicite:3]{index=3}


    final_text = None
    while engine.has_unfinished_requests():  # :contentReference[oaicite:4]{index=4}
        request_outputs = engine.step()      # :contentReference[oaicite:5]{index=5}
        for out in request_outputs:
            if out.request_id == request_id and out.finished:
                # In v0.5.5 RequestOutput.outputs is a list of CompletionOutput(s).
                final_text = out.outputs[0].text
                break

    print("\n=== Output ===")
    print(final_text)


if __name__ == "__main__":
    main()
