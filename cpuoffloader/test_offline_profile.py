#!/usr/bin/env python3
"""Profile a batch of CPU-offloaded requests with vLLM's offline interface.

This script mirrors ``test_patch.py`` but exercises the recommended offline
``LLM`` interface instead of the synchronous ``LLMEngine`` API. It configures
vLLM to run with the PyTorch profiler enabled so you can capture traces for
each request in the batch.

Example:
    $ python cpuoffloader/test_offline_profile.py \\
        --model jamesdborin/Qwen3-30B-A3B-4layers \\
        --tensor-parallelism 1 \\
        --batch-size 4 \\
        --profiler-dir ./vllm_profile
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List
import torch
from transformers import AutoTokenizer


# Ensure the CPU-offload plugin is registered before importing vLLM.
os.environ.setdefault("VLLM_LOGGING_LEVEL", "DEBUG")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a batch of prompts using the offline LLM interface, " "cpu-offload plugin, and PyTorch profiler."
        )
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "jamesdborin/Qwen3-30B-A3B-4layers"),
        help="Model identifier or local path to load with vLLM.",
    )
    parser.add_argument(
        "--tensor-parallelism",
        type=int,
        default=int(os.environ.get("TP", "1")),
        help="Tensor parallelism degree to use during generation.",
    )
    parser.add_argument(
        "--cpu-offload-num-experts",
        type=int,
        default=int(os.environ.get("VLLM_CPU_OFFLOAD_NUM_EXPERTS", "64")),
        help=("Number of experts to offload when using the CPU offload plugin."),
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=int(os.environ.get("MAX_LEN", "512")),
        help="Maximum model context length to allocate.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate per request.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts to send in the batch.",
    )
    parser.add_argument(
        "--prompt",
        default=("Write a one-sentence explanation of what a Mixture-of-Experts model is."),
        help=("Optional text to prepend between the UUID prefix and the random body of each prompt."),
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help=("Number of tokens for the random portion of each prompt before detokenizing."),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sleep-after",
        type=float,
        default=5.0,
        help=("Seconds to sleep after profiling to allow background workers " "time to flush trace files."),
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Enable tensor shape recording in the PyTorch profiler.",
    )
    parser.add_argument(
        "--profile-save-dir",
        type=str,
        default="./vllm_profile",
        help="Directory to save the PyTorch profiler traces.",
    )
    parser.add_argument("--enable-plugin", action="store_true", help="Enable the cpuoffloader plugin")

    return parser.parse_args()


def build_prompts(
    tokenizer: AutoTokenizer,
    prompt_len: int,
    batch_size: int,
) -> List[str]:
    vocab_size = tokenizer.vocab_size
    prompts: List[str] = []

    max_id = min(vocab_size, 100_000)
    ids = torch.randint(1, max_id, (batch_size, prompt_len))
    prompts = tokenizer.batch_decode(ids)
    return prompts


def main() -> None:
    args = parse_args()

    os.environ["VLLM_CPU_OFFLOAD_NUM_EXPERTS"] = str(args.cpu_offload_num_experts)
    os.environ["VLLM_PLUGINS"] = "cpuoffloader" if args.enable_plugin else ""
    os.environ["VLLM_TORCH_PROFILER_DIR"] = args.profile_save_dir
    os.environ["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "1" if args.record_shapes else "0"
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = build_prompts(
        tokenizer=tokenizer,
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_tokens=args.max_tokens,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallelism,
        trust_remote_code=True,
        max_model_len=args.max_len,
        enforce_eager=True,
        enable_expert_parallel=True,
    )

    llm.start_profile()
    _ = llm.generate(prompts, sampling_params)
    llm.stop_profile()


if __name__ == "__main__":
    main()
