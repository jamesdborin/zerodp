#!/usr/bin/env python3
"""
Profile a batch of CPU-offloaded requests with sglang's offline interface.

This script mirrors test_offline_profile.py but uses sglang instead of vLLM.
It configures sglang to run with the PyTorch profiler enabled so you can capture traces for each request in the batch.

Example:
    $ python cpuoffloader/test_offline_profile_sglang.py \
        --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
        --tp-size 1 \
        --batch-size 4 \
        --profile-save-dir ./sglang_profile
"""

import argparse
import os
import time
import torch

# Set up environment variables for sglang profiling and CPU offload
os.environ.setdefault("SGLANG_LOGGING_LEVEL", "DEBUG")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile a batch of prompts using sglang offline interface and PyTorch profiler."
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("MODEL", "jamesdborin/Qwen3-30B-A3B-4layers"),
        help="Model identifier or local path to load with sglang.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=int(os.environ.get("TP", "1")),
        help="Tensor parallelism degree to use during generation.",
    )
    parser.add_argument(
        "--cpu-offload-num-experts",
        type=int,
        default=int(os.environ.get("SGLANG_CPU_OFFLOAD_NUM_EXPERTS", "64")),
        help="Number of experts to offload when using the CPU offload plugin.",
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
        default="Write a one-sentence explanation of what a Mixture-of-Experts model is.",
        help="Optional text to use as the prompt.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Number of tokens for the random portion of each prompt before detokenizing.",
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
        help="Seconds to sleep after profiling to allow background workers to flush trace files.",
    )
    parser.add_argument(
        "--record-shapes",
        action="store_true",
        help="Enable tensor shape recording in the PyTorch profiler.",
    )
    parser.add_argument(
        "--profile-save-dir",
        type=str,
        default=None,
        help="Directory to save the PyTorch profiler traces.",
    )
    parser.add_argument(
        "--data-parallel-rank",
        type=int,
        default=None,
        help="Which rank to send requests to.",
    )
    return parser.parse_args()

def build_prompts(tokenizer, prompt_len, batch_size, prompt_prefix):
    vocab_size = tokenizer.vocab_size
    max_id = min(vocab_size, 100_000)
    ids = torch.randint(1, max_id, (batch_size, prompt_len))
    prompts = tokenizer.batch_decode(ids)
    prompts = [f"{prompt_prefix} {p}" for p in prompts]
    return prompts


def main():
    args = parse_args()

    do_profile = args.profile_save_dir is not None
    if do_profile:
        os.environ["SGLANG_TORCH_PROFILER_DIR"] = args.profile_save_dir
        os.environ["SGLANG_TORCH_PROFILER_RECORD_SHAPES"] = "0"
    
    os.environ["SGLANG_OFFLOAD_NUM_EXPERTS"] = str(args.cpu_offload_num_experts)
    os.environ["SGLANG_USE_ZERODP"] = "1"
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer
    from sglang.srt.entrypoints.engine import Engine

    tokenizer = get_tokenizer(args.model_path)
    prompts = build_prompts(
        tokenizer=tokenizer,
        prompt_len=args.prompt_len,
        batch_size=args.batch_size,
        prompt_prefix=args.prompt,
    )

    sampling_params = {
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "ignore_eos": True,
    }
    engine = Engine(
        model_path=args.model_path,
        tp_size=args.tp_size,
        context_length=args.max_len,
        trust_remote_code=True,
        log_level="info",
        disable_cuda_graph=False,
        dp_size=2
    )
    start = time.time()
    
    if do_profile:
        engine.start_profile()
    
    _ = engine.generate(prompts, sampling_params, data_parallel_rank=args.data_parallel_rank)
    
    if do_profile:
        engine.stop_profile()
    end = time.time()

    print(f"Finished in {end - start:.2f} seconds.")
    time.sleep(args.sleep_after)


if __name__ == "__main__":
    main()
