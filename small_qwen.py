#!/usr/bin/env python3
"""
Create a 4-layer Qwen3-30B-A3B-derived model:
- download config + tokenizer
- set num_hidden_layers=4
- initialize fresh weights
- count parameters
- push config, tokenizer, and weights to Hugging Face

WARNING:
Weights are randomly initialized (not sliced from the 30B model).
"""

import argparse
import os
import tempfile
import json

import torch
from huggingface_hub import HfApi, upload_folder
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Qwen/Qwen3-30B-A3B")
    ap.add_argument("--dst", required=True, help="e.g. your-username/Qwen3-30B-A3B-L4")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = ap.parse_args()

    token = os.environ.get("HUGGING_FACE_TOKEN")
    torch_dtype = getattr(torch, args.dtype)

    # ------------------------------------------------------------
    # 1) Load config + tokenizer
    # ------------------------------------------------------------
    config = AutoConfig.from_pretrained(args.src, token=token)
    tokenizer = AutoTokenizer.from_pretrained(args.src, token=token)

    old_layers = config.num_hidden_layers
    config.num_hidden_layers = args.layers

    if hasattr(config, "max_window_layers") and config.max_window_layers is not None:
        config.max_window_layers = min(config.max_window_layers, args.layers)

    # ------------------------------------------------------------
    # 2) Initialize a fresh model with the modified config
    # ------------------------------------------------------------
    model = AutoModelForCausalLM.from_config(
        config,
        torch_dtype=torch_dtype,
    )

    n_params = sum(p.numel() for p in model.parameters())

    print(f"Old layers: {old_layers}")
    print(f"New layers: {config.num_hidden_layers}")
    print(f"Parameter count: {n_params:,}")

    # ------------------------------------------------------------
    # 3) Create HF repo
    # ------------------------------------------------------------
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.dst,
        exist_ok=True,
        private=args.private,
    )

    # ------------------------------------------------------------
    # 4) Save everything and upload
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save model + config
        model.save_pretrained(tmpdir, safe_serialization=True)
        tokenizer.save_pretrained(tmpdir)

        # README
        readme = f"""---
license: apache-2.0
tags:
- qwen
- qwen3
- causal-lm
- randomly-initialized
---

# {args.dst}

This model is derived from `{args.src}` with:

- `num_hidden_layers = {args.layers}`
- Freshly initialized weights (⚠️ **not** the original 30B weights)
- Total parameters: **{n_params:,}**

This checkpoint is suitable for:
- research
- fine-tuning
- architecture experiments
"""
        with open(os.path.join(tmpdir, "README.md"), "w") as f:
            f.write(readme)

        # Metadata report
        with open(os.path.join(tmpdir, "report.json"), "w") as f:
            json.dump(
                {
                    "source_model": args.src,
                    "old_num_hidden_layers": old_layers,
                    "new_num_hidden_layers": args.layers,
                    "parameter_count": n_params,
                    "dtype": args.dtype,
                    "weights": "randomly_initialized",
                },
                f,
                indent=2,
            )

        upload_folder(
            repo_id=args.dst,
            folder_path=tmpdir,
            commit_message=f"Initial commit: {args.layers}-layer model with tokenizer and weights",
            token=token,
        )

    print(f"\n✅ Uploaded to https://huggingface.co/{args.dst}")

if __name__ == "__main__":
    main()
