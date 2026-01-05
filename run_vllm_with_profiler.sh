#!/usr/bin/env bash

set -euo pipefail

MODEL="${MODEL:-jamesdborin/Qwen3-30B-A3B-4layers}"
MAX_LEN="${MAX_LEN:-512}"

export VLLM_PLUGINS="${VLLM_PLUGINS:-""}"
export VLLM_CPU_OFFLOAD_NUM_EXPERTS="${VLLM_CPU_OFFLOAD_NUM_EXPERTS:-8}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-DEBUG}"
export VLLM_USE_CUDA_GRAPH="${VLLM_USE_CUDA_GRAPH:-1}"

mkdir -p ./vllm_profile

VLLM_TORCH_PROFILER_DIR=./vllm_profile \
  vllm serve "$MODEL" \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code \
  --enforce-eager \
  --port 8000 \
  --host 0.0.0.0 \
  --compilation-config '{"use_inductor": false}'
