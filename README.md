# ZeroDP: Zero-Copy Data Parallel Expert Offloading

This repository contains the code accompanying the blog post at [mainlymatmul.com/blog/zerodp](https://mainlymatmul.com/blog/zerodp).

ZeroDP implements an efficient zero-copy data parallel approach for serving Mixture-of-Experts (MoE) models, where expert weights are shared across data parallel ranks via CUDA IPC (Inter-Process Communication) handles. This eliminates the need to duplicate expert weights across GPUs, significantly reducing memory requirements.

## Overview

Traditional data parallel serving duplicates model weights across all GPUs. For MoE models with large expert parameters, this is extremely memory-inefficient. ZeroDP solves this by:

1. **Loading experts on a single GPU** (rank 0)
2. **Sharing IPC handles** via torch multiprocessing queues to other ranks
3. **Asynchronous overlapped copying** during the forward pass for efficient computation
4. **Double-buffered transfers** to hide memory copy latency

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Parallel Rank 0                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  All Expert Weights (Source)                              │  │
│  │  • Layer 0-N: w13_weight [num_experts, 2*inter, hidden]  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           │ IPC Handles via Queue               │
│                           ▼                                     │
├─────────────────────────────────────────────────────────────────┤
│                      Data Parallel Rank 1                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Double Buffers (Destination)                             │  │
│  │  • Buffer 0 ──┐                                           │  │
│  │  • Buffer 1 ──┤  Ping-pong between layers                │  │
│  └───────────────┴───────────────────────────────────────────┘  │
│                                                                 │
│  Forward Pass with Overlap:                                    │
│  1. Stream 0 (Compute): Wait for buffer, run MLP              │
│  2. Stream 1 (Copy): Async copy next layer to other buffer    │
│  3. Synchronize with CUDA events for coordination             │
└─────────────────────────────────────────────────────────────────┘
```

## Key Implementation Files

### 1. `sglang/python/sglang/srt/models/qwen3_moe.py`

**Primary Addition**: `offload_experts()` method

This file implements the IPC handle transfer mechanism:

- **Rank 0 (Source)**: Creates expert weight tensors and sends IPC handles through the multiprocessing queue
- **Rank 1 (Receiver)**: Receives IPC handles and reconstructs remote tensor references
- Uses CUDA events to signal when the transfer is complete
- Sets up `source_experts` attribute on receiving rank for later async copies

Key code pattern:
```python
# Rank 0: Send IPC handles
ipc_queue.put(tensor)  # Send tensor reference
transfer_complete.record(send_stream)
ipc_queue.put(transfer_complete.ipc_handle())  # Send completion event

# Rank 1: Receive and setup
received_tensor = ipc_queue.get()
layer.mlp.experts.source_experts = received_tensor
ready_evt = torch.cuda.Event.from_ipc_handle(...)
```

### 2. `sglang/python/sglang/srt/models/qwen2_moe.py`

**Primary Addition**: Asynchronous overlapped copy in `Qwen2MoeModel.forward()`

Implements double-buffered expert loading with stream synchronization:

- Creates two buffers and two sets of CUDA events (`copy_done`, `compute_done`)
- **Copy stream**: Asynchronously copies next layer's experts while current layer computes
- **Compute stream**: Waits for current buffer to be ready, then runs MLP forward
- Events coordinate between streams to prevent race conditions

Key code pattern:
```python
# Double buffering with events
copy_done = [torch.cuda.Event(), torch.cuda.Event()]
compute_done = [torch.cuda.Event(), torch.cuda.Event()]

# Overlap: prefetch next while computing current
with torch.cuda.stream(self.copy_stream):
    self.copy_stream.wait_event(compute_done[next_buf])
    combined_experts.copy_(source_src, non_blocking=True)
    copy_done[next_buf].record(self.copy_stream)

compute_stream.wait_event(copy_done[this_buf])
# ... run computation ...
compute_done[this_buf].record(compute_stream)
```

### 3. `sglang/python/sglang/srt/managers/data_parallel_controller.py`

**Primary Addition**: IPC queue creation and distribution

Creates torch multiprocessing queues and passes them to each data parallel worker:

- Spawns one queue per tensor parallel rank (line 258-263)
- Queues are created with `mp.get_context("spawn").Queue()`
- Each DP worker receives its corresponding queue during process creation
- Queues are passed through the launch chain to reach the model loader

Key code pattern:
```python
# Create queues for IPC communication
ctx = mp.get_context("spawn")
num_parallel_queues = server_args.tp_size
ipc_queues = [ctx.Queue() for _ in range(num_parallel_queues)]

# Pass to scheduler process
proc = mp.Process(
    target=self.run_scheduler_process_func,
    args=(..., ipc_queue),
)
```

### 4. `sglang/python/sglang/srt/managers/tp_worker.py`

**Pass-through Layer**: Queue propagation

Receives `ipc_queue` parameter and forwards it to the model runner:

- Accepts `ipc_queue` in `__init__()` (line 219)
- Stores as instance variable (line 230)
- Passes to `ModelRunner` creation (line 267)
- Ensures queue reaches the model loading stage

### 5. `sglang/python/sglang/srt/model_loader/loader.py`

**Primary Addition**: Expert offloading invocation

Calls the model's `offload_experts()` method after weight loading:

- Receives `ipc_queue` and `dp_rank` parameters in `load_model()` (line 579-580)
- Checks if model has `use_zerodp` flag enabled (line 600)
- Invokes `model.offload_experts()` with queue, rank, and GPU ID (line 601)
- This triggers the IPC handle transfer implemented in `qwen3_moe.py`

Key code pattern:
```python
if hasattr(model.model, "use_zerodp") and model.model.use_zerodp:
    model.offload_experts(
        ipc_queue=ipc_queue,
        dp_rank=dp_rank,
        gpu_id=device_config.gpu_id
    )
```

### 6. `zerodp/test_offline_profile_sglang.py`

**Testing Script**: Profiling and validation

Provides command-line interface to test the ZeroDP implementation:

- Configures sglang with `SGLANG_USE_ZERODP=1` environment variable
- Supports targeting specific data parallel ranks via `--data-parallel-rank`
- Enables PyTorch profiler for performance analysis with `--profile-save-dir`
- Creates randomized prompts for consistent benchmarking
- Allows measuring individual DP rank performance to verify load balancing

Usage examples:
```bash
# Run on all DP ranks
python zerodp/test_offline_profile_sglang.py \
    --model-path Qwen/Qwen3-30B-A3B \
    --tp-size 1 \
    --batch-size 4

# Target specific DP rank for profiling
python zerodp/test_offline_profile_sglang.py \
    --model-path Qwen/Qwen3-30B-A3B \
    --data-parallel-rank 0 \
    --profile-save-dir ./profiles
```

## How It Works

### 1. Initialization Phase

1. **Data Parallel Controller** creates multiprocessing queues (one per TP rank)
2. Queues are passed through `tp_worker.py` to reach the **Loader**
3. **Loader** calls `offload_experts()` on the model with the queue

### 2. Weight Transfer Phase (qwen3_moe.py)

1. **Rank 0** sends tensor IPC handles through the queue
2. **Rank 1** receives handles and reconstructs tensor references
3. Completion event coordinates the handoff
4. Rank 1 stores remote tensors in `source_experts` attribute

### 3. Inference Phase (qwen2_moe.py)

1. **First Layer**: Copy experts from remote GPU to local buffer 0
2. **Loop**: For each layer:
   - Compute stream: Wait for current buffer, run MLP
   - Copy stream: Async copy next layer to alternate buffer
   - Events prevent buffer overwrites and ensure readiness
3. **Synchronization**: Wait for final copy stream completion

## Benefits

- **50% Memory Reduction**: Expert weights stored on only one GPU
- **Zero Copy Overhead**: Uses CUDA IPC instead of explicit copies
- **Latency Hiding**: Double buffering overlaps transfers with computation
- **Stream Parallelism**: Independent compute and copy streams maximize throughput

## Installation

### Prerequisites

- CUDA-capable GPUs with P2P access enabled
- PyTorch with CUDA support
- SGLang dependencies

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/jamesdborin/zerodp.git
cd zerodp

# Initialize and update submodules
git submodule update --init --recursive

# Install SGLang from the submodule
cd sglang
pip install -e "python[all]"
cd ..
```

### Verify Installation

```bash
# Check CUDA IPC availability
python -c "import torch; print('IPC available:', torch.cuda.is_available())"

# Run basic test
python zerodp/test_offline_profile_sglang.py \
    --model-path Qwen/Qwen3-30B-A3B \
    --batch-size 1 \
    --max-tokens 16
```

## Usage

### Profiling

```bash
# Profile all DP ranks
python zerodp/test_offline_profile_sglang.py \
    --model-path Qwen/Qwen3-30B-A3B \
    --tp-size 1 \
    --batch-size 4 \
    --profile-save-dir ./traces

# Profile specific rank
python zerodp/test_offline_profile_sglang.py \
    --model-path Qwen/Qwen3-30B-A3B \
    --data-parallel-rank 1 \
    --profile-save-dir ./traces/rank1
```

View traces with:
```bash
tensorboard --logdir ./traces
# Navigate to http://localhost:6006
```

## Configuration Options

### Environment Variables

- `SGLANG_USE_ZERODP`: Enable ZeroDP mode (set to `1`)
- `SGLANG_LOGGING_LEVEL`: Set to `DEBUG` for detailed logs

### Test Script Arguments

- `--model-path`: HuggingFace model ID or local path
- `--tp-size`: Tensor parallelism degree
- `--batch-size`: Number of concurrent requests
- `--data-parallel-rank`: Target specific DP rank (0 or 1)
- `--profile-save-dir`: Directory for PyTorch profiler traces
- `--max-tokens`: Maximum tokens to generate per request

## Performance Tips

1. **GPU Affinity**: Ensure GPUs are on the same NUMA node for faster IPC
2. **Buffer Size**: Larger buffers reduce streaming overhead but increase memory
3. **Batch Size**: Larger batches better hide transfer latency
4. **Profiling**: Use `--profile-save-dir` to identify bottlenecks

## Limitations

- Currently supports Qwen2/Qwen3 MoE models
- Requires at least 2 data parallel ranks
- Assumes all experts fit in GPU memory on rank 0
- P2P access must be enabled between GPUs

## Extending to Other Models

To add ZeroDP support to other MoE architectures:

1. Add `use_zerodp` flag to model config
2. Implement `offload_experts()` method with IPC handle transfer
3. Add double-buffered async copy in the model's forward pass
4. Ensure proper CUDA stream synchronization with events

## Citation

If you use this code, please cite the blog post:

```
@misc{zerodp2025,
  title={ZeroDP: Zero-Copy Data Parallel Expert Offloading},
  author={James Dborin},
  year={2025},
  url={https://mainlymatmul.com/blog/zerodp}
}
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

For major changes, please open an issue first to discuss.

## Support

- **Blog Post**: [mainlymatmul.com/blog/zerodp](https://mainlymatmul.com/blog/zerodp)
- **Issues**: [GitHub Issues](https://github.com/jamesdborin/zerodp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jamesdborin/zerodp/discussions)
