# vLLM CPU Offload MoE Plugin

This plugin provides CPU offloading for MoE (Mixture of Experts) layers in vLLM, specifically designed for Qwen3 MoE models. It uses mapped memory to store expert weights on CPU while maintaining GPU-accessible pointers for efficient data transfer during inference.

## Features

- **CPU Offloading**: Store expert weights in CPU mapped memory to reduce GPU VRAM usage
- **CUDA Graph Compatible**: Always copies ALL experts to GPU before computation
- **Efficient Transfer**: Uses Triton kernels for fast CPU→GPU data transfer
- **Zero-Copy Access**: Leverages `cudaHostAllocMapped` for GPU-accessible host memory

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  OffloadFusedMoE Layer                      │
├─────────────────────────────────────────────────────────────┤
│  CPU Mapped Memory           │  GPU Staging Buffers        │
│  ├─ w13_weight_cpu           │  ├─ w13_weight_gpu          │
│  └─ w2_weight_cpu            │  └─ w2_weight_gpu           │
├─────────────────────────────────────────────────────────────┤
│                    Forward Pass                             │
│  1. Copy ALL expert weights from CPU to GPU                │
│  2. Router selects top-k experts                           │
│  3. Run fused_experts kernel with GPU weights              │
│  4. Return results                                         │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

1. vLLM installed
2. CUDA toolkit with mapped memory support
3. The `mapped_tensor` module (from pagingllm project)

### Build mapped_tensor module

```bash
cd /home/titan-6/pagingllm
python setup.py build_ext --inplace
```

### Install the plugin

```bash
cd vllm_cpu_offload_moe
pip install -e .
```

## Usage

### Enable the plugin

```bash
export VLLM_PLUGINS="cpu_offload_moe"
```

### Using with vLLM

The plugin registers a new model architecture `Qwen3MoeForCausalLMOffload` that you can use:

```python
from vllm import LLM, SamplingParams

# The plugin automatically handles the offloading
llm = LLM(
    model="Qwen/Qwen3-MoE-xxx",  # Your MoE model
    trust_remote_code=True,
)

# Generate as usual
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=100))
```

### Testing the components

```bash
python example_usage.py --test
```

## Pingpong Buffer Support

The plugin provides two ways to enable pingpong buffering:

### Method 1: Quantization Config (Recommended)

Use the `pingpong_moe` quantization method to automatically enable pingpong buffering for all MoE layers:

```python
from vllm import LLM

# Use pingpong_moe quantization
llm = LLM(
    model="your-moe-model",
    quantization="pingpong_moe",  # Automatically uses PingpongUnquantizedFusedMoEMethod
)

# The plugin automatically registers this quantization method
# All MoE layers will use pingpong buffering when w13_buffer is set
```

### Method 2: Manual Method Replacement

For more control, manually replace the quant method on individual layers:

```python
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm_cpu_offload_moe.pingpong_fused_moe_method import PingpongUnquantizedFusedMoEMethod

# Get a FusedMoE layer
moe_layer = get_moe_layer()

# Replace with pingpong method
moe_layer.quant_method = PingpongUnquantizedFusedMoEMethod(moe_layer.moe_config)

# Set up pingpong buffer
buffer_shape = moe_layer.w13_weight.shape
moe_layer.w13_buffer = torch.empty(buffer_shape, dtype=moe_layer.w13_weight.dtype, device='cuda')

# Copy weights to buffer asynchronously
moe_layer.w13_buffer.copy_(moe_layer.w13_weight, non_blocking=True)

# The method automatically uses w13_buffer instead of w13_weight
result = moe_layer(x, router_logits)
```

### Benefits

- **Overlapping Transfers**: Hide weight transfer latency behind computation
- **Double Buffering**: Alternate between two buffers for continuous processing
- **Any MoE Layer**: Works with standard vLLM MoE layers, not just offloaded ones
- **Backward Compatible**: Falls back to standard weights when no buffer is set
- **Automatic Integration**: Quantization config method integrates seamlessly with vLLM server

## Files

- `setup.py` - Package installation configuration
- `vllm_cpu_offload_moe/`
  - `__init__.py` - Plugin registration
  - `offload_fused_moe.py` - OffloadFusedMoE layer implementation
  - `qwen3_moe_offload.py` - Qwen3 MoE model with offloading
  - `triton_kernels.py` - Triton kernels for CPU→GPU transfer
  - `pingpong_fused_moe_method.py` - Pingpong buffer method for any MoE layer
- `example_usage.py` - Usage examples and tests

## How It Works

### 1. Weight Loading

After loading weights normally, the plugin moves expert weights to CPU mapped memory:

```python
# Original GPU tensor
w13_gpu = self.w13_weight.data  # [num_experts, 2*intermediate, hidden]

# Create CPU mapped tensor
w13_cpu = create_cpu_mapped_tensor(w13_gpu.shape, w13_gpu.dtype)
w13_cpu.copy_(w13_gpu.cpu())

# Free GPU memory (replace with placeholder)
self.w13_weight.data = torch.empty(0, dtype=w13_gpu.dtype, device="cuda")
```

### 2. Forward Pass

During inference, weights are copied from CPU to GPU before computation:

```python
def forward_impl(self, hidden_states, router_logits):
    # Copy ALL weights from CPU to GPU (CUDA graph compatible)
    copy_all_experts_to_gpu(self.w13_weight_cpu, self.w13_weight_gpu)
    copy_all_experts_to_gpu(self.w2_weight_cpu, self.w2_weight_gpu)
    
    # Run normal MoE computation with GPU weights
    result = fused_experts(
        hidden_states=hidden_states,
        w1=self.w13_weight_gpu,
        w2=self.w2_weight_gpu,
        ...
    )
    return result
```

### 3. Triton Copy Kernel

The copy uses a simple but efficient Triton kernel:

```python
@triton.jit
def _copy_all_experts_kernel(src_ptr, dst_ptr, total_numel, BLOCK_SIZE):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_numel
    vals = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, vals, mask=mask)
```

## Performance Considerations

- **Memory Savings**: Expert weights stored on CPU, only staging buffers on GPU
- **Overhead**: CPU→GPU copy happens every forward pass
- **Throughput**: ~20-30 GB/s typical for PCIe 4.0 x16

## Limitations

- Currently only supports Qwen3 MoE models
- Requires CUDA device with mapped memory support
- Copy overhead increases inference latency
- Not optimized for very small batch sizes

## Extending to Other Models

To add support for other MoE models:

1. Create a new model file (e.g., `deepseek_moe_offload.py`)
2. Replace `FusedMoE` with `OffloadFusedMoE` in the MoE block
3. Register the model in `__init__.py`

## License

Apache-2.0
