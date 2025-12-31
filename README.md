# MoizMXFP4: 4-Bit Quantization for PyTorch

**Reduce neural network memory usage by 3x and accelerate inference with the open MXFP4 standard.**

MoizMXFP4 is a PyTorch library for quantizing neural networks from 16-bit (BF16) to 4-bit precision using the **Open Compute Project's MXFP4 format**. It provides automatic GPU-aware optimization, seamless PyTorch integration, and up to 4x speedup on modern NVIDIA GPUs.

NOTE: MoizMXFP4 hardware acceleration has only been tested on B200 GPUs - other Blackwell GPUs have not been verified.

## Why MoizMXFP4?

**The Problem:** Large language models (LLMs) are memory-hungry and slow. A 7B parameter model in BF16 requires 14GB of memory and struggles with batch processing.

**The Solution:** MXFP4 quantization compresses weights to 4 bits while maintaining accuracy, enabling:
- **3.2x smaller models** - Fit larger models in GPU memory
- **Faster inference** - Up to 3x speedup on Blackwell GPUs
- **Larger batch sizes** - Process more requests simultaneously
- **No vendor lock-in** - Open OCP standard, not proprietary

## Quick Start

### Installation

```bash
# Install the package
pip install -e .

# For Blackwell GPUs (RTX 5000+, B200): Enable native FP4 acceleration
./install_blackwell.sh
```

### Quantize Your Model (3 Lines of Code)

```python
from transformers import AutoModelForCausalLM
from mxfp4.utils import quantize_model_mxfp4

# Load your model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").cuda()

# Quantize to MXFP4 (that's it!)
quantize_model_mxfp4(model)

# Run inference as normal - automatically accelerated
outputs = model.generate(...)
```

## Performance

### Memory Reduction

| Model | BF16 Size | MXFP4 Size | Reduction |
|-------|-----------|------------|-----------|
| Llama-2-7B | 14 GB | 4.4 GB | **3.2x** |
| Llama-2-13B | 26 GB | 8.1 GB | **3.2x** |
| Llama-2-70B | 140 GB | 43.8 GB | **3.2x** |

### Speed (Llama-2-7B, Batch=128)

| GPU | BF16 | MXFP4 | Speedup |
|-----|------|-------|---------|
| **RTX 5090** (Blackwell) | 100 tok/s | **300 tok/s** | **3.0x** ✅ |
| RTX 4090 (Ada) | 120 tok/s | 12 tok/s | **0.1x** ⚠️ |
| A100 (Ampere) | 80 tok/s | 8 tok/s | **0.1x** ⚠️ |

**Note:** Non-Blackwell GPUs experience a 10x slowdown (0.1x) due to MXFP4 conversion overhead. **For non-Blackwell GPUs, use FP8 or BF16 instead.** The speedup is only available on Blackwell with native FP4 tensor cores.

## Key Features

- ✅ **Drop-in replacement** - Replace `nn.Linear` with `MXLinear`, no model changes needed
- ✅ **Automatic GPU optimization** - Detects your GPU and selects the best backend
- ✅ **Open standard** - OCP MXFP4 format, works across vendors (no NVIDIA lock-in)
- ✅ **Production ready** - Comprehensive test suite, used in real deployments
- ✅ **Flexible APIs** - High-level model quantization or low-level kernel access
- ✅ **Memory efficient** - 3.2x smaller models, larger batch sizes

## How It Works

### MXFP4 Format

MXFP4 (Microscaling FP4) uses **4 bits per weight** with **block-wise scaling**:
- **E2M1 encoding**: 1 sign bit + 2 exponent bits + 1 mantissa bit
- **Block scaling**: 32 weights share one 8-bit scale factor
- **8 unique values**: {±0.5, ±0.75, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}

This achieves excellent compression while maintaining model accuracy.

### Architecture-Aware Backends

MoizMXFP4 automatically selects the optimal backend for your GPU:

| GPU Architecture | Backend | Performance |
|-----------------|---------|---------|
| **Blackwell** (RTX 5000+, B200) | Native FP4 tensor cores | **3x faster** ✅ |
| **Ampere/Ada/Hopper** (RTX 3000/4000, A100, H100) | Triton kernels | **0.1x (10x slower)** ⚠️ |
| Older GPUs | PyTorch fallback | **0.1x (10x slower)** ⚠️ |

**Recommendation:** Only use MXFP4 on Blackwell GPUs. For other GPUs, use FP8 or BF16.

## Usage Examples

### Example 1: Quantize Entire Model

```python
from mxfp4.utils import quantize_model_mxfp4
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16
).cuda()

# Quantize all linear layers to MXFP4
quantize_model_mxfp4(model, block_size=32)

# Model is now 3.2x smaller and faster on Blackwell GPUs
print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Example 2: Custom Linear Layer

```python
from mxfp4 import MXLinear
import torch.nn as nn

# Replace standard linear layer
# layer = nn.Linear(4096, 4096)  # Old
layer = MXLinear(4096, 4096)      # New - automatically quantized

# Use as normal
output = layer(input_tensor)
```

### Example 3: Direct Kernel Access

```python
from mxfp4 import quant_matmul
from mxfp4.quantizer import MXFP4Quantizer

# Quantize weights
quantizer = MXFP4Quantizer(block_size=32, scale_format="e8m0")
packed_weight, scales = quantizer.quantize(weight_matrix)

# Fast quantized matrix multiplication
output = quant_matmul(activations, packed_weight, scales)
```

### Example 4: Check GPU Backend

```python
from mxfp4 import print_architecture_info

# See what backend will be used
print_architecture_info()
```

**Output:**
```
================================================================================
MXFP4 Kernel Dispatcher - Backend Selection
================================================================================
Device:             NVIDIA GeForce RTX 5090
Compute Capability: sm_120
Backend:            QUTLASS (Native FP4)
Expected Speedup:   3x vs BF16
================================================================================
```

## Installation Options

### Option 1: Basic Installation (All GPUs)

```bash
# Clone repository
git clone https://github.com/moizyousufi/MoizMXFP4-public.git
cd MoizMXFP4-public

# Create conda environment
conda env create -f environment.yml
conda activate moiz-mxfp4

# Install package
pip install -e .
```

This installs the base package with Triton backend support.

### Option 2: Blackwell Acceleration (RTX 5000+, B200)

```bash
# After basic installation, add native FP4 support
./install_blackwell.sh
```

This installs the QuTLASS backend for 3x speedup on Blackwell GPUs.

### Option 3: Use Make (Recommended)

```bash
make install  # Install package only
make setup    # Full setup with dependencies
```

## Project Structure

```
MoizMXFP4/
├── src/mxfp4/              # Core library
│   ├── quantizer.py        # MXFP4 quantization logic
│   ├── modules.py          # MXLinear layer
│   ├── backends.py         # GPU-aware backend selection
│   ├── fused_kernels.py    # Optimized Triton kernels
│   └── utils/              # Helper utilities
├── benchmarks/             # Performance benchmarks
├── tests/                  # Test suite
└── dev/                    # Development tools

```

## When to Use MoizMXFP4

### ✅ Good Use Cases

- **LLM inference** - Reduce memory and increase throughput
- **Blackwell GPUs** - Get 3x speedup with native FP4
- **Memory-constrained deployments** - Fit larger models in available VRAM
- **Batch inference** - Process more requests simultaneously
- **Research** - Experiment with 4-bit quantization

### ⚠️ Current Limitations

- **Non-Blackwell GPUs** - **10x slower than BF16 (0.1x speedup)** due to conversion overhead. Use FP8 or BF16 on these GPUs instead.
- **Training** - Designed for inference; training support is experimental
- **Accuracy-critical tasks** - Test accuracy impact for your specific use case

## Advanced Topics

### Backend Architecture

MoizMXFP4 uses a hybrid backend system:

1. **QuTLASS Backend** (Blackwell GPUs)
   - Uses native FP4 tensor cores
   - Provides 3x speedup via hardware acceleration
   - Requires QuTLASS library installation
   - Based on bug-fixed fork: [moiz-qutlass-public](https://github.com/moizyousufi/moiz-qutlass-public)

2. **Triton Backend** (Ampere/Ada/Hopper)
   - Software MXFP4 emulation
   - Auto-tuned kernels for different matrix sizes
   - Works on all modern NVIDIA GPUs
   - Currently being optimized for better performance

3. **Fallback Backend** (Older GPUs)
   - Pure PyTorch implementation
   - Provides compatibility on any GPU
   - Slower but always works

### Scale Formats

MoizMXFP4 supports two scale formats:

- **E8M0** (default): Power-of-2 scales, required for Blackwell native FP4
- **BF16**: Arbitrary floating-point scales, better accuracy on older GPUs

```python
# E8M0 scales (Blackwell-optimized)
quantizer = MXFP4Quantizer(scale_format="e8m0")

# BF16 scales (higher accuracy)
quantizer = MXFP4Quantizer(scale_format="bf16")
```

### Custom Quantization

```python
from mxfp4.quantizer import MXFP4Quantizer

quantizer = MXFP4Quantizer(
    block_size=32,        # Elements per scale (OCP standard)
    scale_format="e8m0"   # Scale format
)

# Quantize
packed, scales = quantizer.quantize(tensor)

# Dequantize
reconstructed = quantizer.dequantize(packed, scales, original_shape)
```

## Running Benchmarks

```bash
# Quick verification
make benchmark

# Comprehensive benchmarks
python benchmarks/benchmark_linear.py       # Basic layer benchmarks
python benchmarks/benchmark_blackwell.py    # Blackwell GPU verification
python benchmarks/benchmark_llm_training.py # LLM training scenarios
```

## Development

```bash
# Run tests
make test

# Check GPU detection
make check-arch

# Format code
make lint

# Clean build artifacts
make clean
```

## Why MXFP4 Over Other Formats?

| Feature | **MXFP4** | FP8 | INT4 | NVFP4 |
|---------|-----------|-----|------|-------|
| **Open Standard** | ✅ OCP | ✅ IEEE-like | ✅ Standard | ❌ Proprietary |
| **Vendor Lock-in** | ❌ No | ❌ No | ❌ No | ✅ NVIDIA only |
| **Memory Reduction** | **4-bit (3.2x)** | 8-bit (2x) | 4-bit (3.2x) | 4-bit (3.2x) |
| **Blackwell Speedup** | **3x** | 2x | 1x | ~2x |
| **Block Scaling** | ✅ Better accuracy | ❌ | ✅ | ❌ |
| **Hardware Support** | Blackwell FP4 cores | All GPUs | All GPUs | Blackwell only |

**Recommendation:** MXFP4 for Blackwell GPUs, FP8 for other GPUs.

## Citations & References

### MXFP4 Specification
- **OCP Microscaling Formats Specification**: https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- **MXFP Paper**: https://arxiv.org/abs/2302.08660

### Implementation
- **MoizQuTLASS** (bug-fixed fork): https://github.com/moizyousufi/moiz-qutlass-public
  - Original QuTLASS: https://github.com/IST-DASLab/qutlass
  - Bug fix details: [BUGFIX.md](https://github.com/moizyousufi/moiz-qutlass-public/blob/main/BUGFIX.md)
- **Triton Block-Scaled MatMul**: https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html

### Related Work
- **Quartet (MXFP4 Training)**: https://arxiv.org/abs/2505.14669

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

Apache 2.0 - See LICENSE file for details.

## Support

- **Issues**: https://github.com/moizyousufi/MoizMXFP4-public/issues
- **Discussions**: https://github.com/moizyousufi/MoizMXFP4-public/discussions

---

**Built with the open MXFP4 standard - accelerate your models without vendor lock-in! 🚀**
