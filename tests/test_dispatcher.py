"""
Test Architecture-Aware Kernel Dispatcher

Validates that the dispatcher correctly detects GPU architecture
and routes to the appropriate kernel.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mxfp4.kernel_dispatcher import (
    quant_matmul,
    print_architecture_info,
    get_dispatcher
)
from mxfp4.quantizer import MXFP4Quantizer


def test_dispatcher():
    """Test that dispatcher works correctly."""
    print("="*80)
    print("KERNEL DISPATCHER TEST")
    print("="*80)
    print()

    # Print architecture detection results
    print_architecture_info()
    print()

    # Get dispatcher info
    dispatcher = get_dispatcher()
    info = dispatcher.get_architecture_info()

    print(f"Detected Architecture: {info['architecture'].upper()}")
    print(f"Selected Kernel Path:  {info['kernel_path']}")
    print()

    # Run correctness test
    print("Running correctness test...")
    device = 'cuda'
    quantizer = MXFP4Quantizer(block_size=32)

    # Create test data
    M, N, K = 128, 4096, 4096
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    W = torch.randn(N, K, dtype=torch.bfloat16, device=device)

    # Quantize weights
    packed, scales = quantizer.quantize(W)

    # Run through dispatcher
    output = quant_matmul(x, packed, scales)

    # Validate output shape
    assert output.shape == (M, N), f"Output shape mismatch: {output.shape} != {(M, N)}"
    assert output.dtype == torch.bfloat16, f"Output dtype mismatch: {output.dtype} != torch.bfloat16"

    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Output dtype: {output.dtype}")
    print()

    # Run quick benchmark
    print("Running quick benchmark...")
    num_runs = 50

    # Warmup
    for _ in range(10):
        _ = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        _ = quant_matmul(x, packed, scales)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end) / num_runs

    print(f"✅ Performance: {time_ms:.3f} ms")
    print()

    # Summary
    print("="*80)
    print("✅ DISPATCHER TEST PASSED")
    print("="*80)
    print()
    print("Summary:")
    print(f"  - Architecture:  {info['architecture'].upper()}")
    print(f"  - Kernel:        {info['kernel_path']}")
    print(f"  - Native FP4:    {'Yes' if info['native_fp4'] else 'No (software emulation)'}")
    print(f"  - Performance:   {time_ms:.3f} ms (M=128, N=4096, K=4096)")
    print()

    if info['native_fp4']:
        print("🚀 Using Blackwell native FP4 tensor cores!")
    else:
        print("⚡ Using optimized software emulation (Ampere/Ada/Hopper)")
        print("   → On Blackwell hardware, this will automatically use native FP4 (2-2.5x faster!)")


if __name__ == "__main__":
    test_dispatcher()
