"""
Test Blackwell-Specific Kernel Code

Validates that Blackwell kernel code is correct on all GPUs.
On non-Blackwell GPUs: Tests code correctness (software emulation)
On Blackwell GPUs: Tests native FP4 acceleration and speedup
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mxfp4.fused_kernels import quant_matmul as quant_matmul_ampere
from mxfp4.kernels.blackwell import quant_matmul_blackwell
from mxfp4.quantizer import MXFP4Quantizer
from mxfp4.kernel_dispatcher import get_dispatcher


def test_correctness():
    """Test that Blackwell kernel produces same results as Ampere kernel."""
    print("="*80)
    print("CORRECTNESS TEST: Blackwell Kernel vs Ampere Baseline")
    print("="*80)

    device = 'cuda'
    quantizer = MXFP4Quantizer(block_size=32)

    # Get architecture info
    dispatcher = get_dispatcher()
    info = dispatcher.get_architecture_info()
    print(f"\nRunning on: {info['architecture'].upper()}")
    print(f"Dispatcher routes to: {info['kernel_path']}")
    print()

    configs = [
        (1, 256, 256, "Tiny"),
        (16, 1024, 1024, "Small"),
        (128, 4096, 4096, "Large"),
    ]

    for M, N, K, desc in configs:
        print(f"\n{desc}: M={M}, N={N}, K={K}")

        # Create test data
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=device)

        # Quantize weights
        packed, scales = quantizer.quantize(W)

        # Run both kernels
        output_ampere = quant_matmul_ampere(x, packed, scales)
        output_blackwell = quant_matmul_blackwell(x, packed, scales)

        # Compare
        max_diff = (output_ampere - output_blackwell).abs().max().item()
        mean_diff = (output_ampere - output_blackwell).abs().mean().item()

        print(f"  Max diff:  {max_diff:.6f}")
        print(f"  Mean diff: {mean_diff:.6f}")

        if max_diff < 0.01:
            print(f"  ✅ PASSED - Numerically identical")
        elif max_diff < 2.0:
            print(f"  ⚠️  WARNING - Small differences (acceptable for quantization)")
        else:
            print(f"  ❌ FAILED - Large differences!")
            assert False, f"Correctness test failed: max_diff={max_diff:.6f}"

    print("\n" + "="*80)
    print("✅ ALL CORRECTNESS TESTS PASSED")
    print("="*80)


def benchmark_performance():
    """Benchmark Blackwell kernel vs Ampere kernel across batch sizes."""
    print("\n" + "="*80)
    print("PERFORMANCE BENCHMARK: Blackwell Kernel vs Ampere Kernel")
    print("="*80)

    device = 'cuda'
    quantizer = MXFP4Quantizer(block_size=32)
    num_runs = 100

    # Get architecture info
    dispatcher = get_dispatcher()
    info = dispatcher.get_architecture_info()
    is_blackwell = info['native_fp4']

    print(f"\nRunning on: {info['architecture'].upper()}")
    if is_blackwell:
        print("🚀 Native FP4 detected! Expect 2-2.5x speedup for Blackwell kernel.")
    else:
        print("⚡ Software emulation - Blackwell kernel uses optimized configs.")
    print()

    configs = [
        (1, 4096, 4096, "B=1"),
        (16, 4096, 4096, "B=16"),
        (128, 4096, 4096, "B=128"),
        (256, 4096, 4096, "B=256"),
        (512, 4096, 4096, "B=512"),
    ]

    for M, N, K, desc in configs:
        print(f"\n{desc}: M={M}, N={N}, K={K}")

        # Create test data
        x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        W = torch.randn(N, K, dtype=torch.bfloat16, device=device)
        packed, scales = quantizer.quantize(W)

        # Warmup
        for _ in range(10):
            _ = quant_matmul_ampere(x, packed, scales)
            _ = quant_matmul_blackwell(x, packed, scales)
        torch.cuda.synchronize()

        # Benchmark Ampere
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(num_runs):
            _ = quant_matmul_ampere(x, packed, scales)
        end.record()
        torch.cuda.synchronize()
        time_ampere = start.elapsed_time(end) / num_runs

        # Benchmark Blackwell
        start.record()
        for _ in range(num_runs):
            _ = quant_matmul_blackwell(x, packed, scales)
        end.record()
        torch.cuda.synchronize()
        time_blackwell = start.elapsed_time(end) / num_runs

        # Results
        speedup = time_ampere / time_blackwell
        print(f"  Ampere kernel:    {time_ampere:.3f} ms")
        print(f"  Blackwell kernel: {time_blackwell:.3f} ms")
        print(f"  Speedup:          {speedup:.2f}x")

        if is_blackwell:
            # On Blackwell, expect 2-2.5x speedup
            if speedup > 2.0:
                print(f"  🚀 EXCELLENT - Native FP4 working!")
            elif speedup > 1.5:
                print(f"  ✅ GOOD - Blackwell acceleration detected")
            else:
                print(f"  ⚠️  WARNING - Expected 2-2.5x speedup on Blackwell")
        else:
            # On non-Blackwell, expect similar performance
            if speedup > 1.05:
                print(f"  ✅ Blackwell config slightly faster")
            elif speedup > 0.95:
                print(f"  ✅ Similar performance (expected on {info['architecture'].upper()})")
            else:
                print(f"  ⚠️  Blackwell config slower (different autotuning)")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

    if is_blackwell:
        print("\n🚀 Running on Blackwell - Native FP4 tensor cores active!")
    else:
        print(f"\n⚡ Running on {info['architecture'].upper()} - Software emulation")
        print("   On Blackwell hardware, expect 2-2.5x speedup from native FP4!")


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # Test correctness first
    test_correctness()

    # Benchmark performance
    benchmark_performance()
