#!/usr/bin/env python3
"""
Profile the fused MXFP4 kernel using NVIDIA Nsight Compute.

Usage:
    # Profile with nsys (high-level timeline)
    nsys profile --trace=cuda,nvtx python dev/profile_kernel.py

    # Profile with ncu (detailed metrics)
    ncu --set full -o profile_report python dev/profile_kernel.py

    # View report
    ncu-ui profile_report.ncu-rep
"""

import torch
from mxfp4.quantizer import MXFP4Quantizer
from mxfp4.fused_kernels import quant_matmul

def profile_single_run():
    """Profile a single kernel execution."""
    # Configuration matching benchmark
    M, N, K = 128, 4096, 4096
    device = 'cuda'

    print(f"Profiling configuration: M={M}, N={N}, K={K}")

    # Setup
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K, device=device)
    packed, scales = quantizer.quantize(w)

    # Warmup
    for _ in range(10):
        _ = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()

    # Profile this section
    print("Starting profiled section...")
    torch.cuda.nvtx.range_push("quant_matmul_kernel")

    for _ in range(100):  # Multiple runs for better statistics
        output = quant_matmul(x, packed, scales)

    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print(f"Output shape: {output.shape}")
    print("Profiling complete!")

    return output

def profile_breakdown():
    """Profile different components separately."""
    M, N, K = 128, 4096, 4096
    device = 'cuda'

    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K, device=device)
    packed, scales = quantizer.quantize(w)

    # Warmup
    for _ in range(10):
        _ = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()

    # Profile fused kernel
    torch.cuda.nvtx.range_push("fused_matmul")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(100):
        output = quant_matmul(x, packed, scales)
    end.record()
    torch.cuda.synchronize()

    fused_time = start.elapsed_time(end) / 100
    torch.cuda.nvtx.range_pop()

    # Profile unfused (dequant + matmul)
    from mxfp4.kernels import dequantize_triton

    torch.cuda.nvtx.range_push("dequant_only")
    start.record()
    for _ in range(100):
        w_dequant = dequantize_triton(packed.view(-1), scales.view(-1), 32)
        w_dequant = w_dequant.view(N, K).to(torch.bfloat16)
    end.record()
    torch.cuda.synchronize()
    dequant_time = start.elapsed_time(end) / 100
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("matmul_only")
    w_dequant = dequantize_triton(packed.view(-1), scales.view(-1), 32)
    w_dequant = w_dequant.view(N, K).to(torch.bfloat16)

    start.record()
    for _ in range(100):
        output = torch.matmul(x, w_dequant.T)
    end.record()
    torch.cuda.synchronize()
    matmul_time = start.elapsed_time(end) / 100
    torch.cuda.nvtx.range_pop()

    print("\n=== Performance Breakdown ===")
    print(f"Fused kernel:       {fused_time:.3f} ms")
    print(f"Dequant only:       {dequant_time:.3f} ms")
    print(f"Matmul only (BF16): {matmul_time:.3f} ms")
    print(f"Unfused total:      {dequant_time + matmul_time:.3f} ms")
    print(f"\nFused overhead:     {fused_time - dequant_time:.3f} ms (custom matmul)")
    print(f"vs cuBLAS matmul:   {matmul_time:.3f} ms")
    print(f"Slowdown factor:    {(fused_time - dequant_time) / matmul_time:.2f}x")

if __name__ == "__main__":
    import sys

    if not torch.cuda.is_available():
        print("CUDA not available!")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print()

    # Run breakdown first
    profile_breakdown()
    print("\n" + "="*60 + "\n")

    # Then run single profiled execution
    profile_single_run()
