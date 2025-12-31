#!/usr/bin/env python3
"""
Profile using PyTorch's built-in profiler (works on all platforms).
"""

import torch
from torch.profiler import profile, ProfilerActivity
from mxfp4.quantizer import MXFP4Quantizer
from mxfp4.fused_kernels import quant_matmul

def profile_with_pytorch():
    """Profile using PyTorch profiler."""
    device = 'cuda'
    M, N, K = 128, 4096, 4096

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

    print("\nStarting profiling...")

    # Profile with PyTorch profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(10):
            output = quant_matmul(x, packed, scales)
        torch.cuda.synchronize()

    # Print summary
    print("\n" + "="*80)
    print("CUDA Kernel Summary (sorted by CUDA time)")
    print("="*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Export to Chrome trace (can view in chrome://tracing)
    prof.export_chrome_trace("pytorch_trace.json")
    print("\nTrace exported to: pytorch_trace.json")
    print("View in Chrome: chrome://tracing")

    # Detailed kernel stats
    print("\n" + "="*80)
    print("Kernel Statistics")
    print("="*80)

    events = prof.key_averages()
    for evt in events:
        if evt.device_type == torch.profiler.DeviceType.CUDA:
            print(f"\nKernel: {evt.key}")
            # Use self_cuda_time_total (correct attribute name)
            cuda_time_us = evt.self_cuda_time_total if hasattr(evt, 'self_cuda_time_total') else 0
            print(f"  CUDA time: {cuda_time_us / 1000:.3f} ms")
            print(f"  Calls: {evt.count}")
            if evt.count > 0:
                print(f"  Avg time: {cuda_time_us / evt.count / 1000:.3f} ms")
            if hasattr(evt, 'cuda_memory_usage') and evt.cuda_memory_usage != 0:
                print(f"  Memory: {evt.cuda_memory_usage / 1024 / 1024:.2f} MB")

    # Compare with cuBLAS
    print("\n" + "="*80)
    print("Comparison: cuBLAS Matmul")
    print("="*80)

    w_dequant = quantizer.dequantize(packed, scales).view(N, K).to(torch.bfloat16)

    with profile(activities=[ProfilerActivity.CUDA]) as prof_cublas:
        for _ in range(10):
            _ = torch.matmul(x, w_dequant.T)
        torch.cuda.synchronize()

    print(prof_cublas.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    print("\nProfiling complete!")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print()

    profile_with_pytorch()
