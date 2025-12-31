#!/usr/bin/env python3
"""
Minimal profiling target for Nsight Compute.

Usage:
    ncu --set full -o profile_report python dev/profile_for_ncu.py

    # Or for specific metrics:
    ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
        --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
        --metrics l1tex__throughput.avg.pct_of_peak_sustained_elapsed \
        -o profile_report python dev/profile_for_ncu.py
"""

import torch
from mxfp4.quantizer import MXFP4Quantizer
from mxfp4.fused_kernels import quant_matmul

def profile_single_config():
    """Profile a single representative configuration."""
    device = 'cuda'

    # Configuration: Large batch (most representative)
    M, N, K = 128, 4096, 4096

    print(f"Profiling configuration: M={M}, N={N}, K={K}")
    print(f"This represents: batch_size={M}, hidden_dim={N}")

    # Setup
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K, device=device)
    packed, scales = quantizer.quantize(w)

    # Warmup (ensure kernel is compiled)
    print("Warming up...")
    for _ in range(20):
        _ = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()

    print("Starting profiled execution...")
    print("(Nsight Compute will capture this kernel)")

    # Single execution for profiling
    # NCU will automatically capture the kernel
    output = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()

    print(f"Output shape: {output.shape}")
    print("Profiling target complete!")

    return output

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print("="*60)

    profile_single_config()
