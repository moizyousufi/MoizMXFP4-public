#!/usr/bin/env python3
"""
Investigate the discrepancy between benchmark and profiling results.
"""

import torch
from mxfp4 import MXLinear
from mxfp4.quantizer import MXFP4Quantizer
from mxfp4.fused_kernels import quant_matmul
from mxfp4.kernels import dequantize_triton

def detailed_benchmark():
    """Match the exact benchmark configuration."""
    # Configuration from benchmark
    batch_size = 128
    in_features = 4096
    out_features = 4096
    device = 'cuda'
    num_runs = 100

    print(f"Configuration: batch={batch_size}, in={in_features}, out={out_features}")
    print("="*60)

    # Setup input
    x = torch.randn(batch_size, in_features, dtype=torch.bfloat16, device=device)

    # 1. BF16 Baseline
    linear_bf16 = torch.nn.Linear(in_features, out_features, bias=False).to(device).to(torch.bfloat16)

    for _ in range(10):
        _ = linear_bf16(x)
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    for _ in range(num_runs):
        _ = linear_bf16(x)
    end_ev.record()
    torch.cuda.synchronize()
    time_bf16 = start_ev.elapsed_time(end_ev) / num_runs

    print(f"BF16 (F.linear):        {time_bf16:.3f} ms")

    # 2. Unfused MXFP4 (using MXLinear)
    linear_mx = MXLinear(in_features, out_features, bias=False, block_size=32).to(device)
    linear_mx.quantize()

    for _ in range(10):
        _ = linear_mx(x)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(num_runs):
        _ = linear_mx(x)
    end_ev.record()
    torch.cuda.synchronize()
    time_unfused = start_ev.elapsed_time(end_ev) / num_runs

    print(f"Unfused (MXLinear):     {time_unfused:.3f} ms")

    # 3. Fused MXFP4
    packed = linear_mx.packed_weight
    scales = linear_mx.weight_scales

    for _ in range(10):
        _ = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(num_runs):
        _ = quant_matmul(x, packed, scales)
    end_ev.record()
    torch.cuda.synchronize()
    time_fused = start_ev.elapsed_time(end_ev) / num_runs

    print(f"Fused (quant_matmul):   {time_fused:.3f} ms")

    print("\n" + "="*60)
    print("Breakdown of Unfused Path:")
    print("="*60)

    # 4. Measure dequant separately
    packed_flat = packed.view(-1)
    scales_flat = scales.view(-1)

    for _ in range(10):
        _ = dequantize_triton(packed_flat, scales_flat, 32)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(num_runs):
        w_dequant = dequantize_triton(packed_flat, scales_flat, 32)
    end_ev.record()
    torch.cuda.synchronize()
    time_dequant = start_ev.elapsed_time(end_ev) / num_runs

    print(f"Dequant kernel only:    {time_dequant:.3f} ms")

    # 5. Measure matmul with pre-dequantized weights
    w_dequant = dequantize_triton(packed_flat, scales_flat, 32)
    w_dequant = w_dequant.view(out_features, in_features).to(torch.bfloat16)

    for _ in range(10):
        _ = torch.nn.functional.linear(x, w_dequant)
    torch.cuda.synchronize()

    start_ev.record()
    for _ in range(num_runs):
        _ = torch.nn.functional.linear(x, w_dequant)
    end_ev.record()
    torch.cuda.synchronize()
    time_matmul = start_ev.elapsed_time(end_ev) / num_runs

    print(f"F.linear (dequant BF16): {time_matmul:.3f} ms")
    print(f"Dequant + Matmul total:  {time_dequant + time_matmul:.3f} ms")

    print("\n" + "="*60)
    print("Analysis:")
    print("="*60)
    print(f"Unfused overhead:        {time_unfused - (time_dequant + time_matmul):.3f} ms")
    print(f"Fused vs Unfused:        {time_fused / time_unfused:.2f}x")
    print(f"Fused vs BF16:           {time_fused / time_bf16:.2f}x")
    print(f"Unfused vs BF16:         {time_unfused / time_bf16:.2f}x")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA not available!")
        exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}\n")

    detailed_benchmark()
