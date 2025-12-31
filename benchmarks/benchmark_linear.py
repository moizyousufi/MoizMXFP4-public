import torch
import torch.nn as nn
import time
import pandas as pd
from mxfp4 import MXLinear
from mxfp4.fused_kernels import quant_matmul

def benchmark_layer(batch_size, in_features, out_features, device='cuda', num_runs=100):
    # Setup inputs
    x = torch.randn(batch_size, in_features, device=device, dtype=torch.bfloat16)
    
    # 1. Baseline: BF16 Linear
    linear_bf16 = nn.Linear(in_features, out_features, bias=False).to(device).to(torch.bfloat16)
    
    # Warmup
    for _ in range(10):
        _ = linear_bf16(x)
    torch.cuda.synchronize()
    
    # Measure Latency
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    
    start_ev.record()
    for _ in range(num_runs):
        _ = linear_bf16(x)
    end_ev.record()
    torch.cuda.synchronize()
    time_bf16 = start_ev.elapsed_time(end_ev) / num_runs
    
    # Measure Memory (Weight only)
    mem_bf16 = linear_bf16.weight.element_size() * linear_bf16.weight.nelement() / (1024**2)
    
    del linear_bf16
    torch.cuda.empty_cache()
    
    # 2. MXFP4 Linear (Unfused: Dequant -> Matmul)
    linear_mx = MXLinear(in_features, out_features, bias=False, block_size=32).to(device)
    linear_mx.quantize() 
    
    # Warmup
    for _ in range(10):
        _ = linear_mx(x)
    torch.cuda.synchronize()
    
    start_ev.record()
    for _ in range(num_runs):
        _ = linear_mx(x)
    end_ev.record()
    torch.cuda.synchronize()
    time_mx = start_ev.elapsed_time(end_ev) / num_runs
    
    # Capture packed weights for fused benchmark
    packed = linear_mx.packed_weight
    scales = linear_mx.weight_scales
    
    # 3. MXFP4 Fused (Kernel)
    # Warmup
    for _ in range(10):
        _ = quant_matmul(x, packed, scales)
    torch.cuda.synchronize()
    
    start_ev.record()
    for _ in range(num_runs):
        _ = quant_matmul(x, packed, scales)
    end_ev.record()
    torch.cuda.synchronize()
    time_fused = start_ev.elapsed_time(end_ev) / num_runs
    
    # Measure Memory
    mem_mx = linear_mx.packed_weight.element_size() * linear_mx.packed_weight.nelement()
    mem_mx += linear_mx.weight_scales.element_size() * linear_mx.weight_scales.nelement()
    mem_mx /= (1024**2)
    
    del linear_mx
    torch.cuda.empty_cache()
    
    return {
        "Batch": batch_size,
        "Hidden": in_features,
        "BF16 (ms)": time_bf16,
        "Unfused (ms)": time_mx,
        "Fused (ms)": time_fused,
        "Speedup (Unfused)": time_bf16 / time_mx,
        "Speedup (Fused)": time_bf16 / time_fused,
        "Compression": mem_bf16 / mem_mx
    }

def run_benchmarks():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark.")
        return

    print(f"Benchmarking on {torch.cuda.get_device_name(0)}...")
    
    configs = [
        (1, 4096, 4096),    # Inference (Llama-2-7B size)
        (1, 11008, 4096),   # Inference MLP Up
        (16, 4096, 4096),   # Small Batch
        (128, 4096, 4096),  # Training Batch
        (128, 11008, 4096)  # Training MLP
    ]
    
    results = []
    for bs, inf, outf in configs:
        try:
            res = benchmark_layer(bs, inf, outf)
            results.append(res)
            print(f"B={bs}, H={inf}->{outf}: "
                  f"BF16={res['BF16 (ms)']:.3f}ms, "
                  f"Fused={res['Fused (ms)']:.3f}ms "
                  f"(Speedup: {res['Speedup (Fused)']:.2f}x)")
        except Exception as e:
            print(f"Failed B={bs}: {e}")
            
    df = pd.DataFrame(results)
    print("\nSummary:")
    print(df.to_markdown(index=False, floatfmt=".3f"))
    
if __name__ == "__main__":
    run_benchmarks()