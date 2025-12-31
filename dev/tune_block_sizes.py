#!/usr/bin/env python3
"""
Block Size Tuning Script - Phase 3.7 (Refactored)

Tests different BLOCK_M, BLOCK_N, BLOCK_K configurations to find the best performance.
Uses subprocesses to ensure fresh compilation and loading of kernels.
"""

import sys
import time
from pathlib import Path
import subprocess
import re
import torch

# Block size configurations to test
# Constraints:
# 1. Warp coverage: (BLOCK_M/16) * (BLOCK_N/16) = 16 warps total
# 2. Shared memory must fit in <48KB
CONFIGS = [
    # Baseline
    (64, 64, 32, "Baseline 64x64"),
    # Wide configs (promising!)
    (128, 32, 16, "Wide 128x32 K=16"),
    (128, 32, 24, "Wide 128x32 K=24"),
    (128, 32, 32, "Wide 128x32 K=32"),
    (128, 32, 40, "Wide 128x32 K=40"),
    # Tall config for comparison
    (32, 128, 32, "Tall 32x128"),
]

SRC_DIR = Path(__file__).parent.parent / "src" / "mxfp4" / "cuda"
KERNEL_FILE = SRC_DIR / "mxfp4_kernel_advanced.cu"

# Minimal binding file content to avoid linking errors
MINIMAL_BINDING_CPP = """
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Forward declaration
void launch_mxfp4_matmul_advanced(
    const __nv_bfloat16* input,
    const uint8_t* packed_weight,
    const __nv_bfloat16* weight_scales,
    __nv_bfloat16* output,
    int M, int N, int K,
    cudaStream_t stream
);

torch::Tensor mxfp4_matmul_advanced(
    torch::Tensor input,
    torch::Tensor packed_weight,
    torch::Tensor weight_scales
) {
    // Minimal validation for benchmarking
    int M = input.size(0);
    int K = input.size(1);
    int N = packed_weight.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kBFloat16).device(input.device());
    torch::Tensor output = torch::empty({M, N}, options);
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    
    launch_mxfp4_matmul_advanced(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
        reinterpret_cast<const uint8_t*>(packed_weight.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(weight_scales.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        M, N, K,
        stream
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp4_matmul_advanced", &mxfp4_matmul_advanced, "MXFP4 advanced kernel");
}
"""

BENCHMARK_RUNNER_SCRIPT = """
import torch
from torch.utils.cpp_extension import load
import time
from pathlib import Path
import sys

# Load the kernel JIT
sources = [
    r"{kernel_path}",
    r"{binding_path}"
]

# print(f"Compiling {{sources}}")
try:
    mxfp4_cuda = load(
        name="mxfp4_cuda_tuned",
        sources=sources,
        extra_cuda_cflags=["-O3", "-use_fast_math", "--expt-relaxed-constexpr", "-std=c++17"],
        verbose=False
    )
except Exception as e:
    print(f"COMPILATION_FAILED: {{e}}")
    sys.exit(1)

def run_bench():
    device = 'cuda'
    M, N, K = 128, 4096, 4096  # Large batch

    # Create dummy data
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    packed = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
    scales = torch.randn(N, K // 32, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(10):
        _ = mxfp4_cuda.mxfp4_matmul_advanced(x, packed, scales)
    torch.cuda.synchronize()

    # Benchmark
    num_runs = 100
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_runs):
        _ = mxfp4_cuda.mxfp4_matmul_advanced(x, packed, scales)
    end.record()
    torch.cuda.synchronize()

    time_ms = start.elapsed_time(end) / num_runs
    print(f"RESULT_TIME: {{time_ms}}")

if __name__ == "__main__":
    run_bench()
"""

def modify_kernel_constants(kernel_file, block_m, block_n, block_k):
    """Modify the block size constants in a kernel file."""
    with open(kernel_file, 'r') as f:
        content = f.read()

    # Replace the constants
    content = re.sub(
        r'constexpr int BLOCK_M = \d+;',
        f'constexpr int BLOCK_M = {block_m};',
        content
    )
    content = re.sub(
        r'constexpr int BLOCK_N = \d+;',
        f'constexpr int BLOCK_N = {block_n};',
        content
    )
    content = re.sub(
        r'constexpr int BLOCK_K = \d+;',
        f'constexpr int BLOCK_K = {block_k};',
        content
    )

    with open(kernel_file, 'w') as f:
        f.write(content)

def run_benchmark_subprocess(kernel_path):
    """Run the benchmark in a separate process."""
    
    # Create temp binding file
    binding_path = Path("temp_binding.cpp")
    with open(binding_path, 'w') as f:
        f.write(MINIMAL_BINDING_CPP)
    
    script_content = BENCHMARK_RUNNER_SCRIPT.format(
        kernel_path=str(kernel_path.absolute()),
        binding_path=str(binding_path.absolute())
    )
    
    # Write temp script
    tmp_script = Path("temp_runner.py")
    with open(tmp_script, 'w') as f:
        f.write(script_content)
        
    try:
        # Run with timeout to catch hangs
        result = subprocess.run(
            [sys.executable, str(tmp_script)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60 # 60s timeout for compilation + run
        )
        
        if result.returncode != 0:
            print(f"Benchmark process failed:\n{result.stdout}\n{result.stderr}")
            return None
            
        # Parse output
        for line in result.stdout.splitlines():
            if "RESULT_TIME:" in line:
                return float(line.split(":")[1].strip())
        
        print(f"Could not find result in output:\n{result.stdout}")
        return None
    except subprocess.TimeoutExpired:
        print("Timeout expired!")
        return None
    finally:
        if tmp_script.exists():
            tmp_script.unlink()
        if binding_path.exists():
            binding_path.unlink()

def main():
    print("="*80)
    print("BLOCK SIZE TUNING - Phase 3.7 (Subprocess + Minimal Binding)")
    print("="*80)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Kernel: {KERNEL_FILE}")
    
    # Backup original kernel
    original_content = KERNEL_FILE.read_text()
    
    results = []
    
    try:
        for block_m, block_n, block_k, name in CONFIGS:
            print(f"\nTesting: {name} (M={block_m}, N={block_n}, K={block_k})")
            
            modify_kernel_constants(KERNEL_FILE, block_m, block_n, block_k)
            
            time_ms = run_benchmark_subprocess(KERNEL_FILE)
            
            if time_ms is not None:
                print(f"⏱️  Time: {time_ms:.3f} ms")
                results.append((block_m, block_n, block_k, name, time_ms))
            else:
                print("❌ Failed")

        # Summary
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"{ 'Config':<25} {'BLOCK_M':>8} {'BLOCK_N':>8} {'BLOCK_K':>8} {'Time (ms)':>12}")
        print("-"*80)

        results.sort(key=lambda x: x[4])

        for block_m, block_n, block_k, name, time_ms in results:
            print(f"{name:<25} {block_m:>8} {block_n:>8} {block_k:>8} {time_ms:>12.3f}")

        if results:
            best = results[0]
            print("\n" + "="*80)
            print(f"🏆 BEST CONFIGURATION: {best[3]}")
            print(f"   BLOCK_M={best[0]}, BLOCK_N={best[1]}, BLOCK_K={best[2]}")
            print(f"   Time: {best[4]:.3f} ms")
            
            # Restore best
            modify_kernel_constants(KERNEL_FILE, best[0], best[1], best[2])
            print(f"\n✅ Applied best configuration to {KERNEL_FILE.name}")
            
    except KeyboardInterrupt:
        print("\nInterrupted. Restoring original kernel...")
        KERNEL_FILE.write_text(original_content)
    except Exception as e:
        print(f"\nError: {e}")
        KERNEL_FILE.write_text(original_content)

if __name__ == "__main__":
    main()
