"""
Kernel Tuning Script for MXFP4 Matmul

Systematically tests different block configurations to beat Triton.
"""

import subprocess
import re
import tempfile
import shutil
from pathlib import Path

# Base kernel path
BASE_KERNEL = Path(__file__).parent.parent / "src/mxfp4/cuda/mxfp4_kernel_optimized_dequant.cu"
BACKUP_KERNEL = BASE_KERNEL.with_suffix(".cu.backup")

# Test configurations
# Format: (BLOCK_M, BLOCK_N, BLOCK_K, PAD, threads_per_block)
CONFIGS = [
    # Current baseline
    (128, 32, 32, 8, 512),

    # Square tiles (better cache locality)
    (64, 64, 32, 8, 512),
    (64, 64, 32, 0, 512),

    # Wider tiles (more N reuse)
    (128, 64, 32, 8, 512),
    (128, 64, 32, 0, 512),
    (64, 128, 32, 8, 512),

    # Taller tiles (more M reuse)
    (256, 32, 32, 8, 512),
    (256, 32, 32, 0, 512),

    # Different thread counts
    (128, 32, 32, 8, 256),
    (128, 64, 32, 8, 256),
    (64, 64, 32, 8, 256),

    # No padding variants
    (128, 32, 32, 0, 512),
    (256, 64, 32, 0, 512),
]


def backup_kernel():
    """Backup original kernel."""
    if not BACKUP_KERNEL.exists():
        shutil.copy(BASE_KERNEL, BACKUP_KERNEL)
        print(f"✅ Backed up kernel to {BACKUP_KERNEL}")


def restore_kernel():
    """Restore original kernel."""
    if BACKUP_KERNEL.exists():
        shutil.copy(BACKUP_KERNEL, BASE_KERNEL)
        print(f"✅ Restored kernel from backup")


def modify_kernel(block_m, block_n, block_k, pad, threads):
    """Modify kernel constants."""
    with open(BASE_KERNEL, 'r') as f:
        content = f.read()

    # Replace constants
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
    content = re.sub(
        r'constexpr int PAD = \d+;',
        f'constexpr int PAD = {pad};',
        content
    )
    content = re.sub(
        r'dim3 block\(\d+\);',
        f'dim3 block({threads});',
        content
    )

    with open(BASE_KERNEL, 'w') as f:
        f.write(content)


def run_benchmark():
    """Run benchmark and extract Phase 3.8 performance."""
    try:
        result = subprocess.run(
            ['python', 'dev/test_cuda_kernel.py'],
            capture_output=True,
            text=True,
            timeout=180,
            cwd=Path(__file__).parent.parent
        )

        output = result.stdout + result.stderr

        # Extract Phase 3.8 time for large batch
        # Looking for: "CUDA Optimized Dequant (Phase 3.8): X.XXX ms"
        pattern = r'CUDA Optimized Dequant \(Phase 3\.8\):\s+([\d.]+) ms'

        # Find all matches and get the last one (large batch)
        matches = re.findall(pattern, output)
        if matches:
            time_ms = float(matches[-1])  # Last match is large batch

            # Check if correctness passed
            if "ALL CORRECTNESS TESTS PASSED" in output:
                return time_ms, True
            else:
                return time_ms, False
        else:
            print(f"⚠️  Could not parse benchmark output")
            return None, False

    except subprocess.TimeoutExpired:
        print(f"⚠️  Benchmark timed out")
        return None, False
    except Exception as e:
        print(f"⚠️  Error running benchmark: {e}")
        return None, False


def main():
    print("=" * 80)
    print("MXFP4 Kernel Tuning - Beating Triton!")
    print("=" * 80)
    print(f"\nTarget: Beat Triton's 2.070 ms")
    print(f"Current best: Phase 3.8 at 2.501 ms (gap: 21%)\n")

    # Backup original kernel
    backup_kernel()

    results = []

    try:
        for i, (block_m, block_n, block_k, pad, threads) in enumerate(CONFIGS, 1):
            print(f"\n[{i}/{len(CONFIGS)}] Testing config: "
                  f"BLOCK={block_m}x{block_n}x{block_k}, "
                  f"PAD={pad}, threads={threads}")

            # Calculate shared memory
            smem_a = 2 * block_m * (block_k + pad) * 2
            smem_b_packed = 2 * (block_k // 2) * (block_n + pad) * 1
            smem_b_scales = 2 * 1 * (block_n + pad) * 2  # Assumes BLOCK_K=32
            smem_b_temp = block_k * (block_n + pad) * 2
            smem_output = block_m * (block_n + pad) * 4
            total_smem = smem_a + smem_b_packed + smem_b_scales + smem_b_temp + smem_output

            print(f"  Shared memory: {total_smem / 1024:.1f} KB")

            if total_smem > 99 * 1024:  # 99 KB limit on Ampere
                print(f"  ⚠️  SKIPPED - Exceeds shared memory limit")
                continue

            # Modify kernel
            modify_kernel(block_m, block_n, block_k, pad, threads)

            # Run benchmark
            time_ms, passed = run_benchmark()

            if time_ms is not None:
                if passed:
                    speedup_vs_baseline = 2.501 / time_ms
                    vs_triton = time_ms / 2.070

                    status = "🎉 BEATS TRITON!" if time_ms < 2.070 else "✅"
                    print(f"  {status} Time: {time_ms:.3f} ms "
                          f"({speedup_vs_baseline:.2f}x vs baseline, "
                          f"{vs_triton:.2f}x vs Triton)")

                    results.append({
                        'config': (block_m, block_n, block_k, pad, threads),
                        'time_ms': time_ms,
                        'smem_kb': total_smem / 1024,
                        'passed': True
                    })
                else:
                    print(f"  ❌ FAILED correctness tests")
                    results.append({
                        'config': (block_m, block_n, block_k, pad, threads),
                        'time_ms': time_ms,
                        'smem_kb': total_smem / 1024,
                        'passed': False
                    })
            else:
                print(f"  ❌ Benchmark failed")

    finally:
        # Restore original kernel
        restore_kernel()

    # Print summary
    print("\n" + "=" * 80)
    print("TUNING RESULTS SUMMARY")
    print("=" * 80)

    # Filter passing configs
    passing = [r for r in results if r['passed']]

    if passing:
        # Sort by time
        passing.sort(key=lambda x: x['time_ms'])

        print(f"\n✅ {len(passing)}/{len(results)} configs passed correctness\n")
        print(f"{'Rank':<6} {'BLOCK_M':<8} {'BLOCK_N':<8} {'BLOCK_K':<8} "
              f"{'PAD':<6} {'Threads':<8} {'Time (ms)':<12} {'vs Triton':<12}")
        print("-" * 80)

        for i, result in enumerate(passing[:10], 1):  # Top 10
            config = result['config']
            time_ms = result['time_ms']
            vs_triton = time_ms / 2.070

            marker = "🎉" if time_ms < 2.070 else "  "
            print(f"{marker}{i:<5} {config[0]:<8} {config[1]:<8} {config[2]:<8} "
                  f"{config[3]:<6} {config[4]:<8} {time_ms:<12.3f} {vs_triton:<12.2f}x")

        best = passing[0]
        print(f"\n🏆 Best config: BLOCK={best['config'][0]}x{best['config'][1]}x{best['config'][2]}, "
              f"PAD={best['config'][3]}, threads={best['config'][4]}")
        print(f"   Time: {best['time_ms']:.3f} ms")

        if best['time_ms'] < 2.070:
            improvement = (2.070 - best['time_ms']) / 2.070 * 100
            print(f"   🎉🎉🎉 BEATS TRITON by {improvement:.1f}%! 🎉🎉🎉")
        else:
            gap = (best['time_ms'] / 2.070 - 1) * 100
            print(f"   Gap to Triton: {gap:.1f}% slower")
    else:
        print("\n❌ No configs passed correctness tests")


if __name__ == "__main__":
    main()
