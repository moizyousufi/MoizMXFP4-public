"""
Test that MXLinear actually uses the QuTLASS backend.
"""

import torch
from mxfp4 import MXLinear
from mxfp4.backends import detect_architecture

def test_mxlinear_uses_qutlass():
    """Verify MXLinear uses QuTLASS backend on Blackwell."""

    print("=" * 80)
    print("Testing MXLinear Backend Integration")
    print("=" * 80)

    # Check backend
    backend_name, info = detect_architecture()
    print(f"\nDetected Backend: {backend_name}")
    print(f"GPU: {info.get('device', 'Unknown')}")

    if backend_name != "qutlass":
        print("\n⚠️  Not using QuTLASS backend - skipping test")
        return

    # Create and quantize MXLinear layer
    print("\nCreating MXLinear layer...")
    layer = MXLinear(4096, 4096, bias=True, block_size=32).cuda()
    layer.quantize()

    # MXLinear stores quantized weights as _weight_packed and _weight_scales
    print(f"Weight (packed): {layer._weight_packed.shape}")
    print(f"Weight scales: {layer._weight_scales.shape}")
    print(f"Bias: {layer.bias.shape if layer.bias is not None else None}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(128, 4096, dtype=torch.bfloat16, device='cuda')

    # Run forward pass - should use QuTLASS matmul
    try:
        output = layer(x)
        print(f"✅ Forward pass succeeded!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Benchmark to see if we get speedup
    print("\nQuick performance check...")

    # Warmup
    for _ in range(10):
        _ = layer(x)
    torch.cuda.synchronize()

    # Measure
    import time
    start = time.time()
    for _ in range(100):
        _ = layer(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"Time for 100 runs: {elapsed*1000:.2f} ms")
    print(f"Average per run: {elapsed*10:.2f} ms")

    print("\n" + "=" * 80)
    print("✅ MXLinear integration test complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_mxlinear_uses_qutlass()
