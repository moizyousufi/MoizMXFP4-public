"""
Test script to verify QuTLASS matmul is working correctly.
"""

import torch
from mxfp4.quantizer import MXFP4Quantizer
from mxfp4.backends import get_backend, detect_architecture

def test_qutlass_matmul():
    """Test QuTLASS matmul implementation."""

    print("=" * 80)
    print("Testing QuTLASS MXFP4 Matmul")
    print("=" * 80)

    # Check GPU and backend
    backend_name, info = detect_architecture()
    print(f"\nDetected GPU: {info.get('device', 'Unknown')}")
    print(f"Architecture: {info.get('architecture', 'unknown')}")
    print(f"Backend: {backend_name}")
    print(f"Compute Capability: {info.get('compute_capability', 'unknown')}")

    if backend_name != "qutlass":
        print("\n⚠️  QuTLASS backend not selected. This test requires Blackwell GPU + QuTLASS installed.")
        print("Install with: ./install_blackwell.sh")
        return

    # Initialize backend
    try:
        backend = get_backend("qutlass")
        print("\n✅ QuTLASS backend loaded successfully")
    except Exception as e:
        print(f"\n❌ Failed to load QuTLASS backend: {e}")
        return

    # Create test tensors
    print("\nCreating test tensors...")
    M, N, K = 128, 256, 4096  # K must be divisible by 32

    device = torch.device("cuda:0")
    input_tensor = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    weight_tensor = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    bias = torch.randn(N, dtype=torch.bfloat16, device=device)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight_tensor.shape}")
    print(f"Bias shape: {bias.shape}")

    # Quantize weights
    print("\nQuantizing weights...")
    quantizer = MXFP4Quantizer(block_size=32, scale_format="e8m0")
    packed_weight, weight_scales = quantizer.quantize(weight_tensor)

    print(f"Packed weight shape: {packed_weight.shape}")
    print(f"Weight scales shape: {weight_scales.shape}")

    # Test QuTLASS matmul
    print("\nTesting QuTLASS matmul...")
    try:
        output_qutlass = backend.matmul(input_tensor, packed_weight, weight_scales, bias)
        print(f"✅ QuTLASS matmul succeeded!")
        print(f"Output shape: {output_qutlass.shape}")
    except Exception as e:
        print(f"❌ QuTLASS matmul failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare with reference (dequantize + standard matmul)
    print("\nComparing with reference implementation...")
    weight_dequant = quantizer.dequantize(packed_weight, weight_scales, weight_tensor.shape)
    output_ref = torch.matmul(input_tensor, weight_dequant.T.to(torch.bfloat16)) + bias

    # Calculate error
    abs_error = torch.abs(output_qutlass - output_ref)
    rel_error = abs_error / (torch.abs(output_ref) + 1e-8)

    print(f"\nError Analysis:")
    print(f"  Max absolute error: {abs_error.max().item():.6f}")
    print(f"  Mean absolute error: {abs_error.mean().item():.6f}")
    print(f"  Max relative error: {rel_error.max().item():.6f}")
    print(f"  Mean relative error: {rel_error.mean().item():.6f}")

    # Check if errors are acceptable (should be very small for quantization error)
    if abs_error.max().item() < 1.0 and rel_error.mean().item() < 0.1:
        print("\n✅ QuTLASS matmul output matches reference within acceptable tolerance!")
    else:
        print("\n⚠️  Large errors detected - may indicate implementation issue")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)

if __name__ == "__main__":
    test_qutlass_matmul()
