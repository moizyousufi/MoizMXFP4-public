
import torch
import pytest
from mxfp4.fused_kernels import quant_matmul
from mxfp4.quantizer import MXFP4Quantizer

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_matmul_correctness():
    # Setup
    M, N, K = 128, 128, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    # Create random weights in FP32, quantize them
    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K, device='cuda') # [Out, In]

    packed, scales = quantizer.quantize(w)

    # Reference: Dequantize manually and matmul
    w_dequant = quantizer.dequantize(packed, scales)
    w_dequant = w_dequant.view(N, K).to(torch.bfloat16)

    # Target: x @ w.T
    ref_output = torch.matmul(x, w_dequant.T)

    # Fused Kernel
    # Note: fused kernel expects packed weight [N, K/2] which `packed` is [N, K/2] if flattened blocks?
    # Quantizer returns packed shape [..., NumBlocks, BlockSize/2] -> flattened to [..., NumBlocks*BlockSize/2]
    # If input W was [N, K], output packed is [N, K/2].
    # Scales is [N, K/32].
    # This matches fused kernel expectation.

    fused_output = quant_matmul(x, packed, scales)

    # Compare
    diff = torch.abs(ref_output.float() - fused_output.float())
    max_diff = torch.max(diff)

    # Tolerance: BF16 matmul has some noise.
    assert max_diff < 0.5, f"Fused kernel mismatch. Max diff: {max_diff}"

    # Check MSE
    mse = torch.mean(diff**2)
    assert mse < 0.1, f"Fused kernel MSE too high: {mse}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_matmul_non_aligned_k():
    """Test fused matmul with K not divisible by 32 (tests masking logic)."""
    # Use K=100 (not divisible by 32 or BLOCK_SIZE_K)
    M, N, K_original = 64, 64, 100

    # Quantize weights
    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K_original, device='cuda')
    packed, scales = quantizer.quantize(w)

    # Determine padded K from packed shape
    K_padded = packed.shape[-1] * 2  # packed is [N, K_padded/2]

    # Pad input to match (required for matmul)
    x_original = torch.randn(M, K_original, dtype=torch.bfloat16, device='cuda')
    if K_padded > K_original:
        padding = K_padded - K_original
        x = torch.nn.functional.pad(x_original, (0, padding), value=0)  # Pad last dim
    else:
        x = x_original

    # Reference: Dequantize and matmul with padded dimensions
    w_dequant = quantizer.dequantize(packed, scales)
    w_dequant_2d = w_dequant.view(N, K_padded).to(torch.bfloat16)
    ref_output = torch.matmul(x, w_dequant_2d.T)

    # Fused (should handle padded K correctly with masking)
    fused_output = quant_matmul(x, packed, scales)

    # Compare
    diff = torch.abs(ref_output.float() - fused_output.float())
    max_diff = torch.max(diff)

    assert max_diff < 0.5, f"Non-aligned K failed. Max diff: {max_diff}"
    assert not torch.isnan(fused_output).any(), "NaN in output (masking bug!)"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_matmul_small_k():
    """Test fused matmul with very small K (edge case)."""
    M, N, K = 32, 32, 32  # K exactly one block
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K, device='cuda')

    packed, scales = quantizer.quantize(w)

    # Reference
    w_dequant = quantizer.dequantize(packed, scales)
    w_dequant = w_dequant.view(N, K).to(torch.bfloat16)
    ref_output = torch.matmul(x, w_dequant.T)

    # Fused
    fused_output = quant_matmul(x, packed, scales)

    # Compare
    diff = torch.abs(ref_output.float() - fused_output.float())
    max_diff = torch.max(diff)

    assert max_diff < 0.5, f"Small K failed. Max diff: {max_diff}"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_matmul_with_bias():
    """Test fused matmul with bias."""
    M, N, K = 128, 128, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
    bias = torch.randn(N, dtype=torch.bfloat16, device='cuda')

    quantizer = MXFP4Quantizer(block_size=32)
    w = torch.randn(N, K, device='cuda')
    packed, scales = quantizer.quantize(w)

    # Reference
    w_dequant = quantizer.dequantize(packed, scales)
    w_dequant = w_dequant.view(N, K).to(torch.bfloat16)
    ref_output = torch.matmul(x, w_dequant.T) + bias

    # Fused (with bias)
    fused_output = quant_matmul(x, packed, scales, bias=bias)

    # Compare
    diff = torch.abs(ref_output.float() - fused_output.float())
    max_diff = torch.max(diff)

    assert max_diff < 0.5, f"Bias test failed. Max diff: {max_diff}"

if __name__ == "__main__":
    pytest.main([__file__])
