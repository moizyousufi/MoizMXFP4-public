import torch
import pytest
from mxfp4.quantizer import MXFP4Quantizer
# Import the triton function, but handle import error if triton not compiled/avail
try:
    from mxfp4.kernels import dequantize_triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

@pytest.fixture
def quantizer():
    return MXFP4Quantizer(block_size=32)

def test_quantizer_initialization(quantizer):
    assert quantizer.block_size == 32
    assert len(quantizer.E2M1_VALUES) == 8

def test_padding(quantizer):
    x = torch.randn(33)
    padded, pad_len = quantizer._pad_tensor(x)
    assert padded.shape[0] == 64
    assert pad_len == 31
    
    x = torch.randn(32)
    padded, pad_len = quantizer._pad_tensor(x)
    assert padded.shape[0] == 32
    assert pad_len == 0

def test_quantize_output_shapes(quantizer):
    x = torch.randn(10, 64)
    packed, scales = quantizer.quantize(x)
    
    assert packed.dtype == torch.uint8
    assert packed.shape == (10, 32)
    assert scales.shape == (10, 2)

def test_roundtrip_accuracy(quantizer):
    torch.manual_seed(42)
    x = torch.randn(128, 128)

    packed, scales = quantizer.quantize(x)
    x_rec = quantizer.dequantize(packed, scales)

    assert x_rec.shape == x.shape

    mse = torch.mean((x - x_rec)**2)
    assert mse < 0.1, f"MSE {mse} is too high for MXFP4"

    block_idx = 0
    scale = scales[0, block_idx]

    # Decode E8M0 scale to float if needed
    if quantizer.scale_format == "e8m0":
        scale_float = quantizer._from_e8m0(scale.unsqueeze(0)).item()
    else:
        scale_float = scale.item()

    block_vals = x_rec[0, 0:32]
    normalized_rec = torch.abs(block_vals / scale_float)

    for val in normalized_rec:
        dist = torch.min(torch.abs(val - quantizer.E2M1_VALUES.to(x.device)))
        assert dist < 1e-5, f"Reconstructed value {val} not in E2M1 set"

def test_specific_values(quantizer):
    data = torch.tensor([0.5, 1.0, 2.0, 6.0] * 8)

    packed, scales = quantizer.quantize(data)
    rec = quantizer.dequantize(packed, scales)

    assert torch.allclose(data, rec, atol=1e-6)

    # Decode E8M0 scale to float if needed
    if quantizer.scale_format == "e8m0":
        scale_float = quantizer._from_e8m0(scales).item()
        # E8M0 format: 127 = 2^0 = 1.0
        assert scales.item() == 127, f"Expected E8M0 encoded 1.0 (127), got {scales.item()}"
        assert abs(scale_float - 1.0) < 1e-6, f"Expected decoded scale 1.0, got {scale_float}"
    else:
        assert scales.item() == 1.0

def test_device_movement(quantizer):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    x = torch.randn(64, 64).cuda()
    packed, scales = quantizer.quantize(x)
    
    assert packed.device.type == 'cuda'
    assert scales.device.type == 'cuda'
    
    rec = quantizer.dequantize(packed, scales)
    assert rec.device.type == 'cuda'

def test_triton_kernel(quantizer):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available/installed")

    # Test data
    x = torch.randn(128, 128).cuda()
    packed, scales = quantizer.quantize(x)

    # Reference Python Dequant
    rec_ref = quantizer.dequantize(packed, scales)

    # Cast ref to BF16 to simulate the kernel's output precision
    rec_ref = rec_ref.to(torch.bfloat16)

    # Triton Dequant
    # Flatten packed and scales for the kernel wrapper
    packed_flat = packed.view(-1)
    scales_flat = scales.view(-1)

    rec_triton = dequantize_triton(packed_flat, scales_flat)

    # Reshape back
    rec_triton = rec_triton.view_as(x)

    # Compare
    # Should be bit-exact or extremely close (BF16 precision diffs)
    # We cast everything to float32 for comparison
    diff = torch.abs(rec_ref.float() - rec_triton.float())
    max_diff = torch.max(diff)

    assert max_diff < 1e-3, f"Triton dequant mismatches Python ref. Max diff: {max_diff}"

def test_negative_values(quantizer):
    """Test that negative values are handled correctly (sign bit)."""
    # Create data with exact E2M1 values (negative)
    data = torch.tensor([-0.5, -1.0, -2.0, -6.0] * 8)

    packed, scales = quantizer.quantize(data)
    rec = quantizer.dequantize(packed, scales)

    # Should reconstruct exactly (within floating point error)
    assert torch.allclose(data, rec, atol=1e-6), f"Negative values not preserved. Max diff: {torch.max(torch.abs(data - rec))}"

    # Verify scale is correct (max abs = 6.0, scale = 6.0/6.0 = 1.0)
    if quantizer.scale_format == "e8m0":
        # E8M0: 127 = 2^0 = 1.0
        assert scales.item() == 127
        scale_float = quantizer._from_e8m0(scales).item()
        assert abs(scale_float - 1.0) < 1e-6
    else:
        assert torch.allclose(scales, torch.tensor([1.0]), atol=1e-6)

def test_zeros(quantizer):
    """Test handling of zero values."""
    x = torch.zeros(128)
    packed, scales = quantizer.quantize(x)
    rec = quantizer.dequantize(packed, scales)

    # Zeros should map to 0.0 or very close (might map to ±0.5 depending on rounding)
    # The scale should be very small (clamped to 1e-8)
    assert torch.allclose(rec, torch.zeros_like(rec), atol=0.1), \
        f"Zeros not preserved. Max abs value: {torch.max(torch.abs(rec))}"

def test_mixed_signs(quantizer):
    """Test blocks with mixed positive and negative values."""
    # Create block with mixed signs
    x = torch.tensor([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0] * 4)

    packed, scales = quantizer.quantize(x)
    rec = quantizer.dequantize(packed, scales)

    # Check signs are preserved
    assert torch.all(torch.sign(x) == torch.sign(rec)), \
        "Signs not preserved in mixed sign block"

    # Check values are reasonably close
    # E8M0 has lower precision than BF16 due to power-of-2 constraint
    mse = torch.mean((x - rec)**2)
    tolerance = 1.0 if quantizer.scale_format == "e8m0" else 0.1
    assert mse < tolerance, f"MSE too high for mixed signs: {mse}"

def test_very_small_values(quantizer):
    """Test handling of very small values (potential underflow)."""
    x = torch.tensor([1e-6, 1e-7, 1e-8] * 10 + [0.0])
    packed, scales = quantizer.quantize(x)
    rec = quantizer.dequantize(packed, scales)

    # Very small values should be clamped/quantized but not cause errors
    assert not torch.isnan(rec).any(), "NaN values in reconstruction"
    assert not torch.isinf(rec).any(), "Inf values in reconstruction"

def test_very_large_values(quantizer):
    """Test handling of very large values.

    Note: MXFP4 assumes values within a block have similar magnitudes.
    Testing with values in the same order of magnitude (1e6 to 6e6).
    """
    # Values with similar magnitude (realistic for a single block)
    x = torch.tensor([1e6, 2e6, 3e6, 6e6] * 8)  # 32 elements, same order of magnitude
    packed, scales = quantizer.quantize(x)
    rec = quantizer.dequantize(packed, scales)

    # Large values should scale properly without overflow
    assert not torch.isnan(rec).any(), "NaN values in reconstruction"
    assert not torch.isinf(rec).any(), "Inf values in reconstruction"

    # Should reconstruct reasonably well
    # E8M0 rounds scale to nearest power of 2, so there's inherent error
    # Max = 6e6, scale = 6e6/6 = 1e6
    # E8M0 rounds 1e6 to nearest power of 2: 2^19 = 524288 or 2^20 = 1048576
    # This causes larger reconstruction error than BF16
    if quantizer.scale_format == "e8m0":
        # E8M0: allow larger relative error due to power-of-2 rounding
        assert torch.allclose(x, rec, rtol=0.1), \
            f"Large values not preserved. Max diff: {torch.max(torch.abs(x - rec))}"
    else:
        # BF16: exact reconstruction expected
        assert torch.allclose(x, rec, rtol=1e-5), \
            f"Large values not preserved. Max diff: {torch.max(torch.abs(x - rec))}"

def test_multi_magnitude_limitation(quantizer):
    """Document MXFP4's limitation with multi-magnitude values in same block.

    MXFP4 uses block-wise scaling, so values with vastly different magnitudes
    in the same block will have high quantization error. This is expected behavior.
    """
    # Mix small and large values in same block (pathological case)
    x = torch.tensor([1.0, 1e6] * 16)  # 6 orders of magnitude difference
    packed, scales = quantizer.quantize(x)
    rec = quantizer.dequantize(packed, scales)

    # Should not crash or produce NaN/Inf
    assert not torch.isnan(rec).any(), "NaN in multi-magnitude block"
    assert not torch.isinf(rec).any(), "Inf in multi-magnitude block"

    # BUT: Quantization error will be LARGE for small values
    # This is expected and documented!
    # Small values get crushed to near-zero when block max is 1e6
    small_val_indices = [i for i in range(len(x)) if x[i] < 10]
    large_val_indices = [i for i in range(len(x)) if x[i] > 1e5]

    # Large values should be relatively accurate
    large_rel_error = torch.abs((x[large_val_indices] - rec[large_val_indices]) / x[large_val_indices])
    assert torch.max(large_rel_error) < 0.5, "Large values have unexpectedly high error"

    # Small values will have HUGE error (this is the limitation!)
    # We just check they don't become negative or NaN
    assert torch.all(rec[small_val_indices] >= 0), "Small values became negative"

@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_different_block_sizes(block_size):
    """Test that different block sizes work correctly."""
    quantizer = MXFP4Quantizer(block_size=block_size)

    # Create data divisible by block_size
    x = torch.randn(10, block_size * 4)
    packed, scales = quantizer.quantize(x)
    rec = quantizer.dequantize(packed, scales)

    # Check shapes
    assert rec.shape == x.shape
    assert scales.shape[-1] == (x.shape[-1] // block_size)

    # Check accuracy
    mse = torch.mean((x - rec)**2)
    assert mse < 0.1, f"MSE too high for block_size={block_size}: {mse}"

def test_shape_validation_triton():
    """Test that Triton kernel validates shapes correctly."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not TRITON_AVAILABLE:
        pytest.skip("Triton not available/installed")

    # Create mismatched shapes
    packed = torch.randint(0, 255, (64,), dtype=torch.uint8).cuda()  # 128 elements when unpacked
    scales_wrong = torch.randn(2).cuda()  # Should be 4 scales (128/32)

    # Should raise ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        dequantize_triton(packed, scales_wrong, quant_block_size=32)
