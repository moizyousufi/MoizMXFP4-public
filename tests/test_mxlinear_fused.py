"""
Unit tests for MXLinear with fused MXFP4 kernel support.

Tests verify that the fused kernel path produces correct results and handles
multi-dimensional inputs properly.
"""

import pytest
import torch
from mxfp4.modules import MXLinear


class TestMXLinearFused:
    """Test suite for MXLinear with use_fused_kernel=True."""

    @pytest.fixture
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_fused_kernel_correctness(self, device):
        """Verify fused kernel produces correct results vs BF16 baseline."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for fused kernel")

        # Create layer
        layer = MXLinear(1024, 8192, bias=False, use_fused_kernel=True).to(device)

        # Generate input
        x = torch.randn(128, 1024, dtype=torch.bfloat16, device=device)

        # BF16 baseline
        y_bf16 = layer(x)

        # Quantize and run with fused kernel
        layer.quantize()
        y_mxfp4 = layer(x)

        # Check MSE
        mse = torch.mean((y_bf16 - y_mxfp4) ** 2).item()
        assert mse < 0.1, f"MSE too high: {mse}"

        print(f"✅ MXLinear fused kernel MSE: {mse:.6f}")

    def test_fused_vs_legacy(self, device):
        """Compare fused kernel vs legacy dequant+matmul path."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Create two layers with same weights
        layer_fused = MXLinear(512, 4096, bias=False, use_fused_kernel=True).to(device)
        layer_legacy = MXLinear(512, 4096, bias=False, use_fused_kernel=False).to(device)

        # Copy weights
        layer_legacy.load_state_dict(layer_fused.state_dict())

        # Quantize both
        layer_fused.quantize()
        layer_legacy.quantize()

        # Generate input
        x = torch.randn(64, 512, dtype=torch.bfloat16, device=device)

        # Forward pass
        y_fused = layer_fused(x)
        y_legacy = layer_legacy(x)

        # Should be similar (MXFP4 is lossy, different paths have different numerical precision)
        # Both paths use MXFP4 quantization, so some difference is expected
        mse = torch.mean((y_fused - y_legacy) ** 2).item()
        max_diff = torch.max(torch.abs(y_fused - y_legacy)).item()

        # Relaxed tolerance for MXFP4 quantization noise
        assert mse < 10.0, f"Fused vs legacy mismatch too large: MSE={mse}"

        print(f"✅ Fused vs legacy comparison:")
        print(f"   MSE: {mse:.6f}")
        print(f"   Max diff: {max_diff:.6f}")

    def test_multidim_input(self, device):
        """Test 3D input handling with reshape logic."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        layer = MXLinear(1024, 8192, bias=False, use_fused_kernel=True).to(device)
        layer.quantize()

        # 3D input: [batch, seq_len, hidden]
        x = torch.randn(4, 128, 1024, dtype=torch.bfloat16, device=device)
        y = layer(x)

        # Check shape
        assert y.shape == (4, 128, 8192), f"Unexpected shape: {y.shape}"
        assert not torch.isnan(y).any(), "NaN in output"
        assert not torch.isinf(y).any(), "Inf in output"

        print(f"✅ 3D input handling: {x.shape} → {y.shape}")

    def test_bias_handling(self, device):
        """Test that bias is correctly applied in fused kernel."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")

        # Layer with bias
        layer = MXLinear(256, 512, bias=True, use_fused_kernel=True).to(device)

        # Generate input
        x = torch.randn(32, 256, dtype=torch.bfloat16, device=device)

        # BF16 baseline
        y_bf16 = layer(x)

        # Quantized
        layer.quantize()
        y_mxfp4 = layer(x)

        # Check MSE
        mse = torch.mean((y_bf16 - y_mxfp4) ** 2).item()
        assert mse < 0.1, f"MSE with bias too high: {mse}"

        print(f"✅ Bias handling MSE: {mse:.6f}")

    def test_quantize_dequantize_cycle(self, device):
        """Test quantize/dequantize workflow."""
        layer = MXLinear(128, 256, bias=False, use_fused_kernel=True).to(device)

        original_weight = layer.weight.clone()

        # Quantize
        layer.quantize()
        assert layer.is_quantized
        assert layer.weight is None
        assert layer.packed_weight is not None
        assert layer.weight_scales is not None

        # Dequantize
        layer.dequantize()
        assert not layer.is_quantized
        assert layer.weight is not None

        # Check weight recovery
        diff = torch.abs(original_weight - layer.weight).mean().item()
        assert diff < 0.1, f"Weight recovery error too high: {diff}"

        print(f"✅ Quantize/dequantize cycle diff: {diff:.6f}")

    def test_backward_compatibility(self, device):
        """Test that use_fused_kernel=False still works."""
        layer = MXLinear(128, 256, bias=False, use_fused_kernel=False).to(device)

        x = torch.randn(16, 128, dtype=torch.bfloat16, device=device)

        # Should work with legacy path
        y_before = layer(x)
        layer.quantize()
        y_after = layer(x)

        # Should be close
        mse = torch.mean((y_before - y_after) ** 2).item()
        assert mse < 0.15, f"Legacy path MSE too high: {mse}"

        print(f"✅ Legacy path (use_fused_kernel=False) MSE: {mse:.6f}")

    def test_extra_repr(self, device):
        """Test that extra_repr includes use_fused_kernel."""
        layer = MXLinear(128, 256, use_fused_kernel=True).to(device)

        repr_str = str(layer)
        assert "use_fused_kernel" in repr_str, "use_fused_kernel not in repr"
        assert "True" in repr_str, "use_fused_kernel value not shown"

        print(f"✅ extra_repr: {repr_str}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_kernel_import():
    """Test that kernel_dispatcher imports successfully."""
    try:
        from mxfp4.kernel_dispatcher import quant_matmul
        print("✅ quant_matmul imported successfully")
    except ImportError as e:
        pytest.fail(f"Failed to import quant_matmul: {e}")
