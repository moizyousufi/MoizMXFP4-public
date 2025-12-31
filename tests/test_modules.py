
import torch
import torch.nn as nn
import pytest
from mxfp4.modules import MXLinear

def test_initialization():
    layer = MXLinear(32, 64, block_size=32)
    assert layer.in_features == 32
    assert layer.out_features == 64
    assert layer.weight.shape == (64, 32)
    assert layer.bias.shape == (64,)
    assert not layer.is_quantized
    assert layer.packed_weight is None

def test_quantize_dequantize_flow():
    layer = MXLinear(64, 64, block_size=32)
    original_weight = layer.weight.clone()
    
    # Quantize
    layer.quantize()
    assert layer.is_quantized
    assert layer.weight is None # Should be deleted
    assert layer.packed_weight is not None
    assert layer.weight_scales is not None
    
    # Check packed shape
    # 64*64 = 4096 elements. 4096 / 2 = 2048 bytes.
    assert layer.packed_weight.numel() == 2048
    
    # Dequantize
    layer.dequantize()
    assert not layer.is_quantized
    assert layer.weight is not None
    assert layer.packed_weight is None
    
    # Check accuracy
    diff = torch.abs(original_weight - layer.weight)
    assert torch.mean(diff) < 0.1 # Rough check

def test_forward_pass_cpu():
    # Test fallback path on CPU
    layer = MXLinear(32, 32, block_size=32)
    x = torch.randn(10, 32)
    
    # Standard output
    y_std = layer(x)
    
    # Quantized output
    layer.quantize()
    y_quant = layer(x)
    
    # Should be close
    diff = torch.abs(y_std - y_quant)
    assert torch.mean(diff) < 0.5 

def test_forward_pass_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    layer = MXLinear(128, 128, block_size=32).cuda()
    x = torch.randn(16, 128).cuda()
    
    # Standard output
    y_std = layer(x)
    
    # Quantize (activates Triton path if available)
    layer.quantize()
    assert layer.packed_weight.is_cuda
    
    y_quant = layer(x)
    
    # Should be close
    # Note: Triton BF16 precision vs Python FP32 precision differences apply
    diff = torch.abs(y_std - y_quant)
    mse = torch.mean(diff**2)
    
    assert mse < 0.1, f"MSE {mse} too high on GPU"

def test_save_load_state_dict():
    # Test that we can save a quantized model and load it back
    layer = MXLinear(32, 32, block_size=32)
    layer.quantize()
    
    state = layer.state_dict()
    
    # Check keys
    assert 'packed_weight' in state
    assert 'weight_scales' in state
    assert 'weight' not in state # Parameter should be gone
    
    # Create new layer
    new_layer = MXLinear(32, 32, block_size=32)
    # We must manually put it in quantized mode to load these keys?
    # PyTorch load_state_dict usually requires keys to match.
    # Since packed_weight is a buffer, it exists in new_layer (as None).
    # But 'weight' is a parameter in new_layer.
    # If we load, strict=True might fail because 'weight' is missing in state_dict.
    
    # This is a known issue with dynamic quantization state.
    # Strategy: User must quantize() (or prepare) before loading? 
    # Or we handle strict=False.
    
    # Let's simulate the "Deployment" flow:
    # 1. Initialize model
    # 2. Quantize (empty weights)
    # 3. Load state dict
    
    new_layer.quantize() # Prepares buffers, deletes weight
    new_layer.load_state_dict(state)
    
    assert torch.equal(new_layer.packed_weight, layer.packed_weight)

if __name__ == "__main__":
    pytest.main([__file__])
