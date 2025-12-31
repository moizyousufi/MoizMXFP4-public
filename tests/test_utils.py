
import torch
import torch.nn as nn
import pytest
from mxfp4.utils import quantize_model_mxfp4, get_model_size_mb
from mxfp4.modules import MXLinear

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.head = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.head(x)

def test_quantize_whole_model():
    model = SimpleMLP()
    
    # Initial check
    assert isinstance(model.fc1, nn.Linear)
    initial_size = get_model_size_mb(model)
    
    # Quantize
    # Exclude 'head' just to test exclusion
    model = quantize_model_mxfp4(model, exclude_layers=['head'])
    
    # Verify replacement
    assert isinstance(model.fc1, MXLinear)
    assert isinstance(model.fc2, MXLinear)
    assert isinstance(model.head, nn.Linear) # Should remain Linear
    
    # Verify state
    assert model.fc1.is_quantized
    
    # Verify Size Reduction
    # Weights are 4-bit vs 32-bit (assuming float32 default)
    # Reduction should be significant (~4-6x roughly depending on overhead)
    final_size = get_model_size_mb(model)
    
    # fc1: 64*128 = 8192 params. 32KB (float32) -> 4KB (4bit) + Scales
    # fc2: 128*64 = 8192 params. 32KB -> 4KB
    # head: 64*10 = 640 params. 2.5KB (stays)
    
    assert final_size < initial_size
    print(f"Compressed model from {initial_size:.3f}MB to {final_size:.3f}MB")

def test_quantized_model_forward():
    model = SimpleMLP()
    x = torch.randn(4, 64)
    
    # Standard output
    with torch.no_grad():
        y_ref = model(x)
    
    # Quantize
    quantize_model_mxfp4(model)
    
    # Forward pass should work
    y_quant = model(x)
    
    assert y_quant.shape == y_ref.shape

if __name__ == "__main__":
    pytest.main([__file__])
