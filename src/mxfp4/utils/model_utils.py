
import torch
import torch.nn as nn
from typing import List, Optional, Union

from ..modules import MXLinear

def quantize_model_mxfp4(
    model: nn.Module,
    block_size: int = 32,
    exclude_layers: Optional[List[str]] = None,
    verbose: bool = True
) -> nn.Module:
    """
    Recursively replace nn.Linear layers with MXLinear and quantize them.
    
    Args:
        model: PyTorch model to quantize.
        block_size: Block size for MXFP4 (default 32).
        exclude_layers: List of layer names (suffixes) to exclude from quantization.
                       e.g. ['lm_head', 'output'].
        verbose: Print progress.
        
    Returns:
        The modified model (in-place modification).
    """
    exclude_layers = exclude_layers or []
    
    # iterate through named_modules to find parents and replace children
    def _replace_recursive(module, prefix=""):
        replaced_count = 0
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # check exclusion
            if any(ex in full_name for ex in exclude_layers):
                continue
                
            if isinstance(child, nn.Linear):
                if verbose:
                    # print(f"Quantizing {full_name}...")
                    pass
                    
                # create replacement
                new_layer = MXLinear(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=(child.bias is not None),
                    block_size=block_size
                )
                
                # copy weights/bias
                # copy data to avoid graph connection issues
                new_layer.weight.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                
                new_layer.to(child.weight.device)
                
                # quantize immediately
                new_layer.quantize()
                
                setattr(module, name, new_layer)
                replaced_count += 1
            else:
                replaced_count += _replace_recursive(child, full_name)
        return replaced_count

    total_replaced = _replace_recursive(model)
    if verbose:
        print(f"Quantized {total_replaced} Linear layers.")
        
    return model

def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in Megabytes (MB).
    Counts Parameters and Buffers (packed weights).
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)
