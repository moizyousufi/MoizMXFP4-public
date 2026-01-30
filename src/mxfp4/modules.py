
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .quantizer import MXFP4Quantizer
# Try to import kernels, fallback to None if not available (CPU mode or build issues)
try:
    from .kernels import dequantize_triton
    TRITON_AVAILABLE = True
except ImportError:
    dequantize_triton = None
    TRITON_AVAILABLE = False

class MXLinear(nn.Module):
    """
    Linear layer with MXFP4 quantization support.

    Modes:
    1. Standard (Training): Weights stored in BF16/FP32.
       - If `simulate_quantization` is True: Forward pass quantizes and dequantizes weights (FakeQuant) to simulate noise.
       - If False: Acts like standard nn.Linear.

    2. Quantized (Inference): Weights stored in MXFP4 (packed uint8 + scales).
       - Original .weight is deleted to save memory.
       - Forward pass dequantizes on-the-fly (using Triton if avail) and computes matmul.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, block_size: int = 32, use_fused_kernel: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.use_fused_kernel = use_fused_kernel

        # standard initialization (BF16 master weights)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # quantization state
        self.quantizer = MXFP4Quantizer(block_size=block_size)
        self.is_quantized = False
        self.simulate_quantization = False # For QAT (Future)

        # buffers for quantized weights (initially empty/None)
        # we register them as buffers so they are saved in state_dict
        self.register_buffer('packed_weight', None)
        self.register_buffer('weight_scales', None)
        self.register_buffer('weight_scales_blocked', None)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def quantize(self, device=None):
        """
        Convert the BF16/FP32 .weight to MXFP4 packed format.
        Deletes .weight to save memory.
        """
        if self.is_quantized:
            return

        if self.weight is None:
            raise ValueError("No weights to quantize.")

        # move to device if requested (or stay on current)
        if device:
            self.to(device)

        with torch.no_grad():
            # 1. quantize
            packed, scales = self.quantizer.quantize(self.weight)

            # 2. store as buffers
            self.packed_weight = packed
            self.weight_scales = scales

            if self.use_fused_kernel:
                try:
                    # Import QuTLASS utils for scale conversion
                    import qutlass.utils

                    # Convert scales to QuTLASS's blocked layout format
                    # This is expensive but only done once during quantize()
                    weight_scales_fp8 = self.weight_scales.view(torch.float8_e8m0fnu)
                    self.weight_scales_blocked = qutlass.utils.to_blocked(
                        weight_scales_fp8, use_triton_kernel=True
                    )

                    # Register as buffer so it's saved in state_dict
                    self.register_buffer('weight_scales_blocked', self.weight_scales_blocked)
                except ImportError:
                    # QuTLASS not available, skip pre-conversion
                    self.weight_scales_blocked = None
            else:
                self.weight_scales_blocked = None

            # 3. delete original weight
            del self.weight
            self.register_parameter('weight', None)

            self.is_quantized = True

    def dequantize(self):
        """
        Convert back to BF16 .weight from packed format.
        Restores .weight Parameter.
        """
        if not self.is_quantized:
            return

        with torch.no_grad():
            # 1. dequantize
            w = self.quantizer.dequantize(self.packed_weight, self.weight_scales)

            # reshape to (Out, In)
            w = w.view(self.out_features, self.in_features)

            # 2. restore parameter
            self.weight = nn.Parameter(w)

            # 3. clear buffers
            self.packed_weight = None
            self.weight_scales = None

            self.is_quantized = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_quantized:
            # inference mode with compressed weights

            if self.use_fused_kernel:
                # FAST PATH: Use fused MXFP4 kernel (2-3x faster!)
                # quant_matmul() auto-selects backend (QuTLASS/Triton/Fallback)
                from .kernel_dispatcher import quant_matmul

                original_shape = input.shape
                input_2d = input.view(-1, input.size(-1))

                weight_scales = self.weight_scales
                if self.weight_scales_blocked is not None:
                    weight_scales._mxlinear_blocked = self.weight_scales_blocked

                output_2d = quant_matmul(input_2d, self.packed_weight,
                                        weight_scales, self.bias)

                output = output_2d.view(*original_shape[:-1], self.out_features)
                return output
            else:
                # LEGACY PATH: Dequantize + standard matmul
                # This is slower but kept for compatibility

                # 1. dequantize Weights
                if TRITON_AVAILABLE and input.is_cuda:
                    # flatten packed/scales for kernel
                    packed_flat = self.packed_weight.view(-1)
                    scales_flat = self.weight_scales.view(-1)

                    # dequantize
                    w_dequant = dequantize_triton(packed_flat, scales_flat, self.block_size)

                    # reshape
                    w_dequant = w_dequant.view(self.out_features, self.in_features)
                else:
                    # fallback to Python (slow but works on CPU)
                    w_dequant = self.quantizer.dequantize(self.packed_weight, self.weight_scales)
                    w_dequant = w_dequant.view(self.out_features, self.in_features)

                # cast to input dtype (if input is FP32, promote BF16 weight to FP32)
                if w_dequant.dtype != input.dtype:
                    w_dequant = w_dequant.to(input.dtype)

                # 2. standard matmul
                return F.linear(input, w_dequant, self.bias)

        else:
            # standard mode (or QAT FakeQuant)
            if self.simulate_quantization:
                # fake quantization: quant -> dequant -> forward
                # this adds noise without saving memory
                packed, scales = self.quantizer.quantize(self.weight)
                w_fake = self.quantizer.dequantize(packed, scales)
                w_fake = w_fake.view(self.out_features, self.in_features)
                return F.linear(input, w_fake, self.bias)
            else:
                # plain forward
                # cast weight to match input dtype if needed
                weight = self.weight
                if weight.dtype != input.dtype:
                    weight = weight.to(input.dtype)
                return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'bias={self.bias is not None}, block_size={self.block_size}, '
                f'quantized={self.is_quantized}, use_fused_kernel={self.use_fused_kernel}')
