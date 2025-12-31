"""
MXFP4 Backend Dispatcher - Hybrid MoizQuTLASS + Triton

Architecture-aware routing for optimal MXFP4 performance:
- Blackwell (sm_120+): MoizQuTLASS native kernels (4x faster)
- Ampere/Ada/Hopper (sm_80-90): Triton software emulation (1.5-2x faster)
- Older GPUs: Fallback to BF16

This provides best performance on each architecture while maintaining
the open MXFP4 standard (no vendor lock-in).

Note: MoizQuTLASS is a bug-fixed fork of the original QuTLASS with
critical scale tensor dimension corrections.
"""

import torch
import warnings
from typing import Optional, Literal

# Try importing backends
try:
    import qutlass
    QUTLASS_AVAILABLE = True
except ImportError:
    QUTLASS_AVAILABLE = False

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def detect_architecture() -> tuple[str, dict]:
    """
    Detect GPU architecture and recommend backend.

    Returns:
        backend: "qutlass", "triton", or "fallback"
        info: Dictionary with GPU details
    """
    if not torch.cuda.is_available():
        return "fallback", {"device": "CPU", "reason": "No CUDA"}

    compute_cap = torch.cuda.get_device_capability(0)
    gpu_name = torch.cuda.get_device_name(0)

    # Determine architecture name
    if compute_cap[0] >= 10:
        arch_name = "blackwell"
    elif compute_cap[0] == 9:
        arch_name = "hopper"
    elif compute_cap[0] == 8:
        if compute_cap[1] >= 9:
            arch_name = "ada"
        else:
            arch_name = "ampere"
    elif compute_cap[0] == 7:
        arch_name = "volta"
    else:
        arch_name = "legacy"

    info = {
        "device": gpu_name,
        "architecture": arch_name,
        "compute_capability": f"sm_{compute_cap[0]}{compute_cap[1]}",
        "major": compute_cap[0],
        "minor": compute_cap[1],
    }

    # Blackwell architecture handling
    # SM100 (B200), SM120+ (RTX 6000 Pro): Use QuTLASS if available
    # SM103 (B300): Use Triton (SM100 binaries don't run on SM103 - NVCC limitation)
    sm_version = compute_cap[0] * 10 + compute_cap[1]

    if compute_cap[0] >= 10:
        # SM103 (B300): Binary incompatibility with SM100
        # NVCC doesn't support SM103, and SM100/SM101 binaries don't execute on SM103
        # Fall back to Triton until NVIDIA adds proper SM103 support
        if sm_version == 103:
            warnings.warn(
                f"B300 (SM103) detected. QuTLASS not supported due to NVCC limitations. "
                f"Using Triton backend. QuTLASS works on B200 (SM100) and RTX 6000 Pro (SM120+)."
            )
            if TRITON_AVAILABLE:
                info["backend"] = "triton"
                info["speedup"] = "1.5-2x vs BF16"
                info["mode"] = "software"
                info["note"] = "SM103 binary incompatibility - awaiting NVCC support"
                return "triton", info
            else:
                return "fallback", info

        # SM100, SM120+: QuTLASS supported
        if QUTLASS_AVAILABLE:
            info["backend"] = "qutlass"
            info["speedup"] = "4x vs BF16"
            info["mode"] = "native"
            return "qutlass", info
        else:
            warnings.warn(
                f"Blackwell GPU detected ({gpu_name}) but MoizQuTLASS not installed. "
                f"Install for 4x speedup: pip install --no-build-isolation git+https://github.com/moizyousufi/moiz-qutlass-public.git"
            )
            if TRITON_AVAILABLE:
                info["backend"] = "triton"
                info["speedup"] = "1.5-2x vs BF16"
                info["mode"] = "software"
                return "triton", info
            else:
                return "fallback", info

    # Hopper/Ampere/Ada (sm_80-90): Use Triton
    elif compute_cap[0] >= 8:
        if TRITON_AVAILABLE:
            info["backend"] = "triton"
            info["speedup"] = "1.5-2x vs BF16"
            info["mode"] = "software"
            return "triton", info
        else:
            warnings.warn(
                f"Ampere/Ada/Hopper GPU detected ({gpu_name}) but Triton not installed. "
                f"Install for 2x speedup: pip install triton"
            )
            return "fallback", info

    # Older GPUs: Fallback
    else:
        info["backend"] = "fallback"
        info["reason"] = f"GPU too old (need sm_80+)"
        return "fallback", info


class MXFP4Backend:
    """Base class for MXFP4 backends."""

    def quantize(self, tensor: torch.Tensor, block_size: int = 32):
        """Quantize tensor to MXFP4."""
        raise NotImplementedError

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor, shape: tuple):
        """Dequantize MXFP4 tensor."""
        raise NotImplementedError

    def matmul(self, input: torch.Tensor, packed_weight: torch.Tensor,
               weight_scales: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Fused quantized matrix multiplication."""
        raise NotImplementedError


class QuTLASSBackend(MXFP4Backend):
    """QuTLASS native MXFP4 backend for Blackwell."""

    def __init__(self):
        if not QUTLASS_AVAILABLE:
            raise ImportError(
                "QuTLASS not available. Install with: "
                "pip install --no-build-isolation git+https://github.com/moizyousufi/moiz-qutlass-public.git"
            )
        self.qutlass = qutlass

    def quantize(self, tensor: torch.Tensor, block_size: int = 32):
        """Quantize using QuTLASS fused kernel (native speedup)."""
        if block_size != 32:
            return super().quantize(tensor, block_size)

        K = tensor.shape[-1]
        
        # requires K divisible by 32
        if K % 32 != 0:
            return super().quantize(tensor, block_size)

        # Create Identity matrix for H (no rotation)
        # TODO: Cache this for performance?
        H = torch.eye(K, dtype=tensor.dtype, device=tensor.device)
        
        # Use fused kernel (auto-detects v2 for large K)
        # method="abs_max" matches default behavior of MXFP4Quantizer
        packed, scales = self.qutlass.fusedQuantizeMx(
            tensor, H, method="abs_max", use_v2=None
        )
        
        return packed, scales

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor, shape: tuple):
        """Dequantize using QuTLASS."""
        from mxfp4.quantizer import MXFP4Quantizer
        quantizer = MXFP4Quantizer(scale_format="e8m0")
        return quantizer.dequantize(packed, scales, shape)

    def matmul(self, input: torch.Tensor, packed_weight: torch.Tensor,
               weight_scales: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Native QuTLASS MXFP4 matmul using hardware FP4 tensor cores."""
        M, K = input.shape
        N = packed_weight.shape[0]

        if K % 32 != 0:
            # requires K divisible by 32 (MXFP4 block size)
            warnings.warn(
                f"K={K} not divisible by 32. QuTLASS requires K % 32 == 0. "
                f"Falling back to Triton kernel."
            )
            from mxfp4.fused_kernels import quant_matmul
            return quant_matmul(input, packed_weight, weight_scales, bias)

        # quantize input activations to MXFP4 using QuTLASS
        try:
            packed_input, input_scales = self.quantize(input, block_size=32)
        except Exception as e:
            warnings.warn(
                f"QuTLASS quantization failed: {e}. "
                f"Falling back to Triton kernel."
            )
            from mxfp4.fused_kernels import quant_matmul
            return quant_matmul(input, packed_weight, weight_scales, bias)

        # call native QuTLASS MXFP4 matmul (uses Blackwell FP4 tensor cores)
        # alpha is a tensor array (not scalar) - QuTLASS API requirement
        alpha = torch.tensor([1.0], device=input.device)

        # convert scales from uint8 (E8M0) to float8_e8m0fnu dtype
        # then apply to_blocked() to convert to QuTLASS's blocked layout
        input_scales_fp8 = input_scales.view(torch.float8_e8m0fnu)
        weight_scales_fp8 = weight_scales.view(torch.float8_e8m0fnu)

        # CRITICAL: to_blocked() converts scales to QuTLASS's expected layout
        input_scales_blocked = self.qutlass.utils.to_blocked(input_scales_fp8, use_triton_kernel=True)
        weight_scales_blocked = self.qutlass.utils.to_blocked(weight_scales_fp8, use_triton_kernel=True)

        try:
            output = self.qutlass.matmul_mxf4_bf16_tn(
                a=packed_input,              # [M, K//2] packed MXFP4
                b=packed_weight,             # [N, K//2] packed MXFP4
                a_sf=input_scales_blocked,   # Blocked E8M0 scales
                b_sf=weight_scales_blocked,  # Blocked E8M0 scales
                alpha=alpha,                 # scaling factor [1.0]
                backend="cutlass"            # use CUTLASS kernels
            )
        except Exception as e:
            warnings.warn(
                f"QuTLASS matmul failed: {e}. "
                f"Falling back to Triton kernel."
            )
            from mxfp4.fused_kernels import quant_matmul
            return quant_matmul(input, packed_weight, weight_scales, bias)

        # Add bias if provided
        if bias is not None:
            output = output + bias.unsqueeze(0)

        return output


class TritonBackend(MXFP4Backend):
    """Triton software emulation backend for Ampere/Ada/Hopper."""

    def __init__(self):
        if not TRITON_AVAILABLE:
            raise ImportError("Triton not available. Install with: pip install triton")

    def quantize(self, tensor: torch.Tensor, block_size: int = 32):
        """Quantize using our E8M0 quantizer."""
        from mxfp4.quantizer import MXFP4Quantizer
        quantizer = MXFP4Quantizer(block_size=block_size, scale_format="e8m0")
        return quantizer.quantize(tensor)

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor, shape: tuple):
        """Dequantize using our quantizer."""
        from mxfp4.quantizer import MXFP4Quantizer
        quantizer = MXFP4Quantizer(scale_format="e8m0")
        return quantizer.dequantize(packed, scales, shape)

    def matmul(self, input: torch.Tensor, packed_weight: torch.Tensor,
               weight_scales: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Triton fused MXFP4 matmul."""
        from mxfp4.fused_kernels import quant_matmul
        return quant_matmul(input, packed_weight, weight_scales, bias)


class FallbackBackend(MXFP4Backend):
    """Fallback backend (dequantize to BF16)."""

    def quantize(self, tensor: torch.Tensor, block_size: int = 32):
        """Quantize (still works, just slow)."""
        from mxfp4.quantizer import MXFP4Quantizer
        quantizer = MXFP4Quantizer(block_size=block_size, scale_format="e8m0")
        return quantizer.quantize(tensor)

    def dequantize(self, packed: torch.Tensor, scales: torch.Tensor, shape: tuple):
        """Dequantize."""
        from mxfp4.quantizer import MXFP4Quantizer
        quantizer = MXFP4Quantizer(scale_format="e8m0")
        return quantizer.dequantize(packed, scales, shape)

    def matmul(self, input: torch.Tensor, packed_weight: torch.Tensor,
               weight_scales: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """Fallback: dequantize then standard matmul."""
        from mxfp4.quantizer import MXFP4Quantizer
        quantizer = MXFP4Quantizer(scale_format="e8m0")

        # Dequantize weights
        N, K_packed = packed_weight.shape
        K = K_packed * 2
        weight_dequant = quantizer.dequantize(packed_weight, weight_scales, (N, K))

        # Standard matmul
        output = torch.matmul(input, weight_dequant.T.to(input.dtype))

        if bias is not None:
            output += bias

        return output


def get_backend(backend: Optional[Literal["qutlass", "triton", "fallback", "auto"]] = "auto") -> MXFP4Backend:
    """
    Get MXFP4 backend instance.

    Args:
        backend: Backend to use ("qutlass", "triton", "fallback", or "auto")
                 "auto" selects based on GPU architecture

    Returns:
        Backend instance

    Example:
        >>> backend = get_backend("auto")  # Selects optimal for your GPU
        >>> backend = get_backend("qutlass")  # Force QuTLASS
    """
    if backend == "auto":
        backend_name, info = detect_architecture()
        backend = backend_name
    else:
        _, info = detect_architecture()

    if backend == "qutlass":
        return QuTLASSBackend()
    elif backend == "triton":
        return TritonBackend()
    elif backend == "fallback":
        return FallbackBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}")


def print_backend_info():
    """Print information about available backends and GPU."""
    print("=" * 70)
    print("MXFP4 Backend Detection")
    print("=" * 70)

    # Check availability
    print("\nBackend Availability:")
    print(f"  QuTLASS: {'✅' if QUTLASS_AVAILABLE else '❌'}")
    print(f"  Triton:  {'✅' if TRITON_AVAILABLE else '❌'}")

    # Detect architecture
    backend_name, info = detect_architecture()

    print(f"\nGPU: {info.get('device', 'Unknown')}")
    print(f"Compute Capability: {info.get('compute_capability', 'Unknown')}")
    print(f"\nRecommended Backend: {backend_name.upper()}")

    if "speedup" in info:
        print(f"Expected Speedup: {info['speedup']}")
    if "mode" in info:
        print(f"Mode: {info['mode']}")
    if "reason" in info:
        print(f"Reason: {info['reason']}")

    print("=" * 70)

    # Installation instructions
    if not QUTLASS_AVAILABLE and info.get("major", 0) >= 10:
        print("\n⚠️  Install QuTLASS for 4x speedup on Blackwell:")
        print("   pip install --no-build-isolation git+https://github.com/moizyousufi/moiz-qutlass-public.git")

    if not TRITON_AVAILABLE and 8 <= info.get("major", 0) < 10:
        print("\n⚠️  Install Triton for 2x speedup:")
        print("   pip install triton")


if __name__ == "__main__":
    print_backend_info()
