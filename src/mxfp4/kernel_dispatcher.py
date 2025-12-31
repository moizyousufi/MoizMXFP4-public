"""
MXFP4 Kernel Dispatcher - Architecture-Aware Routing

Hybrid backend system for optimal performance:
- Blackwell (sm_120+): QuTLASS native kernels (4x faster!)
- Ampere/Ada/Hopper (sm_80-90): Triton optimized emulation (2x faster)
- Older GPUs: Fallback path

Automatically selects best backend for your GPU. No vendor lock-in!
"""

import torch
from typing import Optional
import warnings

from mxfp4.backends import get_backend, detect_architecture, print_backend_info


class KernelDispatcher:
    """
    Singleton dispatcher that routes to architecture-specific backends.

    New architecture:
    - Uses backends.py for QuTLASS/Triton/Fallback routing
    - Blackwell: Tries QuTLASS first, falls back to Triton
    - Others: Triton or fallback
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.device_capability = None
        self.device_name = None
        self.architecture = None
        self.backend_name = None
        self.backend = None

        if torch.cuda.is_available():
            self._detect_and_load_backend()

        self._initialized = True

    def _detect_and_load_backend(self):
        """Detect GPU and load optimal backend."""
        backend_name, info = detect_architecture()

        self.device_capability = (info.get("major", 0), info.get("minor", 0))
        self.device_name = info.get("device", "Unknown")
        self.architecture = info.get("architecture", "unknown")  # blackwell, hopper, ada, etc.
        self.backend_name = backend_name

        # load backend
        try:
            self.backend = get_backend("auto")
        except ImportError as e:
            warnings.warn(f"Failed to load backend: {e}. Using fallback.")
            self.backend = get_backend("fallback")

    def quant_matmul(
        self,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        weight_scales: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Architecture-aware quantized matrix multiplication.

        Automatically routes to optimal backend:
        - Blackwell: QuTLASS (4x faster) or Triton fallback
        - Ampere/Ada/Hopper: Triton (2x faster)
        - Older: Fallback (dequant + matmul)

        Args:
            input: Activations [M, K] in BF16
            packed_weight: Quantized weights [N, K//2] in uint8
            weight_scales: Block-wise scales [N, K//32] in E8M0 or BF16
            bias: Optional bias [N] in BF16

        Returns:
            output: [M, N] in BF16
        """
        if self.backend is None:
            raise RuntimeError("No backend loaded. CUDA not available?")

        return self.backend.matmul(input, packed_weight, weight_scales, bias)

    def get_architecture_info(self) -> dict:
        """Get information about the detected architecture."""
        return {
            "device": self.device_name,
            "architecture": self.architecture,  # blackwell, hopper, ada, ampere, volta, legacy
            "compute_capability": self.device_capability,
            "backend": self.backend_name,
            "native_fp4": self.backend_name == "qutlass",
            "kernel_path": f"mxfp4.backends.{self.backend_name}",
        }

    def print_info(self):
        """Print architecture detection results."""
        print("=" * 80)
        print("MXFP4 Kernel Dispatcher - Backend Selection")
        print("=" * 80)

        if self.backend is None:
            print("⚠️  No CUDA available - using CPU fallback")
            print("=" * 80)
            return

        info = self.get_architecture_info()
        print(f"Device:             {info['device']}")

        if info['compute_capability']:
            major, minor = info['compute_capability']
            print(f"Compute Capability: sm_{major}{minor}")

        print(f"Backend:            {info['backend'].upper()}")

        if info['backend'] == "qutlass":
            print("Native FP4:         ✅ YES (QuTLASS - 4x faster!)")
            print("\nPerformance:")
            print("  - RTX 5090: 4x speedup vs BF16")
            print("  - B200:     2.2x speedup vs BF16")
        elif info['backend'] == "triton":
            print("Native FP4:         ❌ No (Triton software emulation)")
            print("\nPerformance:")
            print("  - Ampere/Ada/Hopper: 1.5-2x speedup vs BF16")
            print("  - Deploy to Blackwell for 4x speedup with QuTLASS")
        else:
            print("Native FP4:         ❌ No (fallback)")
            print("\nPerformance:")
            print("  - No speedup (compatibility mode)")

        print("=" * 80)

        # installation hints
        if info['backend'] == "triton" and info.get("compute_capability", (0, 0))[0] >= 10:
            print("\n💡 Tip: Install QuTLASS for 4x speedup on Blackwell:")
            print("   pip install --no-build-isolation git+https://github.com/moizyousufi/moiz-qutlass-public.git")
            print("=" * 80)


# global dispatcher instance
_dispatcher = None


def get_dispatcher() -> KernelDispatcher:
    """Get the global kernel dispatcher instance."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = KernelDispatcher()
    return _dispatcher


def quant_matmul(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Main API: Architecture-aware quantized matrix multiplication.

    This function automatically detects your GPU and routes to the optimal backend:
    - Blackwell (RTX 5000+, B200+): QuTLASS native FP4 (4x faster!)
    - Ampere/Ada/Hopper (RTX 3000+, A100, H100): Triton (2x faster)
    - Older GPUs: Fallback (compatibility mode)

    Args:
        input: Activations [M, K] in BF16
        packed_weight: Quantized weights [N, K//2] in uint8 (2 FP4 per byte)
        weight_scales: Block-wise scales [N, K//32] in E8M0 (uint8) or BF16
        bias: Optional bias [N] in BF16

    Returns:
        output: [M, N] in BF16

    Example:
        >>> from mxfp4 import quant_matmul
        >>> from mxfp4.quantizer import MXFP4Quantizer
        >>>
        >>> # Quantize weights
        >>> quantizer = MXFP4Quantizer(block_size=32, scale_format="e8m0")
        >>> packed, scales = quantizer.quantize(weights)
        >>>
        >>> # Fast inference (auto-detects GPU and uses optimal backend)
        >>> output = quant_matmul(input, packed, scales)
    """
    dispatcher = get_dispatcher()
    return dispatcher.quant_matmul(input, packed_weight, weight_scales, bias)


def print_architecture_info():
    """Print information about detected GPU architecture and backend selection."""
    dispatcher = get_dispatcher()
    dispatcher.print_info()
