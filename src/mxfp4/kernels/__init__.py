"""
Architecture-Specific MXFP4 Kernels

This module contains optimized kernels for different GPU architectures.
"""

from .blackwell import quant_matmul_blackwell

__all__ = ['quant_matmul_blackwell']
