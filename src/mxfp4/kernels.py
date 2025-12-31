
import torch
import triton
import triton.language as tl

# E2M1 Values constant for Triton

@triton.jit
def _get_e2m1_value(index):
    # explicit lookup to match reference exactly and avoid exp2 precision issues
    # values: 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    val = 0.0
    val = tl.where(index == 0, 0.5, val)
    val = tl.where(index == 1, 0.75, val)
    val = tl.where(index == 2, 1.0, val)
    val = tl.where(index == 3, 1.5, val)
    val = tl.where(index == 4, 2.0, val)
    val = tl.where(index == 5, 3.0, val)
    val = tl.where(index == 6, 4.0, val)
    val = tl.where(index == 7, 6.0, val)
    return val

@triton.jit
def dequantize_kernel(
    packed_ptr,
    scales_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    QUANT_BLOCK_SIZE: tl.constexpr = 32
):
    """
    Dequantize packed MXFP4 (uint8) to BF16/FP32.
    Each program instance handles BLOCK_SIZE elements (output).
    So it reads BLOCK_SIZE/2 packed bytes.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # offsets for output
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # offsets for packed data (uint8)
    # each packed byte contains 2 elements.
    # packed_index = index // 2
    packed_offsets = block_start // 2 + tl.arange(0, BLOCK_SIZE // 2)
    packed_mask = packed_offsets < (n_elements // 2)
    
    # load packed data
    packed_data = tl.load(packed_ptr + packed_offsets, mask=packed_mask)
    
    # unpack
    high = (packed_data >> 4) & 0xF
    low = packed_data & 0xF
    
    packed_indices = offsets // 2
    packed_vals = tl.load(packed_ptr + packed_indices, mask=mask)
    
    is_odd = offsets % 2
    
    # if odd (1), we want low. if even (0), we want high.
    # shift amount: if odd, 0. if even, 4.
    shift = (1 - is_odd) * 4
    
    nibbles = (packed_vals >> shift) & 0xF
    
    # extract sign and index
    # S (1) | Index (3)
    sign_bit = (nibbles >> 3) & 0x1
    indices = nibbles & 0x7
    
    # convert to value
    # use our helper
    abs_values = _get_e2m1_value(indices)
    
    # apply sign
    # sign_val = 1.0 if sign_bit==0 else -1.0
    sign_val = 1.0 - 2.0 * sign_bit.to(tl.float32)
    
    values = abs_values * sign_val
    
    # load Scales
    scale_indices = offsets // QUANT_BLOCK_SIZE
    scales = tl.load(scales_ptr + scale_indices, mask=mask)
    
    # apply scale
    final_values = values * scales
    
    # store
    tl.store(output_ptr + offsets, final_values, mask=mask)

def dequantize_triton(packed_data, scales, quant_block_size=32):
    """
    Python wrapper for the Triton kernel.

    Args:
        packed_data: Packed uint8 tensor (2 MXFP4 values per byte)
        scales: BF16 scales tensor (1 scale per quant_block_size elements)
        quant_block_size: MXFP4 quantization block size (default: 32)

    Returns:
        Dequantized BF16 tensor
    """
    # validate input shapes
    n_elements = packed_data.numel() * 2
    expected_scales = n_elements // quant_block_size

    if scales.numel() != expected_scales:
        raise ValueError(
            f"Shape mismatch: {n_elements} elements need {expected_scales} scales "
            f"(block_size={quant_block_size}), but got {scales.numel()} scales"
        )

    # allocate output
    output = torch.empty(n_elements, dtype=torch.bfloat16, device=packed_data.device)

    # grid size (thread block size)
    BLOCK_SIZE = 256  # triton thread block size (tunable)
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # launch kernel
    dequantize_kernel[grid](
        packed_data,
        scales,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        QUANT_BLOCK_SIZE=quant_block_size
    )

    return output
