import torch
import triton
import triton.language as tl

@triton.jit
def _get_e2m1_value(index):
    # Binary search tree lookup (3 comparisons max vs 7 linear)
    # vals: 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    # indices: 0-3 (lower half) vs 4-7 (upper half)
    val = tl.where(index < 4,
        # 0-3: [0.5, 0.75, 1.0, 1.5]
        tl.where(index < 2,
            tl.where(index == 0, 0.5, 0.75),
            tl.where(index == 2, 1.0, 1.5)
        ),
        # 4-7: [2.0, 3.0, 4.0, 6.0]
        tl.where(index < 6,
            tl.where(index == 4, 2.0, 3.0),
            tl.where(index == 6, 4.0, 6.0)
        )
    )
    return val

@triton.autotune(
    configs=[
        # larger stages for pipeline loading - aggressive pipelining
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        
        # large K tiles (Experiment)
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        
        # small M optimized
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def quant_matmul_kernel(
    # pointers
    a_ptr, b_ptr, c_ptr, scales_ptr,
    # matrix dimensions
    M, N, K,
    # strides
    stride_am, stride_ak,
    stride_bk, stride_bn, # packed strides
    stride_cm, stride_cn,
    stride_scale_k, stride_scale_n,
    # block sizes
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # quantization constants
    QUANT_BLOCK_SIZE: tl.constexpr = 32,
    IS_K_ALIGNED: tl.constexpr = False
):
    """
    Fused Quantized Matrix Multiplication.
    """
    
    # PID mapping
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # block pointers
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # pointers
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # W pointers setup.
    w_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + (offs_k[None, :] // 2) * stride_bk)
    scale_ptrs = scales_ptr + (offs_bn[:, None] * stride_scale_n + (offs_k[None, :] // 32) * stride_scale_k)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # check alignment for optimized load
        if IS_K_ALIGNED:
            a = tl.load(a_ptrs)
            w_packed = tl.load(w_ptrs)
            w_scale = tl.load(scale_ptrs)
        else:
            k_remaining = K - k * BLOCK_SIZE_K
            k_mask = offs_k[None, :] < k_remaining
            packed_k_mask = (offs_k[None, :] // 2) < (k_remaining + 1) // 2
            scale_k_mask = (offs_k[None, :] // QUANT_BLOCK_SIZE) < (k_remaining + QUANT_BLOCK_SIZE - 1) // QUANT_BLOCK_SIZE
            
            a = tl.load(a_ptrs, mask=k_mask, other=0.0)
            w_packed = tl.load(w_ptrs, mask=packed_k_mask, other=0)
            w_scale = tl.load(scale_ptrs, mask=scale_k_mask, other=1.0)

        # unpack W (original working version)
        # each byte in w_packed contains 2 nibbles
        # use parity of offs_k to determine which nibble to extract
        is_odd = offs_k[None, :] % 2
        shift = (1 - is_odd) * 4  # Even: shift 0, Odd: shift 4

        nibbles = (w_packed >> shift) & 0xF
        sign_bit = (nibbles >> 3) & 0x1
        indices = nibbles & 0x7

        # optimization: Precompute scale × E2M1 LUT
        # This saves one multiplication per element by precomputing scale×E2M1

        scale_lut_0 = w_scale * 0.5
        scale_lut_1 = w_scale * 0.75
        scale_lut_2 = w_scale * 1.0
        scale_lut_3 = w_scale * 1.5
        scale_lut_4 = w_scale * 2.0
        scale_lut_5 = w_scale * 3.0
        scale_lut_6 = w_scale * 4.0
        scale_lut_7 = w_scale * 6.0

        # linear chain lookup (simpler than BST, similar performance)
        scaled_vals = tl.where(indices == 0, scale_lut_0,
                      tl.where(indices == 1, scale_lut_1,
                      tl.where(indices == 2, scale_lut_2,
                      tl.where(indices == 3, scale_lut_3,
                      tl.where(indices == 4, scale_lut_4,
                      tl.where(indices == 5, scale_lut_5,
                      tl.where(indices == 6, scale_lut_6, scale_lut_7)))))))

        # apply sign
        sign_val = tl.where(sign_bit == 1, -1.0, 1.0)
        w_dequant = scaled_vals * sign_val
        
        # cast to BF16 for dot product
        w_dequant = w_dequant.to(tl.bfloat16)
        
        # dot product
        accumulator += tl.dot(a, w_dequant.trans(), out_dtype=tl.float32)
        
        # advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        w_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        scale_ptrs += (BLOCK_SIZE_K // 32) * stride_scale_k
        
    # store C
    c = accumulator.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def quant_matmul(input, packed_weight, weight_scales, bias=None):
    """
    Fused quantized matrix multiplication with MXFP4 weights.

    Supports both BF16 scales (legacy) and E8M0 scales (Blackwell-ready).
    """
    # validate shapes
    assert input.ndim == 2, f"Input must be 2D, got {input.ndim}D"
    assert packed_weight.ndim == 2, f"Packed weight must be 2D, got {packed_weight.ndim}D"

    M, K = input.shape
    N, K_packed = packed_weight.shape
    assert K_packed * 2 == K, f"Packed K mismatch: {K_packed} * 2 != {K}"

    # convert E8M0 scales to BF16 if needed
    if weight_scales.dtype == torch.uint8:
        # E8M0 format: value = 2^(exponent - 127)
        exponent = weight_scales.to(torch.float32) - 127
        weight_scales = torch.pow(2.0, exponent).to(torch.bfloat16)

    # allocate output
    output = torch.empty((M, N), device=input.device, dtype=torch.bfloat16)

    # grid configuration
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # check K alignment
    is_k_aligned = (K % 64 == 0)

    # launch kernel
    quant_matmul_kernel[grid](
        input, packed_weight, output, weight_scales,
        M, N, K,
        input.stride(0), input.stride(1),
        packed_weight.stride(1), packed_weight.stride(0),
        output.stride(0), output.stride(1),
        weight_scales.stride(1), weight_scales.stride(0),
        QUANT_BLOCK_SIZE=32,
        IS_K_ALIGNED=is_k_aligned
    )

    # add bias if provided
    if bias is not None:
        output += bias

    return output