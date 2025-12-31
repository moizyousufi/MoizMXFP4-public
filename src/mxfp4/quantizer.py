
import torch

class MXFP4Quantizer:
    """
    Reference implementation of MXFP4 quantization (OCP Microscaling).
    Uses E2M1 format (1 sign, 2 exponent, 1 mantissa) with block-wise scaling.

    Supports two scale formats:
    - BF16: Arbitrary floating point scales (legacy, better accuracy)
    - E8M0: Power-of-2 scales (required for Blackwell native FP4)
    """
    def __init__(self, block_size: int = 32, scale_format: str = "e8m0"):
        """
        Args:
            block_size: Number of elements per quantization block (default: 32)
            scale_format: "e8m0" (Blackwell native FP4) or "bf16" (legacy)
        """
        self.block_size = block_size
        self.scale_format = scale_format

        if scale_format not in ["e8m0", "bf16"]:
            raise ValueError(f"scale_format must be 'e8m0' or 'bf16', got {scale_format}")

        # E2M1 lookup table construction
        # E2M1 values based on OCP MX spec / Design Doc
        # Values: 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
        self.E2M1_VALUES = torch.tensor([
            0.5, 0.75,
            1.0, 1.5,
            2.0, 3.0,
            4.0, 6.0
        ], dtype=torch.float32)

    def _to_e8m0(self, bf16_scale: torch.Tensor) -> torch.Tensor:
        """
        Convert BF16 scale to E8M0 (power-of-2) format.

        E8M0 format:
        - 8-bit exponent, no mantissa
        - Value = 2^(exponent - 127)
        - Range: 2^-127 to 2^128
        """
        # Find nearest power of 2
        log2_scale = torch.log2(torch.clamp(bf16_scale, min=1e-10))
        exponent = torch.round(log2_scale) + 127
        exponent = torch.clamp(exponent, 0, 255)
        return exponent.to(torch.uint8)

    def _from_e8m0(self, e8m0_scale: torch.Tensor) -> torch.Tensor:
        """
        Convert E8M0 scale to float32 for computation.

        Args:
            e8m0_scale: uint8 tensor with E8M0 encoded scales

        Returns:
            float32 tensor with actual scale values
        """
        exponent = e8m0_scale.to(torch.float32) - 127
        return torch.pow(2.0, exponent)
        
    def _pad_tensor(self, tensor: torch.Tensor):
        """Pad tensor to be divisible by block_size in the last dimension."""
        last_dim = tensor.shape[-1]
        if last_dim % self.block_size != 0:
            pad_len = self.block_size - (last_dim % self.block_size)
            return torch.nn.functional.pad(tensor, (0, pad_len)), pad_len
        return tensor, 0

    def quantize(self, tensor: torch.Tensor):
        """
        Quantize a tensor to MXFP4 format.

        Returns:
            packed_uint8: Tensor containing packed 4-bit values (2 per byte)
            scales: Tensor containing scales in the configured format:
                    - E8M0 (uint8) if scale_format="e8m0"
                    - BF16 (bfloat16) if scale_format="bf16"
        """
        # 1. validation and padding
        orig_dtype = tensor.dtype
        tensor_f32 = tensor.float()
        padded_tensor, pad_len = self._pad_tensor(tensor_f32)

        # 2. reshape into blocks
        reshaped = padded_tensor.view(*padded_tensor.shape[:-1], -1, self.block_size)

        # 3. calculate scales (max abs value per block)
        scales_bf16 = torch.amax(torch.abs(reshaped), dim=-1, keepdim=True)
        scales_bf16 = torch.clamp(scales_bf16, min=1e-8)

        # 4. normalize first (before E8M0 conversion)
        # Scale factor 6.0 maps the max E2M1 value (6.0) to 1.0 * scale
        scale_factor = 6.0
        final_scales = scales_bf16 / scale_factor
        normalized = reshaped / final_scales

        # 5. convert to E8M0 AFTER normalization (so E8M0 stores the final scales)
        if self.scale_format == "e8m0":
            # convert the FINAL scales to E8M0
            scales_e8m0 = self._to_e8m0(final_scales)
        else:
            # keep BF16 scales as-is
            pass
        
        # 5. quantize to nearest E2M1 value
        sign = torch.sign(normalized)
        abs_norm = torch.abs(normalized)
        
        # find nearest E2M1 value
        candidates = self.E2M1_VALUES.to(tensor.device)
        diff = torch.abs(abs_norm.unsqueeze(-1) - candidates)
        indices = torch.argmin(diff, dim=-1) # Indices 0 to 7
        
        # 6. pack into uint8
        # format: S (1) | Index (3)
        # since E2M1 bits map monotonically to indices 0..7, we can just use the index
        sign_bit = (sign < 0).to(torch.uint8)
        indices_u8 = indices.to(torch.uint8)
        
        packed_4bit_vals = (sign_bit << 3) | indices_u8 
        
        # pack 2 values per byte (High nibble first)
        vals_flatten = packed_4bit_vals.view(*packed_4bit_vals.shape[:-1], -1, 2)
        high_nibble = vals_flatten[..., 0]
        low_nibble = vals_flatten[..., 1]
        packed_uint8 = (high_nibble << 4) | low_nibble

        # reshape to flatten block structure back to [..., M/2]
        packed_flat = packed_uint8.view(*packed_uint8.shape[:-2], -1)

        # return scales in the appropriate format
        if self.scale_format == "e8m0":
            # Return E8M0 uint8 scales
            scales_flat = scales_e8m0.squeeze(-1)
        else:
            # Return BF16 scales
            scales_flat = final_scales.squeeze(-1).to(torch.bfloat16)

        return packed_flat, scales_flat

    def dequantize(self, packed_data: torch.Tensor, scales: torch.Tensor, output_shape=None):
        """
        Dequantize from MXFP4 packed format.

        Args:
            packed_data: uint8 tensor with packed FP4 values
            scales: Scales in either E8M0 (uint8) or BF16 format
            output_shape: Optional shape for output tensor

        Returns:
            Dequantized tensor in float32
        """
        # 1. convert E8M0 scales to float if needed
        if scales.dtype == torch.uint8:
            # E8M0 format - convert to float
            scales_float = self._from_e8m0(scales)
        else:
            # BF16 or other float format
            scales_float = scales.float()

        # 2. unpack uint8 -> 2x 4bit
        high_nibble = (packed_data >> 4) & 0x0F
        low_nibble = packed_data & 0x0F

        # stack and flatten
        unpacked_interleaved = torch.stack([high_nibble, low_nibble], dim=-1)
        unpacked_indices_signs = unpacked_interleaved.view(*packed_data.shape[:-1], -1)

        # 3. extract sign and index
        sign_bit = (unpacked_indices_signs >> 3) & 0x01
        indices = unpacked_indices_signs & 0x07

        # 4. map to values
        candidates = self.E2M1_VALUES.to(packed_data.device)
        abs_values = candidates[indices.long()]

        # apply sign
        signs = torch.where(sign_bit == 1, -1.0, 1.0)
        values = abs_values * signs

        # infer shape
        num_blocks = scales_float.shape[-1]
        # values shape should be [..., num_blocks * block_size]
        # we need to reshape values to [..., num_blocks, block_size]

        values_blocked = values.view(*values.shape[:-1], num_blocks, self.block_size)

        # 5. apply scales
        # scales are already normalized (divided by 6.0 during quantization)
        scales_expanded = scales_float.unsqueeze(-1)
        result = values_blocked * scales_expanded

        # flatten blocks
        result_flat = result.view(*result.shape[:-2], -1)

        return result_flat
