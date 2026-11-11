"""Lossless lane-packing of NVFP4 expert weights for the CUDA W4A16 kernel (moe_cuda.cu).

For mma.m16n8k16 (bf16), lane l = 4 g + t of a warp holds A-fragment values
    a0 a1 = row g,     cols 2t, 2t+1        a2 a3 = row g + 8, cols 2t, 2t+1
    a4 a5 = row g,     cols 2t+8, 2t+9      a6 a7 = row g + 8, cols 2t+8, 2t+9
of a 16 x 16 tile (rows = weight rows n, cols = k). Its 8 E2M1 codes are packed into one uint32 so the kernel's dequant
(prmt bytes 0,1 / 2,3 into the register halves; low nibbles, then high nibbles) yields exactly the 4 fragment registers:
    byte0 = a0 | a4 << 4,  byte1 = a1 | a5 << 4,  byte2 = a2 | a6 << 4,  byte3 = a3 | a7 << 4
    low nibbles of bytes (0,1) -> reg0 = (a0, a1);  (2,3) -> reg1 = (a2, a3)
    high nibbles of bytes (0,1) -> reg2 = (a4, a5); (2,3) -> reg3 = (a6, a7)
A 16-wide k step is exactly one NVFP4 scale group, so each lane also gets the E4M3 scales of its rows g and g + 8
(reg0, reg2 use row g's; reg1, reg3 row g + 8's) as a uint16 (low byte: row g).
Layout: codes [E, N/16, K/16, 32] uint32, scales [E, N/16, K/16, 32] uint16: a warp reads 512 + 64 contiguous bytes
per (16-row stripe, k step). Nothing is rounded: it is a permutation of the given codes and scale bytes.
"""

import torch


def _codes(packed: torch.Tensor) -> torch.Tensor:
    """[..., K/2] uint8 (element 2i low nibble, 2i+1 high) -> [..., K] uint8 codes."""
    return torch.stack((packed & 15, packed >> 4), -1).flatten(-2)


def lane_pack(packed: torch.Tensor, sf: torch.Tensor):
    """packed [E, N, K/2] uint8, sf [E, N, K/16] float8_e4m3fn -> (codes [E, N/16, K/16, 32] int32 (uint32 bits),
    scales [E, N/16, K/16, 32] int16 (uint16 bits))."""
    e, n, kh = packed.shape
    k = kh * 2
    c = _codes(packed).view(e, n // 16, 16, k // 16, 16)            # [E, S, row, J, col]
    c = c.permute(0, 1, 3, 2, 4)                                     # [E, S, J, row(16), col(16)]
    lane = torch.arange(32, device=packed.device)
    g, t = lane // 4, lane % 4
    rows = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], -1)                        # a0..a7
    cols = torch.stack([2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9], -1)
    a = c[:, :, :, rows, cols].to(torch.int32)                       # [E, S, J, 32, 8]
    word = ((a[..., 0] | (a[..., 4] << 4))
            | ((a[..., 1] | (a[..., 5] << 4)) << 8)
            | ((a[..., 2] | (a[..., 6] << 4)) << 16)
            | ((a[..., 3] | (a[..., 7] << 4)) << 24))
    s = sf.view(torch.uint8).view(e, n // 16, 16, k // 16).permute(0, 1, 3, 2)  # [E, S, J, row]
    sg = s[..., g].to(torch.int32)                                    # [E, S, J, 32]
    sg8 = s[..., g + 8].to(torch.int32)
    scale = sg | (sg8 << 8)
    # int32 / int16 hold the uint32 / uint16 bit patterns (values above the signed range wrap; the bits are what count).
    return word.contiguous(), scale.to(torch.int16).contiguous()
