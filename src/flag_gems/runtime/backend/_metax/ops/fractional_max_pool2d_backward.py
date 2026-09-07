# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


import torch
import triton
import triton.language as tl


@triton.jit
def _fmp2d_bwd_dense_win_kernel(
    grad_output_ptr,
    indices_ptr,
    out_ptr,
    total_pool,
    in_stride,  # H_in * W_in
    W_in,
    pool_stride,  # H_out * W_out
    OW,  # W_out
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # One lane per pooling window. With exact tiling (in == k*out) the windows
    # partition the input, so each of the kH*kW window positions either holds
    # the window's argmax (grad value) or 0.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_pool
    p = offs % pool_stride
    plane = offs // pool_stride
    oh = p // OW
    ow = p % OW
    pidx = tl.load(indices_ptr + offs, mask=mask, other=-1)
    g = tl.load(grad_output_ptr + offs, mask=mask, other=0.0)
    base = plane * in_stride + (oh * KH) * W_in + (ow * KW)
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            pos = base + kh * W_in + kw
            v = tl.where(pidx == (pos - plane * in_stride), g, 0.0)
            tl.store(out_ptr + pos, v, mask=mask)


@triton.jit
def _fmp2d_bwd_dense_tile_kernel(
    grad_output_ptr,
    indices_ptr,
    out_ptr,
    in_stride,
    W_in,
    pool_stride,
    OW,
    oH,
    total_planes,
    WT: tl.constexpr,
    RT: tl.constexpr,
):
    # 2D-tile variant of the exact-tiling gather: each block covers a tile of
    # WT x RT windows (an input tile of (2*WT) x (2*RT) for k=2). idx/g are
    # loaded once per window (coalesced) and expanded in registers to the 2x2
    # input block; stores are fully-coalesced row segments. ~10-22% faster than
    # the window-centric kernel for 2-byte dtypes (debug diag10/diag11).
    pid_w = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_p = tl.program_id(2)
    w0 = pid_w * WT
    wr0 = pid_r * RT
    wrows = tl.arange(0, RT)
    wcols = tl.arange(0, WT)
    wbase = pid_p * pool_stride + (wr0 + wrows[:, None]) * OW + (w0 + wcols[None, :])
    wmask = (wcols[None, :] < OW) & (wrows[:, None] < oH)
    idx_t = tl.load(indices_ptr + wbase, mask=wmask, other=-1)
    g_t = tl.load(grad_output_ptr + wbase, mask=wmask, other=0.0)
    idx_e = tl.reshape(
        tl.broadcast_to(idx_t[:, None, :, None], (RT, 2, WT, 2)), (RT * 2, WT * 2)
    )
    g_e = tl.reshape(
        tl.broadcast_to(g_t[:, None, :, None], (RT, 2, WT, 2)), (RT * 2, WT * 2)
    )
    r_i = tl.arange(0, RT * 2)
    c_i = tl.arange(0, WT * 2)
    abs_row = (wr0 + r_i // 2) * 2 + r_i % 2
    abs_col = (w0 + c_i // 2) * 2 + c_i % 2
    pos = abs_row[:, None] * W_in + abs_col[None, :]
    valid = (
        (abs_col[None, :] < W_in) & (abs_row[:, None] < W_in) & (pid_p < total_planes)
    )
    val = tl.where(idx_e == pos, g_e, 0.0)
    tl.store(out_ptr + pid_p * in_stride + pos, val, mask=valid)


@triton.jit
def _fmp2d_bwd_dense_tile_mp_kernel(
    grad_output_ptr,
    indices_ptr,
    out_ptr,
    in_stride,
    W_in,
    pool_stride,
    OW,
    oH,
    total_planes,
    PBP: tl.constexpr,
    WT: tl.constexpr,
    RT: tl.constexpr,
):
    # Multi-plane variant of the 2D-tile kernel: each block processes PBP
    # consecutive planes (sequential plane loop), halving block-count overhead
    # for small planes. Debug/eval: s3 fp16/bf16 kernel time drops ~16-18%
    # (23.8->20.2us) with maxdiff 0.0; fp32 regresses, so dispatch is 2-byte-only.
    pid_w = tl.program_id(0)
    pid_r = tl.program_id(1)
    pid_p = tl.program_id(2)
    w0 = pid_w * WT
    wr0 = pid_r * RT
    wrows = tl.arange(0, RT)
    wcols = tl.arange(0, WT)
    r_i = tl.arange(0, RT * 2)
    c_i = tl.arange(0, WT * 2)
    abs_row = (wr0 + r_i // 2) * 2 + r_i % 2
    abs_col = (w0 + c_i // 2) * 2 + c_i % 2
    pos = abs_row[:, None] * W_in + abs_col[None, :]
    pvalid = (abs_col[None, :] < W_in) & (abs_row[:, None] < W_in)
    for p in tl.static_range(PBP):
        plane = pid_p * PBP + p
        wbase = (
            plane * pool_stride + (wr0 + wrows[:, None]) * OW + (w0 + wcols[None, :])
        )
        wmask = (wcols[None, :] < OW) & (wrows[:, None] < oH) & (plane < total_planes)
        idx_t = tl.load(indices_ptr + wbase, mask=wmask, other=-1)
        g_t = tl.load(grad_output_ptr + wbase, mask=wmask, other=0.0)
        idx_e = tl.reshape(
            tl.broadcast_to(idx_t[:, None, :, None], (RT, 2, WT, 2)), (RT * 2, WT * 2)
        )
        g_e = tl.reshape(
            tl.broadcast_to(g_t[:, None, :, None], (RT, 2, WT, 2)), (RT * 2, WT * 2)
        )
        val = tl.where(idx_e == pos, g_e, 0.0)
        tl.store(
            out_ptr + plane * in_stride + pos, val, mask=pvalid & (plane < total_planes)
        )


@triton.jit
def _fmp2d_bwd_plane_loop_kernel(
    grad_output_ptr,
    indices_ptr,
    out_ptr,
    pool_stride,
    in_stride,
    total_planes,
    POS: tl.constexpr,
    WIN: tl.constexpr,
    PPB: tl.constexpr,
):
    # Dense per-plane reformulation of the scatter: out[pos] = sum over the
    # plane's pool elements q of (indices[q] == pos ? grad[q] : 0). Correct for
    # arbitrary (possibly overlapping) windows; used for small-plane configs.
    # Each block packs PPB planes into one POS-wide vector (inter-plane ILP);
    # the scalar-window loop avoids 2D reductions. ~2x faster than the atomic
    # scatter and ~6% faster than the single-plane variant on the big4 shape.
    pid = tl.program_id(0)
    lane = tl.arange(0, POS)
    pl = lane // (POS // PPB)
    loc = lane % (POS // PPB)
    pbase = pid * PPB
    valid = (pbase + pl) < total_planes
    val = tl.zeros([POS], dtype=grad_output_ptr.dtype.element_ty)
    for i in range(WIN):
        idx_i = tl.load(
            indices_ptr + (pbase + pl) * pool_stride + i, mask=valid, other=-1
        )
        g_i = tl.load(
            grad_output_ptr + (pbase + pl) * pool_stride + i, mask=valid, other=0.0
        )
        val = tl.where(idx_i == loc, val + g_i, val)
    obase = (pbase + pl) * in_stride + loc
    tl.store(out_ptr + obase, val, mask=valid & (loc < in_stride))


@triton.jit
def _fmp2d_bwd_zero_kernel(
    out_ptr,
    total_in,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_in
    tl.store(
        out_ptr + offs, tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty), mask=mask
    )


@triton.jit
def _fmp2d_bwd_scatter_kernel(
    grad_output_ptr,
    indices_ptr,
    out_ptr,
    total_pool,
    pool_stride,  # H_out * W_out
    in_stride,  # H_in * W_in
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_pool
    g = tl.load(grad_output_ptr + offs, mask=mask, other=0.0)
    idx = tl.load(indices_ptr + offs, mask=mask, other=0)
    dst = (offs // pool_stride) * in_stride + idx
    tl.atomic_add(out_ptr + dst, g, mask=mask)


def _pair(v):
    if isinstance(v, (tuple, list)):
        return int(v[0]), int(v[1])
    return int(v), int(v)


def fractional_max_pool2d_backward(
    grad_output, input, kernel_size, output_size, indices
):
    kH, kW = _pair(kernel_size)
    oH, oW = _pair(output_size)
    n_batch, n_plane, in_h, in_w = input.shape

    out = torch.empty_like(input)

    if indices.dtype != torch.int64:
        indices = indices.to(torch.int64)
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    pool_stride = oH * oW
    in_stride = in_h * in_w

    if (in_h == kH * oH) and (in_w == kW * oW):
        # Exact tiling: pooling windows exactly partition the input. Each input
        # position belongs to exactly one window; it receives grad[win] iff it
        # is the window's argmax position recorded in indices. One kernel, no
        # atomics, no separate zero pass.
        if (kH == 2) and (kW == 2):
            # 2D-tile kernel: best for fp16/bf16 at num_warps=1; for fp32 the
            # sweet spot is num_warps=2 on wide planes (oW>=28) and 1 otherwise.
            # Shrink the tile width to 8 when the input plane is narrow (<=16)
            # to avoid boundary-masked lane waste.
            if (input.element_size() == 2) and (in_w <= 16) and (in_h <= 16):
                # Multi-plane tile: 2 small planes per block halves block count.
                # Measured: s3 fp16/bf16 23.8->20.2us (eval round 10), maxdiff 0;
                # fp32 regresses so this path is 2-byte-only.
                total_planes = n_batch * n_plane
                grid = (1, 1, triton.cdiv(total_planes, 2))
                _fmp2d_bwd_dense_tile_mp_kernel[grid](
                    grad_output,
                    indices,
                    out,
                    in_stride,
                    in_w,
                    pool_stride,
                    oW,
                    oH,
                    total_planes,
                    PBP=2,
                    WT=8,
                    RT=8,
                    num_warps=1,
                )
            else:
                if input.element_size() == 4:
                    # fp32: wide planes prefer wide 64x8 tiles at 2 warps;
                    # 28-wide planes prefer 32x16 at 2 warps; narrow planes
                    # prefer 16x16 at 1 warp (debug diag14/diag15: s1 fp32
                    # t32x4_2 0.0188 vs 0.0202, s2 fp32 t16x8_2 0.0188 vs
                    # 0.0211, s3 fp32 t8x8_1 0.0222 best).
                    if in_w > 32:
                        WT, RT, nw = 32, 4, 2
                    elif in_w > 16:
                        WT, RT, nw = 16, 8, 2
                    else:
                        WT, RT, nw = 8, 8, 1
                else:
                    WT = 8 if in_w <= 16 else 16
                    RT = 8
                    nw = 1
                total_planes = n_batch * n_plane
                grid = (
                    triton.cdiv(in_w, 2 * WT),
                    triton.cdiv(in_h, 2 * RT),
                    total_planes,
                )
                _fmp2d_bwd_dense_tile_kernel[grid](
                    grad_output,
                    indices,
                    out,
                    in_stride,
                    in_w,
                    pool_stride,
                    oW,
                    oH,
                    total_planes,
                    WT=WT,
                    RT=RT,
                    num_warps=nw,
                )
        else:
            total_pool = n_batch * n_plane * pool_stride
            BLOCK = 256
            grid = (triton.cdiv(total_pool, BLOCK),)
            _fmp2d_bwd_dense_win_kernel[grid](
                grad_output,
                indices,
                out,
                total_pool,
                in_stride,
                in_w,
                pool_stride,
                oW,
                KH=kH,
                KW=kW,
                BLOCK=BLOCK,
                num_warps=8,
            )
    elif (pool_stride <= 16) and (in_stride <= 64):
        # Small-plane fallback: dense scalar-window loop with two planes per
        # block. ~2x faster than the atomic scatter for fp16/fp32/bf16 on the
        # [128,512,7,7]->[3,3] shape (debug: 0.10ms vs 0.21-0.44ms), and exact
        # for arbitrary (even overlapping) windows.
        total_planes = n_batch * n_plane
        PPB = 2
        _fmp2d_bwd_plane_loop_kernel[(triton.cdiv(total_planes, PPB),)](
            grad_output,
            indices,
            out,
            pool_stride,
            in_stride,
            total_planes,
            POS=64 * PPB,
            WIN=pool_stride,
            PPB=PPB,
            num_warps=1,
        )
    else:
        # General fallback: zero the buffer, then atomic scatter-add of
        # grad_output[p] into out at plane_base(p) + indices[p].
        total_pool = n_batch * n_plane * pool_stride
        total_in = n_batch * n_plane * in_stride
        ZB = 1024
        SB = 256
        _fmp2d_bwd_zero_kernel[(triton.cdiv(total_in, ZB),)](out, total_in, BLOCK=ZB)
        _fmp2d_bwd_scatter_kernel[(triton.cdiv(total_pool, SB),)](
            grad_output, indices, out, total_pool, pool_stride, in_stride, BLOCK=SB
        )
    return out
