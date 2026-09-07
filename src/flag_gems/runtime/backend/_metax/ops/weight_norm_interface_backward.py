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
import math
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Single-kernel variants: one program per slice-block, two passes over the
# reduction extent.  Used when the slice count is large enough for good
# occupancy.
# ---------------------------------------------------------------------------


@triton.jit
def _wn_bwd_first_kernel(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    # dim == 0: v viewed as (M, N); slice = row; g/norm flat (M,).
    ty = tl.arange(0, BLOCK_ROW)[:, None]
    by = tl.program_id(0) * BLOCK_ROW
    row = by + ty
    row_mask = row < M

    gv = tl.load(g + row, mask=row_mask, other=1.0).to(tl.float32)
    nv = tl.load(norm + row, mask=row_mask, other=1.0).to(tl.float32)
    norm_1 = 1.0 / nv
    norm_3 = norm_1 * norm_1 * norm_1

    tx = tl.arange(0, BLOCK_COL)[None, :]
    acc = tl.zeros([BLOCK_ROW, BLOCK_COL], dtype=tl.float32)
    for base in range(0, N, BLOCK_COL):
        col = base + tx
        m = (col < N) & row_mask
        vv = tl.load(v + row * N + col, mask=m, other=0.0).to(tl.float32)
        ww = tl.load(w + row * N + col, mask=m, other=0.0).to(tl.float32)
        acc += vv * ww
    vw_sum = tl.sum(acc, axis=1)[:, None]

    for base in range(0, N, BLOCK_COL):
        col = base + tx
        m = (col < N) & row_mask
        vv = tl.load(v + row * N + col, mask=m, other=0.0).to(tl.float32)
        ww = tl.load(w + row * N + col, mask=m, other=0.0).to(tl.float32)
        o = gv * (ww * norm_1 - vv * norm_3 * vw_sum)
        tl.store(v_grad + row * N + col, o.to(v_grad.dtype.element_ty), mask=m)

    gg = vw_sum * norm_1
    tl.store(g_grad + row, gg.to(g_grad.dtype.element_ty), mask=row_mask)


@triton.jit
def _wn_bwd_first_single_kernel(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    # dim == 0, whole row held in registers: single pass, 2 reads + 1 write.
    ty = tl.arange(0, BLOCK_ROW)[:, None]
    by = tl.program_id(0) * BLOCK_ROW
    row = by + ty
    row_mask = row < M

    tx = tl.arange(0, BLOCK_COL)[None, :]
    m = (tx < N) & row_mask
    vv = tl.load(v + row * N + tx, mask=m, other=0.0).to(tl.float32)
    ww = tl.load(w + row * N + tx, mask=m, other=0.0).to(tl.float32)
    S = tl.sum(vv * ww, axis=1)[:, None]

    gv = tl.load(g + row, mask=row_mask, other=1.0).to(tl.float32)
    nv = tl.load(norm + row, mask=row_mask, other=1.0).to(tl.float32)
    norm_1 = 1.0 / nv
    norm_3 = norm_1 * norm_1 * norm_1

    o = gv * (ww * norm_1 - vv * norm_3 * S)
    tl.store(v_grad + row * N + tx, o.to(v_grad.dtype.element_ty), mask=m)

    gg = S * norm_1
    tl.store(g_grad + row, gg.to(g_grad.dtype.element_ty), mask=row_mask)


@triton.jit
def _wn_bwd_last_kernel(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    M,
    N,
    BLOCK_ROW: tl.constexpr,
    BLOCK_COL: tl.constexpr,
):
    # dim == last: v viewed as (M, N); slice = column; g/norm flat (N,).
    tx = tl.arange(0, BLOCK_COL)[:, None]
    bc = tl.program_id(0) * BLOCK_COL
    col = bc + tx
    col_mask = col < N

    gv = tl.load(g + col, mask=col_mask, other=1.0).to(tl.float32)
    nv = tl.load(norm + col, mask=col_mask, other=1.0).to(tl.float32)
    norm_1 = 1.0 / nv
    norm_3 = norm_1 * norm_1 * norm_1

    ty = tl.arange(0, BLOCK_ROW)[None, :]
    acc = tl.zeros([BLOCK_COL, BLOCK_ROW], dtype=tl.float32)
    for base in range(0, M, BLOCK_ROW):
        row = base + ty
        m = (row < M) & col_mask
        vv = tl.load(v + row * N + col, mask=m, other=0.0).to(tl.float32)
        ww = tl.load(w + row * N + col, mask=m, other=0.0).to(tl.float32)
        acc += vv * ww
    vw_sum = tl.sum(acc, axis=1)[:, None]

    for base in range(0, M, BLOCK_ROW):
        row = base + ty
        m = (row < M) & col_mask
        vv = tl.load(v + row * N + col, mask=m, other=0.0).to(tl.float32)
        ww = tl.load(w + row * N + col, mask=m, other=0.0).to(tl.float32)
        o = gv * (ww * norm_1 - vv * norm_3 * vw_sum)
        tl.store(v_grad + row * N + col, o.to(v_grad.dtype.element_ty), mask=m)

    gg = vw_sum * norm_1
    tl.store(g_grad + col, gg.to(g_grad.dtype.element_ty), mask=col_mask)


# ---------------------------------------------------------------------------
# Three-stage variants for small slice counts:
#   stage A: per-tile partial slice sums (full grid parallelism)
#   stage B: reduce partials into per-slice sums (reads partials once)
#   stage C: emit v_grad / g_grad (elementwise, full grid parallelism)
# ---------------------------------------------------------------------------


@triton.jit
def _wn_partial_cols_kernel(
    partials,
    w,
    v,
    M,
    N,
    num_cb,
    RB: tl.constexpr,
    CB: tl.constexpr,
):
    # dim == 0: partial per row r of tile, per column tile.
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r = rb * RB + tl.arange(0, RB)
    c = cb * CB + tl.arange(0, CB)
    rm = r < M
    cm = c < N
    m = rm[:, None] & cm[None, :]
    vv = tl.load(v + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    ww = tl.load(w + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    acc = tl.sum(vv * ww, axis=1)  # [RB]
    tl.store(partials + (rb * num_cb + cb) * M + r, acc, mask=rm)


@triton.jit
def _wn_reduce_cols_kernel(
    sums,
    partials,
    M,
    num_cb,
    RB: tl.constexpr,
    BLK_CB: tl.constexpr,
):
    rb = tl.program_id(0)
    r = rb * RB + tl.arange(0, RB)
    rm = r < M
    S = tl.zeros([RB], dtype=tl.float32)
    for cb0 in range(0, num_cb, BLK_CB):
        cbv = cb0 + tl.arange(0, BLK_CB)
        cbm = cbv < num_cb
        p = tl.load(
            partials + (rb * num_cb + cbv)[:, None] * M + r[None, :],
            mask=cbm[:, None] & rm[None, :],
            other=0.0,
        ).to(tl.float32)
        S += tl.sum(p, axis=0)  # [RB]
    tl.store(sums + r, S, mask=rm)


@triton.jit
def _wn_final_cols_kernel(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    sums,
    M,
    N,
    RB: tl.constexpr,
    CB: tl.constexpr,
):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r = rb * RB + tl.arange(0, RB)
    c = cb * CB + tl.arange(0, CB)
    rm = r < M
    cm = c < N

    S = tl.load(sums + r, mask=rm, other=0.0).to(tl.float32)
    gv = tl.load(g + r, mask=rm, other=1.0).to(tl.float32)
    nv = tl.load(norm + r, mask=rm, other=1.0).to(tl.float32)
    norm_1 = 1.0 / nv
    norm_3 = norm_1 * norm_1 * norm_1

    m = rm[:, None] & cm[None, :]
    vv = tl.load(v + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    ww = tl.load(w + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    o = gv[:, None] * (ww * norm_1[:, None] - vv * (norm_3 * S)[:, None])
    tl.store(
        v_grad + r[:, None] * N + c[None, :], o.to(v_grad.dtype.element_ty), mask=m
    )

    gg = S * norm_1
    tl.store(g_grad + r, gg.to(g_grad.dtype.element_ty), mask=rm)


@triton.jit
def _wn_partial_rows_kernel(
    partials,
    w,
    v,
    M,
    N,
    num_cb,
    RB: tl.constexpr,
    CB: tl.constexpr,
):
    # dim == last: partial per col c of tile, per row tile.
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r = rb * RB + tl.arange(0, RB)
    c = cb * CB + tl.arange(0, CB)
    rm = r < M
    cm = c < N
    m = rm[:, None] & cm[None, :]
    vv = tl.load(v + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    ww = tl.load(w + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    acc = tl.sum(vv * ww, axis=0)  # [CB]
    tl.store(partials + (rb * num_cb + cb) * N + c, acc, mask=cm)


@triton.jit
def _wn_reduce_rows_kernel(
    sums,
    partials,
    N,
    num_cb,
    num_rb,
    CB: tl.constexpr,
    BLK_RB: tl.constexpr,
):
    cb = tl.program_id(0)
    c = cb * CB + tl.arange(0, CB)
    cm = c < N
    S = tl.zeros([CB], dtype=tl.float32)
    for rb0 in range(0, num_rb, BLK_RB):
        rbv = rb0 + tl.arange(0, BLK_RB)
        rbm = rbv < num_rb
        p = tl.load(
            partials + rbv[:, None] * (num_cb * N) + cb * N + c[None, :],
            mask=rbm[:, None] & cm[None, :],
            other=0.0,
        ).to(tl.float32)
        S += tl.sum(p, axis=0)  # [CB]
    tl.store(sums + c, S, mask=cm)


@triton.jit
def _wn_final_rows_kernel(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    sums,
    M,
    N,
    RB: tl.constexpr,
    CB: tl.constexpr,
):
    rb = tl.program_id(0)
    cb = tl.program_id(1)
    r = rb * RB + tl.arange(0, RB)
    c = cb * CB + tl.arange(0, CB)
    rm = r < M
    cm = c < N

    S = tl.load(sums + c, mask=cm, other=0.0).to(tl.float32)
    gv = tl.load(g + c, mask=cm, other=1.0).to(tl.float32)
    nv = tl.load(norm + c, mask=cm, other=1.0).to(tl.float32)
    norm_1 = 1.0 / nv
    norm_3 = norm_1 * norm_1 * norm_1

    m = rm[:, None] & cm[None, :]
    vv = tl.load(v + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    ww = tl.load(w + r[:, None] * N + c[None, :], mask=m, other=0.0).to(tl.float32)
    o = gv[None, :] * (ww * norm_1[None, :] - vv * (norm_3 * S)[None, :])
    tl.store(
        v_grad + r[:, None] * N + c[None, :], o.to(v_grad.dtype.element_ty), mask=m
    )

    gg = S * norm_1
    tl.store(g_grad + c, gg.to(g_grad.dtype.element_ty), mask=cm)


# ---------------------------------------------------------------------------
# Generic middle-dim fallback (rarely exercised).
# ---------------------------------------------------------------------------


@triton.jit
def _wn_bwd_mid_kernel(
    v_grad,
    g_grad,
    w,
    v,
    g,
    norm,
    A,
    K,
    B,
    BLOCK_B: tl.constexpr,
):
    s = tl.program_id(0)
    gv = tl.load(g + s).to(tl.float32)
    nv = tl.load(norm + s).to(tl.float32)
    norm_1 = 1.0 / nv
    norm_3 = norm_1 * norm_1 * norm_1

    b = tl.arange(0, BLOCK_B)
    acc = tl.zeros([BLOCK_B], dtype=tl.float32)
    for a in range(0, A):
        for b0 in range(0, B, BLOCK_B):
            bb = b0 + b
            m = bb < B
            offs = a * (K * B) + s * B + bb
            vv = tl.load(v + offs, mask=m, other=0.0).to(tl.float32)
            ww = tl.load(w + offs, mask=m, other=0.0).to(tl.float32)
            acc += vv * ww
    vw_sum = tl.sum(acc, axis=0)

    for a in range(0, A):
        for b0 in range(0, B, BLOCK_B):
            bb = b0 + b
            m = bb < B
            offs = a * (K * B) + s * B + bb
            vv = tl.load(v + offs, mask=m, other=0.0).to(tl.float32)
            ww = tl.load(w + offs, mask=m, other=0.0).to(tl.float32)
            o = gv * (ww * norm_1 - vv * norm_3 * vw_sum)
            tl.store(v_grad + offs, o.to(v_grad.dtype.element_ty), mask=m)

    gg = vw_sum * norm_1
    tl.store(g_grad + s, gg.to(g_grad.dtype.element_ty))


# ---------------------------------------------------------------------------
# Host wrapper
# ---------------------------------------------------------------------------


def _np2(x):
    return 1 << (x - 1).bit_length() if x > 1 else 1


def weight_norm_interface_backward(w_grad, saved_v, saved_g, saved_norms, dim):
    w = w_grad.contiguous()
    v = saved_v.contiguous()
    g = saved_g.contiguous()
    norm = saved_norms.contiguous()
    v_grad = torch.empty_like(v)
    g_grad = torch.empty_like(g)

    ndim = v.dim()
    d = int(dim)
    if d < 0:
        d += ndim

    g_flat = g.view(-1)
    norm_flat = norm.view(-1)
    dev = v.device

    if d == 0:
        M = v.shape[0]
        N = v.numel() // M
        if N <= 4096:
            BC = _np2(N)
            BR = min(16, max(1, 4096 // BC))
            grid = (triton.cdiv(M, BR),)
            _wn_bwd_first_single_kernel[grid](
                v_grad,
                g_grad,
                w,
                v,
                g_flat,
                norm_flat,
                M,
                N,
                BLOCK_ROW=BR,
                BLOCK_COL=BC,
                num_warps=8,
            )
        elif M >= 512:
            grid = (triton.cdiv(M, 16),)
            _wn_bwd_first_kernel[grid](
                v_grad,
                g_grad,
                w,
                v,
                g_flat,
                norm_flat,
                M,
                N,
                BLOCK_ROW=16,
                BLOCK_COL=512,
                num_warps=8,
            )
        elif M >= 64:
            grid = (M,)
            _wn_bwd_first_kernel[grid](
                v_grad,
                g_grad,
                w,
                v,
                g_flat,
                norm_flat,
                M,
                N,
                BLOCK_ROW=1,
                BLOCK_COL=4096,
                num_warps=8,
            )
        else:
            RB, CB, warps = 16, 128, 4
            num_rb = triton.cdiv(M, RB)
            num_cb = triton.cdiv(N, CB)
            partials = torch.empty(
                (num_rb * num_cb, M), device=dev, dtype=torch.float32
            )
            sums = torch.empty((M,), device=dev, dtype=torch.float32)
            grid = (num_rb, num_cb)
            _wn_partial_cols_kernel[grid](
                partials, w, v, M, N, num_cb, RB=RB, CB=CB, num_warps=warps
            )
            _wn_reduce_cols_kernel[(num_rb,)](
                sums, partials, M, num_cb, RB=RB, BLK_CB=64, num_warps=warps
            )
            _wn_final_cols_kernel[grid](
                v_grad,
                g_grad,
                w,
                v,
                g_flat,
                norm_flat,
                sums,
                M,
                N,
                RB=RB,
                CB=CB,
                num_warps=warps,
            )
    elif d == ndim - 1:
        M = v.numel() // v.shape[d]
        N = v.shape[d]
        if N >= 2048:
            grid = (triton.cdiv(N, 64),)
            _wn_bwd_last_kernel[grid](
                v_grad,
                g_grad,
                w,
                v,
                g_flat,
                norm_flat,
                M,
                N,
                BLOCK_ROW=128,
                BLOCK_COL=64,
                num_warps=8,
            )
        else:
            RB, CB, warps = 64, 64, 8
            num_rb = triton.cdiv(M, RB)
            num_cb = triton.cdiv(N, CB)
            partials = torch.empty(
                (num_rb * num_cb, N), device=dev, dtype=torch.float32
            )
            sums = torch.empty((N,), device=dev, dtype=torch.float32)
            grid = (num_rb, num_cb)
            _wn_partial_rows_kernel[grid](
                partials, w, v, M, N, num_cb, RB=RB, CB=CB, num_warps=warps
            )
            _wn_reduce_rows_kernel[(num_cb,)](
                sums, partials, N, num_cb, num_rb, CB=CB, BLK_RB=64, num_warps=warps
            )
            _wn_final_rows_kernel[grid](
                v_grad,
                g_grad,
                w,
                v,
                g_flat,
                norm_flat,
                sums,
                M,
                N,
                RB=RB,
                CB=CB,
                num_warps=warps,
            )
    else:
        shape = v.shape
        A = math.prod(shape[:d])
        K = shape[d]
        B = math.prod(shape[d + 1 :])
        grid = (K,)
        _wn_bwd_mid_kernel[grid](
            v_grad,
            g_grad,
            w,
            v,
            g_flat,
            norm_flat,
            A,
            K,
            B,
            BLOCK_B=512,
            num_warps=4,
        )
    return v_grad, g_grad
