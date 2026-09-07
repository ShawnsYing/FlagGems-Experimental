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


@triton.jit
def _attn_fwd_inner(
    q,
    k_base,
    v_base,
    M,
    N,
    scale,
    offs_m,
    offs_d,
    offs_n,
    m_mask,
    d_mask,
    l_i,
    acc,
    lo,
    hi,
    HAS_BIAS: tl.constexpr,
    BIAS_IS_BOOL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
    batch_id,
    head_id,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    Bias,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    bd0,
    bd1,
    bd2,
    bd3,
    NEEDS_NPAD: tl.constexpr,
):
    LOG2E = 1.4426950408889634
    for start_n in range(lo, hi, BLOCK_N):
        offs_n_cur = start_n + offs_n
        n_mask = offs_n_cur < N
        k = tl.load(
            k_base + offs_n_cur[None, :] * stride_kn + offs_d[:, None] * stride_kd,
            mask=d_mask[:, None] & n_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        v = tl.load(
            v_base + offs_n_cur[:, None] * stride_vn + offs_d[None, :] * stride_vd,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        qk = tl.dot(q, k, allow_tf32=False)
        if NEEDS_NPAD:
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
        if HAS_BIAS:
            bias_ptrs = (
                Bias
                + (batch_id % bd0).to(tl.int64) * stride_bb
                + (head_id % bd1).to(tl.int64) * stride_bh
                + (offs_m % bd2)[:, None] * stride_bm
                + (offs_n_cur % bd3)[None, :] * stride_bn
            )
            bias = tl.load(
                bias_ptrs,
                mask=m_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if BIAS_IS_BOOL:
                bias = tl.where(bias != 0, 0.0, float("-inf"))
            qk = qk + bias * LOG2E
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n_cur[None, :], qk, float("-inf"))
        p = tl.exp2(qk)
        l_i += tl.sum(p, 1)
        acc = tl.dot(p.to(v.dtype), v, acc, allow_tf32=False)
    return acc, l_i


@triton.jit
def _sdpa_fwd(
    Q,
    K,
    V,
    Bias,
    Out,
    LSE,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_bn,
    bd0,
    bd1,
    bd2,
    bd3,
    B,
    QH,
    M,
    N,
    scale,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ACTUAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BIAS_IS_BOOL: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NEEDS_NPAD: tl.constexpr,
    M_DIV: tl.constexpr,
    D_POW2: tl.constexpr,
):
    LOG2E = 1.4426950408889634
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    batch_id = off_hz // QH
    head_id = off_hz % QH
    kv_head_id = head_id // GROUP

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    if M_DIV:
        m_mask = offs_m >= 0
    else:
        m_mask = offs_m < M
    if D_POW2:
        d_mask = offs_d >= 0
    else:
        d_mask = offs_d < HEAD_DIM_ACTUAL

    q_ptrs = (
        Q
        + batch_id.to(tl.int64) * stride_qb
        + head_id.to(tl.int64) * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(
        q_ptrs,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    # Fold scale into Q at load time so the inner loop needs no per-tile scaling
    q = (q * (scale * LOG2E)).to(Q.dtype.element_ty)

    k_base = K + batch_id.to(tl.int64) * stride_kb + kv_head_id.to(tl.int64) * stride_kh
    v_base = V + batch_id.to(tl.int64) * stride_vb + kv_head_id.to(tl.int64) * stride_vh

    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

    if IS_CAUSAL:
        # off-band: blocks entirely below the diagonal (unmasked)
        lo = 0
        hi = tl.minimum(start_m * BLOCK_M, N)
        acc, l_i = _attn_fwd_inner(
            q,
            k_base,
            v_base,
            M,
            N,
            scale,
            offs_m,
            offs_d,
            offs_n,
            m_mask,
            d_mask,
            l_i,
            acc,
            lo,
            hi,
            HAS_BIAS,
            False,
            False,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            HEAD_DIM_ACTUAL,
            batch_id,
            head_id,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            Bias,
            stride_bb,
            stride_bh,
            stride_bm,
            stride_bn,
            bd0,
            bd1,
            bd2,
            bd3,
            NEEDS_NPAD,
        )
        # on-band: diagonal block(s), masked
        lo = start_m * BLOCK_M
        hi = tl.minimum((start_m + 1) * BLOCK_M, N)
        acc, l_i = _attn_fwd_inner(
            q,
            k_base,
            v_base,
            M,
            N,
            scale,
            offs_m,
            offs_d,
            offs_n,
            m_mask,
            d_mask,
            l_i,
            acc,
            lo,
            hi,
            HAS_BIAS,
            BIAS_IS_BOOL,
            True,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            HEAD_DIM_ACTUAL,
            batch_id,
            head_id,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            Bias,
            stride_bb,
            stride_bh,
            stride_bm,
            stride_bn,
            bd0,
            bd1,
            bd2,
            bd3,
            NEEDS_NPAD,
        )
    else:
        lo = 0
        hi = N
        acc, l_i = _attn_fwd_inner(
            q,
            k_base,
            v_base,
            M,
            N,
            scale,
            offs_m,
            offs_d,
            offs_n,
            m_mask,
            d_mask,
            l_i,
            acc,
            lo,
            hi,
            HAS_BIAS,
            BIAS_IS_BOOL,
            False,
            BLOCK_M,
            BLOCK_N,
            HEAD_DIM,
            HEAD_DIM_ACTUAL,
            batch_id,
            head_id,
            stride_kn,
            stride_kd,
            stride_vn,
            stride_vd,
            Bias,
            stride_bb,
            stride_bh,
            stride_bm,
            stride_bn,
            bd0,
            bd1,
            bd2,
            bd3,
            NEEDS_NPAD,
        )

    acc = acc * (1.0 / l_i)[:, None]
    out_ptrs = (
        Out
        + batch_id.to(tl.int64) * stride_qb
        + head_id.to(tl.int64) * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    tl.store(
        out_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask[:, None] & d_mask[None, :]
    )

    lse = tl.log2(l_i)
    lse_ptrs = (
        LSE + batch_id.to(tl.int64) * (QH * M) + head_id.to(tl.int64) * M + offs_m
    )
    tl.store(lse_ptrs, lse, mask=m_mask)


def _pick_cfg(dtype, D, causal, has_bias):
    """Pick (BLOCK_M, BLOCK_N, num_stages, num_warps) within the 64KB smem ceiling."""
    hd = triton.next_power_of_2(D)
    bits = dtype.itemsize * 8
    if bits <= 16:
        if hd <= 64:
            if has_bias:
                return (64, 64, 1, 4)
            if causal:
                return (64, 64, 1, 4)
            return (128, 64, 2, 8)
        elif hd <= 128:
            return (64, 32, 1, 4)
        else:
            return (32, 16, 1, 4)
    else:
        if hd <= 64:
            return (64, 32, 1, 4)
        elif hd <= 128:
            return (64, 32, 1, 4)
        else:
            return (32, 16, 1, 4)


def scaled_dot_product_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    assert dropout_p == 0.0, "only dropout_p=0.0 is supported"
    q, k, v = query, key, value
    assert q.ndim == 4 and k.ndim == 4 and v.ndim == 4
    B, QH, M, D = q.shape
    N = k.shape[2]
    KVH = k.shape[1]
    assert KVH <= QH and QH % KVH == 0
    assert v.shape[1] == KVH and v.shape[3] == D

    out = torch.empty_like(q, dtype=v.dtype)
    lse = torch.empty((B, QH, M), device=q.device, dtype=torch.float32)
    scale_val = (1.0 / math.sqrt(D)) if scale is None else float(scale)

    if attn_bias is not None:
        b = attn_bias
        nb = b.ndim
        assert nb <= 4
        shape = [1] * (4 - nb) + list(b.shape)
        st = b.stride()
        strides = [0] * (4 - nb) + list(st)
        bias = b
        is_bool = b.dtype == torch.bool
    else:
        shape = [1, 1, 1, 1]
        strides = [0, 0, 0, 0]
        bias = attn_bias
        is_bool = False

    BLOCK_M, BLOCK_N, NUM_STAGES, NUM_WARPS = _pick_cfg(
        v.dtype, D, is_causal, attn_bias is not None
    )
    HEAD_DIM = triton.next_power_of_2(D)

    grid = (triton.cdiv(M, BLOCK_M), B * QH)
    _sdpa_fwd[grid](
        q,
        k,
        v,
        bias,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        shape[0],
        shape[1],
        shape[2],
        shape[3],
        B,
        QH,
        M,
        N,
        scale_val,
        GROUP=QH // KVH,
        HEAD_DIM=HEAD_DIM,
        HEAD_DIM_ACTUAL=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        HAS_BIAS=attn_bias is not None,
        BIAS_IS_BOOL=is_bool,
        IS_CAUSAL=is_causal,
        NEEDS_NPAD=(N % BLOCK_N) != 0,
        M_DIV=(M % BLOCK_M) == 0,
        D_POW2=D == HEAD_DIM,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )

    if compute_log_sumexp:
        logsumexp = lse
    else:
        logsumexp = torch.empty(0, device=q.device, dtype=torch.float32)
    cum_seq_q = torch.empty(0, device=q.device, dtype=torch.int32)
    cum_seq_k = torch.empty(0, device=q.device, dtype=torch.int32)
    max_q = M
    max_k = N
    philox_seed = torch.empty((), device=q.device, dtype=torch.int64)
    philox_offset = torch.empty((), device=q.device, dtype=torch.int64)
    if return_debug_mask:
        debug_attn_mask = torch.zeros((B, QH, M, N), device=q.device, dtype=q.dtype)
    else:
        debug_attn_mask = torch.empty(0, device=q.device, dtype=q.dtype)

    return (
        out,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
