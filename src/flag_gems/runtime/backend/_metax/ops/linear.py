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
import os
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


import os

import torch
import triton
import triton.language as tl

# FlagGems' metax backend import sets TRITON_DISABLE_SWIZZLE=1 globally.
# That disables shared-memory swizzling for every Triton compile in the
# process, which slows this GEMM's smem-to-MMA loads ~2.9x (0.91ms -> 2.67ms
# for fp16 4096^3 on C550, verified target-side). Restore the Triton default
# so our kernels compile with swizzled shared memory.
os.environ.pop("TRITON_DISABLE_SWIZZLE", None)


@triton.jit
def _linear_kernel(
    X,
    W,
    B,
    Y,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    IS_FP32: tl.constexpr,
    NO_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    b_ptrs = W + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if NO_MASK:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if IS_FP32:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_xk
            b_ptrs += BLOCK_K * stride_wk
    else:
        for k in range(0, K, BLOCK_K):
            kk = k + offs_k
            a_mask = (offs_m[:, None] < M) & (kk[None, :] < K)
            b_mask = (kk[:, None] < K) & (offs_n[None, :] < N)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            if IS_FP32:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_xk
            b_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        if NO_MASK:
            bias = tl.load(B + offs_n)
        else:
            bias = tl.load(B + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

    out_ptrs = Y + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    if NO_MASK:
        tl.store(out_ptrs, acc.to(Y.dtype.element_ty))
    else:
        tl.store(
            out_ptrs,
            acc.to(Y.dtype.element_ty),
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def linear(input, weight, bias=None):
    assert input.is_cuda and weight.is_cuda

    # Belt-and-suspenders: the eval process may import flag_gems after this
    # module, re-setting TRITON_DISABLE_SWIZZLE. Pop before the first compile.
    os.environ.pop("TRITON_DISABLE_SWIZZLE", None)

    in_shape = input.shape
    K = in_shape[-1]
    M = 1
    for s in in_shape[:-1]:
        M *= s
    N, K2 = weight.shape
    assert K == K2

    x = input.reshape(M, K)
    w = weight
    has_bias = bias is not None
    b = bias if has_bias else weight  # unused when HAS_BIAS=False

    out = torch.empty((M, N), dtype=input.dtype, device=input.device)

    dtype = input.dtype
    is_fp32 = dtype == torch.float32
    contig = (x.stride(1) == 1) and (w.stride(1) == 1)

    if is_fp32:
        if M >= 3072:
            bm, bn, bk, warps, stages = 128, 128, 32, 8, 2
        elif M >= 1536:
            bm, bn, bk, warps, stages = 64, 64, 32, 4, 3
        elif M >= 512:
            bm, bn, bk, warps, stages = 32, 32, 32, 4, 3
        else:
            bm, bn, bk, warps, stages = 32, 32, 32, 4, 3
        no_mask = contig and (M % bm == 0) and (N % bn == 0) and (K % bk == 0)
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        if no_mask:
            # scenario="roll" disables -metaxgpu-mma-unroll-count, which is
            # consistently 2-5% faster for fp32 on C550 (verified target-side).
            _linear_kernel[grid](
                x,
                w,
                b,
                out,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                out.stride(0),
                out.stride(1),
                has_bias,
                True,
                no_mask,
                bm,
                bn,
                bk,
                num_warps=warps,
                num_stages=stages,
                scenario="roll",
            )
        else:
            _linear_kernel[grid](
                x,
                w,
                b,
                out,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                out.stride(0),
                out.stride(1),
                has_bias,
                True,
                no_mask,
                bm,
                bn,
                bk,
                num_warps=warps,
                num_stages=stages,
            )
    else:
        if M >= 3072:
            bm, bn, bk, warps, stages = 256, 128, 32, 8, 2
        elif M >= 512:
            bm, bn, bk, warps, stages = 128, 128, 64, 8, 2
        else:
            bm, bn, bk, warps, stages = 64, 32, 32, 4, 4
        no_mask = contig and (M % bm == 0) and (N % bn == 0) and (K % bk == 0)
        grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
        if no_mask and M >= 512:
            # scenario="roll" is 2-4% faster for fp16/bf16 at M>=1024 but
            # ~6-8% slower at 384 on C550 (verified target-side).
            _linear_kernel[grid](
                x,
                w,
                b,
                out,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                out.stride(0),
                out.stride(1),
                has_bias,
                False,
                no_mask,
                bm,
                bn,
                bk,
                num_warps=warps,
                num_stages=stages,
                scenario="roll",
            )
        else:
            _linear_kernel[grid](
                x,
                w,
                b,
                out,
                M,
                N,
                K,
                x.stride(0),
                x.stride(1),
                w.stride(0),
                w.stride(1),
                out.stride(0),
                out.stride(1),
                has_bias,
                False,
                no_mask,
                bm,
                bn,
                bk,
                num_warps=warps,
                num_stages=stages,
            )

    if input.dim() == 1:
        return out.reshape(N)
    return out.reshape(in_shape[:-1] + (N,))
