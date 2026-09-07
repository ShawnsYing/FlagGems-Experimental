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
def _addbmm_kernel(
    bias_ptr,
    b1_ptr,
    b2_ptr,
    out_ptr,
    M,
    N,
    K,
    B,
    alpha,
    beta,
    sb1b,
    sb1m,
    sb1k,
    sb2b,
    sb2k,
    sb2n,
    sbiasm,
    sbiask,
    som,
    son,
    DO_GEMM: tl.constexpr,
    HAS_BIAS_TERM: tl.constexpr,
    BIAS_1D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if DO_GEMM:
        num_k_tiles = tl.cdiv(K, BLOCK_K)
        for b in range(0, B):
            b1_base = b1_ptr + b * sb1b
            b2_base = b2_ptr + b * sb2b
            for k0 in range(0, num_k_tiles):
                kk = k0 * BLOCK_K + offs_k
                mask_k = kk < K
                a = tl.load(
                    b1_base + offs_m[:, None] * sb1m + kk[None, :] * sb1k,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                bt = tl.load(
                    b2_base + kk[:, None] * sb2k + offs_n[None, :] * sb2n,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                acc = tl.dot(a, bt, acc, input_precision="ieee")

    res = alpha * acc if DO_GEMM else tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if HAS_BIAS_TERM:
        if BIAS_1D:
            bias = tl.load(
                bias_ptr + offs_n[None, :] * sbiask, mask=mask_n[None, :], other=0.0
            )
            bias = tl.broadcast_to(bias, (BLOCK_M, BLOCK_N))
        else:
            bias = tl.load(
                bias_ptr + offs_m[:, None] * sbiasm + offs_n[None, :] * sbiask,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            )
        res = res + beta * bias

    out_dtype = out_ptr.dtype.element_ty
    tl.store(
        out_ptr + offs_m[:, None] * som + offs_n[None, :] * son,
        res.to(out_dtype),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    return float(x)


def addbmm(bias, batch1, batch2, beta=1.0, alpha=1.0):
    a = _to_scalar(alpha)
    b = _to_scalar(beta)

    B, M, K = batch1.shape
    B2, K2, N = batch2.shape
    assert B == B2 and K == K2

    out_dtype = torch.promote_types(
        torch.promote_types(bias.dtype, batch1.dtype), batch2.dtype
    )
    out = torch.empty((M, N), device=bias.device, dtype=out_dtype)

    dt = batch1.dtype
    mn = M * N
    if dt == torch.float32:
        if mn <= 147456:
            # tiny outputs: launch-bound; 32x32x32_w4_s4 measured best on s0
            # (probe16 same-run: 0.02463 vs 0.02598 for 32x32x16_w4_s4)
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 32, 32, 32, 4, 4
        elif mn <= 1048576:
            # small outputs / short K: s4 pipeline measured best
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 64, 64, 32, 4, 4
        elif mn <= 4194304:
            # medium outputs (s3): s4 pipeline measured best on C550
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 64, 64, 32, 4, 4
        else:
            # large outputs, long K: few-batch case prefers more CTAs
            if B <= 4:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 64, 64, 16, 4, 4
            else:
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 128, 128, 16, 8, 4
    else:
        if mn <= 147456:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 32, 32, 64, 4, 2
        elif mn <= 1048576:
            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 64, 64, 64, 4, 4
        else:
            if K <= 2048:
                # mid-K large outputs: 64x128 tiles with 4-stage pipeline win on s3
                BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = 64, 128, 32, 4, 4
            else:
                # long-K large outputs: bf16 prefers 4 stages (abs. eval 19.82
                # vs 20.08 on s4, matching the probe); fp16 stays at 3 stages
                # where s4 measured worse (19.81 vs 19.49)
                if dt == torch.bfloat16:
                    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
                        128,
                        128,
                        32,
                        8,
                        4,
                    )
                else:
                    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = (
                        128,
                        128,
                        32,
                        8,
                        3,
                    )

    GROUP_M = 8
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

    _addbmm_kernel[grid](
        bias,
        batch1,
        batch2,
        out,
        M,
        N,
        K,
        B,
        a,
        b,
        batch1.stride(0),
        batch1.stride(1),
        batch1.stride(2),
        batch2.stride(0),
        batch2.stride(1),
        batch2.stride(2),
        bias.stride(0),
        bias.stride(1),
        out.stride(0),
        out.stride(1),
        DO_GEMM=(a != 0.0),
        HAS_BIAS_TERM=(b != 0.0),
        BIAS_1D=(bias.dim() == 1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
