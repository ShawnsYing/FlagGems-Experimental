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
    batch1_ptr,
    batch2_ptr,
    M,
    N,
    K,
    B,
    s_b1,
    s_m1,
    s_k1,
    s_b2,
    s_k2,
    s_n2,
    s_m0,
    s_n0,
    beta,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN: tl.constexpr,
    FLAT: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # accumulator for sum_b (batch1[b] @ batch2[b]); bias enters only in the
    # epilogue: res = alpha * acc + beta * bias
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if FLAT:
        # single flattened (b, k) loop: one pipeline drains without b-boundary
        # stalls; used for the huge fp16/bf16 squares where the probe measured
        # ~1% over the nested loop
        NK: tl.constexpr = K // BLOCK_K
        for t in range(B * NK):
            b = t // NK
            k0 = (t % NK) * BLOCK_K
            a_tile = tl.load(
                batch1_ptr
                + b * s_b1
                + offs_m[:, None] * s_m1
                + (k0 + offs_k)[None, :] * s_k1
            )
            b_tile = tl.load(
                batch2_ptr
                + b * s_b2
                + (k0 + offs_k)[:, None] * s_k2
                + offs_n[None, :] * s_n2
            )
            acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
    else:
        for b in range(B):
            b1 = batch1_ptr + b * s_b1
            b2 = batch2_ptr + b * s_b2
            for k0 in range(0, K, BLOCK_K):
                if EVEN:
                    a_tile = tl.load(
                        b1 + offs_m[:, None] * s_m1 + (k0 + offs_k)[None, :] * s_k1
                    )
                    b_tile = tl.load(
                        b2 + (k0 + offs_k)[:, None] * s_k2 + offs_n[None, :] * s_n2
                    )
                else:
                    a_ptrs = b1 + offs_m[:, None] * s_m1 + (k0 + offs_k)[None, :] * s_k1
                    b_ptrs = b2 + (k0 + offs_k)[:, None] * s_k2 + offs_n[None, :] * s_n2
                    mask_a = (offs_m[:, None] < M) & ((k0 + offs_k)[None, :] < K)
                    mask_b = ((k0 + offs_k)[:, None] < K) & (offs_n[None, :] < N)
                    a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)
                    b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)
                acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")

    bias_ptrs = bias_ptr + offs_m[:, None] * s_m0 + offs_n[None, :] * s_n0
    res = alpha.to(tl.float32) * acc + beta.to(tl.float32) * tl.load(
        bias_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0
    )
    tl.store(bias_ptrs, res, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _addbmm_ksplit_kernel(
    batch1_ptr,
    batch2_ptr,
    part_ptr,
    M,
    N,
    K,
    B,
    s_b1,
    s_m1,
    s_k1,
    s_b2,
    s_k2,
    s_n2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT: tl.constexpr,
    FLAT: tl.constexpr,
):
    # split-K variant: each program computes the products for one K-half of
    # every batch, writing an fp32 partial tile; a separate reduce kernel
    # combines SPLIT partials with the bias epilogue.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    KP: tl.constexpr = K // SPLIT
    NK: tl.constexpr = (K // SPLIT) // BLOCK_K
    k_base = pid_s * KP
    if FLAT:
        # flattened (b, k) loop: removes batch-boundary pipeline stalls
        for t in range(B * NK):
            b = t // NK
            k0 = k_base + (t % NK) * BLOCK_K
            a_tile = tl.load(
                batch1_ptr
                + b * s_b1
                + offs_m[:, None] * s_m1
                + (k0 + offs_k)[None, :] * s_k1
            )
            b_tile = tl.load(
                batch2_ptr
                + b * s_b2
                + (k0 + offs_k)[:, None] * s_k2
                + offs_n[None, :] * s_n2
            )
            acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
    else:
        for b in range(B):
            b1 = batch1_ptr + b * s_b1
            b2 = batch2_ptr + b * s_b2
            for k0 in range(0, KP, BLOCK_K):
                a_tile = tl.load(
                    b1 + offs_m[:, None] * s_m1 + (k_base + k0 + offs_k)[None, :] * s_k1
                )
                b_tile = tl.load(
                    b2 + (k_base + k0 + offs_k)[:, None] * s_k2 + offs_n[None, :] * s_n2
                )
                acc = tl.dot(a_tile, b_tile, acc, input_precision="ieee")
    out_ptrs = part_ptr + pid_s * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc)


@triton.jit
def _addbmm_reduce_kernel(
    bias_ptr,
    part_ptr,
    M,
    N,
    SPLIT,
    beta,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT):
        acc += tl.load(part_ptr + s * M * N + offs_m[:, None] * N + offs_n[None, :])
    bias_ptrs = bias_ptr + offs_m[:, None] * N + offs_n[None, :]
    res = alpha.to(tl.float32) * acc + beta.to(tl.float32) * tl.load(bias_ptrs)
    tl.store(bias_ptrs, res)


def addbmm_(bias, batch1, batch2, beta=1.0, alpha=1.0):
    M, N = bias.shape
    B, M1, K = batch1.shape
    B2, K2, N2 = batch2.shape

    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    dt = bias.dtype
    mn = M * N
    flat = False
    ksplit = False
    ksplit_flat = False
    SPLIT = 2
    if dt in (torch.float16, torch.bfloat16):
        if mn < 1024 * 1024:
            # small outputs (384^3): latency-bound, many small CTAs win
            BM, BN, BK, nw, ns = 32, 32, 64, 4, 2
        elif mn >= 4096 * 4096:
            # huge square: wide tiles cut L2 traffic; flat loop drains one pipeline
            BM, BN, BK, nw, ns = 256, 128, 32, 8, 2
            flat = True
        elif mn >= 2 * 1024 * 1024:
            # 2048^3: big tiles sustain the MMA pipeline; split-K doubles CTAs
            BM, BN, BK, nw, ns = 128, 128, 64, 8, 2
            ksplit = True
        else:
            # 1024^3: smaller tiles + split-K win; flat loop removes b-boundary
            # stalls within each K-half (probe 0.415 vs 0.438ms nested)
            BM, BN, BK, nw, ns = 64, 64, 64, 4, 3
            ksplit = True
            ksplit_flat = True
    else:
        if mn >= 4096 * 4096:
            BM, BN, BK, nw, ns = 128, 128, 32, 8, 2
        elif mn >= 2 * 1024 * 1024:
            # 2048^3: split-K doubles CTAs (probe 8.75 -> 8.25ms)
            BM, BN, BK, nw, ns = 64, 64, 32, 8, 2
            ksplit = True
        elif mn >= 1024 * 1024:
            # 1024^3: split-K doubles CTAs (probe 1.49 -> 1.20ms)
            BM, BN, BK, nw, ns = 64, 64, 32, 4, 2
            ksplit = True
        else:
            # small outputs (384^3): latency-bound, many small CTAs win
            BM, BN, BK, nw, ns = 32, 32, 64, 4, 2

    even = (M % BM == 0) and (N % BN == 0) and (K % BK == 0)
    ksplit = (
        ksplit and even and (K % (SPLIT * BK) == 0) and (M % 64 == 0) and (N % 64 == 0)
    )
    if not even:
        BM = max(16, min(triton.next_power_of_2(M), 64))
        BN = max(16, min(triton.next_power_of_2(N), 64))
        BK = max(16, min(triton.next_power_of_2(K), 64))
        nw = 4
        ns = 3
        flat = False
        ksplit = False
        ksplit_flat = False

    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    if ksplit:
        part = torch.empty((SPLIT, M, N), dtype=torch.float32, device=bias.device)
        grid3 = (triton.cdiv(M, BM), triton.cdiv(N, BN), SPLIT)
        _addbmm_ksplit_kernel[grid3](
            batch1,
            batch2,
            part,
            M,
            N,
            K,
            B,
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            SPLIT=SPLIT,
            FLAT=ksplit_flat,
            num_warps=nw,
            num_stages=ns,
        )
        _addbmm_reduce_kernel[(M // 64, N // 64)](
            bias,
            part,
            M,
            N,
            SPLIT,
            float(beta),
            float(alpha),
            BLOCK_M=64,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
    else:
        _addbmm_kernel[grid](
            bias,
            batch1,
            batch2,
            M,
            N,
            K,
            B,
            batch1.stride(0),
            batch1.stride(1),
            batch1.stride(2),
            batch2.stride(0),
            batch2.stride(1),
            batch2.stride(2),
            bias.stride(0),
            bias.stride(1),
            float(beta),
            float(alpha),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            EVEN=even,
            FLAT=flat,
            num_warps=nw,
            num_stages=ns,
        )
    return bias
