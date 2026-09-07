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
def _matmuladd_kernel_v2(
    a_ptr,
    b_ptr,
    bias_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_bm,
    stride_bn2,
    stride_om,
    stride_on,
    BIAS_MODE: tl.constexpr,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (k * BLOCK_K + offs_k) < K
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        b_ptrs += BLOCK_K * stride_bk
        a_ptrs += BLOCK_K * stride_ak

    if BIAS_MODE == 1:
        bias = tl.load(bias_ptr + offs_m, mask=offs_m < M, other=0.0)
        acc += bias[:, None]
    elif BIAS_MODE == 0:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    else:
        bias = tl.load(
            bias_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn2,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc += bias

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


def matmuladd(input, other, bias):
    M, K = input.shape
    K2, N = other.shape
    out = torch.empty((M, N), dtype=input.dtype, device=input.device)

    if bias.ndim == 2:
        bias_mode = 2
        stride_bm = bias.stride(0)
        stride_bn2 = bias.stride(1)
    elif bias.numel() == N:
        bias_mode = 0
        stride_bm = 0
        stride_bn2 = 0
    else:
        bias_mode = 1
        stride_bm = 0
        stride_bn2 = 0

    S = max(M, N)
    lowp = input.dtype in (torch.float16, torch.bfloat16)

    if lowp and S >= 2048 and (M % 128 == 0) and (N % 128 == 0) and (K % 32 == 0):
        BM, BN, BK, GM, W, ST = 128, 128, 32, 8, 8, 6
    elif lowp and S >= 512 and (M % 64 == 0) and (N % 64 == 0) and (K % 64 == 0):
        BM, BN, BK, GM, W, ST = 64, 64, 64, 8, 4, 4
    elif lowp:
        BM, BN, BK, GM, W, ST = 64, 64, 64, 8, 4, 3
    elif S >= 2048 and (M % 64 == 0) and (N % 64 == 0) and (K % 16 == 0):
        BM, BN, BK, GM, W, ST = 64, 64, 16, 8, 4, 4
    elif (M % 64 == 0) and (N % 64 == 0) and (K % 32 == 0):
        BM, BN, BK, GM, W, ST = 64, 64, 32, 8, 4, 4
    else:
        BM, BN, BK, GM, W, ST = 64, 64, 64, 8, 4, 3

    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _matmuladd_kernel_v2[grid](
        input,
        other,
        bias,
        out,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        other.stride(0),
        other.stride(1),
        stride_bm,
        stride_bn2,
        out.stride(0),
        out.stride(1),
        BIAS_MODE=bias_mode,
        BLOCK_M=BM,
        BLOCK_N=BN,
        BLOCK_K=BK,
        GROUP_M=GM,
        num_warps=W,
        num_stages=ST,
    )
    return out
