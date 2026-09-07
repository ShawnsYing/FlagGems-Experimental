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
def _baddbmm_small(
    self_ptr,
    batch1_ptr,
    batch2_ptr,
    M,
    N,
    K,
    stride_sb,
    stride_sm,
    stride_sn,
    stride_a1,
    stride_am,
    stride_ak,
    stride_b1,
    stride_bk,
    stride_bn,
    beta,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    a_ptrs = (
        batch1_ptr
        + pid_b * stride_a1
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        batch2_ptr
        + pid_b * stride_b1
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_mask = (k + offs_k) < K
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        if USE_IEEE:
            acc = tl.dot(a, b, acc, input_precision="ieee")
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    s_ptrs = (
        self_ptr
        + pid_b * stride_sb
        + offs_m[:, None] * stride_sm
        + offs_n[None, :] * stride_sn
    )
    mask = m_mask[:, None] & n_mask[None, :]
    s = tl.load(s_ptrs, mask=mask, other=0.0)
    tl.store(s_ptrs, beta * s + alpha * acc, mask=mask)


@triton.jit
def _baddbmm_large(
    self_ptr,
    batch1_ptr,
    batch2_ptr,
    M,
    N,
    K,
    stride_sb,
    stride_sm,
    stride_sn,
    stride_a1,
    stride_am,
    stride_ak,
    stride_b1,
    stride_bk,
    stride_bn,
    beta,
    alpha,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    USE_IEEE: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

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

    a_ptrs = (
        batch1_ptr
        + pid_b * stride_a1
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        batch2_ptr
        + pid_b * stride_b1
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if EVEN_M and EVEN_N and EVEN_K:
        for k in range(0, K, BLOCK_K):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            if USE_IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        m_mask = offs_m < M
        n_mask = offs_n < N
        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K
            a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            if USE_IEEE:
                acc = tl.dot(a, b, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    s_ptrs = (
        self_ptr
        + pid_b * stride_sb
        + offs_m[:, None] * stride_sm
        + offs_n[None, :] * stride_sn
    )
    if EVEN_M and EVEN_N:
        s = tl.load(s_ptrs)
        tl.store(s_ptrs, beta * s + alpha * acc)
    else:
        mask = (offs_m < M)[:, None] & (offs_n < N)[None, :]
        s = tl.load(s_ptrs, mask=mask, other=0.0)
        tl.store(s_ptrs, beta * s + alpha * acc, mask=mask)


def _launch_large(self, batch1, batch2, M, N, K, beta, alpha, use_ieee):
    s_max = max(M, N, K)
    if use_ieee:
        if s_max < 512:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
            num_warps, num_stages = 4, 3
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
            num_warps, num_stages = 8, 3
    else:
        if s_max >= 1024:
            BLOCK_M, BLOCK_N, BLOCK_K = 256, 128, 32
            num_warps, num_stages = 8, 2
        else:
            BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
            num_warps, num_stages = 4, 2

    even_m = (M % BLOCK_M) == 0
    even_n = (N % BLOCK_N) == 0
    even_k = (K % BLOCK_K) == 0
    if use_ieee or s_max < 1024:
        GROUP_M = 8
    else:
        GROUP_M = 4

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), self.shape[0])
    _baddbmm_large[grid](
        self,
        batch1,
        batch2,
        M,
        N,
        K,
        self.stride(0),
        self.stride(1),
        self.stride(2),
        batch1.stride(0),
        batch1.stride(1),
        batch1.stride(2),
        batch2.stride(0),
        batch2.stride(1),
        batch2.stride(2),
        beta,
        alpha,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        USE_IEEE=use_ieee,
        EVEN_M=even_m,
        EVEN_N=even_n,
        EVEN_K=even_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _launch_small(self, batch1, batch2, M, N, K, beta, alpha, use_ieee):
    if K >= 256:
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 64
        num_warps = 4
    elif K >= 64:
        BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
        num_warps = 4
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32
        num_warps = 2
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), self.shape[0])
    _baddbmm_small[grid](
        self,
        batch1,
        batch2,
        M,
        N,
        K,
        self.stride(0),
        self.stride(1),
        self.stride(2),
        batch1.stride(0),
        batch1.stride(1),
        batch1.stride(2),
        batch2.stride(0),
        batch2.stride(1),
        batch2.stride(2),
        beta,
        alpha,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        USE_IEEE=use_ieee,
        num_warps=num_warps,
    )


def baddbmm_(self, batch1, batch2, *, beta=1, alpha=1):
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    beta = float(beta)
    alpha = float(alpha)

    B, M, N = self.shape
    K = batch1.shape[2]

    use_ieee = self.dtype == torch.float32

    if M >= 128 and N >= 128 and K >= 128:
        _launch_large(self, batch1, batch2, M, N, K, beta, alpha, use_ieee)
    else:
        _launch_small(self, batch1, batch2, M, N, K, beta, alpha, use_ieee)
    return self
