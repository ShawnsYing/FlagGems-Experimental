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
def _leq_flat_kernel(a_ptr, b_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    r = a <= b
    one = tl.full([], 1.0, dtype=a.dtype)
    zero = tl.full([], 0.0, dtype=a.dtype)
    tl.store(a_ptr + offs, tl.where(r, one, zero), mask=mask)


@triton.jit
def _leq_flat_unmasked_kernel(a_ptr, b_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    r = a <= b
    one = tl.full([], 1.0, dtype=a.dtype)
    zero = tl.full([], 0.0, dtype=a.dtype)
    tl.store(a_ptr + offs, tl.where(r, one, zero))


@triton.jit
def _leq_strided_kernel(
    a_ptr, b_ptr, shape_ptr, sa_ptr, sb_ptr, ndim, numel, BLOCK: tl.constexpr
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    rem = offs
    a_idx = tl.zeros([BLOCK], dtype=tl.int64)
    b_idx = tl.zeros([BLOCK], dtype=tl.int64)
    for d in range(0, ndim):
        shp = tl.load(shape_ptr + d)
        sa = tl.load(sa_ptr + d)
        sb = tl.load(sb_ptr + d)
        coord = rem % shp
        a_idx += coord * sa
        b_idx += coord * sb
        rem = rem // shp
    a = tl.load(a_ptr + a_idx, mask=mask)
    b = tl.load(b_ptr + b_idx, mask=mask)
    r = a <= b
    one = tl.full([], 1.0, dtype=a.dtype)
    zero = tl.full([], 0.0, dtype=a.dtype)
    tl.store(a_ptr + a_idx, tl.where(r, one, zero), mask=mask)


def less_equal_scalar_(A, B):
    numel = A.numel()
    if numel == 0:
        return A
    if A.is_contiguous() and B.is_contiguous() and A.shape == B.shape:
        if numel % 1024 == 0:
            _leq_flat_unmasked_kernel[(numel // 1024,)](A, B, BLOCK=1024)
        else:
            _leq_flat_kernel[(numel // 1024 + 1,)](A, B, numel, BLOCK=1024)
    else:
        ndim = A.dim()
        shape = A.shape
        sa = A.stride()
        sb = []
        for d in range(ndim):
            if d < B.dim() and B.shape[d] == shape[d]:
                sb.append(B.stride(d))
            else:
                sb.append(0)
        shape_t = torch.tensor(shape, dtype=torch.int64, device=A.device)
        sa_t = torch.tensor(sa, dtype=torch.int64, device=A.device)
        sb_t = torch.tensor(sb, dtype=torch.int64, device=A.device)
        _leq_strided_kernel[(numel // 1024 + 1,)](
            A, B, shape_t, sa_t, sb_t, ndim, numel, BLOCK=1024
        )
    return A
