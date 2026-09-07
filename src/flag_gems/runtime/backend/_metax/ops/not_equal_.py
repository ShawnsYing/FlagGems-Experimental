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

_DTYPE_MAP = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.int64: tl.int64,
    torch.int32: tl.int32,
    torch.int16: tl.int16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.bool: tl.int1,
}


def _row_major_strides(shape):
    s = 1
    strides = []
    for d in reversed(shape):
        strides.append(s)
        s *= d
    return tuple(reversed(strides))


@triton.jit
def _ne_inplace_kernel(
    A_ptr,
    B_ptr,
    N,
    A_SHAPE: tl.constexpr,
    A_STRIDES: tl.constexpr,
    B_STRIDES: tl.constexpr,
    RANK: tl.constexpr,
    PROMOTE: tl.constexpr,
    PROMOTED: tl.constexpr,
    DECOMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < N

    a_idx = offs
    b_idx = offs
    if DECOMP:
        # Mixed-radix decomposition of the flat index over A's logical shape.
        idx = offs
        a_idx = tl.zeros(offs.shape, dtype=tl.int64)
        b_idx = tl.zeros(offs.shape, dtype=tl.int64)
        for d in tl.static_range(8):
            if d < RANK:
                dim = A_SHAPE[d]
                c = idx % dim
                idx = idx // dim
                a_idx = a_idx + c * A_STRIDES[d]
                b_idx = b_idx + c * B_STRIDES[d]

    a = tl.load(A_ptr + a_idx, mask=mask)
    b = tl.load(B_ptr + b_idx, mask=mask)
    if PROMOTE:
        a = a.to(PROMOTED)
        b = b.to(PROMOTED)
    one = tl.full((BLOCK,), 1, dtype=A_ptr.dtype.element_ty)
    zero = tl.full((BLOCK,), 0, dtype=A_ptr.dtype.element_ty)
    r = tl.where(a != b, one, zero)
    tl.store(A_ptr + a_idx, r, mask=mask)


def not_equal_(A, B):
    N = A.numel()
    if N == 0:
        return A
    RANK = A.dim()
    if RANK > 8:
        raise NotImplementedError("rank > 8 unsupported")

    if isinstance(B, torch.Tensor):
        if A.is_contiguous() and B.is_contiguous() and B.shape == A.shape:
            # Fast path: dense elementwise, no broadcast / no index decomposition.
            B_b = B
            decomp = False
            a_strides = ()
            b_strides = ()
        else:
            B_b = torch.broadcast_to(B, A.shape)
            decomp = True
            a_strides = A.stride()
            b_strides = B_b.stride()
    else:
        B = torch.as_tensor(B, device=A.device)
        B_b = torch.broadcast_to(B, A.shape)
        decomp = not (A.is_contiguous() and B_b.stride() == _row_major_strides(A.shape))
        a_strides = A.stride()
        b_strides = B_b.stride()

    p = torch.promote_types(A.dtype, B.dtype)
    promoted = _DTYPE_MAP[p]
    promote = p != A.dtype

    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    _ne_inplace_kernel[grid](
        A,
        B_b,
        N,
        A.shape,
        a_strides,
        b_strides,
        RANK=RANK,
        PROMOTE=promote,
        PROMOTED=promoted,
        DECOMP=decomp,
        BLOCK=BLOCK,
    )
    return A
