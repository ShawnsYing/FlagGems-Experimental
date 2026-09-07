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

_MAXDIM = 8
_BLOCK = 1024
_NUM_WARPS = 4
# dtype codes used as tl.constexpr branch selectors inside the kernel
_DTYPE_CODES = {
    torch.bool: 0,
    torch.int8: 1,
    torch.uint8: 2,
    torch.int16: 3,
    torch.int32: 4,
    torch.int64: 5,
    torch.float16: 6,
    torch.bfloat16: 7,
    torch.float32: 8,
    torch.float64: 9,
}


@triton.jit
def _cast_operand(x, code: tl.constexpr):
    if code == 0 or code == 1:
        return x.to(tl.int8)
    elif code == 2:
        return x.to(tl.uint8)
    elif code == 3:
        return x.to(tl.int16)
    elif code == 4:
        return x.to(tl.int32)
    elif code == 5:
        return x.to(tl.int64)
    elif code == 6:
        return x.to(tl.float16)
    elif code == 7:
        return x.to(tl.bfloat16)
    elif code == 8:
        return x.to(tl.float32)
    else:
        return x.to(tl.float64)


@triton.jit
def _result_to(x, code: tl.constexpr):
    # x is an i1 comparison result. Integer targets use a plain extend;
    # float targets use a select of 0.0/1.0 constants to avoid the
    # int->float conversion that the backend miscompiles for bf16.
    if code == 0 or code == 1:
        return x.to(tl.int8)
    elif code == 2:
        return x.to(tl.uint8)
    elif code == 3:
        return x.to(tl.int16)
    elif code == 4:
        return x.to(tl.int32)
    elif code == 5:
        return x.to(tl.int64)
    elif code == 6:
        return tl.where(x, tl.full([], 1.0, tl.float16), tl.full([], 0.0, tl.float16))
    elif code == 7:
        return tl.where(x, tl.full([], 1.0, tl.bfloat16), tl.full([], 0.0, tl.bfloat16))
    elif code == 8:
        return tl.where(x, tl.full([], 1.0, tl.float32), tl.full([], 0.0, tl.float32))
    else:
        return tl.where(x, tl.full([], 1.0, tl.float64), tl.full([], 0.0, tl.float64))


@triton.jit
def le_kernel(
    A_ptr,
    B_ptr,
    a_s0,
    a_s1,
    a_s2,
    a_s3,
    a_s4,
    a_s5,
    a_s6,
    a_s7,
    b_s0,
    b_s1,
    b_s2,
    b_s3,
    b_s4,
    b_s5,
    b_s6,
    b_s7,
    sh0,
    sh1,
    sh2,
    sh3,
    sh4,
    sh5,
    sh6,
    sh7,
    cp0,
    cp1,
    cp2,
    cp3,
    cp4,
    cp5,
    cp6,
    cp7,
    ndim: tl.constexpr,
    A_CONTIG: tl.constexpr,
    B_CONTIG: tl.constexpr,
    B_SCALAR: tl.constexpr,
    CT: tl.constexpr,
    AT: tl.constexpr,
    numel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    if A_CONTIG:
        a_off = offs
    else:
        a_strides = (a_s0, a_s1, a_s2, a_s3, a_s4, a_s5, a_s6, a_s7)
        shapes = (sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7)
        cps = (cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7)
        idx = offs
        a_off = tl.zeros([BLOCK], dtype=tl.int32)
        for d in tl.static_range(ndim):
            i_d = (idx // cps[d]) % shapes[d]
            a_off += i_d * a_strides[d]

    if B_SCALAR:
        b_val = _cast_operand(tl.load(B_ptr), CT)
    else:
        if B_CONTIG:
            b_off = offs
        else:
            b_strides = (b_s0, b_s1, b_s2, b_s3, b_s4, b_s5, b_s6, b_s7)
            shapes = (sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7)
            cps = (cp0, cp1, cp2, cp3, cp4, cp5, cp6, cp7)
            idx = offs
            b_off = tl.zeros([BLOCK], dtype=tl.int32)
            for d in tl.static_range(ndim):
                i_d = (idx // cps[d]) % shapes[d]
                b_off += i_d * b_strides[d]
        b_val = _cast_operand(tl.load(B_ptr + b_off, mask=mask, other=0), CT)

    a_val = _cast_operand(tl.load(A_ptr + a_off, mask=mask, other=0), CT)
    res = a_val <= b_val
    out = _result_to(res, AT)
    tl.store(A_ptr + a_off, out, mask=mask)


def le_scalar_(A, B):
    numel = A.numel()
    if numel == 0:
        return A
    ndim = A.dim()
    if ndim > _MAXDIM:
        raise ValueError(f"le_ kernel supports at most {_MAXDIM} dims, got {ndim}")

    shape = A.shape
    A_CONTIG = A.is_contiguous()
    B_SCALAR = B.numel() == 1
    B_CONTIG = (not B_SCALAR) and B.is_contiguous() and (B.shape == shape)

    if B_SCALAR:
        b_strides = (0,) * ndim
    else:
        off = ndim - B.dim()
        b_strides = tuple(
            0 if d < off or B.shape[d - off] == 1 else B.stride(d - off)
            for d in range(ndim)
        )
    a_strides = tuple(A.stride())

    cumprod = [1] * ndim
    for d in range(ndim - 2, -1, -1):
        cumprod[d] = cumprod[d + 1] * shape[d + 1]

    pad = lambda t: tuple(t) + (0,) * (_MAXDIM - len(t))  # noqa: E731
    a_strides_pad = pad(a_strides)
    b_strides_pad = pad(b_strides)
    shape_pad = pad(shape)
    cumprod_pad = pad(cumprod)

    is_bool = A.dtype == torch.bool
    A_k = A.view(torch.uint8) if is_bool else A
    AT = _DTYPE_CODES[torch.uint8] if is_bool else _DTYPE_CODES[A.dtype]
    CT = _DTYPE_CODES[torch.promote_types(A.dtype, B.dtype)]

    BLOCK = _BLOCK
    grid = (triton.cdiv(numel, BLOCK),)
    le_kernel[grid](
        A_k,
        B,
        *a_strides_pad,
        *b_strides_pad,
        *shape_pad,
        *cumprod_pad,
        ndim,
        A_CONTIG,
        B_CONTIG,
        B_SCALAR,
        CT,
        AT,
        numel,
        BLOCK,
        num_warps=_NUM_WARPS,
    )
    return A
