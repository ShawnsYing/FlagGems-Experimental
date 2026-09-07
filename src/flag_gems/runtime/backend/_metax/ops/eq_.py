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
    torch.bool: tl.int1,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int8: tl.int8,
    torch.int16: tl.int16,
    torch.int32: tl.int32,
    torch.int64: tl.int64,
    torch.uint8: tl.uint8,
}

_BLOCK = 1024
_WARPS = 4
_SMALL = 1024
_HUGE = (
    1 << 29
)  # >=512Mi elements: 16-bit dtypes prefer large blocks in the DRAM-streaming regime


@triton.jit
def _eq_kernel(
    a_ptr, b_ptr, n_elements, OUT_DTYPE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    if OUT_DTYPE == tl.bfloat16:
        # metax backend cannot lower bool->bf16 casts; synthesize 1.0/0.0 via
        # an f32 select (IEEE-correct incl. -0.0 and NaN) then cast f32->bf16.
        r = a == b
        res = tl.where(r, 1.0, 0.0).to(tl.bfloat16)
    elif OUT_DTYPE == tl.int1:
        res = a == b
    else:
        res = (a == b).to(OUT_DTYPE)
    tl.store(a_ptr + offs, res, mask=mask)


@triton.jit
def _eq_even_kernel(
    a_ptr, b_ptr, n_elements, OUT_DTYPE: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    # Unmasked specialization for n % BLOCK_SIZE == 0 (all loads/stores in-bounds).
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    if OUT_DTYPE == tl.bfloat16:
        # metax backend cannot lower bool->bf16 casts; synthesize 1.0/0.0 via
        # an f32 select (IEEE-correct incl. -0.0 and NaN) then cast f32->bf16.
        r = a == b
        res = tl.where(r, 1.0, 0.0).to(tl.bfloat16)
    elif OUT_DTYPE == tl.int1:
        res = a == b
    else:
        res = (a == b).to(OUT_DTYPE)
    tl.store(a_ptr + offs, res)


@triton.jit
def _eq_bcast_kernel(
    a_ptr,
    b_ptr,
    meta_ptr,
    n_elements,
    OUT_DTYPE: tl.constexpr,
    NDIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # meta layout: [shape(NDIM), a_stride(NDIM), b_stride(NDIM)] int32 scalars.
    # b_stride[d] is 0 when B broadcasts along dim d (size-1 dims / missing dims).
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    rem = offs
    a_off = offs * 0
    b_off = offs * 0
    for d in tl.static_range(NDIM):
        sh = tl.load(meta_ptr + d)
        c = rem % sh
        rem = rem // sh
        a_off += c * tl.load(meta_ptr + NDIM + d)
        b_off += c * tl.load(meta_ptr + 2 * NDIM + d)
    a = tl.load(a_ptr + a_off, mask=mask)
    b = tl.load(b_ptr + b_off, mask=mask)
    if OUT_DTYPE == tl.bfloat16:
        # metax backend cannot lower bool->bf16 casts; synthesize 1.0/0.0 via
        # an f32 select (IEEE-correct incl. -0.0 and NaN) then cast f32->bf16.
        r = a == b
        res = tl.where(r, 1.0, 0.0).to(tl.bfloat16)
    elif OUT_DTYPE == tl.int1:
        res = a == b
    else:
        res = (a == b).to(OUT_DTYPE)
    tl.store(a_ptr + a_off, res, mask=mask)


def _pick_config(n, dtype):
    if n < _SMALL:
        blk = max(1, triton.next_power_of_2(n))
        warps = 1
    elif dtype == torch.float16 and n >= _HUGE:
        # probe (2adc5ec): fp16 1G b2048w8 4.2934 vs b4096w8 4.2966 vs b1024w4 4.3061
        blk = 2048
        warps = 8
    elif dtype == torch.bfloat16 and n >= _HUGE:
        # probe (2adc5ec): bf16 1G b4096w8 4.3080 vs b2048w8 4.3219 vs b1024w4 4.3288
        blk = 4096
        warps = 8
    elif dtype == torch.float32 and n >= (1 << 24):
        # probe (2adc5ec): fp32 16M ~0.7% faster with b512/w4 than b1024/w4
        blk = 512
        warps = 4
    else:
        blk = _BLOCK
        warps = _WARPS
    return blk, warps


def eq_scalar_(A, B):
    n = A.numel()
    if n == 0:
        return A
    out_dtype = _DTYPE_MAP.get(A.dtype)
    if A.shape == B.shape and A.is_contiguous() and B.is_contiguous():
        blk, warps = _pick_config(n, A.dtype)
        grid = (triton.cdiv(n, blk),)
        if n % blk == 0:
            _eq_even_kernel[grid](
                A,
                B,
                n,
                OUT_DTYPE=out_dtype,
                BLOCK_SIZE=blk,
                num_warps=warps,
            )
        else:
            _eq_kernel[grid](
                A,
                B,
                n,
                OUT_DTYPE=out_dtype,
                BLOCK_SIZE=blk,
                num_warps=warps,
            )
    else:
        ndim = A.dim()
        if B.dim() < ndim:
            b_shape = (1,) * (ndim - B.dim()) + tuple(B.shape)
            b_strides = [0] * (ndim - B.dim()) + list(B.stride())
        else:
            b_shape = tuple(B.shape)
            b_strides = list(B.stride())
        for d in range(ndim):
            if b_shape[d] == 1:
                b_strides[d] = 0
        meta = torch.empty(3 * ndim, dtype=torch.int32, device=A.device)
        meta[0:ndim] = torch.tensor(list(A.shape), dtype=torch.int32)
        meta[ndim : 2 * ndim] = torch.tensor(list(A.stride()), dtype=torch.int32)
        meta[2 * ndim : 3 * ndim] = torch.tensor(list(b_strides), dtype=torch.int32)
        blk, warps = _pick_config(n, A.dtype)
        grid = (triton.cdiv(n, blk),)
        _eq_bcast_kernel[grid](
            A,
            B,
            meta,
            n,
            OUT_DTYPE=out_dtype,
            NDIM=ndim,
            BLOCK_SIZE=blk,
            num_warps=warps,
        )
    return A
