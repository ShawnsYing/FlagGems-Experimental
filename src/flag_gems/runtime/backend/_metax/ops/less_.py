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
def _less_flat_kernel(
    a_ptr, b_ptr, n_elements, EVICT: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    r = tl.where(a < b, 1.0, 0.0).to(a.dtype)
    if EVICT:
        tl.store(a_ptr + offs, r, mask=mask, eviction_policy="evict_first")
    else:
        tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _less_bcast_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    ad0,
    ad1,
    ad2,
    ad3,
    ad4,
    ad5,
    ad6,
    ad7,
    as0,
    as1,
    as2,
    as3,
    as4,
    as5,
    as6,
    as7,
    bd0,
    bd1,
    bd2,
    bd3,
    bd4,
    bd5,
    bd6,
    bd7,
    bs0,
    bs1,
    bs2,
    bs3,
    bs4,
    bs5,
    bs6,
    bs7,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    rem = offs
    a_off = tl.zeros([BLOCK], dtype=tl.int32)
    b_off = tl.zeros([BLOCK], dtype=tl.int32)
    for d in tl.static_range(8):
        if d == 0:
            ad = ad0
            astr = as0
            bd = bd0
            bstr = bs0
        elif d == 1:
            ad = ad1
            astr = as1
            bd = bd1
            bstr = bs1
        elif d == 2:
            ad = ad2
            astr = as2
            bd = bd2
            bstr = bs2
        elif d == 3:
            ad = ad3
            astr = as3
            bd = bd3
            bstr = bs3
        elif d == 4:
            ad = ad4
            astr = as4
            bd = bd4
            bstr = bs4
        elif d == 5:
            ad = ad5
            astr = as5
            bd = bd5
            bstr = bs5
        elif d == 6:
            ad = ad6
            astr = as6
            bd = bd6
            bstr = bs6
        else:
            ad = ad7
            astr = as7
            bd = bd7
            bstr = bs7
        c = rem % ad
        rem = rem // ad
        a_off += c * astr
        b_off += tl.where(bd > 1, c, 0) * bstr
    a = tl.load(a_ptr + a_off, mask=mask)
    b = tl.load(b_ptr + b_off, mask=mask)
    r = tl.where(a < b, 1.0, 0.0).to(a.dtype)
    tl.store(a_ptr + a_off, r, mask=mask)


_LARGE = (
    1 << 28
)  # 268M elements: above this the 2^30 shapes dominate and want fewer, fatter CTAs

_CONFIG = {
    torch.float16: {False: (1024, 4), True: (2048, 8)},
    torch.bfloat16: {False: (1024, 4), True: (2048, 4)},
    torch.float32: {False: (512, 4), True: (1024, 4)},
}


def _config(dtype, n):
    cfg = _CONFIG.get(dtype)
    if cfg is not None:
        return cfg[n > _LARGE]
    return (1024, 4)


def less_scalar_(A, B):
    n = A.numel()
    if n == 0:
        return A

    if A.shape == B.shape and A.is_contiguous() and B.is_contiguous():
        block, warps = _config(A.dtype, n)
        grid = (triton.cdiv(n, block),)
        _less_flat_kernel[grid](
            A, B, n, EVICT=(A.dtype == torch.float16), BLOCK=block, num_warps=warps
        )
        return A

    ndim = max(A.ndim, B.ndim)
    ap = ndim - A.ndim
    bp = ndim - B.ndim
    a_dims = [1] * ap + list(A.shape) + [1] * (8 - ndim)
    b_dims = [1] * bp + list(B.shape) + [1] * (8 - ndim)
    a_strides = [0] * ap + list(A.stride()) + [0] * (8 - ndim)
    b_strides = [0] * bp + list(B.stride()) + [0] * (8 - ndim)
    args = []
    for d in range(8):
        args.append(a_dims[d])
    for d in range(8):
        args.append(a_strides[d])
    for d in range(8):
        args.append(b_dims[d])
    for d in range(8):
        args.append(b_strides[d])
    block, warps = _config(A.dtype, n)
    grid = (triton.cdiv(n, block),)
    _less_bcast_kernel[grid](A, B, n, *args, BLOCK=block, num_warps=warps)
    return A
