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
def _copy_flat_kernel(in_ptr, out_ptr, total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    v = tl.load(in_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _copy_flat_nomask_kernel(in_ptr, out_ptr, BLOCK: tl.constexpr):
    # Specialization for n % BLOCK == 0: no mask ALU in the timed path.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, v)


@triton.jit
def _copy_strided_kernel(
    in_ptr,
    out_ptr,
    total,
    SHAPE: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Handles non-contiguous inputs: decompose flat logical index into
    # per-dimension coordinates and gather via the input's actual strides.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    idx = offs
    in_off = tl.zeros([BLOCK], dtype=tl.int64)
    for d in tl.static_range(len(SHAPE)):
        s = SHAPE[d]
        st = STRIDES[d]
        in_off += (idx % s) * st
        idx = idx // s
    v = tl.load(in_ptr + in_off, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


def _launch_flat(src, dst, n, block, warps):
    if n % block == 0:
        _copy_flat_nomask_kernel[(n // block,)](src, dst, BLOCK=block, num_warps=warps)
    else:
        _copy_flat_kernel[(triton.cdiv(n, block),)](
            src, dst, n, BLOCK=block, num_warps=warps
        )


def lift_fresh(self):
    n = self.numel()
    out = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    if n == 0:
        return out
    if self.is_contiguous():
        if n <= 65536:
            # latency-bound regime: many small blocks with few warps launch fastest
            _launch_flat(self, out, n, 512, 2)
        elif self.element_size() == 2:
            # fp16/bf16: 16 elems/thread (two 128-bit accesses) for the
            # not-yet-saturated medium range; very large copies are already
            # bandwidth-saturated and prefer the plain 1024/4 shape.
            if n >= (1 << 28):
                _launch_flat(self, out, n, 1024, 4)
            else:
                _launch_flat(self, out, n, 2048, 4)
        else:
            _launch_flat(self, out, n, 1024, 4)
    else:
        _copy_strided_kernel[(triton.cdiv(n, 1024),)](
            self,
            out,
            n,
            SHAPE=tuple(self.shape),
            STRIDES=tuple(self.stride()),
            BLOCK=1024,
            num_warps=4,
        )
    return out
