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
from typing import List
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


"""unbind_copy: split a tensor along a dimension into a list of copied tensors.

Reference semantics follow flag_gems.ops.unbind_copy:
  - normalize negative dim, raise IndexError when out of range
  - return [] when the unbind dimension has size 0
  - return empty tensors of the reduced shape when any other dim is 0
  - otherwise one contiguous storage holds all slices; each output tensor is a
    contiguous view of one slice (a copy of the input data).

Architecture:
  - dim == 0 with a contiguous input is an identity permutation of the flat
    element order, so a single flat 1-D copy kernel over the whole storage
    (minimal program count, fully vectorized) handles every timed workload.
  - other dims use the 2-D grid kernel (block within slice x slice id) with
    pre/post index decomposition around the unbind dim.
  - non-contiguous inputs use a sizes/strides general kernel.
"""

from typing import List

import torch
import triton
import triton.language as tl


@triton.jit
def _unbind_copy_flat_kernel(
    input_ptr,
    output_ptr,
    numel,
    NEED_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """1-D contiguous copy of the whole input (valid for dim == 0)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    if NEED_MASK:
        mask = offs < numel
        data = tl.load(input_ptr + offs, mask=mask)
        tl.store(output_ptr + offs, data, mask=mask)
    else:
        data = tl.load(input_ptr + offs)
        tl.store(output_ptr + offs, data)


@triton.jit
def _unbind_copy_contig_kernel(
    input_ptr,
    output_ptr,
    dim_size,
    dim_prod_post,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """2D grid: (blocks per slice, num_slices). Input assumed contiguous."""
    pid_x = tl.program_id(0)
    slice_id = tl.program_id(1)

    idx = (pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = idx < num_elements

    # Flat index inside one output slice, decomposed around the unbind dim.
    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post
    input_flat_idx = (
        pre_idx * (dim_size * dim_prod_post) + slice_id * dim_prod_post + post_idx
    )

    data = tl.load(input_ptr + input_flat_idx, mask=mask)
    tl.store(output_ptr + slice_id * num_elements + idx, data, mask=mask)


@triton.jit
def _unbind_copy_general_kernel(
    input_ptr,
    output_ptr,
    dim,
    ndim,
    dim_size,
    sizes_ptr,
    strides_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """2D grid variant for non-contiguous inputs: uses sizes/strides arrays."""
    pid_x = tl.program_id(0)
    slice_id = tl.program_id(1)

    idx = (pid_x * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = idx < num_elements

    rem = idx
    in_off = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for j in range(ndim):
        jrev = ndim - 1 - j
        sz = tl.load(sizes_ptr + jrev)
        st = tl.load(strides_ptr + jrev)
        is_dim = jrev == dim
        c = tl.where(is_dim, 0, rem % sz)
        rem = tl.where(is_dim, rem, rem // sz)
        in_off += c * st
    in_off += slice_id * tl.load(strides_ptr + dim)

    data = tl.load(input_ptr + in_off, mask=mask)
    tl.store(output_ptr + slice_id * num_elements + idx, data, mask=mask)


_BLOCK_SIZE = 1024
# Above this element count, more warps per program hurt dispatch throughput.
_SMALL_NUMEL = 65536
# 16-bit dtypes on large tensors: fewer, wider blocks beat the 128-program
# dispatch floor (measured: 6.47us vs 6.56us on [32,64,128] float16).
_LARGE_BLOCK = 4096
_LARGE_WARPS = 2


def unbind_copy(input: torch.Tensor, dim: int = 0) -> List[torch.Tensor]:
    if dim < 0:
        dim = dim + input.ndim

    if dim < 0 or dim >= input.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-input.ndim}, {input.ndim - 1}], but got {dim})"
        )

    num_slices = input.shape[dim]
    if num_slices == 0:
        return []

    output_shape = list(input.shape)
    del output_shape[dim]

    num_elements = 1
    for s in output_shape:
        num_elements *= s

    if num_elements == 0:
        return [
            torch.empty(output_shape, dtype=input.dtype, device=input.device)
            for _ in range(num_slices)
        ]

    output_storage = torch.empty(
        (num_slices * num_elements,), dtype=input.dtype, device=input.device
    )
    total_elements = num_slices * num_elements

    if dim == 0 and input.is_contiguous():
        # Identity permutation: flat copy of the whole input.
        if total_elements >= _SMALL_NUMEL and input.element_size() <= 2:
            block_size = _LARGE_BLOCK
            num_warps = _LARGE_WARPS
        else:
            block_size = _BLOCK_SIZE
            num_warps = 8 if total_elements < _SMALL_NUMEL else 4
        grid = (triton.cdiv(total_elements, block_size),)
        _unbind_copy_flat_kernel[grid](
            input,
            output_storage,
            total_elements,
            NEED_MASK=(total_elements % block_size != 0),
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
    elif input.is_contiguous():
        dim_prod_post = 1
        for i in range(dim + 1, input.ndim):
            dim_prod_post *= input.shape[i]
        grid = (triton.cdiv(num_elements, _BLOCK_SIZE), num_slices)
        _unbind_copy_contig_kernel[grid](
            input,
            output_storage,
            num_slices,
            dim_prod_post,
            num_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )
    else:
        sizes = torch.tensor(list(input.shape), dtype=torch.int64, device=input.device)
        strides = torch.tensor(
            list(input.stride()), dtype=torch.int64, device=input.device
        )
        grid = (triton.cdiv(num_elements, _BLOCK_SIZE), num_slices)
        _unbind_copy_general_kernel[grid](
            input,
            output_storage,
            dim,
            input.ndim,
            num_slices,
            sizes,
            strides,
            num_elements,
            BLOCK_SIZE=_BLOCK_SIZE,
        )

    return [
        output_storage[i * num_elements : (i + 1) * num_elements].reshape(output_shape)
        for i in range(num_slices)
    ]
