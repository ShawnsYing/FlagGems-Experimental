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
import math
import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# scatter_add (torch.scatter_add semantics):
#   out = inp.clone()
#   for every multi-index position p in `index`:
#       out[..., p[d] := index[p], ...] += src[p]
# The iteration space is `index`'s shape; `src` may be larger than `index`
# along every dimension and `inp` may be larger along dims != dim.
# ---------------------------------------------------------------------------


@triton.jit
def _copy_kernel(inp_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    v = tl.load(inp_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, v, mask=mask)


@triton.jit
def _scatter_add_fused_kernel(
    inp_ptr,
    src_ptr,
    index_ptr,
    out_ptr,
    N,  # out columns (= inp.shape[-1])
    DN,  # index columns (= index.shape[-1])
    SN,  # src row stride (= src.shape[-1])
    MI,  # index rows (may be < inp rows)
    BI: tl.constexpr,
):
    # One CTA per OUTPUT ROW (grid = inp.shape[0]). The row is copied from
    # `inp` with a write-through store so it is L2-visible, then every src
    # contribution of that row is atomically added. Rows with no index entry
    # (pid_r >= MI) only copy. Single launch instead of copy+scatter.
    pid_r = tl.program_id(0)
    j = tl.arange(0, BI)
    mask_n = j < N
    mask_j = j < DN
    has_idx = pid_r < MI

    row_base = pid_r.to(tl.int64) * DN
    src_row = pid_r.to(tl.int64) * SN
    out_row = pid_r.to(tl.int64) * N

    # Issue the index/src loads first so their DRAM latency overlaps the copy
    # phase; the barrier still orders the copy stores before the atomics.
    jm = mask_j & has_idx
    idx = tl.load(index_ptr + row_base + j, mask=jm, other=0)
    val = tl.load(src_ptr + src_row + j, mask=jm, other=0.0)

    v_inp = tl.load(inp_ptr + out_row + j, mask=mask_n, other=0.0)
    tl.store(out_ptr + out_row + j, v_inp, mask=mask_n, cache_modifier=".wt")
    tl.debug_barrier()

    tl.atomic_add(out_ptr + out_row + idx, val, mask=jm)


@triton.jit
def _scatter_add2d_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    N,  # out columns (= inp.shape[-1])
    DN,  # index columns (= index.shape[-1])
    SN,  # src row stride (= src.shape[-1])
    BI: tl.constexpr,
):
    # Specialized for 2-D tensors with scatter along the LAST dim.
    #   index: (MI, DN) int64, values in [0, N)
    #   src:   (M_src, SN) with M_src >= MI, SN >= DN
    #   out:   (MI, N)
    pid_j = tl.program_id(0)
    pid_r = tl.program_id(1)

    row_base = pid_r.to(tl.int64) * DN
    src_row = pid_r.to(tl.int64) * SN
    out_row = pid_r.to(tl.int64) * N

    j = pid_j * BI + tl.arange(0, BI)
    mask = j < DN

    idx = tl.load(index_ptr + row_base + j, mask=mask, other=0)
    val = tl.load(src_ptr + src_row + j, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + out_row + idx, val, mask=mask)


@triton.jit
def _scatter_add_generic_kernel(
    src_ptr,
    index_ptr,
    out_ptr,
    nidx,
    d,
    ostr_d,
    ish0,
    ish1,
    ish2,
    ish3,
    ish4,
    ish5,
    istr0,
    istr1,
    istr2,
    istr3,
    istr4,
    istr5,
    sstr0,
    sstr1,
    sstr2,
    sstr3,
    sstr4,
    sstr5,
    ostr0,
    ostr1,
    ostr2,
    ostr3,
    ostr4,
    ostr5,
    BLOCK: tl.constexpr,
):
    # General rank <= 6 fallback. Shapes are padded to length 6 with
    # shape=1 / stride=0 so a single unrolled decomposition works for any rank.
    pid = tl.program_id(0).to(tl.int64)
    p = pid * BLOCK + tl.arange(0, BLOCK)
    mask = p < nidx

    rem = p
    c0 = rem % ish0
    rem = rem // ish0
    c1 = rem % ish1
    rem = rem // ish1
    c2 = rem % ish2
    rem = rem // ish2
    c3 = rem % ish3
    rem = rem // ish3
    c4 = rem % ish4
    rem = rem // ish4
    c5 = rem % ish5

    idx_off = (
        c0 * istr0 + c1 * istr1 + c2 * istr2 + c3 * istr3 + c4 * istr4 + c5 * istr5
    )
    src_off = (
        c0 * sstr0 + c1 * sstr1 + c2 * sstr2 + c3 * sstr3 + c4 * sstr4 + c5 * sstr5
    )
    out_off = (
        c0 * ostr0 + c1 * ostr1 + c2 * ostr2 + c3 * ostr3 + c4 * ostr4 + c5 * ostr5
    )

    cd = tl.where(
        d == 0,
        c0,
        tl.where(
            d == 1,
            c1,
            tl.where(d == 2, c2, tl.where(d == 3, c3, tl.where(d == 4, c4, c5))),
        ),
    )

    idx = tl.load(index_ptr + idx_off, mask=mask, other=0)
    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    out_off = out_off + (idx - cd) * ostr_d
    tl.atomic_add(out_ptr + out_off, val, mask=mask)


_MAX_RANK = 6


def _pad_to(vals, n, fill):
    v = list(vals) + [fill] * (n - len(vals))
    return v[:n]


def _scatter_generic(out, index, src, d):
    nd = index.ndim
    if nd > _MAX_RANK:
        raise NotImplementedError("rank > %d not supported" % _MAX_RANK)
    ish = _pad_to(index.shape, _MAX_RANK, 1)
    istr = _pad_to(index.stride(), _MAX_RANK, 0)
    sstr = _pad_to(src.stride(), _MAX_RANK, 0)
    ostr = _pad_to(out.stride(), _MAX_RANK, 0)
    ostr_d = out.stride(d)
    nidx = index.numel()
    BLOCK = 1024
    grid = (triton.cdiv(nidx, BLOCK),)
    _scatter_add_generic_kernel[grid](
        src,
        index,
        out,
        nidx,
        d,
        ostr_d,
        *ish,
        *istr,
        *sstr,
        *ostr,
        BLOCK=BLOCK,
    )


def scatter_add(inp, dim, index, src):
    shape = inp.shape
    nd = len(shape)
    d = dim if dim >= 0 else dim + nd

    out = torch.empty_like(inp)
    n = inp.numel()
    if n > 0:
        COPY_BLOCK = 1024
        fast2d = (
            nd == 2
            and d == 1
            and index.numel() > 0
            and index.is_contiguous()
            and src.is_contiguous()
        )
        if fast2d:
            N = shape[1]
            M = shape[0]
            MI = index.shape[0]
            DN = index.shape[1]
            SN = src.shape[1]
            BI = triton.next_power_of_2(max(N, DN))
            if BI <= 2048:
                # Fused single launch: copy + scatter in one kernel.
                NW = 4 if BI <= 256 else 8
                _scatter_add_fused_kernel[(M,)](
                    inp, src, index, out, N, DN, SN, MI, BI=BI, num_warps=NW
                )
            else:
                _copy_kernel[(triton.cdiv(n, COPY_BLOCK),)](
                    inp, out, n, BLOCK=COPY_BLOCK
                )
                if DN <= 2048:
                    BI2 = 256
                    NW2 = 4
                else:
                    BI2 = 512
                    NW2 = 8
                grid = (triton.cdiv(DN, BI2), MI)
                _scatter_add2d_kernel[grid](
                    src, index, out, N, DN, SN, BI=BI2, num_warps=NW2
                )
        else:
            _copy_kernel[(triton.cdiv(n, COPY_BLOCK),)](inp, out, n, BLOCK=COPY_BLOCK)
            if index.numel() > 0:
                _scatter_generic(out, index, src, d)

    return out
