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

# torch dtype -> triton dtype map (real dtypes)
_TL_DT = {
    torch.float64: tl.float64,
    torch.float32: tl.float32,
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.float8_e5m2: tl.float8e5,
    torch.int64: tl.int64,
    torch.int32: tl.int32,
    torch.int16: tl.int16,
    torch.int8: tl.int8,
    torch.uint8: tl.uint8,
    torch.bool: tl.int1,
}

_FP_DTYPES = (
    torch.float64,
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
)

_BLOCK = 1024
_NUM_WARPS = 4
# Small-size path uses a smaller block so each thread does exactly one
# 128-bit vector load/store per stream (more CTAs/threads to hide launch
# latency on tiny tensors; the 4096-elem cases are launch-bound).
_SMALL_BLOCK = 512
# Below this element count use the one-tile kernel (more CTAs hide launch
# latency at small sizes); at/above use the two-tile kernel which measures
# 1-2% faster on streaming workloads (crossover probe: tied through 4M,
# two-tile wins at >= 8M).
_T2_MIN = 1 << 21


@triton.jit
def _emit_out(
    r,
    BLOCK: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    """Materialize the i1 comparison as 0/1 in the output dtype.

    The metax backend cannot lower int->fp casts (e.g. uitofp i1 -> bf16),
    so we select typed constants instead of casting.
    """
    if OUT_IS_BOOL:
        return r
    if OUT_IS_FP:
        return tl.where(
            r, tl.full([BLOCK], 1.0, dtype=OUT_DT), tl.full([BLOCK], 0.0, dtype=OUT_DT)
        )
    return tl.where(
        r, tl.full([BLOCK], 1, dtype=OUT_DT), tl.full([BLOCK], 0, dtype=OUT_DT)
    )


@triton.jit
def _ne_flat(
    a_ptr,
    b_ptr,
    numel,
    BLOCK: tl.constexpr,
    COMMON_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    a = tl.load(a_ptr + offs, mask=mask).to(COMMON_DT)
    b = tl.load(b_ptr + offs, mask=mask).to(COMMON_DT)
    r = _emit_out(a != b, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _ne_flat2(
    a_ptr,
    b_ptr,
    numel,
    BLOCK: tl.constexpr,
    COMMON_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    """Two-tile-per-program flat compare (halved CTA count, more bytes in
    flight per thread). Masks handle arbitrary numel; for power-of-2 sizes
    the mask is all-true and the compiler emits unconditional vector code."""
    pid = tl.program_id(0)
    base = pid * (2 * BLOCK)
    offs0 = base + tl.arange(0, BLOCK)
    m0 = offs0 < numel
    a0 = tl.load(a_ptr + offs0, mask=m0).to(COMMON_DT)
    b0 = tl.load(b_ptr + offs0, mask=m0).to(COMMON_DT)
    r0 = _emit_out(a0 != b0, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + offs0, r0, mask=m0)
    offs1 = base + BLOCK + tl.arange(0, BLOCK)
    m1 = offs1 < numel
    a1 = tl.load(a_ptr + offs1, mask=m1).to(COMMON_DT)
    b1 = tl.load(b_ptr + offs1, mask=m1).to(COMMON_DT)
    r1 = _emit_out(a1 != b1, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + offs1, r1, mask=m1)


@triton.jit
def _ne_bscalar(
    a_ptr,
    b_ptr,
    numel,
    BLOCK: tl.constexpr,
    COMMON_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    b = tl.load(b_ptr).to(COMMON_DT)
    a = tl.load(a_ptr + offs, mask=mask).to(COMMON_DT)
    r = _emit_out(a != b, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _ne_bnum(
    a_ptr,
    numel,
    b_val,
    BLOCK: tl.constexpr,
    COMMON_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    b = tl.full([BLOCK], b_val, dtype=COMMON_DT)
    a = tl.load(a_ptr + offs, mask=mask).to(COMMON_DT)
    r = _emit_out(a != b, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _ne_mod(
    a_ptr,
    b_ptr,
    numel,
    b_numel,
    BLOCK: tl.constexpr,
    COMMON_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    b_offs = offs % b_numel
    a = tl.load(a_ptr + offs, mask=mask).to(COMMON_DT)
    b = tl.load(b_ptr + b_offs, mask=mask).to(COMMON_DT)
    r = _emit_out(a != b, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + offs, r, mask=mask)


@triton.jit
def _ne_general(
    a_ptr,
    b_ptr,
    numel,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6,
    a7,
    b0,
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
    COMMON_DT: tl.constexpr,
    OUT_DT: tl.constexpr,
    OUT_IS_BOOL: tl.constexpr,
    OUT_IS_FP: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < numel
    i = offs
    a_off = tl.zeros([BLOCK], dtype=tl.int64)
    b_off = tl.zeros([BLOCK], dtype=tl.int64)
    if RANK > 0:
        idx = i % s0
        i = i // s0
        a_off += idx * a0
        b_off += idx * b0
    if RANK > 1:
        idx = i % s1
        i = i // s1
        a_off += idx * a1
        b_off += idx * b1
    if RANK > 2:
        idx = i % s2
        i = i // s2
        a_off += idx * a2
        b_off += idx * b2
    if RANK > 3:
        idx = i % s3
        i = i // s3
        a_off += idx * a3
        b_off += idx * b3
    if RANK > 4:
        idx = i % s4
        i = i // s4
        a_off += idx * a4
        b_off += idx * b4
    if RANK > 5:
        idx = i % s5
        i = i // s5
        a_off += idx * a5
        b_off += idx * b5
    if RANK > 6:
        idx = i % s6
        i = i // s6
        a_off += idx * a6
        b_off += idx * b6
    if RANK > 7:
        idx = i % s7
        i = i // s7
        a_off += idx * a7
        b_off += idx * b7
    a = tl.load(a_ptr + a_off, mask=mask).to(COMMON_DT)
    b = tl.load(b_ptr + b_off, mask=mask).to(COMMON_DT)
    r = _emit_out(a != b, BLOCK, OUT_DT, OUT_IS_BOOL, OUT_IS_FP)
    tl.store(a_ptr + a_off, r, mask=mask)


def _broadcasts_to(a_shape, b_shape):
    if len(b_shape) > len(a_shape):
        return False
    for i in range(1, len(b_shape) + 1):
        if b_shape[-i] != 1 and b_shape[-i] != a_shape[-i]:
            return False
    return True


def _suffix_ok(a_shape, b_shape):
    """True if b_offs = flat_index % b_numel is the correct B offset for the
    flat index (B contiguous, broadcast only via leading size-1 dims)."""
    rb = len(b_shape)
    if rb > len(a_shape):
        return False
    first = 0
    while first < rb and b_shape[first] == 1:
        first += 1
    return a_shape[len(a_shape) - (rb - first) :] == b_shape[first:]


def _bcast_strides(a_shape, b_shape, b_stride):
    ra, rb = len(a_shape), len(b_shape)
    off = ra - rb
    res = []
    for i in range(ra):
        if i < off:
            res.append(0)
        else:
            j = i - off
            res.append(0 if b_shape[j] == 1 else b_stride[j])
    return res


def not_equal_scalar_(A, B):
    a = A
    numel = a.numel()
    if numel == 0:
        return a

    a_dt = a.dtype
    out_tl = _TL_DT[a_dt]
    out_is_bool = a_dt == torch.bool
    out_is_fp = a_dt in _FP_DTYPES

    if not torch.is_tensor(B):
        if isinstance(B, bool):
            sdt = torch.bool
        elif isinstance(B, int):
            sdt = torch.int64
        elif isinstance(B, float):
            sdt = torch.float32
        else:
            raise TypeError("ne_: unsupported scalar B type %r" % type(B))
        common_tl = _TL_DT[torch.promote_types(a_dt, sdt)]
        _ne_bnum[(triton.cdiv(numel, _BLOCK),)](
            a,
            numel,
            B,
            BLOCK=_BLOCK,
            COMMON_DT=common_tl,
            OUT_DT=out_tl,
            OUT_IS_BOOL=out_is_bool,
            OUT_IS_FP=out_is_fp,
            num_warps=_NUM_WARPS,
        )
        return a

    b = B
    # Hot path: same dtype + same shape + contiguous + int32-safe numel.
    # Skips promote_types / broadcasts_to / suffix_ok / b.numel() / grid.
    if (
        numel < (1 << 31)
        and a_dt == b.dtype
        and a.shape == b.shape
        and a.is_contiguous()
        and b.is_contiguous()
    ):
        if numel < _T2_MIN:
            # Small sizes: one tile per program with a smaller block (one
            # 128-bit vector op per thread) keeps more CTAs for latency
            # hiding (eval: 5.89us vs 6.4us at 4096 elems for the 2-tile form).
            gs = (triton.cdiv(numel, _SMALL_BLOCK),)
            _ne_flat[gs](
                a,
                b,
                numel,
                BLOCK=_SMALL_BLOCK,
                COMMON_DT=out_tl,
                OUT_DT=out_tl,
                OUT_IS_BOOL=out_is_bool,
                OUT_IS_FP=out_is_fp,
                num_warps=_NUM_WARPS,
            )
            return a
        # Large sizes: two tiles per program. Measured 1.1-2.1% faster than
        # one-tile at 16M..1G elements. Warp count by element width: 4-byte
        # items want 8 warps (1528 GB/s @1G fp32 vs 1497), 2-byte items want
        # 4 warps (1524 GB/s @1G fp16/bf16 vs 1497).
        nw = 8 if a_dt.itemsize == 4 else _NUM_WARPS
        grid2 = (triton.cdiv(numel, 2 * _BLOCK),)
        _ne_flat2[grid2](
            a,
            b,
            numel,
            BLOCK=_BLOCK,
            COMMON_DT=out_tl,
            OUT_DT=out_tl,
            OUT_IS_BOOL=out_is_bool,
            OUT_IS_FP=out_is_fp,
            num_warps=nw,
        )
        return a

    if not _broadcasts_to(a.shape, b.shape):
        raise RuntimeError("ne_: B is not broadcastable to A")
    common_tl = (
        out_tl if a_dt == b.dtype else _TL_DT[torch.promote_types(a_dt, b.dtype)]
    )
    b_numel = b.numel()
    grid = (triton.cdiv(numel, _BLOCK),)

    if numel >= 2**31:
        # int64-offset path for huge tensors
        rank = a.dim()
        sizes = list(reversed(a.shape)) + [1] * (8 - rank)
        sa = list(reversed(a.stride())) + [0] * (8 - rank)
        sb = list(reversed(_bcast_strides(a.shape, b.shape, b.stride()))) + [0] * (
            8 - rank
        )
        _ne_general[grid](
            a,
            b,
            numel,
            *sizes,
            *sa,
            *sb,
            RANK=rank,
            BLOCK=_BLOCK,
            COMMON_DT=common_tl,
            OUT_DT=out_tl,
            OUT_IS_BOOL=out_is_bool,
            OUT_IS_FP=out_is_fp,
            num_warps=_NUM_WARPS,
        )
        return a

    if b_numel == 1:
        _ne_bscalar[grid](
            a,
            b,
            numel,
            BLOCK=_BLOCK,
            COMMON_DT=common_tl,
            OUT_DT=out_tl,
            OUT_IS_BOOL=out_is_bool,
            OUT_IS_FP=out_is_fp,
            num_warps=_NUM_WARPS,
        )
        return a

    a_contig = a.is_contiguous()
    b_contig = b.is_contiguous()
    if a_contig and b_contig and _suffix_ok(a.shape, b.shape):
        _ne_mod[grid](
            a,
            b,
            numel,
            b_numel,
            BLOCK=_BLOCK,
            COMMON_DT=common_tl,
            OUT_DT=out_tl,
            OUT_IS_BOOL=out_is_bool,
            OUT_IS_FP=out_is_fp,
            num_warps=_NUM_WARPS,
        )
        return a

    rank = a.dim()
    sizes = list(reversed(a.shape)) + [1] * (8 - rank)
    sa = list(reversed(a.stride())) + [0] * (8 - rank)
    sb = list(reversed(_bcast_strides(a.shape, b.shape, b.stride()))) + [0] * (8 - rank)
    _ne_general[grid](
        a,
        b,
        numel,
        *sizes,
        *sa,
        *sb,
        RANK=rank,
        BLOCK=_BLOCK,
        COMMON_DT=common_tl,
        OUT_DT=out_tl,
        OUT_IS_BOOL=out_is_bool,
        OUT_IS_FP=out_is_fp,
        num_warps=_NUM_WARPS,
    )
    return a
