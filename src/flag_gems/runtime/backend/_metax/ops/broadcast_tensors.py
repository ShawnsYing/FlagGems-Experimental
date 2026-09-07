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


"""broadcast_tensors: broadcast a sequence of tensors to a common shape.

Entry point: run(*tensors) -> tuple of broadcasted tensors.

Implementation: one fused Triton kernel per call writes every output.
The host only allocates outputs, reads tensor metadata, computes scalar/grid
values, and launches the JIT kernel.

Kernel structure (memory-bound streaming copy):
  * 2D grid: axis0 = blocks over the inner (contiguous) run of output dims,
    axis1 = outer coordinates. Outer coordinates are decomposed once per
    program (not per element).
  * All shape/stride values are tl.constexpr so per-dim math constant-folds
    (broadcast dims vanish) and pointer alignment/divisibility is provable for
    128-bit vectorized loads and stores.
  * Per input, three inner-source modes: identity (+offs, vectorized),
    all-broadcast (+0), or general mixed-radix fallback (small tensors).
"""

import torch
import triton
import triton.language as tl

_BLOCK_MAX = 1024


@triton.jit
def _outer_base(
    pid1,
    SH: tl.constexpr,
    OS: tl.constexpr,
    SK: tl.constexpr,
    TK: tl.constexpr,
    NOD: tl.constexpr,
):
    """Source offset contributed by the outer dims for one input (per program)."""
    acc = pid1 * 0
    for d in tl.static_range(NOD):
        c = (pid1 // OS[d]) % SH[d]
        acc += tl.minimum(c, SK[d] - 1) * TK[d]
    return acc


@triton.jit
def _inner_general(
    offs,
    SH: tl.constexpr,
    IS: tl.constexpr,
    SK: tl.constexpr,
    TK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Source offset contributed by the inner run for one input (per element).

    All tuples are host-sliced to length SPLIT (indexed by the loop var j).
    """
    acc = offs * 0
    for j in tl.static_range(SPLIT):
        c = (offs // IS[j]) % SH[j]
        acc += tl.minimum(c, SK[j] - 1) * TK[j]
    return acc


@triton.jit
def _bcast_fused(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    NDIM: tl.constexpr,
    NOD: tl.constexpr,
    SPLIT: tl.constexpr,
    N_IN: tl.constexpr,
    IM0: tl.constexpr,
    IM1: tl.constexpr,
    IM2: tl.constexpr,
    BLOCK: tl.constexpr,
    SH: tl.constexpr,
    OS: tl.constexpr,
    IS: tl.constexpr,
    SK0: tl.constexpr,
    TK0: tl.constexpr,
    SK1: tl.constexpr,
    TK1: tl.constexpr,
    SK2: tl.constexpr,
    TK2: tl.constexpr,
    SHI: tl.constexpr,
    ISI: tl.constexpr,
    SKI0: tl.constexpr,
    TKI0: tl.constexpr,
    SKI1: tl.constexpr,
    TKI1: tl.constexpr,
    SKI2: tl.constexpr,
    TKI2: tl.constexpr,
    numel: tl.constexpr,
    P_outer: tl.constexpr,
    R: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    offs = pid0 * BLOCK + tl.arange(0, BLOCK)
    total = pid1 * R + offs
    mask = total < numel

    base0 = _outer_base(pid1, SH, OS, SK0, TK0, NOD)
    if IM0 == 1:
        src0 = base0 + offs
    elif IM0 == 2:
        src0 = base0 + offs * 0
    else:
        src0 = base0 + _inner_general(offs, SHI, ISI, SKI0, TKI0, SPLIT)
    v0 = tl.load(in0_ptr + src0, mask=mask)
    tl.store(out0_ptr + total, v0, mask=mask)

    if N_IN >= 2:
        base1 = _outer_base(pid1, SH, OS, SK1, TK1, NOD)
        if IM1 == 1:
            src1 = base1 + offs
        elif IM1 == 2:
            src1 = base1 + offs * 0
        else:
            src1 = base1 + _inner_general(offs, SHI, ISI, SKI1, TKI1, SPLIT)
        v1 = tl.load(in1_ptr + src1, mask=mask)
        tl.store(out1_ptr + total, v1, mask=mask)

    if N_IN >= 3:
        base2 = _outer_base(pid1, SH, OS, SK2, TK2, NOD)
        if IM2 == 1:
            src2 = base2 + offs
        elif IM2 == 2:
            src2 = base2 + offs * 0
        else:
            src2 = base2 + _inner_general(offs, SHI, ISI, SKI2, TKI2, SPLIT)
        v2 = tl.load(in2_ptr + src2, mask=mask)
        tl.store(out2_ptr + total, v2, mask=mask)


def _padded(t, ndim):
    """Right-aligned padded (input_shape, input_stride) tuples of length ndim."""
    off = ndim - t.dim()
    s = [1] * ndim
    st = [0] * ndim
    for i in range(t.dim()):
        s[off + i] = t.shape[i]
        st[off + i] = t.stride(i)
    return tuple(s), tuple(st)


# ---------------------------------------------------------------------------
# Tiled broadcast path (2-input "row x column" pattern).
#
# For the pattern in0 = (1, S1, [S2..])  (broadcasts over dim 0 only) and
# in1 = (S0, 1, [S2..]) (broadcasts over dim 1 only), each program materializes
# a TI x TJ x TK tile of BOTH outputs while loading only a (TJ x TK) plane from
# in0 and a (TI x TK) strip from in1.  This removes the per-element load and
# index traffic of the fused kernel and, because tiles are exact, needs no
# masks.  The 2D case is a separate kernel so the contiguous store axis is the
# last block axis (vectorizable).
# ---------------------------------------------------------------------------


@triton.jit
def _bcast_tile2d(
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    TI: tl.constexpr,
    TJ: tl.constexpr,
    P1: tl.constexpr,
):
    pid1 = tl.program_id(0)
    it = pid1 // P1
    jt = pid1 % P1
    i0 = it * TI
    j0 = jt * TJ
    r = tl.arange(0, TI)
    s = tl.arange(0, TJ)
    row = tl.load(in0_ptr + j0 + s)
    col = tl.load(in1_ptr + i0 + r)
    offs = (i0 + r)[:, None] * S1 + (j0 + s)[None, :]
    tl.store(out0_ptr + offs, tl.broadcast_to(row[None, :], (TI, TJ)))
    tl.store(out1_ptr + offs, tl.broadcast_to(col[:, None], (TI, TJ)))


@triton.jit
def _bcast_tile3d(
    in0_ptr,
    in1_ptr,
    out0_ptr,
    out1_ptr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    K: tl.constexpr,
    TI: tl.constexpr,
    TJ: tl.constexpr,
    TK: tl.constexpr,
    P1: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)
    it = pid1 // P1
    jt = pid1 % P1
    i0 = it * TI
    j0 = jt * TJ
    k0 = pid0 * TK
    r = tl.arange(0, TI)
    s = tl.arange(0, TJ)
    t = tl.arange(0, TK)
    plane = tl.load(in0_ptr + (j0 + s)[:, None] * K + (k0 + t)[None, :])
    strip = tl.load(in1_ptr + (i0 + r)[:, None] * K + (k0 + t)[None, :])
    offs = (
        (i0 + r)[:, None, None] * (S1 * K)
        + (j0 + s)[None, :, None] * K
        + (k0 + t)[None, None, :]
    )
    tl.store(out0_ptr + offs, tl.broadcast_to(plane[None, :, :], (TI, TJ, TK)))
    tl.store(out1_ptr + offs, tl.broadcast_to(strip[:, None, :], (TI, TJ, TK)))


def _tile_cfg2d(S0, S1):
    def pick(lim):
        v = 1
        while v * 2 <= lim and v * 2 <= 128:
            v *= 2
        return v

    TI = pick(S0)
    TJ = pick(S1)
    while TI * TJ > 4096:
        if TI >= TJ:
            TI //= 2
        else:
            TJ //= 2
    while (S0 // TI) * (S1 // TJ) < 16 and (TI > 1 or TJ > 1):
        if TI >= TJ:
            TI //= 2
        else:
            TJ //= 2
    if S0 % TI or S1 % TJ:
        return None
    return TI, TJ


def _tile_cfg3d(S0, S1, K):
    def pick_t(lim):
        v = 1
        while v * 2 <= lim and v * 2 <= 16:
            v *= 2
        return v

    TI = pick_t(S0)
    TJ = pick_t(S1)
    TK = 1
    while TK * 2 <= K:
        TK *= 2
    while TI * TJ * TK > 32768:
        TK //= 2
    if S0 % TI or S1 % TJ or K % TK:
        return None
    return TI, TJ, TK


def _try_tile(tensors, SH, ndim):
    """Build a tiled 2-input launch closure when the pattern matches.

    Returns a callable plan(tensors, outs) or None when the tiled path does
    not apply.  Pure function of tensor metadata so it can be cached.
    """
    n = len(tensors)
    if n != 2 or ndim < 2:
        return None
    t0, t1 = tensors
    if not t0.is_contiguous() or not t1.is_contiguous():
        return None
    K = 1
    for d in range(2, ndim):
        K *= SH[d]

    sk0, tk0 = _padded(t0, ndim)
    sk1, tk1 = _padded(t1, ndim)

    def plane_role(sk, tk):
        if sk[0] != 1:
            return False
        for d in range(1, ndim):
            if sk[d] != SH[d]:
                return False
        if ndim == 2:
            return tk[1] == 1
        if tk[1] != K:
            return False
        for d in range(2, ndim):
            if tk[d] != 1:
                return False
        return True

    def strip_role(sk, tk):
        if sk[0] != SH[0] or sk[1] != 1:
            return False
        for d in range(2, ndim):
            if sk[d] != SH[d]:
                return False
        if ndim == 2:
            return tk[0] == 1
        if tk[0] != K:
            return False
        for d in range(2, ndim):
            if tk[d] != 1:
                return False
        return True

    if plane_role(sk0, tk0) and strip_role(sk1, tk1):
        which = 0
    elif plane_role(sk1, tk1) and strip_role(sk0, tk0):
        which = 1
    else:
        return None

    if ndim == 2:
        # 2-byte dtypes benefit from the tile at any size; 4-byte dtypes only
        # when the launch-argument savings matter (small outputs) -- for large
        # 4-byte outputs the fused kernel already runs at the bandwidth ceiling
        # and the tile is slightly slower.
        if t0.element_size() > 2 and SH[0] * SH[1] >= (1 << 20):
            return None
        cfg = _tile_cfg2d(SH[0], SH[1])
        if cfg is None:
            return None
        TI, TJ = cfg
        P1 = SH[1] // TJ
        grid = ((SH[0] // TI) * P1,)
        nw = max(1, min(8, (TI * TJ) // 256))

        def plan2d(ts, os_):
            _bcast_tile2d[grid](
                ts[which],
                ts[1 - which],
                os_[which],
                os_[1 - which],
                S0=SH[0],
                S1=SH[1],
                TI=TI,
                TJ=TJ,
                P1=P1,
                num_warps=nw,
            )

        return plan2d

    # ndim >= 3: use the tile only when the fused kernel would launch an
    # excessive number of small programs (measured: 1M single-row programs are
    # ~30% slower than the tile on (1024,1024,1024), while 32K programs are
    # already near peak on (64,512,512)).
    R = 1
    split = 0
    for d in range(ndim - 1, -1, -1):
        if split >= 1 and R * SH[d] > 4 * _BLOCK_MAX:
            break
        R *= SH[d]
        split += 1
    P_outer = 1
    for d in range(ndim - split):
        P_outer *= SH[d]
    b = 1
    while b < R and b < _BLOCK_MAX:
        b *= 2
    fused_progs = ((R + b - 1) // b) * P_outer
    if fused_progs <= 262144:
        return None

    cfg = _tile_cfg3d(SH[0], SH[1], K)
    if cfg is None:
        return None
    TI, TJ, TK = cfg
    P1 = SH[1] // TJ
    grid = (K // TK, (SH[0] // TI) * P1)
    nw = max(1, min(8, (TI * TJ * TK) // 256))

    def plan3d(ts, os_):
        _bcast_tile3d[grid](
            ts[which],
            ts[1 - which],
            os_[which],
            os_[1 - which],
            S0=SH[0],
            S1=SH[1],
            K=K,
            TI=TI,
            TJ=TJ,
            TK=TK,
            P1=P1,
            num_warps=nw,
        )

    return plan3d


def _make_fused_plan(tensors, SH, ndim, numel):
    """Build the fused-kernel launch closure (general fallback)."""
    n = len(tensors)

    # ---- choose inner/outer split (inner = trailing dims with product ~ BLOCK)
    R = 1
    split = 0
    for d in range(ndim - 1, -1, -1):
        if split >= 1 and R * SH[d] > 4 * _BLOCK_MAX:
            break
        R *= SH[d]
        split += 1
    NOD = ndim - split
    P_outer = 1
    for d in range(NOD):
        P_outer *= SH[d]

    b = 1
    while b < R and b < _BLOCK_MAX:
        b *= 2
    BLOCK = b
    # ~32B of data per thread: for 2-byte dtypes this needs 16 elems/thread
    # (e.g. BLOCK=512 -> 1 warp), for 4-byte 8 elems/thread (BLOCK=512 -> 2
    # warps).  Measured: fp16 (64,512,512) fused runs 16% faster with 1 warp
    # vs 2 (63.5us -> 53.5us in round 7), while fp32 is flat.
    num_warps = max(1, (BLOCK * tensors[0].element_size()) // 1024)

    # outer strides: OS[d] = prod(SH[d+1 .. NOD))
    OS = []
    for d in range(NOD):
        p = 1
        for e in range(d + 1, NOD):
            p *= SH[e]
        OS.append(p)
    OS = tuple(OS)

    # inner strides: IS[dd] = prod(SH[dd+1 .. ndim)) for every dd
    IS = []
    for dd in range(ndim):
        p = 1
        for e in range(dd + 1, ndim):
            p *= SH[e]
        IS.append(p)
    IS = tuple(IS)

    def inner_mode(sk, tk):
        ident = True
        all_one = True
        for j in range(split):
            dd = NOD + j
            if sk[dd] != 1:
                all_one = False
            if sk[dd] != SH[dd] or tk[dd] != IS[dd]:
                ident = False
        if ident:
            return 1
        if all_one:
            return 2
        return 0

    pads = [_padded(t, ndim) for t in tensors]
    ims = [inner_mode(sk, tk) for sk, tk in pads]

    # inner-run slices (indexed by j in [0, SPLIT) inside _inner_general)
    SHI = tuple(SH[NOD:])
    ISI = tuple(IS[NOD:])
    slices = [(tuple(sk[NOD:]), tuple(tk[NOD:])) for sk, tk in pads]

    grid = (triton.cdiv(R, BLOCK), P_outer)
    kw = dict(
        NDIM=ndim,
        NOD=NOD,
        SPLIT=split,
        N_IN=n,
        BLOCK=BLOCK,
        SH=tuple(SH),
        OS=OS,
        IS=IS,
        SHI=SHI,
        ISI=ISI,
        numel=numel,
        P_outer=P_outer,
        R=R,
        num_warps=num_warps,
    )

    if n == 1:
        sk0, tk0 = pads[0]
        si0, ti0 = slices[0]

        def plan1(ts, os_):
            _bcast_fused[grid](
                ts[0],
                ts[0],
                ts[0],
                os_[0],
                os_[0],
                os_[0],
                IM0=ims[0],
                IM1=0,
                IM2=0,
                SK0=sk0,
                TK0=tk0,
                SK1=sk0,
                TK1=tk0,
                SK2=sk0,
                TK2=tk0,
                SKI0=si0,
                TKI0=ti0,
                SKI1=si0,
                TKI1=ti0,
                SKI2=si0,
                TKI2=ti0,
                **kw,
            )

        return plan1
    if n == 2:
        sk0, tk0 = pads[0]
        sk1, tk1 = pads[1]
        si0, ti0 = slices[0]
        si1, ti1 = slices[1]

        def plan2(ts, os_):
            _bcast_fused[grid](
                ts[0],
                ts[1],
                ts[0],
                os_[0],
                os_[1],
                os_[0],
                IM0=ims[0],
                IM1=ims[1],
                IM2=0,
                SK0=sk0,
                TK0=tk0,
                SK1=sk1,
                TK1=tk1,
                SK2=sk0,
                TK2=tk0,
                SKI0=si0,
                TKI0=ti0,
                SKI1=si1,
                TKI1=ti1,
                SKI2=si0,
                TKI2=ti0,
                **kw,
            )

        return plan2
    sk0, tk0 = pads[0]
    sk1, tk1 = pads[1]
    sk2, tk2 = pads[2]
    si0, ti0 = slices[0]
    si1, ti1 = slices[1]
    si2, ti2 = slices[2]

    def plan3(ts, os_):
        _bcast_fused[grid](
            ts[0],
            ts[1],
            ts[2],
            os_[0],
            os_[1],
            os_[2],
            IM0=ims[0],
            IM1=ims[1],
            IM2=ims[2],
            SK0=sk0,
            TK0=tk0,
            SK1=sk1,
            TK1=tk1,
            SK2=sk2,
            TK2=tk2,
            SKI0=si0,
            TKI0=ti0,
            SKI1=si1,
            TKI1=ti1,
            SKI2=si2,
            TKI2=ti2,
            **kw,
        )

    return plan3


_PLAN_CACHE = {}


def _make_plan(tensors, bshape, ndim, numel):
    plan = _try_tile(tensors, bshape, ndim)
    if plan is not None:
        return plan
    kshape = (1,) if ndim == 0 else bshape
    kndim = 1 if ndim == 0 else ndim
    return _make_fused_plan(tensors, kshape, kndim, numel)


def broadcast_tensors(*tensors):
    if len(tensors) == 1 and not isinstance(tensors[0], torch.Tensor):
        tensors = tuple(tensors[0])
    if not tensors:
        return []

    ndim = max(t.dim() for t in tensors)
    bshape = [1] * ndim
    for t in tensors:
        off = ndim - t.dim()
        for i, d in enumerate(t.shape):
            if d > bshape[off + i]:
                bshape[off + i] = d
    bshape = tuple(bshape)

    numel = 1
    for s in bshape:
        numel *= s
    if numel >= (1 << 31):
        raise RuntimeError("broadcast_tensors: output too large for int32 kernel")

    outs = []
    for t in tensors:
        outs.append(torch.empty(bshape, dtype=t.dtype, device=t.device))
    if numel == 0:
        return tuple(outs)

    key = (bshape, tuple((t.shape, t.stride(), t.dtype) for t in tensors))
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        plan = _make_plan(tensors, bshape, ndim, numel)
        _PLAN_CACHE[key] = plan
    plan(tensors, outs)
    return tuple(outs)
