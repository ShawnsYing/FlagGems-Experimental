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

"""Triton implementation of aten::_foreach_round (round-half-to-even)."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _libdev

_BLOCK = 1024
_MAX_RANK = 8
_WARPS = 4
# Small contiguous same-dtype tensors are fused into one launch: up to
# _GROUP_K tensors per launch while every tensor stays below _GROUP_MAX
# elements (above that the per-launch cost is negligible anyway).
_GROUP_K = 16
_GROUP_MAX = 1 << 20


@triton.jit
def _round_he(x):
    """round-half-to-even via rint/nearbyint.

    Target-probed bit-exact against torch.round (fp16/bf16 are widened to
    fp32 exactly) including +-0.0, +-inf and NaN, in a single libdevice call
    instead of a floor/compare/select/fixup sequence.
    """
    return _libdev.rint(x)


# Bounded configuration search for the streaming kernel. The key encodes the
# dtype class and a size bucket (<=1M, <=64M, larger), so a winner chosen for a
# 16.7M-element shape is never reused for a 1e9-element one. Winners are cached
# per process after the first invocation of each key.
_CONFIGS = [
    triton.Config({"BLOCK": 1024}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 2048}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK": 4096}, num_warps=8, num_stages=1),
    triton.Config({"BLOCK": 8192}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_CONFIGS, key=["cls"])
@triton.jit
def _round_contig_kernel(
    x_ptr,
    out_ptr,
    numel,
    cls,
    OTYPE: tl.constexpr,
    CTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    I64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if I64:
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    else:
        offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    if IS_FLOAT:
        res = _round_he(x.to(CTYPE)).to(OTYPE)
    else:
        res = x
    tl.store(out_ptr + offs, res, mask=mask)


@triton.jit
def _round_group_kernel(
    x0, x1, x2, x3, x4, x5, x6, x7,
    x8, x9, x10, x11, x12, x13, x14, x15,
    o0, o1, o2, o3, o4, o5, o6, o7,
    o8, o9, o10, o11, o12, o13, o14, o15,
    n0, n1, n2, n3, n4, n5, n6, n7,
    n8, n9, n10, n11, n12, n13, n14, n15,
    c0, c1, c2, c3, c4, c5, c6, c7,
    c8, c9, c10, c11, c12, c13, c14,
    OTYPE: tl.constexpr,
    CTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch for a group of small same-dtype contiguous tensors.

    Grid covers the total block count of the group; the owning tensor of this
    program is found by comparing the program id against cumulative block
    counts, so all tensors of the group stream through a single launch.
    """
    pid = tl.program_id(0)
    xp = x0
    op = o0
    n = n0
    base = 0
    if pid >= c0:
        xp = x1
        op = o1
        n = n1
        base = c0
    if pid >= c1:
        xp = x2
        op = o2
        n = n2
        base = c1
    if pid >= c2:
        xp = x3
        op = o3
        n = n3
        base = c2
    if pid >= c3:
        xp = x4
        op = o4
        n = n4
        base = c3
    if pid >= c4:
        xp = x5
        op = o5
        n = n5
        base = c4
    if pid >= c5:
        xp = x6
        op = o6
        n = n6
        base = c5
    if pid >= c6:
        xp = x7
        op = o7
        n = n7
        base = c6
    if pid >= c7:
        xp = x8
        op = o8
        n = n8
        base = c7
    if pid >= c8:
        xp = x9
        op = o9
        n = n9
        base = c8
    if pid >= c9:
        xp = x10
        op = o10
        n = n10
        base = c9
    if pid >= c10:
        xp = x11
        op = o11
        n = n11
        base = c10
    if pid >= c11:
        xp = x12
        op = o12
        n = n12
        base = c11
    if pid >= c12:
        xp = x13
        op = o13
        n = n13
        base = c12
    if pid >= c13:
        xp = x14
        op = o14
        n = n14
        base = c13
    if pid >= c14:
        xp = x15
        op = o15
        n = n15
        base = c14
    offs = (pid - base) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(xp + offs, mask=mask, other=0)
    if IS_FLOAT:
        res = _round_he(x.to(CTYPE)).to(OTYPE)
    else:
        res = x
    tl.store(op + offs, res, mask=mask)


@triton.jit
def _round_strided_kernel(
    x_ptr,
    out_ptr,
    last_dim,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    d0,
    d1,
    d2,
    d3,
    d4,
    d5,
    d6,
    OTYPE: tl.constexpr,
    CTYPE: tl.constexpr,
    IS_FLOAT: tl.constexpr,
    OUT_LINEAR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rem = row
    c6 = rem % d6
    rem = rem // d6
    c5 = rem % d5
    rem = rem // d5
    c4 = rem % d4
    rem = rem // d4
    c3 = rem % d3
    rem = rem // d3
    c2 = rem % d2
    rem = rem // d2
    c1 = rem % d1
    rem = rem // d1
    c0 = rem % d0
    in_off = (
        c0.to(tl.int64) * s0
        + c1.to(tl.int64) * s1
        + c2.to(tl.int64) * s2
        + c3.to(tl.int64) * s3
        + c4.to(tl.int64) * s4
        + c5.to(tl.int64) * s5
        + c6.to(tl.int64) * s6
    )
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < last_dim
    x = tl.load(x_ptr + in_off + col.to(tl.int64) * s7, mask=mask, other=0)
    if IS_FLOAT:
        res = _round_he(x.to(CTYPE)).to(OTYPE)
    else:
        res = x
    if OUT_LINEAR:
        out_off = row.to(tl.int64) * last_dim + col.to(tl.int64)
    else:
        out_off = in_off + col.to(tl.int64) * s7
    tl.store(out_ptr + out_off, res, mask=mask)


_DTYPES = {
    torch.float16: (tl.float16, tl.float32, True, "Half"),
    torch.bfloat16: (tl.bfloat16, tl.float32, True, "BFloat16"),
    torch.float32: (tl.float32, tl.float32, True, "Float"),
    torch.float64: (tl.float64, tl.float64, True, "Double"),
    torch.int64: (tl.int64, tl.int64, False, "Long"),
    torch.int32: (tl.int32, tl.int32, False, "Int"),
    torch.int16: (tl.int16, tl.int16, False, "Short"),
    torch.int8: (tl.int8, tl.int8, False, "Char"),
    torch.uint8: (tl.uint8, tl.uint8, False, "Byte"),
}

_DTYPE_CLS = {
    torch.float16: 0,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
}


def _collapse(shape, strides):
    cs = []
    ct = []
    for sz, st in zip(shape, strides):
        if sz == 1:
            continue
        if cs and ct[-1] == st * sz:
            cs[-1] *= sz
            ct[-1] = st
        else:
            cs.append(sz)
            ct.append(st)
    return cs, ct


def _launch_group(idxs, tensors, outs):
    """Fuse a chunk of <= _GROUP_K small contiguous same-dtype tensors."""
    t0 = tensors[idxs[0]]
    otype, ctype, is_float, _ = _DTYPES[t0.dtype]
    xs = []
    os_ = []
    ns = []
    cs = []
    cum = 0
    for i in idxs:
        n = tensors[i].numel()
        xs.append(tensors[i])
        os_.append(outs[i])
        ns.append(n)
        cum += (n + _BLOCK - 1) // _BLOCK
        cs.append(cum)
    # pad unused slots with a valid pointer and zero work
    while len(xs) < _GROUP_K:
        xs.append(t0)
        os_.append(os_[0])
        ns.append(0)
        cs.append(cum)
    _round_group_kernel[(cum,)](
        xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7],
        xs[8], xs[9], xs[10], xs[11], xs[12], xs[13], xs[14], xs[15],
        os_[0], os_[1], os_[2], os_[3], os_[4], os_[5], os_[6], os_[7],
        os_[8], os_[9], os_[10], os_[11], os_[12], os_[13], os_[14], os_[15],
        ns[0], ns[1], ns[2], ns[3], ns[4], ns[5], ns[6], ns[7],
        ns[8], ns[9], ns[10], ns[11], ns[12], ns[13], ns[14], ns[15],
        cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], cs[6], cs[7],
        cs[8], cs[9], cs[10], cs[11], cs[12], cs[13], cs[14],
        OTYPE=otype,
        CTYPE=ctype,
        IS_FLOAT=is_float,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _launch_contig(t, out):
    otype, ctype, is_float, _ = _DTYPES[t.dtype]
    numel = t.numel()
    if numel <= _GROUP_MAX:
        bucket = 0
    elif numel <= 1 << 26:
        bucket = 1
    else:
        bucket = 2
    cls = _DTYPE_CLS[t.dtype] * 3 + bucket
    _round_contig_kernel[lambda meta: ((numel + meta["BLOCK"] - 1) // meta["BLOCK"],)](
        t,
        out,
        numel,
        cls,
        OTYPE=otype,
        CTYPE=ctype,
        IS_FLOAT=is_float,
        I64=numel >= 2147483648,
    )


def _round_one(t, out):
    entry = _DTYPES.get(t.dtype)
    if entry is None:
        raise NotImplementedError('"round_cuda" not implemented for \'%s\'' % t.dtype)
    otype, ctype, is_float, _ = entry

    numel = t.numel()
    if numel == 0:
        return out

    if t.is_contiguous():
        _launch_contig(t, out)
        return out

    cs, ct = _collapse(t.shape, t.stride())
    if not cs:
        cs, ct = [1], [1]
    rank = len(cs)
    if rank > _MAX_RANK:
        raise RuntimeError("foreach_round: unsupported tensor rank")

    last = cs[-1]
    rows = numel // last
    out_linear = out.is_contiguous()
    if not out_linear and out.stride() != t.stride():
        out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
        out_linear = True

    block = _BLOCK
    blocks = (last + block - 1) // block
    if blocks > 65535:
        block = triton.next_power_of_2((last + 65534) // 65535)
        blocks = (last + block - 1) // block

    pad = _MAX_RANK - rank
    strides = [0] * pad + ct
    dims = [1] * pad + cs[:-1]

    _round_strided_kernel[(rows, blocks)](
        t,
        out,
        last,
        strides[0],
        strides[1],
        strides[2],
        strides[3],
        strides[4],
        strides[5],
        strides[6],
        strides[7],
        dims[0],
        dims[1],
        dims[2],
        dims[3],
        dims[4],
        dims[5],
        dims[6],
        OTYPE=otype,
        CTYPE=ctype,
        IS_FLOAT=is_float,
        OUT_LINEAR=out_linear,
        BLOCK=block,
        num_warps=_WARPS,
    )
    return out


def foreach_round(self):
    if isinstance(self, torch.Tensor):
        raise TypeError(
            "_foreach_round(): argument 'self' (position 1) must be tuple of Tensors, not Tensor"
        )
    tensors = list(self)
    if not tensors:
        raise RuntimeError("Tensor list must have at least one tensor.")

    outs = [torch.empty_like(t, memory_format=torch.preserve_format) for t in tensors]
    groups = {}
    for i, t in enumerate(tensors):
        if _DTYPES.get(t.dtype) is None:
            raise NotImplementedError(
                '"round_cuda" not implemented for \'%s\'' % t.dtype
            )
        n = t.numel()
        if n == 0:
            continue
        if t.is_contiguous():
            if n <= _GROUP_MAX:
                g = groups.get(t.dtype)
                if g is None:
                    groups[t.dtype] = [i]
                else:
                    g.append(i)
            else:
                _launch_contig(t, outs[i])
        else:
            outs[i] = _round_one(t, outs[i])

    for idxs in groups.values():
        if len(idxs) == 1:
            i = idxs[0]
            _launch_contig(tensors[i], outs[i])
            continue
        for s in range(0, len(idxs), _GROUP_K):
            chunk = idxs[s : s + _GROUP_K]
            if len(chunk) == 1:
                i = chunk[0]
                _launch_contig(tensors[i], outs[i])
            else:
                _launch_group(chunk, tensors, outs)

    return tuple(outs)
