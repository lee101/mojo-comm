"""Byte-level buffer kernels for the Python-facing `comm` comm-buffer subset.

`comm` is a messaging protocol, not a numeric library: the whole of
`comm.base_comm` is dict plumbing around `publish_msg`. The only per-message
work with a byte loop in it is the `buffers` payload, which
`comm.base_comm` types as `BuffersType = Optional[List[bytes]]` and which
callers (ipywidgets, jupyter kernels) fill with the raw memory of numpy
arrays. That is the surface ported here.

Every exported symbol takes buffer addresses as plain `Int` values and
rebuilds the pointer inside the body, because `@export` rejects parametric
functions and an inferred pointer origin would make the symbol parametric.

These kernels move bytes; they never interpret them. A byte copy is exact, so
the parity tests for this port assert byte equality rather than a tolerance.
"""

comptime U8 = Pointer[UInt8, AnyOrigin[mut=True]]
comptime U64 = Pointer[UInt64, AnyOrigin[mut=True]]
comptime IT = Pointer[Int, AnyOrigin[mut=True]]


def bp(addr: Int) -> U8:
    return U8(unsafe_from_address=addr)


def copy_block(dst_addr: Int, src_addr: Int, n: Int):
    """Copy `n` bytes from `src_addr` to `dst_addr`.

    Address-based rather than pointer-based so the same range can be opened
    as `UInt64` and moved eight bytes per iteration. A byte-at-a-time loop
    measured at roughly a quarter of glibc's memcpy rate and these kernels
    are pure memcpy; the sub-word remainder is copied a byte at a time.
    Pointer deref is byte-aligned, so an unaligned source address is fine.
    """
    var words = n // 8
    if words > 0:
        var d = U64(unsafe_from_address=dst_addr)
        var s = U64(unsafe_from_address=src_addr)
        for i in range(words):
            d[unsafe_offset=i] = s[unsafe_offset=i]
    var i = words * 8
    while i < n:
        bp(dst_addr)[unsafe_offset=i] = bp(src_addr)[unsafe_offset=i]
        i += 1


@export("comm_concat")
def comm_concat(
    ptr_addr: Int, len_addr: Int, count: Int, dst_addr: Int
) abi("C"):
    """Gather `count` separate byte blocks into one contiguous buffer.

    `ptr` is an int64 table of *absolute* source addresses and `len` the
    matching byte lengths, so the sources may be unrelated allocations --
    which is exactly the case for a comm message, whose `buffers` are
    independent `bytes` objects. The result is the concatenation of
    `bytes(ptr[i] .. ptr[i] + len[i])` for `i` in `0..count`, and the caller
    sizes the destination as `sum(len)`.
    """
    var ptrs = IT(unsafe_from_address=ptr_addr)
    var lens = IT(unsafe_from_address=len_addr)
    var at = 0
    for i in range(count):
        var n = lens[unsafe_offset=i]
        copy_block(dst_addr + at, ptrs[unsafe_offset=i], n)
        at += n


@export("comm_split")
def comm_split(
    src_addr: Int, ptr_addr: Int, len_addr: Int, count: Int, dst_addr: Int,
    dst_off_addr: Int
) abi("C"):
    """Scatter `count` byte ranges out of one contiguous buffer.

    Source `ptr[i] .. ptr[i] + len[i])` of `src` is written to
    `dst[dst_off[i] .. dst_off[i] + len[i])`. `dst_off` is supplied by the
    caller, normally the exclusive prefix sum of `len`.
    """
    var ptrs = IT(unsafe_from_address=ptr_addr)
    var lens = IT(unsafe_from_address=len_addr)
    var offs = IT(unsafe_from_address=dst_off_addr)
    for i in range(count):
        var n = lens[unsafe_offset=i]
        copy_block(
            dst_addr + offs[unsafe_offset=i], src_addr + ptrs[unsafe_offset=i], n
        )


@export("comm_strided_gather")
def comm_strided_gather(
    dst_addr: Int, src_addr: Int, itemsize: Int, ndim: Int, shape_addr: Int,
    src_stride_addr: Int, dst_stride_addr: Int
) abi("C"):
    """Copy an `ndim`-dimensional strided region of `src` into `dst`.

    `shape`, `src_stride` and `dst_stride` are int64 arrays of length `ndim`
    with strides already expressed in *bytes*. Iteration is row-major over
    `shape`. The outermost `ndim - 1` dimensions are walked as a flat
    counter; the innermost dimension is copied `itemsize` bytes at a time,
    and collapses to a single block copy when its source stride is already
    `itemsize` (the contiguous case).

    `ndim` is clamped to 4; the Python shim rejects anything larger.
    """
    var shape = IT(unsafe_from_address=shape_addr)
    var sst = IT(unsafe_from_address=src_stride_addr)
    var dts = IT(unsafe_from_address=dst_stride_addr)
    var nd = ndim
    if nd > 4:
        nd = 4
    if nd <= 0:
        return

    var inner_n = shape[unsafe_offset=nd - 1]
    var inner_s = sst[unsafe_offset=nd - 1]
    var inner_d = dts[unsafe_offset=nd - 1]
    var outer_total = 1
    for d in range(nd - 1):
        outer_total *= shape[unsafe_offset=d]
    if outer_total <= 0 or inner_n <= 0:
        return

    var contiguous = inner_s == itemsize
    for o in range(outer_total):
        var rem = o
        var s_off = 0
        var d_off = 0
        for d in range(nd - 2, -1, -1):
            var ext = shape[unsafe_offset=d]
            var idx = 0
            if ext > 0:
                idx = rem % ext
                rem = rem // ext
            s_off += idx * sst[unsafe_offset=d]
            d_off += idx * dts[unsafe_offset=d]
        if contiguous:
            copy_block(dst_addr + d_off, src_addr + s_off, inner_n * itemsize)
        else:
            for k in range(inner_n):
                copy_block(
                    dst_addr + d_off + k * inner_d,
                    src_addr + s_off + k * inner_s,
                    itemsize,
                )


@export("comm_prefix_sum")
def comm_prefix_sum(len_addr: Int, count: Int, out_addr: Int) abi("C"):
    """Exclusive prefix sum of `count` int64 lengths into `out`.

    `out[0] == 0`, `out[i] == out[i-1] + len[i-1]`. This is the byte-offset
    table a comm needs to say where each buffer sits in a flat arena.
    """
    var lens = IT(unsafe_from_address=len_addr)
    var dst = IT(unsafe_from_address=out_addr)
    var at = 0
    for i in range(count):
        dst[unsafe_offset=i] = at
        at += lens[unsafe_offset=i]
