"""ctypes bridge to the compiled Mojo buffer kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a
64-bit address, so the address argtypes below must stay `c_int64`; `c_int`
truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-comm.so"

_MAX_NDIM = 4


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    for name in ("comm_concat", "comm_split", "comm_strided_gather"):
        getattr(lib, name).restype = None
    lib.comm_concat.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.comm_split.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64,
    ]
    lib.comm_strided_gather.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    lib.comm_prefix_sum.restype = None
    lib.comm_prefix_sum.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
    ]
    return lib


lib = _load()


def _addr(buf) -> int:
    """Byte address of a contiguous buffer, as a 64-bit C integer."""
    return ctypes.c_int64(buf.ctypes.data).value


def _flat(ar: np.ndarray) -> np.ndarray:
    """Contiguous 1-D uint8 view of an array's raw bytes."""
    return np.ascontiguousarray(ar).view(np.uint8).reshape(-1)


def concat_buffers(buffers) -> bytes:
    """Concatenate a list of buffers into one flat byte blob.

    This is the shape a comm hands to a transport that wants a single write
    for all of a message's buffers. Mirrors ``b"".join(buffers)`` byte for
    byte; the copy loop lives in Mojo and the sources are read in place,
    never staged through a second copy.
    """
    items = list(buffers)
    if not items:
        return b""
    views, ptrs, lens = _address_table(items)
    arena = np.empty(int(lens.sum()), dtype=np.uint8)
    lib.comm_concat(_addr(ptrs), _addr(lens), len(items), _addr(arena))
    return arena.tobytes()


def concat_buffers_into(arena: np.ndarray, buffers) -> int:
    """Coalesce buffers into a caller-owned arena, returning bytes written.

    A comm that reuses one staging buffer across messages should use this
    rather than :func:`concat_buffers`: the destination is allocated once,
    so a batch of small buffers is not paying for a fresh multi-megabyte
    allocation on every message. `arena` must be a contiguous uint8 array
    of at least `sum(len(b) for b in buffers)` bytes.
    """
    items = list(buffers)
    if not items:
        return 0
    views, ptrs, lens = _address_table(items)
    total = int(lens.sum())
    if total > arena.size:
        raise ValueError(f"arena holds {arena.size} bytes, buffers need {total}")
    lib.comm_concat(_addr(ptrs), _addr(lens), len(items), _addr(arena))
    return total


def _address_table(items):
    """Absolute source addresses and byte lengths for a list of buffers."""
    views = []  # zero-copy uint8 views; also pin the source addresses
    ptrs = np.empty(len(items), dtype=np.int64)
    lens = np.empty(len(items), dtype=np.int64)
    for i, b in enumerate(items):
        v = np.frombuffer(b, dtype=np.uint8)
        views.append(v)
        ptrs[i] = v.ctypes.data
        lens[i] = v.size
    return views, ptrs, lens


def split_buffers(blob: bytes, lengths) -> list:
    """Cut one flat byte blob back into buffers of the given lengths.

    The inverse of :func:`concat_buffers`.
    """
    lens = np.ascontiguousarray(lengths, dtype=np.int64)
    count = int(lens.size)
    if count == 0:
        return []
    src = np.ascontiguousarray(np.frombuffer(blob, dtype=np.uint8))
    total = int(lens.sum())
    if total != src.size:
        raise ValueError(
            f"buffer lengths sum to {total}, blob holds {src.size} bytes"
        )
    dst = np.empty(total, dtype=np.uint8)
    ptrs = np.ascontiguousarray(
        np.cumsum(np.concatenate(([0], lens[:-1]))), dtype=np.int64
    )
    lib.comm_split(
        _addr(src), _addr(ptrs), _addr(lens), count, _addr(dst), _addr(ptrs)
    )
    return [dst[ptrs[i]:ptrs[i] + lens[i]].tobytes() for i in range(count)]


def buffer_offsets(lengths) -> np.ndarray:
    """Exclusive prefix sum of buffer byte lengths.

    The offset table describing where each buffer lands in a flat arena.
    """
    lens = np.ascontiguousarray(lengths, dtype=np.int64)
    out = np.empty(lens.size, dtype=np.int64)
    lib.comm_prefix_sum(_addr(lens), int(lens.size), _addr(out))
    return out


def pack_buffer(ar: np.ndarray) -> bytes:
    """Serialise one array to the contiguous C-order bytes a comm carries.

    Contiguous inputs are returned as a view, no copy. Non-contiguous inputs
    (the case with a loop in it: a sliced widget buffer) are gathered by the
    Mojo strided kernel in one pass.
    """
    if ar.ndim > _MAX_NDIM:
        raise ValueError(f"comm buffers support at most {_MAX_NDIM} dimensions")
    if ar.flags["C_CONTIGUOUS"]:
        return ar.tobytes()
    return _strided_gather(ar).tobytes()


def _strided_gather(ar: np.ndarray) -> np.ndarray:
    """Gather an arbitrary strided view into a fresh C-contiguous array."""
    ar = np.asanyarray(ar)
    nd = max(ar.ndim, 1)
    if ar.ndim == 0:
        ar = ar.reshape(1)
        nd = 1
    if nd > _MAX_NDIM:
        raise ValueError(f"comm buffers support at most {_MAX_NDIM} dimensions")
    shape = np.ascontiguousarray(ar.shape, dtype=np.int64)
    sstride = np.ascontiguousarray(ar.strides, dtype=np.int64)
    dstride = np.ascontiguousarray(
        np.array(
            [
                ar.itemsize * int(np.prod(ar.shape[i + 1:], dtype=np.int64))
                for i in range(nd)
            ],
            dtype=np.int64,
        )
    )
    dst = np.empty(int(ar.size), dtype=ar.dtype)
    lib.comm_strided_gather(
        ctypes.c_int64(dst.ctypes.data).value,
        ctypes.c_int64(ar.ctypes.data).value,
        ctypes.c_int64(ar.itemsize),
        ctypes.c_int64(nd),
        _addr(shape),
        _addr(sstride),
        _addr(dstride),
    )
    return dst


def unpack_buffer(blob: bytes, dtype, shape=(-1,)) -> np.ndarray:
    """Rebuild an array from a comm buffer and its dtype/shape descriptor."""
    ar = np.frombuffer(blob, dtype=dtype)
    return ar.reshape(shape)


def pack_buffers(arrays) -> list:
    """Serialise a batch of arrays, one comm buffer each."""
    return [pack_buffer(a) for a in arrays]
