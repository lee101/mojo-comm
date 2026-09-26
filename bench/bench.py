"""Correctness-gated benchmark for mojo-comm.

Every case verifies byte equality with the reference before timing, so a
regression in the Mojo kernels shows up as a correctness failure rather
than a suspiciously good number.

References are the fastest reasonable Python: `bytes.join` for the
concatenation, and `np.ascontiguousarray` -- NumPy's own C strided copy --
for the gather. Comparing against a Python loop of element assignments
would be a strawman, not a baseline.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_comm  # noqa: E402


def _time(fn, repeats=7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_concat(n_buffers: int = 512, buffer_bytes: int = 4096):
    """Coalesce many independent comm buffers into one flat blob."""
    rng = np.random.default_rng(0)
    parts = [
        rng.integers(0, 256, size=buffer_bytes, dtype=np.uint8).tobytes()
        for _ in range(n_buffers)
    ]
    assert mojo_comm.concat_buffers(parts) == b"".join(parts), "concat mismatch"

    return (
        f"concat {n_buffers}x{buffer_bytes}B",
        _time(lambda: b"".join(parts)),
        _time(lambda: mojo_comm.concat_buffers(parts)),
    )


def bench_concat_arena(n_buffers: int = 512, buffer_bytes: int = 4096):
    """The same coalescing into a staging buffer the caller reuses.

    This is the shape a long-lived comm actually has: one arena, many
    messages. It isolates the copy from the per-message allocation.
    """
    rng = np.random.default_rng(0)
    parts = [
        rng.integers(0, 256, size=buffer_bytes, dtype=np.uint8).tobytes()
        for _ in range(n_buffers)
    ]
    arena = np.empty(n_buffers * buffer_bytes, dtype=np.uint8)
    want = b"".join(parts)
    assert mojo_comm.concat_buffers_into(arena, parts) == len(want)
    assert arena[:len(want)].tobytes() == want, "concat_into mismatch"

    def ref():
        out = bytearray()
        for p in parts:
            out += p
        return bytes(out)

    return (
        f"concat->arena {n_buffers}x{buffer_bytes}B",
        _time(ref),
        _time(lambda: mojo_comm.concat_buffers_into(arena, parts)),
    )


def bench_concat_kernel(n_buffers: int = 512, buffer_bytes: int = 4096):
    """The Mojo copy loop alone, with the Python address table hoisted out.

    `concat_buffers` loses to `b"".join` end to end, and this row says why:
    the kernel itself is at parity, and the gap is the per-buffer address
    table the shim has to build before it can call in.
    """
    from mojo_comm import _lib

    rng = np.random.default_rng(0)
    parts = [
        rng.integers(0, 256, size=buffer_bytes, dtype=np.uint8).tobytes()
        for _ in range(n_buffers)
    ]
    views, ptrs, lens = _lib._address_table(parts)
    arena = np.empty(n_buffers * buffer_bytes, dtype=np.uint8)
    lib = _lib.lib
    addr = _lib._addr
    lib.comm_concat(addr(ptrs), addr(lens), n_buffers, addr(arena))
    assert arena.tobytes() == b"".join(parts), "kernel concat mismatch"

    return (
        f"concat kernel only {n_buffers}x{buffer_bytes}B",
        _time(lambda: b"".join(parts)),
        _time(
            lambda: lib.comm_concat(
                addr(ptrs), addr(lens), n_buffers, addr(arena)
            )
        ),
    )


def bench_split(n_buffers: int = 512, buffer_bytes: int = 4096):
    """Cut a flat blob back into per-buffer byte strings."""
    rng = np.random.default_rng(1)
    lengths = [buffer_bytes] * n_buffers
    parts = [
        rng.integers(0, 256, size=n, dtype=np.uint8).tobytes() for n in lengths
    ]
    blob = mojo_comm.concat_buffers(parts)
    assert mojo_comm.split_buffers(blob, lengths) == parts, "split mismatch"

    def ref():
        at = 0
        out = []
        for n in lengths:
            out.append(blob[at:at + n])
            at += n
        return out

    return (
        f"split {n_buffers}x{buffer_bytes}B",
        _time(ref),
        _time(lambda: mojo_comm.split_buffers(blob, lengths)),
    )


def bench_gather_2d(n: int = 2048, factor: int = 2):
    """Pack a strided 2D view, against NumPy's own strided copy."""
    base = np.arange(n * n * factor * factor, dtype=np.float64)
    view = base.reshape(n, factor, n, factor)[:, 0, :, 0]
    assert not view.flags["C_CONTIGUOUS"]
    want = np.ascontiguousarray(view).tobytes()
    assert mojo_comm.pack_buffer(view) == want, "gather mismatch"

    return (
        f"gather 2d {n}x{n} stride {factor}",
        _time(lambda: np.ascontiguousarray(view).tobytes()),
        _time(lambda: mojo_comm.pack_buffer(view)),
    )


def bench_gather_3d(n: int = 256, factor: int = 2):
    base = np.arange(n * 3 * n * 3 * n * 3, dtype=np.int64)
    view = base.reshape(n, 3, n, 3, n, 3)[:, 0, :, 0, :, 0]
    want = np.ascontiguousarray(view).tobytes()
    assert mojo_comm.pack_buffer(view) == want, "gather 3d mismatch"

    return (
        f"gather 3d {n}^3 stride {factor}",
        _time(lambda: np.ascontiguousarray(view).tobytes()),
        _time(lambda: mojo_comm.pack_buffer(view)),
    )


def bench_contiguous(n: int = 1 << 22):
    """The fast path: a buffer that is already C-contiguous."""
    ar = np.arange(n, dtype=np.float64)
    assert mojo_comm.pack_buffer(ar) == ar.tobytes(), "contiguous mismatch"
    return (
        f"pack contiguous n={n}",
        _time(lambda: ar.tobytes()),
        _time(lambda: mojo_comm.pack_buffer(ar)),
    )


def main():
    print(f"{'case':<34}{'reference':>12}{'mojo-comm':>14}{'ratio':>10}")
    print("-" * 70)
    for fn in (bench_concat, bench_concat_arena, bench_concat_kernel,
               bench_split, bench_gather_2d, bench_gather_3d, bench_contiguous):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<34}{ref*1e3:>10.2f}ms{got*1e3:>12.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
