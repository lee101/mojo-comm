# mojo-comm

`mojo-comm` is the compute-oriented subset of
[`comm`](https://pypi.org/project/comm/): the byte-level comm-buffer
payload, with the copy loops implemented in Mojo and callable from Python.

The Python package is named `mojo_comm`, so it installs alongside the real
`comm` and the tests drive a real `comm.base_comm.BaseComm` to check that
the buffers this port produces are the bytes a real comm actually receives.

## Is there anything numeric in `comm`?

No, and this port does not pretend otherwise. `comm` is 326 lines of
messaging protocol: `BaseComm` holds a `comm_id`, a pair of callbacks and a
`publish_msg` hook; `CommManager` is a dict registry keyed by comm id; the
whole package is dict plumbing around one subclass hook. There is not a
single arithmetic operation in it, and inventing a numeric kernel to make
the port look substantial would be a lie.

What `comm` *does* have is exactly one place where bytes move in a loop: the
message payload. `comm.base_comm` types it as
`BuffersType = Optional[List[bytes]]`, and every real caller (ipywidgets,
jupyter kernels) fills it with the raw memory of numpy arrays. Two shapes of
that are genuine per-byte work:

- a **non-contiguous** array has to be gathered into C order before it can be
  written, which is a strided copy with a row loop in it, and
- a message's buffers are **independent allocations** that a transport often
  wants as one flat blob, which is a gather/scatter over an offset table.

That is the port. Everything else in `comm` is left where it is.

## Covered subset

| area | implemented API | Mojo kernel |
| --- | --- | --- |
| Buffer serialisation | `pack_buffer`, `pack_buffers`, `unpack_buffer` | `comm_strided_gather` |
| Buffer coalescing | `concat_buffers`, `concat_buffers_into` | `comm_concat` |
| Buffer splitting | `split_buffers` | `comm_split` |
| Buffer layout | `buffer_offsets` | `comm_prefix_sum` |
| Message envelope | `MojoComm` (`open`/`send`/`close`/`receive`, `on_msg`/`on_close`) | — |

`MojoComm` deliberately does not subclass `comm.base_comm.BaseComm`: the base
class requires a `publish_msg` implementation, which is transport-specific.
`MojoComm` is the transport-agnostic buffer codec that a `BaseComm` subclass
would call from its `publish_msg`, and the tests check that the two agree on
the bytes.

## Not implemented

Everything else, and it is not a short list:

- `BaseComm.open`/`close`/`send` lifecycle, `comm_id` generation, the
  `on_msg`/`on_close` callback registry — real `BaseComm` does all of this
  and it is dict bookkeeping, not compute.
- `CommManager`: `register_comm`, `unregister_comm`, `get_comm`, target
  dispatch. A singleton registry of dicts.
- `create_comm` / `get_comm_manager` and the IPython event hooks in
  `handle_msg` (`post_execute`, `pre_execute`).
- The actual transport. `comm` has none; that is `comm_ipython` or
  `pyzmq`'s job.

## Install

The repository pins its own Mojo toolchain in `pixi.toml`
(`mojo == 1.2.0.dev2026092605`). With the toolchain on `PATH`:

```bash
bash build/build.sh     # -> dist/libmojo-comm.so
PYTHONPATH=python python -m pytest tests -q
```

`build/build.sh` must not be run inside a private `pixi install`; the shared
environment is the only environment.

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib`.

Buffers cross the C ABI as 64-bit addresses and are reconstructed in Mojo as
`Pointer[UInt8, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric. `comm_concat` takes a table of *absolute* source addresses
because a message's buffers are unrelated allocations; `comm_split` takes
offsets into one blob because there the source really is contiguous.

The block copy opens the same range as `UInt64` and moves eight bytes per
iteration, with a byte-at-a-time remainder. A plain `dst[i] = src[i]` loop
measured 1.68 ms on 2 MiB against 0.28 ms for the word loop and 0.21 ms for
glibc `memcpy`; Mojo 1.2.0 has no `memcpy` in `std.memory`, so the word loop
is the fastest available route to the same thing.

Every kernel here moves bytes and never interprets them. A byte copy is
exact, so the parity tests assert byte equality, never a tolerance. (The
"Mojo emits FMA" caveat does not apply to this port — there is no
floating-point arithmetic in it at all.)

## Performance

Best-of-seven wall clock, same process, against the fastest reasonable
Python. Every case verifies byte equality before it times anything.

| case | reference | mojo-comm | result |
| --- | ---: | ---: | ---: |
| concat kernel only, 512x4 KiB | 0.22 ms | 0.26 ms | 0.83x |
| concat->arena, 512x4 KiB | 0.44 ms | 3.23 ms | 0.14x |
| concat, 512x4 KiB | 0.22 ms | 6.02 ms | 0.04x |
| split, 512x4 KiB | 0.46 ms | 1.29 ms | 0.35x |
| gather 2d, 2048x2048 stride 2 | 88.40 ms | 81.72 ms | 1.08x |
| gather 3d, 256^3 stride 2 | 306.27 ms | 294.88 ms | 1.04x |
| pack contiguous, n=4194304 | 40.41 ms | 39.84 ms | 1.01x |

Reproduce with `python bench/bench.py`.

**The coalescing rows are losses, and they are worth reading carefully.**
The Mojo copy loop is at 0.83x of `b"".join` — parity, which is the honest
ceiling for a memcpy against glibc. The end-to-end `concat_buffers` is 0.04x
because the shim has to build an address table with one Python-level buffer
descriptor lookup per buffer before it can make the call, and `b"".join` is
a single C call that needs no table. The `concat kernel only` row exists to
make that attribution visible rather than to hide it: the loss is Python
call overhead for many small buffers, not a slow kernel. If you are
coalescing `bytes` objects, use `b"".join`.

The gather rows are the reason the port exists, and they are parity too. A
strided gather is memory-bound, and NumPy's `ascontiguousarray` is already a
C strided copy at memory speed; there is no headroom to win and Mojo does not
win. It is a real result that the port is at parity, and the point of the
port is that the copy now lives in one Mojo compilation unit rather than
being re-derived per caller.

Reproduce the numbers above; the box is shared, so the gather rows move
between roughly 0.85x and 1.10x run to run.

## Tests

```bash
PYTHONPATH=python python -m pytest tests -q
```

42 tests. Parity is checked against a real `comm.base_comm.BaseComm` (a
subclass that records `publish_msg`) for the message contract, and against
`bytes.join`, `np.ascontiguousarray` and `np.cumsum` for the kernels. The
strided gather is tested over transposed, reversed, Fortran-order, 3-D and
4-D, empty and scalar views, and five dtypes, because a wrong stride, a
dropped dimension or a dropped SIMD tail is exactly the bug that would
survive a contiguous-only test.

## License

MIT
