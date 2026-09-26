"""Parity tests for the comm buffer payload.

`comm.base_comm` types a message's payload as `BuffersType = Optional[List[bytes]]`.
These tests check that the buffers `mojo_comm` produces are the bytes a real
`comm.BaseComm` actually receives, and that every byte-level kernel matches
the obvious NumPy/bytes reference.

Everything here moves bytes and never interprets them, so byte equality is
the correct assertion, not a tolerance: a byte copy is exact.
"""

import numpy as np
import pytest

import mojo_comm
from mojo_comm import _lib

comm = pytest.importorskip("comm")


def _rng():
    return np.random.default_rng(20240501)


# --------------------------------------------------------------------------
# parity with the real comm package
# --------------------------------------------------------------------------


class RecordingComm(comm.base_comm.BaseComm):
    """A real BaseComm that keeps whatever the transport would have seen."""

    def __init__(self, **kwargs):
        self.published = []
        super().__init__(**kwargs)

    def publish_msg(self, msg_type, data=None, metadata=None, buffers=None, **kw):
        # `published` exists before super().__init__() runs, so BaseComm's own
        # comm_open message is the first entry.
        self.published.append(
            {"msg_type": msg_type, "data": data, "metadata": metadata,
             "buffers": buffers}
        )


def test_buffers_reach_a_real_basecomm_byte_identical():
    """Our serialisation is what a real BaseComm's publish_msg receives."""
    arrays = [
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.linspace(0.0, 1.0, 5),
        np.array([1, 2, 3], dtype=np.int32),
    ]
    ours = mojo_comm.pack_buffers(arrays)

    real = RecordingComm()
    # The reference: hand the real comm the same arrays as raw buffers, which
    # is what every comm transport in the wild does.
    real.send(
        data={"method": "update"},
        buffers=[memoryview(a).cast("B") for a in arrays],
    )
    got = real.published[1]["buffers"]
    for mine, theirs in zip(ours, got):
        assert isinstance(mine, bytes)
        assert mine == bytes(theirs)


def test_comm_roundtrip_through_basecomm_callbacks():
    """msg/close callbacks and buffer bytes survive the Mojo comm path."""
    seen = []
    mine = mojo_comm.MojoComm(comm_id="c-1")
    mine.on_msg(seen.append)

    arr = np.arange(12, dtype=np.float64).reshape(3, 4)
    mine.send(data={"method": "update"}, arrays=[arr])
    assert len(seen) == 1
    assert seen[0]["data"] == {"method": "update"}
    assert seen[0]["buffers"][0] == arr.tobytes()

    back = mine.receive(
        data=seen[0]["data"],
        blob=mojo_comm.concat_buffers(seen[0]["buffers"]),
        lengths=[arr.nbytes],
        specs=[(np.float64, (3, 4))],
    )
    np.testing.assert_array_equal(back[0], arr)

    closed = []
    mine.on_close(closed.append)
    mine.close({"comm_id": "c-1"})
    assert closed == [{"comm_id": "c-1"}]
    with pytest.raises(RuntimeError):
        mine.send(data={"method": "update"}, arrays=[arr])


def test_send_without_buffers_matches_comm_default():
    """`buffers=None` is the real package's default and must survive."""
    mine = mojo_comm.MojoComm(comm_id="c-2")
    assert mine.send(data={"method": "ping"})["buffers"] is None
    real = RecordingComm()
    real.send(data={"method": "ping"})
    assert real.published[1]["msg_type"] == "comm_msg"
    assert real.published[1]["buffers"] is None



def test_concat_into_a_reused_arena():
    """`concat_buffers_into` writes the same bytes without reallocating."""
    rng = _rng()
    parts = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
             for n in (7, 0, 300, 1, 64)]
    want = b"".join(parts)
    arena = np.empty(len(want) + 32, dtype=np.uint8)

    assert mojo_comm.concat_buffers_into(arena, parts) == len(want)
    assert arena[:len(want)].tobytes() == want
    after_first = arena.copy()

    # A second, shorter batch rewrites only its own prefix and leaves the
    # rest of the arena, and the slack past the write, untouched.
    short = parts[:2]
    n = mojo_comm.concat_buffers_into(arena, short)
    assert arena[:n].tobytes() == b"".join(short)
    assert arena[n:len(want)].tobytes() == after_first[n:len(want)].tobytes()
    assert arena[len(want):].tobytes() == after_first[len(want):].tobytes()

    assert mojo_comm.concat_buffers_into(arena, []) == 0
    with pytest.raises(ValueError, match="arena holds"):
        mojo_comm.concat_buffers_into(np.empty(3, dtype=np.uint8), parts)

# --------------------------------------------------------------------------
# concat / split / offsets
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lengths",
    [
        [0],
        [1],
        [1, 1, 1, 1],
        [3, 0, 5],
        [0, 0, 0, 0],
        [7, 1, 0, 64, 129, 3],
        list(range(1, 33)),
    ],
)
def test_concat_matches_bytes_join(lengths):
    rng = _rng()
    parts = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
             for n in lengths]
    assert mojo_comm.concat_buffers(parts) == b"".join(parts)
    assert mojo_comm.concat_buffers([]) == b""


def test_concat_of_one_buffer_is_identity():
    rng = _rng()
    part = rng.integers(0, 256, size=1000, dtype=np.uint8).tobytes()
    assert mojo_comm.concat_buffers([part]) == part


@pytest.mark.parametrize(
    "lengths",
    [[3, 0, 5], [0, 0, 7], [1, 1, 1, 1, 1], [64, 65, 1], list(range(1, 20))],
)
def test_split_is_the_inverse_of_concat(lengths):
    rng = _rng()
    parts = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
             for n in lengths]
    blob = mojo_comm.concat_buffers(parts)
    assert mojo_comm.split_buffers(blob, lengths) == parts
    assert mojo_comm.split_buffers(blob, []) == []


def test_split_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="sum to"):
        mojo_comm.split_buffers(b"abcdef", [2, 2])


@pytest.mark.parametrize(
    "lengths",
    [[1], [0, 5], [3, 0, 5], [2, 2, 2, 2], list(range(1, 40))],
)
def test_buffer_offsets_is_the_exclusive_prefix_sum(lengths):
    got = mojo_comm.buffer_offsets(lengths)
    want = np.concatenate(([0], np.cumsum(lengths)[:-1])).astype(np.int64)
    np.testing.assert_array_equal(got, want)
    assert got.dtype == np.int64
    assert mojo_comm.buffer_offsets([]).size == 0


def test_offsets_locate_every_buffer_in_the_blob():
    """The offset table must actually address the concatenated blob."""
    rng = _rng()
    lengths = [5, 0, 11, 2, 30]
    parts = [rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
             for n in lengths]
    blob = mojo_comm.concat_buffers(parts)
    offs = mojo_comm.buffer_offsets(lengths)
    for part, off, n in zip(parts, offs, lengths):
        assert blob[off:off + n] == part


# --------------------------------------------------------------------------
# pack_buffer: the strided gather
# --------------------------------------------------------------------------


def _roundtrip(ar):
    raw = mojo_comm.pack_buffer(ar)
    back = np.frombuffer(raw, dtype=ar.dtype).reshape(ar.shape)
    return back


def test_contiguous_buffer_is_the_raw_c_order_bytes():
    ar = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert mojo_comm.pack_buffer(ar) == ar.tobytes()
    assert np.array_equal(_roundtrip(ar), ar)


def test_zero_dimensional_buffer():
    ar = np.array(3.5)
    assert mojo_comm.pack_buffer(ar) == np.array([3.5]).tobytes()
    np.testing.assert_array_equal(_roundtrip(ar), ar)


@pytest.mark.parametrize(
    "make",
    [
        lambda a: a[::2, ::2],
        lambda a: a[1:, 1:],
        lambda a: a.T,
        lambda a: a[:, ::-1],
        lambda a: a[::3, 1::4],
        lambda a: a.reshape(4, 6)[2:4, 1:5][:, ::-1],
    ],
)
def test_two_dimensional_strided_views(make):
    ar = np.arange(24, dtype=np.int64).reshape(4, 6)
    view = make(ar)
    assert not view.flags["C_CONTIGUOUS"] or view.base is not None
    np.testing.assert_array_equal(_roundtrip(view), view)
    # A wrong stride shows up immediately: the packed bytes are the packed
    # C-order bytes of the view, not of its base.
    assert mojo_comm.pack_buffer(view) == np.ascontiguousarray(view).tobytes()


def test_fortran_order_view_packs_as_c_order():
    ar = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    assert mojo_comm.pack_buffer(ar) == np.ascontiguousarray(ar).tobytes()
    np.testing.assert_array_equal(_roundtrip(ar), ar)


def test_three_and_four_dimensional_strided_views():
    ar = np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5)
    for view in (
        ar[:, ::-1, 1:4, ::3],
        ar[1:, ::2, 1:, ::-1],
        ar[0],
        ar[1, 2, 3, 4],
    ):
        np.testing.assert_array_equal(_roundtrip(view), view)
        assert mojo_comm.pack_buffer(view) == np.ascontiguousarray(view).tobytes()


def test_empty_view_packs_to_nothing():
    ar = np.arange(12, dtype=np.int64).reshape(3, 4)
    assert mojo_comm.pack_buffer(ar[:, :0]) == b""
    assert mojo_comm.pack_buffer(ar[0:0, :]) == b""
    np.testing.assert_array_equal(_roundtrip(ar[0:0, :]), ar[0:0, :])


def test_every_dtype_and_shape_survives_the_gather():
    rng = _rng()
    for dtype in (np.uint8, np.int16, np.float32, np.float64, np.complex128):
        ar = (rng.standard_normal(24) * 100).astype(dtype).reshape(2, 3, 4)
        np.testing.assert_array_equal(_roundtrip(ar), ar)
        np.testing.assert_array_equal(_roundtrip(ar[:, ::-1]), ar[:, ::-1])


def test_strided_gather_rejects_more_than_four_dimensions():
    ar = np.zeros((2,) * 5, dtype=np.int64)
    with pytest.raises(ValueError, match="at most 4 dimensions"):
        mojo_comm.pack_buffer(ar)


def test_pack_buffers_batch_matches_per_array():
    rng = _rng()
    arrays = [
        np.arange(5, dtype=np.int8),
        rng.standard_normal((3, 3)),
        np.arange(24, dtype=np.int32).reshape(4, 6)[::2, ::-1],
    ]
    got = mojo_comm.pack_buffers(arrays)
    assert got == [mojo_comm.pack_buffer(a) for a in arrays]
    assert len(got) == 3


# --------------------------------------------------------------------------
# unpack
# --------------------------------------------------------------------------


def test_unpack_buffer_reshape_and_dtype():
    ar = np.arange(12, dtype=np.float32).reshape(3, 4)
    back = mojo_comm.unpack_buffer(mojo_comm.pack_buffer(ar), np.float32, (3, 4))
    np.testing.assert_array_equal(back, ar)
    flat = mojo_comm.unpack_buffer(mojo_comm.pack_buffer(ar), np.float32)
    np.testing.assert_array_equal(flat, ar.ravel())


def test_split_then_unpack_reproduces_the_arrays():
    rng = _rng()
    arrays = [rng.standard_normal((3, 3)), rng.integers(0, 100, 5, dtype=np.int32)]
    blob = mojo_comm.concat_buffers(mojo_comm.pack_buffers(arrays))
    lengths = [a.nbytes for a in arrays]
    specs = [(a.dtype, a.shape) for a in arrays]
    back = [
        mojo_comm.unpack_buffer(c, dt, sh)
        for c, (dt, sh) in zip(mojo_comm.split_buffers(blob, lengths), specs)
    ]
    for want, got in zip(arrays, back):
        np.testing.assert_array_equal(got, want)


def test_split_kernel_is_exercised_directly():
    """Drive the kernel with a hand-built offset table, not via the helper."""
    lib = _lib.lib
    src = np.arange(20, dtype=np.uint8)
    dst = np.zeros(20, dtype=np.uint8)
    ptrs = np.array([10, 0, 5], dtype=np.int64)
    lens = np.array([4, 10, 6], dtype=np.int64)
    offs = np.array([0, 4, 14], dtype=np.int64)
    lib.comm_split(_lib._addr(src), _lib._addr(ptrs), _lib._addr(lens), 3,
                   _lib._addr(dst), _lib._addr(offs))
    want = np.concatenate([src[10:14], src[0:10], src[5:11]])
    np.testing.assert_array_equal(dst, want)


def test_prefix_sum_kernel_is_exercised_directly():
    lib = _lib.lib
    lens = np.array([3, 0, 7, 1], dtype=np.int64)
    out = np.zeros(4, dtype=np.int64)
    lib.comm_prefix_sum(_lib._addr(lens), 4, _lib._addr(out))
    np.testing.assert_array_equal(out, [0, 3, 3, 10])
