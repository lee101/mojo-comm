"""A comm whose buffer payload runs through the Mojo kernels.

`comm.base_comm.BaseComm` has exactly one subclass hook, `publish_msg`, and
one subclass-provided comm. This module supplies the small piece of that
contract with a loop in it: turning the `buffers` list into the bytes a
transport writes, and turning received bytes back into arrays. Everything
else -- the comm id, the callback registry, `CommManager` -- is dict
plumbing and stays in the real `comm` package.
"""

from ._lib import (
    buffer_offsets,
    concat_buffers,
    pack_buffer,
    pack_buffers,
    split_buffers,
    unpack_buffer,
)


class MojoComm:
    """Minimal comm that carries array payloads as flat comm buffers.

    Mirrors the parts of `comm.base_comm.BaseComm` that touch bytes: `open`,
    `send` and `close` keep the msg/metadata/buffers triple, and `buffers`
    are serialised with the Mojo kernels. The base class is not subclassed
    because it requires a `publish_msg` implementation, which is transport
    specific; this object is the transport-agnostic buffer codec that a
    `BaseComm` subclass would call from its `publish_msg`.
    """

    def __init__(self, comm_id=None, target_name="comm"):
        self.comm_id = comm_id
        self.target_name = target_name
        self._closed = False
        self._msg_callback = None
        self._close_callback = None
        self.sent = []

    def on_msg(self, callback):
        self._msg_callback = callback

    def on_close(self, callback):
        self._close_callback = callback

    def open(self, data=None, metadata=None, arrays=None):
        return self.send(data, metadata, arrays)

    def send(self, data=None, metadata=None, arrays=None):
        """Serialise `arrays` into comm buffers and record the message."""
        if self._closed:
            raise RuntimeError(f"comm {self.comm_id} is closed")
        if arrays is None:
            buffers = None
        else:
            buffers = pack_buffers(arrays) if not _all_bytes(arrays) else list(arrays)
        msg = {
            "comm_id": self.comm_id,
            "data": data,
            "metadata": metadata,
            "buffers": buffers,
        }
        self.sent.append(msg)
        if self._msg_callback is not None:
            self._msg_callback(msg)
        return msg

    def close(self, msg=None):
        if self._msg_callback is not None:
            self._msg_callback(msg)
        if self._close_callback is not None:
            self._close_callback(msg)
        self._closed = True

    def receive(self, data=None, metadata=None, blob=None, lengths=None,
                specs=None):
        """Inverse of :meth:`send`: rebuild the arrays from received bytes.

        `specs` is a list of ``(dtype, shape)`` pairs, one per buffer.
        """
        if blob is None:
            chunks = None
        else:
            chunks = split_buffers(blob, lengths)
        arrays = None
        if chunks is not None:
            arrays = [
                unpack_buffer(c, dtype, shape)
                for c, (dtype, shape) in zip(chunks, specs)
            ]
        msg = {
            "comm_id": self.comm_id,
            "data": data,
            "metadata": metadata,
            "buffers": chunks,
        }
        if self._msg_callback is not None:
            self._msg_callback(msg)
        return arrays


def _all_bytes(items):
    return all(isinstance(b, (bytes, bytearray, memoryview)) for b in items)


__all__ = [
    "MojoComm",
    "buffer_offsets",
    "concat_buffers",
    "pack_buffer",
    "pack_buffers",
    "split_buffers",
    "unpack_buffer",
]
