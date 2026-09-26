"""mojo-comm: the byte-level comm-buffer surface of `comm`, in Mojo.

Installable alongside the real `comm` package, which it is tested against
for parity on the buffer-payload contract. See README.md for the coverage
table and for what deliberately stays in `comm`.
"""

from ._lib import (
    buffer_offsets,
    concat_buffers,
    concat_buffers_into,
    pack_buffer,
    pack_buffers,
    split_buffers,
    unpack_buffer,
)
from .comm import MojoComm

__all__ = [
    "MojoComm",
    "buffer_offsets",
    "concat_buffers",
    "concat_buffers_into",
    "pack_buffer",
    "pack_buffers",
    "split_buffers",
    "unpack_buffer",
]
__version__ = "0.1.0"
