"""Binary frame messages on the /frames socket (see contracts/README.md, "Wire details").

Each message is: chunk number (uint32, big-endian), frame index inside the chunk (uint16, big-endian), JPEG bytes.
"""

import struct

HEADER = struct.Struct(">IH")
HEADER_SIZE = HEADER.size  # 6


def pack_frame(chunk: int, index: int, jpeg: bytes) -> bytes:
    return HEADER.pack(chunk, index) + jpeg


def unpack_frame(message: bytes) -> tuple[int, int, bytes]:
    if len(message) < HEADER_SIZE:
        raise ValueError(f"frame message too short: {len(message)} bytes")
    chunk, index = HEADER.unpack_from(message)
    return chunk, index, message[HEADER_SIZE:]
