import pytest

from engine.framing import HEADER_SIZE, pack_frame, unpack_frame


def test_round_trip_and_layout():
    msg = pack_frame(0x01020304, 0x0506, b"\xff\xd8jpeg")
    assert HEADER_SIZE == 6
    assert msg[:6] == bytes([1, 2, 3, 4, 5, 6])  # big-endian chunk, then big-endian index
    assert unpack_frame(msg) == (0x01020304, 0x0506, b"\xff\xd8jpeg")


def test_limits_and_short_message():
    assert unpack_frame(pack_frame(2**32 - 1, 2**16 - 1, b""))[:2] == (2**32 - 1, 2**16 - 1)
    with pytest.raises(Exception):
        pack_frame(2**32, 0, b"")
    with pytest.raises(ValueError):
        unpack_frame(b"\x00\x01")
