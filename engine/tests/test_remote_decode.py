"""Remote decode: wire format, the client's named states, fallback to the local decoder, reconnect. CPU only."""

import dataclasses
import threading
import time

import numpy as np
import pytest
import torch

from engine.backends.causal_forcing import DecoderState
from engine.remote_decode import (DecodeServer, RemoteDecoder, RemoteState, Route, latents_from_bytes,
                                  latents_to_bytes, parse_address, resync_decoder)

HW = (2, 3)


@dataclasses.dataclass(frozen=True)
class Chunk:
    latents: torch.Tensor
    video_id: int
    chunk_index: int
    ready: object = None


def chunk(video_id, index):
    z = torch.full((1, 1, 16, *HW), float(index), dtype=torch.bfloat16)
    return Chunk(z, video_id, index)


class FakeService:
    """Stands in for WanDecodeService: 4 frames whose pixels are latent value + 100, so tests can see the route."""

    name = "fake"
    latent_hw = HW

    def __init__(self):
        self.seen = []
        self.fail_at = None
        self.delay_s = 0.0
        self.expected = {}

    def decode(self, latents, video_id, chunk_index):
        if chunk_index == self.fail_at:
            raise RuntimeError("boom")
        time.sleep(self.delay_s)
        resynced = chunk_index != 0 and self.expected.get(video_id) != chunk_index
        self.expected[video_id] = chunk_index + 1
        self.seen.append((video_id, chunk_index))
        v = int(latents.float()[0, 0, 0, 0, 0]) + 100
        return np.full((4, 16, 24, 3), v, dtype=np.uint8), resynced


def local_decode(c):
    return np.full((4, 16, 24, 3), int(c.latents.float()[0, 0, 0, 0, 0]), dtype=np.uint8)


def wait(pred, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def start_server(service, port=0):
    server = DecodeServer(service, "127.0.0.1", port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def served():
    service = FakeService()
    server = start_server(service)
    client = RemoteDecoder(f"127.0.0.1:{server.port}", HW, local_decode, connect_timeout=0.5,
                           request_timeout=0.5, retry_interval=0.05)
    yield service, server, client
    client.close()
    server.stop()


def test_latents_round_trip_bf16():
    z = torch.randn(1, 1, 16, 4, 5).to(torch.bfloat16)
    shape, raw = latents_to_bytes(z)
    assert len(raw) == z.numel() * 2
    assert torch.equal(latents_from_bytes(shape, raw), z)


def test_parse_address():
    assert parse_address("10.0.0.2:7801") == ("10.0.0.2", 7801)
    with pytest.raises(ValueError):
        parse_address("nohost")


def test_remote_decodes_in_order(served):
    service, _, client = served
    assert wait(lambda: client.state is RemoteState.READY)
    for i in range(5):
        frames = client.decode(chunk(1, i))
        assert client.last_route is Route.REMOTE
        assert frames.shape == (4, 16, 24, 3) and frames[0, 0, 0, 0] == 100 + i
    assert service.seen == [(1, i) for i in range(5)]
    assert client.stats()["remote_chunks"] == 5 and client.resyncs == 0


def test_service_down_falls_back_to_local_then_reconnects():
    service = FakeService()
    probe = DecodeServer(service, "127.0.0.1", 0)
    port = probe.port
    probe.stop()  # nothing listens on this port now
    client = RemoteDecoder(f"127.0.0.1:{port}", HW, local_decode, connect_timeout=0.2, retry_interval=0.05)
    try:
        assert client.state is RemoteState.CONNECTING
        assert client.decode(chunk(1, 0))[0, 0, 0, 0] == 0
        assert client.last_route is Route.LOCAL
        server = start_server(service, port)
        try:
            assert wait(lambda: client.state is RemoteState.READY)
            frames = client.decode(chunk(1, 1))
            assert client.last_route is Route.REMOTE and frames[0, 0, 0, 0] == 101
            assert client.resyncs == 1  # the service never saw chunk 0, so it restarted at chunk 1
        finally:
            server.stop()
    finally:
        client.close()


def test_service_error_falls_back_for_that_chunk(served):
    service, _, client = served
    assert wait(lambda: client.state is RemoteState.READY)
    client.decode(chunk(1, 0))
    service.fail_at = 1
    frames = client.decode(chunk(1, 1))
    assert client.last_route is Route.LOCAL and frames[0, 0, 0, 0] == 1
    assert client.failures == 1
    service.fail_at = None
    assert wait(lambda: client.state is RemoteState.READY)
    assert client.decode(chunk(1, 2))[0, 0, 0, 0] == 102


def test_slow_service_times_out_to_local(served):
    service, _, client = served
    assert wait(lambda: client.state is RemoteState.READY)
    service.delay_s = 1.0
    t0 = time.perf_counter()
    frames = client.decode(chunk(1, 0))
    assert time.perf_counter() - t0 < 0.9
    assert client.last_route is Route.LOCAL and frames[0, 0, 0, 0] == 0
    assert client.state is not RemoteState.READY or client.failures == 1


def test_wrong_latent_size_never_ready():
    service = FakeService()
    server = start_server(service)
    client = RemoteDecoder(f"127.0.0.1:{server.port}", (9, 9), local_decode, connect_timeout=0.2, retry_interval=0.05)
    try:
        time.sleep(0.3)
        assert client.state is RemoteState.CONNECTING
        client.decode(chunk(1, 0))
        assert client.last_route is Route.LOCAL
    finally:
        client.close()
        server.stop()


def test_close_decodes_locally(served):
    _, _, client = served
    assert wait(lambda: client.state is RemoteState.READY)
    client.close()
    assert client.state is RemoteState.CLOSED
    client.decode(chunk(1, 0))
    assert client.last_route is Route.LOCAL


class DummyDecoder:
    def __init__(self):
        self.video_id, self.next_chunk_index, self.state = None, 0, DecoderState.EMPTY

    def _start_video(self, video_id):
        self.video_id, self.next_chunk_index, self.state = video_id, 0, DecoderState.EMPTY


def test_resync_decoder():
    d = DummyDecoder()
    assert resync_decoder(d, 1, 0) is False  # new video at chunk 0: decode() handles it
    d.video_id, d.next_chunk_index, d.state = 1, 5, DecoderState.STREAMING
    assert resync_decoder(d, 1, 5) is False
    assert resync_decoder(d, 1, 9) is True
    assert (d.video_id, d.next_chunk_index, d.state) == (1, 9, DecoderState.STREAMING)
    assert resync_decoder(d, 2, 3) is True and d.video_id == 2 and d.next_chunk_index == 3


def test_engine_with_unreachable_remote_keeps_streaming():
    from engine.engine import Engine, EngineConfig
    from engine.lifecycle import Lifecycle

    probe = DecodeServer(FakeService(), "127.0.0.1", 0)
    port = probe.port
    probe.stop()
    frames = []
    cfg = EngineConfig(backend="fake", width=64, height=48, decode_remote=f"127.0.0.1:{port}",
                       backend_options=dict(gen_ms=20, decode_ms=5, encode_ms=30, frames_per_chunk=3, fps=50))
    e = Engine(cfg, frames.append, lambda m: None)
    e.start()
    try:
        assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
        assert wait(lambda: len(frames) >= 20)
        s = e.stats()["remote_decode"]
        assert s["state"] == "connecting" and s["local_chunks"] > 0 and s["remote_chunks"] == 0
    finally:
        e.stop()
