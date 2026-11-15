"""Decode latents on another GPU: a small TCP decode service and the engine's client for it.

Why: at 672x384 the stock Wan VAE decoder gives a clearly better picture than TAEHV but takes about
155 ms per chunk. Next to generation (about 210 ms) on one GPU that is over the 250 ms budget, so the
Wan decoder runs in its own process, on any other GPU, behind one TCP connection.

Service (on the decode GPU, from the repo root, with ~/engine-venv):

    python -m engine.remote_decode --port 7801 --width 672 --height 384

Engine: `python -m engine.server ... --decode-remote HOST:7801`. The decode thread then sends every
chunk to the service and gets the frames back. If the service is down, slow or broken, that chunk and
the following ones are decoded by the local decoder (TAEHV) and the client reconnects in the
background, so the stream never stops.

Wire format. One persistent TCP connection, the client sends one request and waits for its reply.
Every message is a 4-byte big-endian length, a UTF-8 JSON header of that length, then `nbytes`
payload bytes (the length of the payload is in the header).
- hello   client -> service  {"op": "hello", "version": 1, "latent_hw": [h, w]}
- ready   service -> client  {"op": "ready", "version": 1, "latent_hw": [h, w], "decoder": "wan"}
- decode  client -> service  {"op": "decode", "video_id", "chunk_index", "shape": [1, F, 16, h, w],
                              "dtype": "bf16", "nbytes"} + the latents as raw little-endian bf16
- frames  service -> client  {"op": "frames", "video_id", "chunk_index", "shape": [T, H, W, 3],
                              "nbytes", "decode_ms", "resynced"} + uint8 RGB frames
- error   service -> client  {"op": "error", "message"}; the client then drops the connection
The service decodes chunks of one video in order. If a chunk does not follow the previous one (the
client fell back to local decode for a while, or the service restarted), the service restarts its
streaming state at that chunk and says so with "resynced": true.
"""

from __future__ import annotations

import argparse
import enum
import json
import logging
import socket
import struct
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

log = logging.getLogger("engine.remote_decode")

VERSION = 1
DEFAULT_PORT = 7801
_LEN = struct.Struct(">I")
MAX_HEADER = 1 << 16
MAX_PAYLOAD = 256 << 20


class ProtocolError(RuntimeError):
    pass


# --- framing -------------------------------------------------------------------------------------

def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    header = dict(header, nbytes=len(payload))
    raw = json.dumps(header).encode()
    sock.sendall(_LEN.pack(len(raw)) + raw)
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("connection closed by peer")
        got += k
    return bytes(buf)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    (n,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if n > MAX_HEADER:
        raise ProtocolError(f"header too large: {n} bytes")
    header = json.loads(_recv_exact(sock, n))
    if not isinstance(header, dict) or "op" not in header:
        raise ProtocolError(f"bad header: {header!r}")
    nbytes = int(header.get("nbytes", 0))
    if nbytes < 0 or nbytes > MAX_PAYLOAD:
        raise ProtocolError(f"bad payload size {nbytes}")
    return header, _recv_exact(sock, nbytes) if nbytes else b""


def latents_to_bytes(latents: torch.Tensor) -> tuple[list[int], bytes]:
    t = latents.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    return list(t.shape), t.view(torch.int16).numpy().tobytes()


def latents_from_bytes(shape: list[int], payload: bytes) -> torch.Tensor:
    expected = int(np.prod(shape)) * 2
    if len(payload) != expected:
        raise ProtocolError(f"latent payload is {len(payload)} bytes, shape {shape} needs {expected}")
    return torch.from_numpy(np.frombuffer(payload, dtype=np.int16).copy()).view(torch.bfloat16).reshape(shape)


def parse_address(text: str) -> tuple[str, int]:
    host, sep, port = text.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"expected HOST:PORT, got {text!r}")
    return host, int(port)


# --- service ---------------------------------------------------------------------------------------

class WanDecodeService:
    """Holds only the stock Wan VAE decoder (the backend's own _Decoder) on one GPU."""

    name = "wan"

    def __init__(self, width: int, height: int, device: str = "cuda:0", repo_dir: Optional[str] = None):
        from engine.backends import causal_forcing as cf

        self._cf = cf
        repo = cf._import_repo(repo_dir or cf.DEFAULT_REPO_DIR)
        self.device = torch.device(device)
        self.latent_hw = (height // cf.SPATIAL_DOWNSAMPLE, width // cf.SPATIAL_DOWNSAMPLE)
        with cf._in_dir(repo):  # the official code opens the VAE weights through a relative path
            self.decoder = cf._Decoder(repo, cf.VaeType.WAN, self.device, self.latent_hw)
        # Warm up (cuDNN autotune, allocator) so the first real chunk is not slow; video id -1 is never used.
        zero = torch.zeros(1, 1, cf.LATENT_CHANNELS, *self.latent_hw, dtype=torch.bfloat16)
        for i in range(3):
            self.decode(zero, -1, i)

    def decode(self, latents: torch.Tensor, video_id: int, chunk_index: int) -> tuple[np.ndarray, bool]:
        resynced = resync_decoder(self.decoder, video_id, chunk_index)
        with torch.cuda.device(self.device):
            z = latents.to(self.device, non_blocking=False)
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            chunk = self._cf.ChunkLatents(latents=z, video_id=video_id, chunk_index=chunk_index, ready=ready)
            return self.decoder.decode(chunk), resynced


def resync_decoder(decoder, video_id: int, chunk_index: int) -> bool:
    """Make a backend _Decoder accept `chunk_index` of `video_id` next, even if it did not see the chunks
    before it (they went to the other decoder). The streaming state restarts empty at that chunk but
    keeps all 4 frames of it (only the first chunk of a video drops 3). Returns True if it restarted.
    This uses the decoder's internal fields because the backend file is owned by another teammate."""
    from engine.backends.causal_forcing import DecoderState

    if decoder.video_id == video_id and decoder.next_chunk_index == chunk_index:
        return False
    if chunk_index == 0:
        return False  # a new video: decode() starts it the normal way
    decoder._start_video(video_id)
    decoder.next_chunk_index = chunk_index
    decoder.state = DecoderState.STREAMING
    return True


class DecodeServer:
    """Accepts one client at a time and decodes its chunks in order. `service.decode(latents, video_id,
    chunk_index) -> (frames uint8 [T, H, W, 3], resynced)` does the work."""

    def __init__(self, service, host: str = "0.0.0.0", port: int = DEFAULT_PORT):
        self.service = service
        self._sock = socket.create_server((host, port), reuse_port=False)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()

    def serve_forever(self) -> None:
        log.info("decode service (%s) listening on port %d", self.service.name, self.port)
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            log.info("client %s connected", addr)
            try:
                self._handle(conn)
            except (ConnectionError, ProtocolError, OSError) as exc:
                log.warning("client %s dropped: %s", addr, exc)
            finally:
                conn.close()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()

    def _handle(self, conn: socket.socket) -> None:
        header, _ = recv_message(conn)
        if header["op"] != "hello" or header.get("version") != VERSION:
            send_message(conn, {"op": "error", "message": f"expected hello version {VERSION}, got {header}"})
            return
        if list(header.get("latent_hw", [])) != list(self.service.latent_hw):
            send_message(conn, {"op": "error", "message": f"latent size {header.get('latent_hw')} != service "
                                                          f"{list(self.service.latent_hw)}"})
            return
        send_message(conn, {"op": "ready", "version": VERSION, "latent_hw": list(self.service.latent_hw),
                            "decoder": self.service.name})
        while not self._stop.is_set():
            header, payload = recv_message(conn)
            if header["op"] != "decode":
                send_message(conn, {"op": "error", "message": f"unknown op {header['op']!r}"})
                return
            video_id, chunk_index = int(header["video_id"]), int(header["chunk_index"])
            try:
                latents = latents_from_bytes(header["shape"], payload)
                t0 = time.perf_counter()
                frames, resynced = self.service.decode(latents, video_id, chunk_index)
                ms = (time.perf_counter() - t0) * 1e3
            except Exception as exc:  # report it, then let the client fall back
                log.exception("decode of video %d chunk %d failed", video_id, chunk_index)
                send_message(conn, {"op": "error", "message": f"{type(exc).__name__}: {exc}"})
                return
            if resynced:
                log.info("video %d: restarted the streaming state at chunk %d", video_id, chunk_index)
            frames = np.ascontiguousarray(frames, dtype=np.uint8)
            send_message(conn, {"op": "frames", "video_id": video_id, "chunk_index": chunk_index,
                                "shape": list(frames.shape), "decode_ms": round(ms, 2), "resynced": resynced},
                         frames.tobytes())


# --- client ------------------------------------------------------------------------------------------

class RemoteState(enum.Enum):
    CONNECTING = "connecting"  # no usable connection; the background thread is (re)connecting
    READY = "ready"            # handshake done; chunks go to the service
    CLOSED = "closed"          # close() was called; everything is decoded locally


class Route(enum.Enum):
    REMOTE = "remote"
    LOCAL = "local"


class RemoteDecoder:
    """The engine's decode function when --decode-remote is set: `decode(chunk) -> frames`.

    `local_decode(chunk)` is the fallback (the backend's own decoder, TAEHV). Transitions:
    CONNECTING -> READY        the background thread connected and the handshake passed
    READY -> CONNECTING        a request failed or timed out; the chunk is decoded locally
    any -> CLOSED              close()
    """

    def __init__(self, address: str, latent_hw: tuple[int, int], local_decode: Callable[[object], np.ndarray],
                 local_resync: Optional[Callable[[int, int], bool]] = None, connect_timeout: float = 2.0,
                 request_timeout: float = 1.0, retry_interval: float = 2.0):
        self.host, self.port = parse_address(address)
        self.latent_hw = tuple(latent_hw)
        self._local_decode = local_decode
        self._local_resync = local_resync
        self.connect_timeout, self.request_timeout, self.retry_interval = connect_timeout, request_timeout, retry_interval
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._sock: Optional[socket.socket] = None
        self.state = RemoteState.CONNECTING
        self.last_route: Optional[Route] = None
        self.remote_chunks = self.local_chunks = self.failures = self.resyncs = 0
        self.last_remote_ms: Optional[float] = None  # service-side decode time of the last remote chunk
        self._thread = threading.Thread(target=self._connect_loop, name="remote-decode-connect", daemon=True)
        self._thread.start()

    # transitions
    def _to_ready(self, sock: socket.socket) -> None:
        self._sock, self.state = sock, RemoteState.READY
        log.info("remote decoder %s:%d ready", self.host, self.port)

    def _to_connecting(self, reason: str) -> None:
        if self._sock is not None:
            self._sock.close()
        self._sock = None
        if self.state is RemoteState.CLOSED:
            return
        self.state = RemoteState.CONNECTING
        self.failures += 1
        log.warning("remote decoder %s:%d lost (%s); decoding locally and reconnecting", self.host, self.port, reason)
        self._wake.notify_all()

    def close(self) -> None:
        with self._lock:
            if self._sock is not None:
                self._sock.close()
            self._sock, self.state = None, RemoteState.CLOSED
            self._wake.notify_all()
        self._thread.join(timeout=5)

    def _connect_loop(self) -> None:
        while True:
            with self._lock:
                while self.state is RemoteState.READY:
                    self._wake.wait()
                if self.state is RemoteState.CLOSED:
                    return
            sock = self._try_connect()
            with self._lock:
                if self.state is RemoteState.CLOSED:
                    if sock is not None:
                        sock.close()
                    return
                if sock is not None:
                    self._to_ready(sock)
                    continue
                self._wake.wait(self.retry_interval)

    def _try_connect(self) -> Optional[socket.socket]:
        sock = None
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send_message(sock, {"op": "hello", "version": VERSION, "latent_hw": list(self.latent_hw)})
            header, _ = recv_message(sock)
            if header["op"] != "ready":
                raise ProtocolError(header.get("message", f"unexpected reply {header['op']!r}"))
            sock.settimeout(self.request_timeout)
            return sock
        except (OSError, ProtocolError, ValueError) as exc:
            log.debug("remote decoder %s:%d not reachable: %s", self.host, self.port, exc)
            if sock is not None:
                sock.close()
            return None

    # decode
    def decode(self, chunk) -> np.ndarray:
        with self._lock:
            sock = self._sock if self.state is RemoteState.READY else None
        if sock is not None:
            try:
                frames = self._remote(sock, chunk)
                self.remote_chunks += 1
                self.last_route = Route.REMOTE
                return frames
            except Exception as exc:  # any failure of the remote path: this chunk is decoded locally
                with self._lock:
                    if self._sock is sock:
                        self._to_connecting(f"{type(exc).__name__}: {exc}")
        if self._local_resync is not None and self._local_resync(chunk.video_id, chunk.chunk_index):
            log.info("local decoder restarted at video %d chunk %d", chunk.video_id, chunk.chunk_index)
        frames = self._local_decode(chunk)
        if self.last_route is Route.REMOTE:
            log.warning("chunk %d decoded locally (remote decoder %s)", chunk.chunk_index, self.state.value)
        self.local_chunks += 1
        self.last_route = Route.LOCAL
        return frames

    def _remote(self, sock: socket.socket, chunk) -> np.ndarray:
        ready = getattr(chunk, "ready", None)
        if ready is not None:
            ready.synchronize()  # the latents were written on the generation stream
        shape, payload = latents_to_bytes(chunk.latents)
        send_message(sock, {"op": "decode", "video_id": int(chunk.video_id), "chunk_index": int(chunk.chunk_index),
                            "shape": shape, "dtype": "bf16"}, payload)
        header, body = recv_message(sock)
        if header["op"] == "error":
            raise ProtocolError(f"service error: {header.get('message')}")
        if header["op"] != "frames" or header["video_id"] != chunk.video_id or header["chunk_index"] != chunk.chunk_index:
            raise ProtocolError(f"reply {header} does not match video {chunk.video_id} chunk {chunk.chunk_index}")
        if header.get("resynced"):
            self.resyncs += 1
        self.last_remote_ms = header.get("decode_ms")
        return np.frombuffer(body, dtype=np.uint8).reshape(header["shape"])

    def stats(self) -> dict:
        return {"state": self.state.value, "remote_chunks": self.remote_chunks, "local_chunks": self.local_chunks,
                "failures": self.failures, "resyncs": self.resyncs, "last_remote_ms": self.last_remote_ms}


def make_remote_decoder(address: str, backend, **kwargs) -> RemoteDecoder:
    """RemoteDecoder for a loaded backend: the fallback is the backend's own decode_chunk."""
    local = getattr(backend, "decoder", None)
    resync = (lambda v, c: resync_decoder(local, v, c)) if local is not None and hasattr(local, "_start_video") else None
    hw = getattr(backend, "latent_hw", (backend.height // 8, backend.width // 8))
    return RemoteDecoder(address, hw, backend.decode_chunk, local_resync=resync, **kwargs)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Wan VAE decode service for the engine (see module docstring)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--width", type=int, default=672)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log-level", default="info")
    a = p.parse_args(argv)
    logging.basicConfig(level=a.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    t0 = time.perf_counter()
    service = WanDecodeService(a.width, a.height, a.device)
    log.info("Wan decoder loaded in %.1f s", time.perf_counter() - t0)
    DecodeServer(service, a.host, a.port).serve_forever()


if __name__ == "__main__":
    main()
