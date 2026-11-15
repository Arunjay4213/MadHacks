"""Round-trip timing of the remote Wan decode service with saved latents (no generator needed).

    python -m engine.tools.bench_remote_decode --address HOST:7801 --latents meadow_latents.pt [--pace 0.25]

Latents come from a list of [1, F, 16, h, w] tensors saved with torch.save. Each chunk is sent as
video 1 in order. `--pace` sends one chunk every that many seconds (0.25 = real time), so the
number shows the latency the engine would see; without it chunks go back to back (throughput).
"""

import argparse
import dataclasses
import json
import time

import numpy as np
import torch

from engine.remote_decode import RemoteDecoder, RemoteState, Route


@dataclasses.dataclass(frozen=True)
class _Chunk:
    latents: torch.Tensor
    video_id: int
    chunk_index: int
    ready: object = None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--address", required=True)
    p.add_argument("--latents", required=True)
    p.add_argument("--pace", type=float, default=0.0)
    p.add_argument("--skip", type=int, default=5, help="chunks left out of the statistics (warm-up)")
    a = p.parse_args()
    lat = torch.load(a.latents)
    hw = tuple(lat[0].shape[-2:])

    def no_local(chunk):
        raise RuntimeError("remote decode failed during the benchmark")

    rd = RemoteDecoder(a.address, hw, no_local, request_timeout=5.0)
    t_end = time.time() + 15
    while rd.state is not RemoteState.READY and time.time() < t_end:
        time.sleep(0.05)
    if rd.state is not RemoteState.READY:
        raise SystemExit(f"could not connect to {a.address}")
    rtt, service = [], []
    t_next = time.perf_counter()
    for i, z in enumerate(lat):
        if a.pace:
            time.sleep(max(0.0, t_next - time.perf_counter()))
            t_next += a.pace
        t0 = time.perf_counter()
        frames = rd.decode(_Chunk(z, 1, i))
        rtt.append((time.perf_counter() - t0) * 1e3)
        service.append(rd.last_remote_ms)
        assert rd.last_route is Route.REMOTE
    r, s = np.array(rtt[a.skip:]), np.array(service[a.skip:], dtype=float)
    out = {"chunks": len(rtt), "frames_last": list(frames.shape), "rtt_ms_median": float(np.median(r)),
           "rtt_ms_p90": float(np.percentile(r, 90)), "rtt_ms_max": float(r.max()),
           "service_ms_median": float(np.median(s)), "network_and_copy_ms_median": float(np.median(r - s)),
           "pace_s": a.pace, "chunks_per_s_back_to_back": float(1e3 / np.mean(r)) if not a.pace else None}
    print(json.dumps(out))
    rd.close()


if __name__ == "__main__":
    main()
