"""Backend registry. Each backend is a module engine/backends/<name>.py with a `load(...)` function.

The interface every backend implements:

    def load(width: int, height: int, steps: int, device: str = "cuda:0", decode_device: str = "cuda:0",
             **options) -> Backend

    class Backend:
        fps: int                 # video frame rate, 16
        frames_per_chunk: int    # usual frames per generate_chunk call; any chunk may return a different count
        width: int
        height: int
        def encode_prompt(self, prompt: str) -> torch.Tensor     # encoder thread, while generation runs
        def encode_prompt_with_ids(self, prompt: str) -> tuple[torch.Tensor, list[int]]  # same, plus token ids
        def reset(self) -> None                                   # new video: clears frame cache and decoder state
        def generate_chunk(self, cond: torch.Tensor) -> object    # generation thread only; returns latents
        def decode_chunk(self, latents: object) -> numpy.ndarray  # decode thread only, in chunk order; uint8 [T,H,W,3]

The module is imported only when chosen, so the fake backend needs no GPU packages.
"""

import importlib

BACKENDS = ("fake", "causal_forcing")


def load_backend(name: str, **kwargs):
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; known: {', '.join(BACKENDS)}")
    module = importlib.import_module(f"{__name__}.{name}")
    return module.load(**kwargs)
