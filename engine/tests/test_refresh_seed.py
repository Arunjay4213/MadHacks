"""The fixed-noise refresh seed: any reset seed works, and a seeded replay repeats."""
import pytest
import torch

from engine.tests.test_causal_forcing import make_backend, no_cuda, reanchored_chunk  # noqa: F401 (no_cuda is a fixture)

COND = torch.full((1, 3, 8), 1.0)


def refresh_input(b, seed, anchor=None):
    """Start a video, run to its clean anchor, refresh; returns (the refresh's first denoiser input, anchor, chunk).

    With `anchor`, the clean anchor and its chunk are replaced first, so only the refresh noise can differ."""
    b.reset(seed=seed)
    for _ in range(6):
        b.generate_chunk(COND)
    if anchor is not None:
        b._clean_anchor, b.clean_anchor_chunk = anchor
    held = (b._clean_anchor.clone(), b.clean_anchor_chunk)
    _, calls, _ = reanchored_chunk(b, COND, 0.5, "clean")
    assert b.last_rebase.from_clean
    return calls[0]["x"], held


@pytest.mark.parametrize("seed", [2**64 - 1, -(2**63)])
def test_any_reset_seed_refreshes_and_replays(no_cuda, seed):
    b = make_backend(refresh_noise="fixed")
    x1, _ = refresh_input(b, seed)
    x2, _ = refresh_input(b, seed)
    assert torch.equal(x1, x2)

