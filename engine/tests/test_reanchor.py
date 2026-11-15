"""The re-anchor wiring (--reanchor) with the fake backend: policy hooks, the backend call and /stats."""

import time

import pytest

import engine.engine as engine_module
from engine.backends.fake import FakeBackend
from engine.commands import CommandStage
from engine.engine import Engine, EngineConfig, ReanchorState, reanchor_config
from engine.lifecycle import Lifecycle
from engine.server import main
from engine.tests.test_engine import wait

FAST = dict(gen_ms=5, decode_ms=1, encode_ms=5, frames_per_chunk=2, fps=100)


def start(reanchor: str, options: dict | None = None) -> Engine:
    cfg = EngineConfig(backend="fake", width=64, height=48, backend_options=FAST, reanchor=reanchor,
                       reanchor_options=options or {})
    e = Engine(cfg, lambda b: None, lambda m: None)
    e.start()
    assert wait(lambda: e.lifecycle.state is Lifecycle.GENERATING)
    return e


@pytest.fixture
def engine_on():
    e = start("on")
    yield e
    e.stop()


def winter(ramp: int) -> dict:
    return {"prompt": "a meadow in deep winter snow", "ramp_chunks": ramp, "provisional": False, "cap": 1.0}


def test_off_never_calls_reanchor():
    e = start("off")
    try:
        assert wait(lambda: e.chunk_generated >= 10)
        rec = e.handle_target(winter(1), time.monotonic_ns())
        assert wait(lambda: rec.stage is CommandStage.SHOWN)
        assert wait(lambda: e.chunk_generated >= 70)  # past the refresh interval too
        assert e.reanchor_state is ReanchorState.OFF and e.policy is None
        assert e.backend.reanchor_calls == 0
        assert e.stats()["reanchor"] == {"state": "off", "count": 0, "last_strength": None, "last_reason": None,
                                         "last_chunk": None, "config": {},
                                         "chunks_since_anchor": None}
    finally:
        e.stop()


@pytest.mark.parametrize("ramp", [1, 2])
def test_sudden_final_reanchors_once_at_full_strength(engine_on, ramp):
    e = engine_on
    assert e.reanchor_state is ReanchorState.ON and e.policy is not None
    assert wait(lambda: e.chunk_generated >= 10)  # past the policy's minimum gap since the start
    rec = e.handle_target(winter(ramp), time.monotonic_ns())
    assert wait(lambda: rec.stage is CommandStage.SHOWN)
    # The anchor fires once the ramp reaches the full target: the ramp's last chunk.
    assert wait(lambda: e.stats()["reanchor"]["count"] >= 1)
    r = e.stats()["reanchor"]
    assert (r["state"], r["count"], r["last_strength"], r["last_reason"]) == ("on", 1, 1.0, "steer")
    assert r["last_chunk"] == rec.applies_from_chunk + ramp - 1
    assert e.backend.reanchor_calls == 1 and e.backend.last_reanchor_strength == 1.0


def test_refresh_after_24_idle_chunks_from_the_clean_anchor(engine_on):
    e = engine_on
    assert wait(lambda: e.stats()["reanchor"]["count"] >= 1, timeout=20)
    r = e.stats()["reanchor"]
    # The policy is built before chunk 0, so its 24th next_chunk call is chunk 23.
    assert (r["last_reason"], r["last_strength"], r["last_chunk"]) == ("refresh", 0.5, 23)
    assert e.backend.reanchor_calls >= 1 and e.backend.last_reanchor_source == "clean"


def test_steer_anchor_goes_to_the_backend_from_the_last_latent(engine_on):
    e = engine_on
    assert wait(lambda: e.chunk_generated >= 10)
    rec = e.handle_target(winter(1), time.monotonic_ns())
    assert wait(lambda: rec.stage is CommandStage.SHOWN)
    assert wait(lambda: e.stats()["reanchor"]["last_reason"] == "steer")
    assert e.backend.last_reanchor_source == "last"


def test_refresh_source_option_reaches_the_backend():
    e = start("on", {"refresh_every_chunks": "10", "refresh_source": "last"})
    try:
        assert wait(lambda: e.stats()["reanchor"]["count"] >= 1, timeout=20)
        assert e.stats()["reanchor"]["config"] == {"refresh_every_chunks": 10, "refresh_source": "last"}
        assert e.backend.last_reanchor_source == "last"
    finally:
        e.stop()


def test_reset_restarts_the_refresh_count():
    # Refresh off, because a refresh also clears the count and could hide a reset that does not.
    e = start("on", {"refresh_every_chunks": "none"})
    try:
        assert wait(lambda: e.policy.chunks_since_anchor >= 30)
        done = e.reset(time.monotonic_ns()).result(5)
        assert done.wait(5)
        assert e.policy.chunks_since_anchor < 20  # counting restarted at the reset
    finally:
        e.stop()


class NoReanchorBackend(FakeBackend):
    reanchor = None  # like a backend with no re-anchor API


def test_backend_without_reanchor_is_unsupported(monkeypatch):
    monkeypatch.setattr(engine_module, "load_backend", lambda name, **kw: NoReanchorBackend(
        kw["width"], kw["height"], kw["steps"], **FAST))
    e = start("on")
    try:
        assert wait(lambda: e.chunk_generated >= 5)
        assert e.reanchor_state is ReanchorState.UNSUPPORTED and e.policy is None
        assert e.stats()["reanchor"]["state"] == "unsupported"
    finally:
        e.stop()


# --- --reanchor-opt ----------------------------------------------------------------------------


def test_reanchor_options_convert_to_the_field_types():
    cfg = reanchor_config({"refresh_every_chunks": "none", "steer_strength": "0.9", "min_gap_chunks": "8"})
    assert cfg.refresh_every_chunks is None
    assert cfg.steer_strength == 0.9 and isinstance(cfg.steer_strength, float)
    assert cfg.min_gap_chunks == 8 and isinstance(cfg.min_gap_chunks, int)
    assert reanchor_config({"refresh_every_chunks": "30"}).refresh_every_chunks == 30
    assert reanchor_config({}) == reanchor_config({"min_gap_chunks": "8"})  # 8 is the default


def test_bad_reanchor_settings_fail_before_anything_starts(monkeypatch):
    with pytest.raises(ValueError):
        Engine(EngineConfig(reanchor="maybe"), lambda b: None, lambda m: None)
    for options in ({"bogus": "1"}, {"min_gap_chunks": "eight"}, {"refresh_every_chunks": "often"},
                    {"steer_strength": "1.5"}, {"refresh_source": "golden"}):
        with pytest.raises(ValueError):
            reanchor_config(options)
        with pytest.raises(ValueError):  # at Engine construction, before the backend loads
            Engine(EngineConfig(reanchor="on", reanchor_options=options), lambda b: None, lambda m: None)

    def no_build(*a, **kw):
        raise AssertionError("the app must not be built")
    monkeypatch.setattr("engine.server.build_app", no_build)
    for argv in (["--reanchor", "on", "--reanchor-opt", "bogus=1"], ["--reanchor-opt", "refresh_every_chunks"]):
        with pytest.raises(SystemExit):
            main(argv)


def test_refresh_off_never_refreshes():
    e = start("on", {"refresh_every_chunks": "none"})
    try:
        assert e.reanchor_state is ReanchorState.ON and e.policy.config.refresh_every_chunks is None
        assert wait(lambda: e.chunk_generated >= 70, timeout=30)  # about three default refresh intervals
        r = e.stats()["reanchor"]
        assert (r["count"], r["last_reason"], r["config"]) == (0, None, {"refresh_every_chunks": None})
        assert r["chunks_since_anchor"] >= 70
        assert e.backend.reanchor_calls == 0
    finally:
        e.stop()
