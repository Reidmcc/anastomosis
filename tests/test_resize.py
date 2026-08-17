"""Resizing the window must not restart the world.

A resize -- dragging an edge, hitting maximise, a monitor's scale factor
changing -- used to rebuild the engine, which meant reseeding every field,
every agent and the tick counter: a hard cut in the middle of a session that is
supposed to run for days. The window size is a property of the *presentation*,
so it may only rebuild the presentation chain.

Three things are asserted here:

* the simulation survives untouched (same layer resources, same agents, same
  tick count) and keeps running afterwards;
* the image survives -- the slew limiter's history is carried across, so the
  first frame at the new size does not start from black and climb back out;
* the application coalesces the burst of sizes a drag produces, and never
  constructs a second engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference as R
from anastomosis import config, engine as engine_module

WIDTH, HEIGHT = 128, 96


def _run(engine, params, target, fmt, frames):
    for _ in range(frames):
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)


def _read_field(device, pingpong):
    """The current half of a ping-pong pair, as float32 RGBA."""
    texture = pingpong.textures[pingpong.index]
    width, height = texture.size[0], texture.size[1]
    raw = device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, 1),
    )
    return np.frombuffer(raw, dtype=np.float16).reshape(height, width, 4)


def _settled_engine(gpu_device, offscreen_target, frames=25):
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 2
    engine = engine_module.Engine(device, WIDTH, HEIGHT, params, seed=31)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    _run(engine, params, target, fmt, frames)
    return engine, params


# ---------------------------------------------------------------------------
# Aspect correction (no GPU needed)
# ---------------------------------------------------------------------------


def test_aspect_correction_is_neutral_when_shapes_agree():
    assert engine_module.aspect_correction(1920, 1080, 1280, 720) == (1.0, 1.0)


def test_aspect_correction_widens_rather_than_stretches():
    """A wider window samples more field horizontally, at unchanged height."""
    scale_x, scale_y = engine_module.aspect_correction(2560, 720, 1280, 720)
    assert scale_x == pytest.approx(2.0)
    assert scale_y == pytest.approx(1.0)

    scale_x, scale_y = engine_module.aspect_correction(1280, 1440, 1280, 720)
    assert scale_x == pytest.approx(1.0)
    assert scale_y == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------


def test_resize_preserves_simulation_state(gpu_device, offscreen_target):
    """The layers, their contents and the tick counter all come through."""
    device, _ = gpu_device
    engine, params = _settled_engine(gpu_device, offscreen_target)

    before_tick = engine.tick_count
    before_specs = [layer.spec for layer in engine.layers]
    before_textures = [layer.pigment.textures[0] for layer in engine.layers]
    before_agents = [layer.agents_buf for layer in engine.layers]
    before_field = _read_field(device, engine.layers[0].reaction)
    before_hue = engine.hue_phase

    engine.resize(WIDTH * 2, HEIGHT * 2)

    assert engine.tick_count == before_tick, "the tick counter was reset"
    assert [layer.spec for layer in engine.layers] == before_specs, (
        "simulation resolution changed; it is fixed at startup by design"
    )
    assert [layer.pigment.textures[0] for layer in engine.layers] == before_textures, (
        "field textures were reallocated, which discards their state"
    )
    assert [layer.agents_buf for layer in engine.layers] == before_agents, (
        "the agent buffers were reallocated, which reseeds every agent"
    )
    assert np.array_equal(_read_field(device, engine.layers[0].reaction), before_field), (
        "the reaction field changed across the resize"
    )
    assert engine.hue_phase == before_hue, "the hue anchor jumped"

    assert (engine.width, engine.height) == (WIDTH * 2, HEIGHT * 2)
    assert (engine.sim_width, engine.sim_height) == (WIDTH, HEIGHT)


def test_resize_keeps_running_afterwards(gpu_device, offscreen_target):
    """The world keeps advancing, and the output is finite at the new size."""
    engine, params = _settled_engine(gpu_device, offscreen_target)
    before = engine.tick_count

    new_w, new_h = 192, 160
    engine.resize(new_w, new_h)
    target, fmt = offscreen_target(new_w, new_h)
    _run(engine, params, target, fmt, 20)

    assert engine.tick_count == before + 20
    image = engine.read_final_rgba()
    assert image.shape == (new_h, new_w, 4)
    assert np.isfinite(image).all(), "output contains non-finite values"

    stats = engine.read_stats()
    assert stats["mean_v"] > 0.02, f"field died across the resize: {stats['mean_v']}"


def test_resize_carries_the_image_across(gpu_device, offscreen_target):
    """The frame on screen seeds the new history, so nothing fades from black.

    The slew limiter emits ``history + bounded step``. With an empty history
    the image would have to climb out of black at ~0.01 lightness per frame --
    about a second of fade, which is exactly the interruption this change
    exists to remove.
    """
    engine, params = _settled_engine(gpu_device, offscreen_target)
    before = float(R.lightness(engine.read_final_rgba()[..., :3]).mean())
    assert before > 0.01, "nothing on screen to begin with; test is vacuous"

    new_w, new_h = WIDTH * 2, HEIGHT * 2
    engine.resize(new_w, new_h)
    target, fmt = offscreen_target(new_w, new_h)
    _run(engine, params, target, fmt, 1)

    after = float(R.lightness(engine.read_final_rgba()[..., :3]).mean())
    step = params.safety.max_luma_delta
    assert abs(after - before) <= 4.0 * step, (
        f"mean lightness jumped {before:.4f} -> {after:.4f} across the resize; "
        f"the per-frame limit is {step:.4f}"
    )


def test_repeated_resizes_do_not_leak_bind_groups(gpu_device, offscreen_target):
    """Caches are keyed by ``id()``, so retired resources must be evicted.

    Left in place, an entry for a freed texture is not just a leak: CPython
    recycles addresses, so a later object can land on one and silently bind a
    destroyed resource.
    """
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 1
    engine = engine_module.Engine(device, 96, 64, params, seed=17)
    target, fmt = offscreen_target(96, 64)
    # The sanitise pass runs every 60 ticks and flips field parity, so the full
    # set of ping-pong combinations only appears after a couple of cycles. Let
    # the cache settle first, or its normal growth masks what is measured here.
    _run(engine, params, target, fmt, 150)

    retired = {id(engine.hdr_view), id(engine.img_partials)}
    retired |= {id(view) for view in engine.final.views}
    retired |= {id(engine._buffer_binding(engine.img_partials))}

    engine.resize(160, 128)

    stale = [key for key in engine._bind_cache if not retired.isdisjoint(key)]
    assert not stale, f"{len(stale)} cached bind groups still name a retired target"
    assert retired.isdisjoint(engine._buf_bindings), (
        "a retired buffer is still held by the binding cache"
    )

    # Alternating between two sizes: whatever the first cycle leaves cached,
    # the second must not add to it.
    counts = []
    for width, height in [(192, 160), (96, 64), (192, 160), (96, 64)]:
        engine.resize(width, height)
        target, fmt = offscreen_target(width, height)
        _run(engine, params, target, fmt, 10)
        counts.append(len(engine._bind_cache))

    assert counts[-1] <= counts[1], (
        f"bind group cache grew across repeated resizes: {counts}"
    )


def test_resize_to_the_same_size_is_a_noop(gpu_device, offscreen_target):
    engine, _ = _settled_engine(gpu_device, offscreen_target, frames=5)
    final_before = engine.final.textures[0]
    engine.resize(WIDTH, HEIGHT)
    assert engine.final.textures[0] is final_before


# ---------------------------------------------------------------------------
# The application's size tracking
# ---------------------------------------------------------------------------


class _FakeCanvas:
    def __init__(self, size):
        self.size = size

    def get_physical_size(self):
        return self.size


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def resize(self, width, height):
        self.calls.append((width, height))


def _fake_app(tmp_path):
    from anastomosis.app import AppOptions, Application

    app = Application(AppOptions(config_path=tmp_path / "config.toml", ui=False))
    app.canvas = _FakeCanvas((1280, 720))
    app.engine = _FakeEngine()
    app._size = (1280, 720)
    return app


def test_drag_is_coalesced_into_one_resize(tmp_path):
    """A drag reports a new size every frame; only the settled one is applied."""
    from anastomosis import app as app_module

    app = _fake_app(tmp_path)
    now = 0.0
    for width in range(1281, 1300):  # the drag
        app.canvas.size = (width, 720)
        now += 0.016
        app._follow_canvas_size(now)
    assert app.engine.calls == [], "reallocated mid-drag"

    now += app_module.RESIZE_SETTLE  # the button comes up and the size holds
    app._follow_canvas_size(now)
    assert app.engine.calls == [(1299, 720)]
    assert app._size == (1299, 720)

    now += 1.0
    app._follow_canvas_size(now)
    assert app.engine.calls == [(1299, 720)], "resized again with nothing changed"


def test_maximise_resizes_without_rebuilding_the_engine(tmp_path):
    app = _fake_app(tmp_path)
    engine = app.engine

    app.canvas.size = (1920, 1080)
    app._follow_canvas_size(0.0)
    app._follow_canvas_size(1.0)

    assert engine.calls == [(1920, 1080)]
    assert app.engine is engine, "the engine was replaced, restarting the world"


def test_degenerate_sizes_are_ignored(tmp_path):
    """A minimised window reports a zero dimension on some backends."""
    app = _fake_app(tmp_path)
    for size in ((0, 0), (1280, 0), (0, 720)):
        app.canvas.size = size
        app._follow_canvas_size(0.0)
        app._follow_canvas_size(1.0)
    assert app.engine.calls == []
