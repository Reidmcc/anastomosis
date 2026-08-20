"""The rhizotron backend -- DESIGN.md §15, build step 1.

What step 1 has to prove, before any root grows:

* the descent is an *exact* integer translation of the stateful fields, at
  any rate the config can ask for (§15.4 -- "no resampling, no accumulation
  of interpolation error");
* the soil generator is deterministic in the seed and inexhaustible in the
  depth counter -- the same world twice, and a different world ten thousand
  rows down;
* the scroll is visible to the safety stage's reprojection (§15.7.1), and
  the flash bound holds under this backend exactly as it does under the
  fungal two -- strictly, with the descent stopped, and by the WCAG area
  criterion with it running;
* moisture stays bounded and behaves like water: rain lands at the surface
  and the wet front moves *down*;
* a checkpoint resumes bit-identically, descent counters included, which is
  the only test that catches a piece of state quietly left out (§4.6).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import reference as R
from anastomosis import checkpoint, config, events
from anastomosis.rhizotron import (
    MARGIN_ROWS,
    RhizotronEngine,
    RhizotronGeometry,
)

WIDTH, HEIGHT = 96, 64

# Same figure, same reasoning as test_flash_safety.py.
F16_TOLERANCE = 0.0005

FLASH_LUMINANCE_FRACTION = 0.10
FLASH_AREA_FRACTION = 0.25

# Overrides that freeze the moisture physics entirely, leaving the pass a
# pure scroll -- the isolation several tests below build on.
STILL_WATER = {
    "rhizotron.percolation_rate": 0.0,
    "rhizotron.lateral_spread": 0.0,
    "rhizotron.drainage_rate": 0.0,
    "rhizotron.rain_base": 0.0,
    "rhizotron.rain_event_gain": 0.0,
}

# The fastest descent the config admits: the rate ceiling, at the slowest
# tick rate, so each tick advances as many rows as a config can ask for.
FULL_SPEED = {
    "rhizotron.descent_rate": 60.0,
    "rhizotron.descent_wander": 0.0,
    "sim_hz": 4.0,
}


def _resolve(overrides=None, **macros) -> config.Params:
    return config.Config(
        macros=config.Macros(**macros), overrides=dict(overrides or {})
    ).resolve()


def _engine(gpu_device, params, seed=4242) -> RhizotronEngine:
    device, _ = gpu_device
    return RhizotronEngine(device, WIDTH, HEIGHT, params, seed=seed)


def _moisture(engine) -> np.ndarray:
    pair = engine.moisture
    return checkpoint._read_texture(engine.device, pair.textures[pair.index])


def _write_moisture(engine, data: np.ndarray) -> None:
    for texture in engine.moisture.textures:
        checkpoint._write_texture(engine.device, texture, data)


def _hdr(engine) -> np.ndarray:
    return checkpoint._read_texture(engine.device, engine.hdr)


def _capture(engine, params, target, fmt, frames, mutate=None, scheduler=None):
    """Run frames, returning the Oklab lightness of each output."""
    scheduler = scheduler or events.EventScheduler(seed=11)
    captured = []
    for index in range(frames):
        if mutate is not None:
            mutate(index, params)
        rows, _ = scheduler.pack(8)
        engine.tick(params, rows)
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)
        scheduler.update(1.0 / params.sim_hz, params.events)
        captured.append(R.lightness(engine.read_final_rgba()[..., :3]))
    return np.stack(captured)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_geometry_is_the_view_plus_its_margins():
    params = _resolve()
    geometry = RhizotronGeometry.derive(WIDTH, HEIGHT, params)
    assert geometry.width % 32 == 0
    assert geometry.height == geometry.view_rows + 2 * MARGIN_ROWS
    assert geometry.problems() == []
    assert "soil column" in geometry.describe()

    # A file claiming margins the build does not have is unusable, not a
    # crash: the bounds check catches it before anything is allocated.
    bent = RhizotronGeometry(
        width=geometry.width, height=geometry.height + 3,
        view_rows=geometry.view_rows)
    assert bent.problems()


# ---------------------------------------------------------------------------
# The descent
# ---------------------------------------------------------------------------


def test_the_scroll_is_an_exact_integer_translation(gpu_device):
    """§15.4's central claim, asserted bit-for-bit.

    With the moisture physics frozen the pass is a pure shift, so after N
    ticks whose descent crossed K whole rows, every surviving texel must be
    *exactly* the texel K rows below it -- the same half-floats, not close
    ones. Anything less would be resampling, and resampling error compounds
    over a descent that never ends.
    """
    params = _resolve(overrides={**STILL_WATER, **FULL_SPEED})
    engine = _engine(gpu_device, params)
    g = engine.geometry

    rng = np.random.default_rng(7)
    pattern = np.zeros((g.height, g.width, 4), dtype=np.float16)
    pattern[..., 0] = rng.uniform(0.05, 1.0, (g.height, g.width))
    # The EMA channel equals the moisture channel, so the shading lowpass is
    # at its fixed point and the pass moves both channels untouched.
    pattern[..., 1] = pattern[..., 0]
    _write_moisture(engine, pattern)

    start = engine.descent_state()["origin"]
    for _ in range(15):
        engine.tick(params, [])
    shifted_rows = engine.descent_state()["origin"] - start
    assert shifted_rows >= 2, "the full-speed descent should cross whole rows"
    assert shifted_rows <= 15 * 2, "no tick may scroll past the margin budget"

    after = _moisture(engine)
    surviving = g.height - shifted_rows
    assert np.array_equal(
        after[:surviving, :, :2], pattern[shifted_rows:, :, :2]
    ), "the descent resampled or disturbed rows it should only have moved"


def test_fresh_rows_are_generated_below(gpu_device):
    """Rows entering from beneath hold generated soil, not stale texels."""
    params = _resolve(overrides={**STILL_WATER, **FULL_SPEED})
    engine = _engine(gpu_device, params)
    g = engine.geometry

    # A sentinel value the generator cannot produce (baselines sit far below
    # it), so any surviving trace of it in a regenerated row is a bug.
    pattern = np.full((g.height, g.width, 4), 1.375, dtype=np.float16)
    pattern[..., 2:] = 0.0
    _write_moisture(engine, pattern)

    start = engine.descent_state()["origin"]
    for _ in range(15):
        engine.tick(params, [])
    shifted_rows = engine.descent_state()["origin"] - start
    assert shifted_rows >= 2

    after = _moisture(engine).astype(np.float32)
    fresh = after[g.height - shifted_rows:, :, 0]
    assert float(fresh.max()) < 1.0, "a regenerated row still holds old data"
    assert float(fresh.min()) > 0.0, "a regenerated row came up empty"


def test_the_soil_is_deterministic_and_never_repeats_with_depth(gpu_device):
    """Same seed and depth: the same world. Ten thousand rows down: another.

    The comparison is on the compositor's own output (the HDR target, before
    the exposure governor and the limiter make it history-dependent), because
    the image is the only place the procedural soil exists.
    """
    params = _resolve(overrides=STILL_WATER)
    engine = _engine(gpu_device, params)

    import wgpu

    device, _ = gpu_device
    target = device.create_texture(
        size=(WIDTH, HEIGHT, 1), format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    ).create_view()

    engine.render(params, frac=0.5, target_view=target, target_format="rgba8unorm")
    here = _hdr(engine)

    state = engine.descent_state()
    engine.restore_descent({**state, "origin": state["origin"] + 10_000})
    engine.render(params, frac=0.5, target_view=target, target_format="rgba8unorm")
    deeper = _hdr(engine)

    engine.restore_descent(state)
    engine.render(params, frac=0.5, target_view=target, target_format="rgba8unorm")
    back = _hdr(engine)

    assert np.array_equal(here, back), "the soil is not a function of its depth"
    difference = np.abs(
        deeper[..., :3].astype(np.float32) - here[..., :3].astype(np.float32)
    ).mean()
    scale = np.abs(here[..., :3].astype(np.float32)).mean()
    assert difference > 0.05 * scale, (
        "soil ten thousand rows down is indistinguishable from soil here"
    )


def test_the_descent_is_reported_to_the_reprojection(gpu_device):
    """§15.7.1: the scroll must land in the velocity the safety stage reads.

    The sign convention is the test: material moves up the screen, so the
    stored velocity's y component must be negative, and its magnitude must
    match the displayed displacement the engine actually produced.
    """
    params = _resolve(overrides=FULL_SPEED)
    engine = _engine(gpu_device, params)

    import wgpu

    device, _ = gpu_device
    target = device.create_texture(
        size=(WIDTH, HEIGHT, 1), format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
    ).create_view()

    frame_dt = 1.0 / 30.0
    displayed = []
    for _ in range(6):
        engine.tick(params, [])
        before = engine._display_prev
        engine.render(params, frac=1.0, target_view=target,
                      target_format="rgba8unorm", frame_dt=frame_dt)
        displayed.append(engine._display_prev - before)

    velocity = checkpoint._read_texture(
        engine.device, engine.scroll_vel).astype(np.float32)
    vy = float(velocity[0, 0, 1])
    assert vy < 0.0, "the scroll's velocity must point up the screen"

    reproject = frame_dt * params.sim_hz
    expected = -(displayed[-1] / engine.geometry.view_rows) / reproject
    assert vy == pytest.approx(expected, rel=0.02), (
        "the velocity texture does not carry the displacement the frame made"
    )


# ---------------------------------------------------------------------------
# Flash safety under this backend
# ---------------------------------------------------------------------------


def test_the_bound_holds_exactly_with_the_descent_stopped(gpu_device, offscreen_target):
    """Descent at zero makes reprojection the identity; the bound is exact.

    The mirror of test_limiter_bound_holds_exactly, with this backend's own
    adversaries stepped un-ramped every frame: the wetting machinery, the
    soil's luminance span, a cloudburst arriving and vanishing, and the
    palette slamming across the family ring. The application ramps all of
    these; the guarantee must not depend on that.
    """
    params = _resolve(overrides={
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
    })

    def mutate(index, p):
        extreme = index % 2 == 0
        p.rhizotron.wet_darken = 0.8 if extreme else 0.0
        p.rhizotron.wet_chroma = 2.0 if extreme else 0.0
        p.rhizotron.soil_l_range = 0.40 if extreme else 0.0
        p.rhizotron.rain_base = 1.0 if extreme else 0.0
        p.render.background_luma = 0.3 if extreme else 0.0
        p.render.l_max = 0.9 if extreme else 0.05
        p.render.c_max = 0.22 if extreme else 0.0
        p.render.hue_anchor = 0.0 if extreme else math.pi
        p.safety.exposure_target = 0.40 if extreme else 0.02

    engine = _engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 40, mutate=mutate)

    deltas = np.abs(np.diff(frames, axis=0))
    limit = params.safety.max_luma_delta + F16_TOLERANCE
    worst = float(deltas.max())
    assert worst <= limit, (
        f"adversarial stepping under the rhizotron produced a lightness step "
        f"of {worst:.5f}, above the limit {limit:.5f}"
    )


def test_wcag_area_criterion_under_full_descent_and_rain(gpu_device, offscreen_target):
    """The fastest config-reachable descent, in the rain, at fixed pixels.

    Honest motion counts against the WCAG area measure even though the
    limiter rightly permits it (§14.5.1), so the fastest scroll plus a rain
    event mid-run is this backend's busy corner.
    """
    params = _resolve(overrides={
        "rhizotron.descent_rate": 60.0,
        "events.attack_seconds": 2.0,
        "events.hold_seconds": 60.0,
    })

    scheduler = events.EventScheduler(seed=5)

    def mutate(index, p):
        if index == 10:
            scheduler.trigger("bloom", p.events)

    engine = _engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(
        engine, params, target, fmt, 45, mutate=mutate, scheduler=scheduler)

    deltas = np.abs(np.diff(frames, axis=0))
    flashing = (deltas >= FLASH_LUMINANCE_FRACTION).mean(axis=(1, 2))
    worst = float(flashing.max())
    assert worst < FLASH_AREA_FRACTION, (
        f"{worst:.1%} of the screen changed by >={FLASH_LUMINANCE_FRACTION:.0%} "
        f"lightness in one frame under the descent; the WCAG area threshold "
        f"is {FLASH_AREA_FRACTION:.0%}"
    )


# ---------------------------------------------------------------------------
# Moisture
# ---------------------------------------------------------------------------


def test_moisture_stays_bounded_and_rain_soaks_downward(gpu_device):
    """Water behaves like water: bounded everywhere, and rain moves down.

    The rain is isolated against a *control run* -- a second engine on the
    same seed that gets no event -- because a young column is still settling
    toward its resting bands, and that drift would otherwise pollute the
    anomaly on both sides of the comparison. The engines are deterministic
    (asserted separately below), so the subtraction is exact. The event's
    envelope is compressed the way the rift recovery test compresses its
    attack: the percolation is what is under test, not the envelope.
    """
    params = _resolve(overrides={
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
        "events.attack_seconds": 1.0,
        "events.hold_seconds": 6.0,
        "events.release_seconds": 5.0,
    })
    rained = _engine(gpu_device, params, seed=13)
    control = _engine(gpu_device, params, seed=13)
    g = rained.geometry

    scheduler = events.EventScheduler(seed=13)

    def run(ticks):
        for _ in range(ticks):
            rows, _ = scheduler.pack(8)
            rained.tick(params, rows)
            control.tick(params, [])
            scheduler.update(1.0 / params.sim_hz, params.events)

    run(60)
    event = scheduler.trigger("bloom", params.events)
    assert event is not None

    def anomaly_centroid():
        wet = np.clip(
            _moisture(rained).astype(np.float32)[..., 0]
            - _moisture(control).astype(np.float32)[..., 0],
            0.0, None)
        rows = wet.sum(axis=1)
        total = rows.sum()
        assert total > 1e-3, "the rain event added no water"
        return float((rows * np.arange(g.height)).sum() / total)

    run(60)  # 3 s: the cloudburst is landing near the surface
    shallow = anomaly_centroid()
    run(400)  # the pulse has ended; 20 s of soaking
    deep = anomaly_centroid()

    assert deep > shallow + 1.0, (
        f"the wet front did not descend (centroid {shallow:.2f} -> {deep:.2f})"
    )

    field = _moisture(rained).astype(np.float32)
    assert np.isfinite(field).all()
    assert float(field[..., 0].min()) >= 0.0
    assert float(field[..., 0].max()) <= 1.5


def test_two_columns_from_one_seed_are_the_same_column(gpu_device):
    params = _resolve()
    a = _engine(gpu_device, params, seed=77)
    b = _engine(gpu_device, params, seed=77)
    for _ in range(20):
        a.tick(params, [])
        b.tick(params, [])
    assert np.array_equal(_moisture(a), _moisture(b))
    assert a.descent_state() == b.descent_state()


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def test_a_resumed_column_evolves_bit_identically(gpu_device):
    """The §4.6 discipline, applied to the new backend.

    The one check that catches a piece of state quietly left out of the
    snapshot -- which for this backend means the descent's counters and walk
    as much as the moisture itself.
    """
    params = _resolve()
    original = _engine(gpu_device, params, seed=31)
    scheduler = events.EventScheduler(seed=3)
    for _ in range(30):
        rows, _ = scheduler.pack(8)
        original.tick(params, rows)
        scheduler.update(1.0 / params.sim_hz, params.events)

    snapshot = checkpoint.capture(
        original, scheduler=scheduler, sim_hz=params.sim_hz)
    assert snapshot.meta["backend"] == "rhizotron"

    geometry = checkpoint.required_geometry(snapshot, backend="rhizotron")
    assert geometry == original.geometry
    # And the other backends must refuse it rather than misread it.
    assert checkpoint.required_geometry(snapshot, backend="layered") is None

    device, _ = gpu_device
    resumed = RhizotronEngine(
        device, WIDTH, HEIGHT, params, seed=1, geometry=geometry)
    assert checkpoint.restore(resumed, snapshot)

    assert resumed.descent_state() == original.descent_state()
    assert resumed.tick_count == original.tick_count

    for _ in range(5):
        original.tick(params, [])
        resumed.tick(params, [])
    assert np.array_equal(_moisture(original), _moisture(resumed)), (
        "a resumed column diverged from the one it was captured from"
    )
    assert resumed.descent_state() == original.descent_state()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_the_backend_is_registered_everywhere():
    from anastomosis import app as app_module
    from anastomosis.__main__ import build_parser

    assert config.normalise_backend("rhizotron") == "rhizotron"
    assert "rhizotron" in config.BACKENDS
    assert checkpoint.default_checkpoint_path(
        "rhizotron").name == "checkpoint-rhizotron.npz"
    assert checkpoint.layout_for("rhizotron").name == "rhizotron"

    engine_class, geometry_class = app_module.backend_classes("rhizotron")
    assert engine_class is RhizotronEngine
    assert geometry_class is RhizotronGeometry

    args = build_parser().parse_args(["--backend", "rhizotron"])
    assert args.backend == "rhizotron"
