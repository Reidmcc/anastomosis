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
    TIP_ALIVE,
    TIP_DTYPE,
    TIP_SPENT,
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


def _structure(engine) -> np.ndarray:
    pair = engine.structure
    return checkpoint._read_texture(engine.device, pair.textures[pair.index])


def _tips(engine) -> np.ndarray:
    raw = engine.device.queue.read_buffer(engine.tips.cur)
    return np.frombuffer(raw, dtype=TIP_DTYPE).copy()


def _write_tips(engine, tips: np.ndarray) -> None:
    for buffer in engine.tips.buffers:
        engine.device.queue.write_buffer(buffer, 0, tips.tobytes())


def _alive(tips: np.ndarray) -> np.ndarray:
    return (tips["flags"] & TIP_ALIVE) != 0


def _orders(tips: np.ndarray) -> np.ndarray:
    return tips["flags"] & 3


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
    assert np.array_equal(_structure(a), _structure(b))
    assert np.array_equal(_tips(a), _tips(b))
    assert a.descent_state() == b.descent_state()


# ---------------------------------------------------------------------------
# The plant (§15.11 step 3)
# ---------------------------------------------------------------------------

# Overrides that isolate one tropism: everything else that can turn a tip is
# off, the water is still, and the world stands still.
BARE_STEERING = {
    **STILL_WATER,
    "rhizotron.descent_rate": 0.0,
    "rhizotron.descent_wander": 0.0,
    "rhizotron.stone_amount": 0.0,
    "rhizotron.hardpan_amount": 0.0,
    "rhizotron.hydro_gain": 0.0,
    "rhizotron.chemo_gain": 0.0,
    "rhizotron.avoid_gain": 0.0,
    "rhizotron.thigmo_gain": 0.0,
    "rhizotron.tip_jitter": 0.0,
    "rhizotron.branch_prob": 0.0,
}


def _single_tip(engine, x, y, heading, order=0, side=0):
    tips = np.zeros(max(engine.geometry.tips_total, 1), dtype=TIP_DTYPE)
    tips["x"][0] = x
    tips["y"][0] = y
    tips["heading"][0] = heading
    tips["rng"][0] = 12345
    tips["flags"][0] = order | (4 if side else 0) | TIP_ALIVE | TIP_SPENT
    tips["generation"][0] = 1.0
    _write_tips(engine, tips)


def test_gravitropism_turns_an_axis_downward(gpu_device):
    """The shape-giving tropism, isolated: a tip launched sideways plunges.

    Every other steering input is zeroed, so the only thing acting is the
    angular relaxation toward the axis's setpoint -- straight down.
    """
    params = _resolve(overrides=BARE_STEERING)
    engine = _engine(gpu_device, params)
    g = engine.geometry
    _single_tip(engine, g.width * 0.5, g.height * 0.3, heading=0.0)

    start_y = g.height * 0.3
    for _ in range(60):
        engine.tick(params, [])

    tips = _tips(engine)
    assert _alive(tips)[0]
    assert abs(tips["heading"][0] - math.pi / 2.0) < 0.05, (
        f"an axis launched sideways is heading {tips['heading'][0]:.2f}, "
        f"not down"
    )
    assert tips["y"][0] > start_y + 5.0, "the axis did not descend"


def test_branching_builds_a_tree_in_its_own_slots(gpu_device):
    """Laterals and fines appear, at their orders, in their parents' blocks.

    The slot arithmetic is the load-bearing part: a child may only exist if
    its parent has handed out its index, which is what makes birth
    deterministic under any GPU scheduling.
    """
    params = _resolve(overrides={
        **STILL_WATER,
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
    })
    engine = _engine(gpu_device, params)
    for _ in range(500):
        engine.tick(params, [])

    tips = _tips(engine)
    g = engine.geometry
    born = (tips["flags"] & TIP_SPENT) != 0
    orders = _orders(tips)
    a, l = g.max_axes, g.laterals_per_axis
    assert born[: params.rhizotron.axes_per_plant].all()
    assert (born & (orders == 1)).sum() >= 4, "no laterals branched"
    assert (born & (orders == 2)).sum() >= 2, "no fines branched"

    children = tips["flags"] >> 8
    generations = tips["generation"].astype(np.int64)
    for slot in np.nonzero(born)[0]:
        if slot < a:
            continue
        if slot < a + a * l:
            parent, index = (slot - a) // l, (slot - a) % l
            cycle = l
        else:
            fi = slot - a - a * l
            f = g.fines_per_lateral
            parent, index = a + fi // f, fi % f
            cycle = f
        # A slot on its n-th life needed the parent's counter past
        # index + (n-1) * cycle when that life began.
        assert children[parent] > index + (generations[slot] - 1) * cycle, (
            f"slot {slot} (life {generations[slot]}) exists but its parent "
            f"{parent} only handed out {children[parent]} children"
        )
        assert orders[slot] == orders[parent] + 1

    # And laterals leave at their own angle: on average, further off vertical
    # than the axes that bore them.
    alive = _alive(tips)
    axis_off = np.abs(
        np.angle(np.exp(1j * (tips["heading"][alive & (orders == 0)]
                              - math.pi / 2))))
    lateral_off = np.abs(
        np.angle(np.exp(1j * (tips["heading"][alive & (orders == 1)]
                              - math.pi / 2))))
    if len(axis_off) and len(lateral_off):
        assert lateral_off.mean() > axis_off.mean()


def test_the_structure_is_built_downward_and_ages_upward(gpu_device):
    """The plant grows down from its crown, and its history reads upward:
    material near the crown is older than material near the front -- the
    pallor-by-age gradient the shading is built on."""
    params = _resolve(overrides={
        **STILL_WATER,
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
    })
    engine = _engine(gpu_device, params)
    for _ in range(600):
        engine.tick(params, [])

    field = _structure(engine).astype(np.float32)
    density = field[..., 0]
    age = field[..., 1]
    assert float(density.sum()) > 1.0, "the plant built nothing"

    rows = np.arange(engine.geometry.height)
    centroid = float((density.sum(axis=1) * rows).sum() / density.sum())
    seed_row = MARGIN_ROWS + engine.geometry.view_rows * 0.12
    assert centroid > seed_row + 3.0, "the structure did not extend downward"

    rooted = density > 0.05
    assert rooted.any()
    row_of = np.broadcast_to(rows[:, None], rooted.shape)
    shallow = rooted & (row_of < centroid)
    deep = rooted & (row_of >= centroid)
    if shallow.any() and deep.any():
        assert age[shallow].mean() > age[deep].mean(), (
            "material near the crown should be older than the growing front"
        )


def test_stones_slow_the_plant(gpu_device):
    """Thigmotropism's other half: stone is resistance, so the same plant in
    stonier ground travels less and lays down less. Same seed, one knob."""
    grown = {}
    for label, amount in (("clear", 0.0), ("stony", 1.0)):
        params = _resolve(overrides={
            **STILL_WATER,
            "rhizotron.descent_rate": 0.0,
            "rhizotron.descent_wander": 0.0,
            "rhizotron.stone_amount": amount,
        })
        engine = _engine(gpu_device, params, seed=21)
        for _ in range(400):
            engine.tick(params, [])
        grown[label] = float(_structure(engine).astype(np.float32)[..., 0].sum())
    assert grown["stony"] < grown["clear"] * 0.985, (
        f"stony ground ({grown['stony']:.1f}) should cost the plant travel "
        f"against clear ground ({grown['clear']:.1f})"
    )


def test_tips_carried_off_the_top_are_done(gpu_device):
    """A tip that cannot outgrow the descent leaves the window and dies."""
    params = _resolve(overrides={
        **STILL_WATER,
        **FULL_SPEED,
        "rhizotron.elong_axis": 0.0,
        "rhizotron.elong_lateral": 0.0,
        "rhizotron.elong_fine": 0.0,
        "rhizotron.branch_prob": 0.0,
    })
    engine = _engine(gpu_device, params)
    for _ in range(150):
        engine.tick(params, [])
    tips = _tips(engine)
    assert not _alive(tips).any(), (
        "tips scrolled off the top of the window are still alive"
    )
    # Spent stays spent: the slots must not be reborn.
    assert ((tips["flags"] & TIP_SPENT) != 0)[
        : params.rhizotron.axes_per_plant].all()


def test_fine_structure_turns_over(gpu_device):
    """Senescence: the same plant, with and without the forgetting.

    With turnover on, the community carries measurably less standing fine
    mass than the identical run with it off -- the fuzz is a process, not an
    accumulating painting. Same seed, one knob, so the comparison is exact.
    """
    grown = {}
    for label, rate in (("kept", 0.0), ("turning", 0.06)):
        params = _resolve(overrides={
            **STILL_WATER,
            "rhizotron.descent_rate": 0.0,
            "rhizotron.descent_wander": 0.0,
            "rhizotron.senescence_rate": rate,
            "rhizotron.senescence_delay": 5.0,
        })
        engine = _engine(gpu_device, params, seed=17)
        for _ in range(700):
            engine.tick(params, [])
        grown[label] = float(_structure(engine).astype(np.float32)[..., 0].sum())
    assert grown["turning"] < grown["kept"] * 0.9, (
        f"senescence removed almost nothing "
        f"({grown['turning']:.1f} vs {grown['kept']:.1f})"
    )


def test_succession_wakes_new_plants(gpu_device):
    """§15.4: plants age out, rest as seeds, and return -- and the seed bank
    fills axis slots the first seeding never used. Accelerated lifetimes, so
    a minute of simulation holds several plant lives."""
    params = _resolve(overrides={
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
        "rhizotron.axis_life": 12.0,
        "rhizotron.regermination_delay": 8.0,
        "rhizotron.germination_rate": 4.0,
        "rhizotron.germination_floor": 0.5,
    })
    engine = _engine(gpu_device, params, seed=23)
    for _ in range(1400):
        engine.tick(params, [])

    tips = _tips(engine)
    count = params.rhizotron.axes_per_plant
    generations = tips["generation"][: engine.geometry.max_axes]
    assert generations[:count].max() >= 2, (
        "no first-seeding axis has lived a second life"
    )
    assert generations[count:].max() >= 1, (
        "the seed bank never woke a reserve axis"
    )


def test_the_descent_follows_the_growing_front(gpu_device):
    """§15.4's front controller: growth surges, and the window leans after
    it. A fast-growing plant deep in the view pulls the rate multiplier up;
    a world with nothing living idles it toward its floor."""
    params = _resolve(overrides={
        "rhizotron.descent_rate": 1.0,
        "rhizotron.descent_wander": 0.0,
        "rhizotron.elong_axis": 20.0,
        "rhizotron.front_tau": 6.0,
    })
    engine = _engine(gpu_device, params)
    for _ in range(400):
        engine.tick(params, [])
    assert engine._descent_mult > 1.2, (
        f"the descent did not lean after a plant growing at the bottom "
        f"(multiplier {engine._descent_mult:.2f})"
    )

    barren = _resolve(overrides={
        "rhizotron.descent_rate": 1.0,
        "rhizotron.descent_wander": 0.0,
        "rhizotron.axis_life": 3.0,
        "rhizotron.elong_axis": 0.0,
        "rhizotron.elong_lateral": 0.0,
        "rhizotron.elong_fine": 0.0,
        "rhizotron.branch_prob": 0.0,
        "rhizotron.germination_rate": 0.0,
        "rhizotron.germination_floor": 0.0,
        "rhizotron.front_tau": 6.0,
    })
    idle = _engine(gpu_device, barren)
    for _ in range(400):
        idle.tick(barren, [])
    assert idle._descent_mult < 0.6, (
        f"an empty world should idle the descent toward its floor "
        f"(multiplier {idle._descent_mult:.2f})"
    )


def test_the_plant_is_not_a_comb(gpu_device):
    """§15.7(5): the morphology hazard is parallel verticals. The grown
    structure must spread across columns rather than condensing into cords --
    the §4.9 anti-condensation spirit, applied to the new world."""
    params = _resolve(overrides={
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
    })
    engine = _engine(gpu_device, params, seed=29)
    for _ in range(700):
        engine.tick(params, [])

    density = _structure(engine).astype(np.float32)[..., 0]
    total = float(density.sum())
    assert total > 1.0
    columns = density.sum(axis=0)
    assert float(columns.max()) / total < 0.20, (
        "a fifth of the plant's mass stands in one column: a cord, not a plant"
    )
    occupied = (columns > 0.02 * columns.max()).mean()
    assert occupied > 0.15, (
        f"the plant spans only {occupied:.0%} of the columns it could"
    )


def test_the_macros_reach_the_rhizotron():
    """§15.6, the deferred half landed: Tempo paces the world, Scale chooses
    a flora, Intensity sets the community's investment -- and the two mode
    tables carry identical rhizotron entries until step 6 earns a retune."""
    quiet = config.Config(macros=config.Macros(
        tempo=0.0, scale=0.0, intensity=0.0)).resolve()
    busy = config.Config(macros=config.Macros(
        tempo=1.0, scale=1.0, intensity=1.0)).resolve()

    assert busy.rhizotron.elong_axis > quiet.rhizotron.elong_axis
    assert busy.rhizotron.descent_rate > quiet.rhizotron.descent_rate
    assert busy.rhizotron.percolation_rate > quiet.rhizotron.percolation_rate
    assert busy.rhizotron.spacing_axis > quiet.rhizotron.spacing_axis
    assert busy.rhizotron.splat_axis > quiet.rhizotron.splat_axis
    assert busy.rhizotron.germination_rate > quiet.rhizotron.germination_rate
    assert busy.rhizotron.branch_prob > quiet.rhizotron.branch_prob
    # The shimmer starts at exactly zero: the bottom of the travel is the
    # plain instrument.
    assert quiet.rhizotron.mycorrhiza == 0.0
    assert busy.rhizotron.mycorrhiza > 0.3
    # The tempo top stays inside the §15.7(2) certificate.
    assert busy.rhizotron.elong_axis <= 24.0

    for mode in ("regulation", "activation"):
        resolved = config.Config(mode=mode, macros=config.Macros(
            tempo=1.0, scale=1.0, intensity=1.0)).resolve()
        assert resolved.rhizotron == busy.rhizotron, (
            f"the {mode} table's rhizotron entries diverged before step 6"
        )

    # An untouched slider stays within a hair of the shipped look.
    mid = config.Config().resolve().rhizotron
    defaults = config.RhizotronParams()
    for name in ("elong_axis", "spacing_axis", "splat_axis",
                 "germination_rate", "branch_prob", "descent_rate"):
        assert abs(getattr(mid, name) - getattr(defaults, name)) \
            <= 0.02 * max(abs(getattr(defaults, name)), 1e-6), name


def test_the_shimmer_spends_chroma_not_lightness(gpu_device, offscreen_target):
    """The mycorrhizal accent and the hairs must colour the fuzz, not
    brighten it: the luminance architecture is shared and stays put."""
    images = {}
    for label, myco in (("plain", 0.0), ("shimmer", 0.9)):
        params = _resolve(overrides={
            **STILL_WATER,
            "rhizotron.descent_rate": 0.0,
            "rhizotron.descent_wander": 0.0,
            "rhizotron.mycorrhiza": myco,
        })
        engine = _engine(gpu_device, params, seed=31)
        target, fmt = offscreen_target(engine.width, engine.height)
        for _ in range(400):
            engine.tick(params, [])
        for _ in range(60):
            engine.render(params, frac=0.5, target_view=target,
                          target_format=fmt)
        images[label] = R.linear_srgb_to_oklab(
            engine.read_final_rgba()[..., :3])
    delta = images["shimmer"] - images["plain"]
    chroma_moved = float(np.abs(delta[..., 1:]).mean())
    luma_moved = float(np.abs(delta[..., 0]).mean())
    assert chroma_moved > 1e-5, "the shimmer did nothing at all"
    assert luma_moved < chroma_moved * 0.75, (
        f"the shimmer moved lightness ({luma_moved:.2e}) more than it "
        f"should against its chroma ({chroma_moved:.2e})"
    )


def test_hardpan_structures_the_water(gpu_device):
    """A band that passes almost nothing makes the moisture field *vertically*
    organised -- water pools above it -- against the same seed without it."""
    spread = {}
    for label, amount in (("open", 0.0), ("panned", 1.0)):
        params = _resolve(overrides={
            "rhizotron.descent_rate": 0.0,
            "rhizotron.descent_wander": 0.0,
            # Heavy rain, slow relaxation: transport-dominated, so what the
            # bands do to the water is what the measurement sees.
            "rhizotron.rain_base": 0.04,
            "rhizotron.drainage_rate": 0.003,
            "rhizotron.hardpan_amount": amount,
            # Thin strata, so the hardpan lattice -- a scale above them --
            # actually crosses a test-sized window a few times.
            "rhizotron.strata_thickness": 0.12,
        })
        engine = _engine(gpu_device, params, seed=37)
        for _ in range(700):
            engine.tick(params, [])
        wet = _moisture(engine).astype(np.float32)[..., 0]
        spread[label] = float(wet.mean(axis=1).std())
    assert spread["panned"] > spread["open"] * 1.1, (
        f"hardpan did not organise the water "
        f"(row spread {spread['panned']:.4f} vs {spread['open']:.4f})"
    )


def _write_structure(engine, data: np.ndarray) -> None:
    for texture in engine.structure.textures:
        checkpoint._write_texture(engine.device, texture, data)


def test_roots_forage_into_richness(gpu_device):
    """§15.3's foraging: a rich half-field draws measurably more growth than
    a poor one. The richness is painted by hand and pinned (no uptake, no
    spread), so the asymmetry in what grows is the tropism's doing."""
    params = _resolve(overrides={
        **STILL_WATER,
        "rhizotron.descent_rate": 0.0,
        "rhizotron.descent_wander": 0.0,
        "rhizotron.nutrient_uptake": 0.0,
        "rhizotron.nutrient_spread": 0.0,
        "rhizotron.chemo_gain": 2.5,
        "rhizotron.forage_gain": 2.0,
        "rhizotron.hydro_gain": 0.0,
        # Clear ground: this isolates the chemotropism, and a seed hashed
        # onto a stone would spend the test forcing its way out instead.
        "rhizotron.stone_amount": 0.0,
        "rhizotron.hardpan_amount": 0.0,
    })
    engine = _engine(gpu_device, params, seed=41)
    g = engine.geometry
    field = np.zeros((g.height, g.width, 4), dtype=np.float16)
    field[:, : g.width // 2, 3] = 0.05
    field[:, g.width // 2:, 3] = 1.5
    _write_structure(engine, field)

    for _ in range(500):
        engine.tick(params, [])

    # The axes' child counters are the record of the foraging: branching is
    # richness-modulated, the axes plunge straight through their own halves,
    # and once they reach the bottom margin the counters freeze -- so what
    # they hold is exactly how eagerly each half's ground was foraged during
    # the transit.
    tips = _tips(engine)
    count = params.rhizotron.axes_per_plant
    axes_x = tips["x"][:count]
    children = (tips["flags"][:count] >> 8).astype(np.int64)
    poor = int(children[axes_x < g.width / 2].sum())
    rich = int(children[axes_x >= g.width / 2].sum())
    assert rich >= 2 + 2 * poor, (
        f"the rich half's axes threw {rich} laterals against the poor "
        f"half's {poor}; foraging should at least have doubled them"
    )


def test_dead_roots_enrich_the_ground(gpu_device):
    """The recycling memory: with the foraging feedback switched off the two
    runs grow *identically*, so the richness they leave behind differs by
    exactly what senescence returned -- ground where a plant died is richer
    ground, and nothing else moved."""
    fields = {}
    for label, recycle in (("kept", 0.0), ("returned", 1.5)):
        params = _resolve(overrides={
            **STILL_WATER,
            "rhizotron.descent_rate": 0.0,
            "rhizotron.descent_wander": 0.0,
            "rhizotron.senescence_rate": 0.06,
            "rhizotron.senescence_delay": 5.0,
            "rhizotron.nutrient_recycle": recycle,
            # Decoupled: richness must not steer this pair, or the growth
            # (and so the uptake) would differ between the runs.
            "rhizotron.chemo_gain": 0.0,
            "rhizotron.forage_gain": 0.0,
        })
        engine = _engine(gpu_device, params, seed=43)
        for _ in range(600):
            engine.tick(params, [])
        fields[label] = _structure(engine).astype(np.float32)
    assert np.array_equal(fields["kept"][..., 0], fields["returned"][..., 0]), (
        "decoupled runs should grow identically; the economy leaked into "
        "the growth"
    )
    gained = fields["returned"][..., 3] - fields["kept"][..., 3]
    assert float(gained.max()) > 0.01, (
        "senescence returned nothing anywhere"
    )
    assert float(gained.min()) >= -1e-3, (
        "recycling somehow *removed* richness"
    )


def test_shipped_endpoints_sit_inside_the_swept_certificate():
    """§15.7(2): crispness and speed are licensed by measurement.

    tests/crisp_sweep.py ran the growth burst at up to 4x the shipped
    elongation rates and the crispest transfer the floor admits, under the
    fastest config-reachable descent, and no swept point put a single pixel
    past even *half* the WCAG flash threshold. The `validate` ceilings sit at
    that swept range's edge; a retune past either re-runs the sweep first,
    which is what this test makes impossible to forget.
    """
    defaults = config.RhizotronParams()
    swept_elong_ceiling = 24.0  # 4x the shipped axis rate, as swept
    swept_edge_floor = 0.02     # the crispest transfer the sweep measured
    assert defaults.elong_axis * 4.0 <= swept_elong_ceiling + 1e-9

    fast = config.Config(overrides={
        "rhizotron.elong_axis": 999.0,
        "rhizotron.elong_lateral": 999.0,
        "rhizotron.root_edge": 0.0,
    }).resolve()
    assert fast.rhizotron.elong_axis == swept_elong_ceiling
    assert fast.rhizotron.elong_lateral == swept_elong_ceiling
    assert fast.rhizotron.root_edge == swept_edge_floor


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
    assert np.array_equal(_structure(original), _structure(resumed)), (
        "a resumed plant's structure diverged"
    )
    assert np.array_equal(_tips(original), _tips(resumed)), (
        "a resumed plant's tips diverged -- a branch decision depended on "
        "something the checkpoint does not carry"
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
