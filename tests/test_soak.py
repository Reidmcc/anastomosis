"""Long-duration properties: non-repetition, numerical survival, no leaks.

These are the tests that speak to the actual use case. A visual that is
beautiful for ten minutes and dead, saturated, or subtly looping after six
hours would fail the brief completely, and none of that is visible in a short
interactive check.
"""

from __future__ import annotations

import re

import numpy as np
import pytest
import wgpu

from anastomosis import config, engine as engine_module, events, shaders

MASK32 = 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Non-repetition, tested at the mechanism rather than by observation
# ---------------------------------------------------------------------------


def test_no_shader_treats_time_as_a_float_phase():
    """The central rule from DESIGN.md §3, enforced statically.

    Slow variation must come from integrating increments into a stored field,
    never from evaluating a function of a clock. A float-valued tick would both
    reintroduce periodicity and quantise visibly: at 30 Hz, one day of runtime
    puts an f32 at ~0.008 resolution.

    A grep is crude, but this is a property that is easy to reintroduce by
    accident and effectively impossible to notice by looking at the output for
    an hour.
    """
    offenders = []
    pattern = re.compile(
        r"f32\s*\(\s*(params\.tick|render\.frame|params\.seed|render\.seed)\s*\)"
    )
    for name in shaders.all_shader_names():
        source = shaders.load(name)
        for lineno, line in enumerate(source.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "tick/seed converted to float, which makes the result a function of "
        "time rather than a stateful process:\n  " + "\n  ".join(offenders)
    )


def test_no_shader_uses_a_trig_function_of_the_tick():
    """The same rule, catching the `sin(t)` formulation directly."""
    offenders = []
    pattern = re.compile(r"(sin|cos)\s*\([^)]*\b(tick|frame)\b")
    for name in shaders.all_shader_names():
        for lineno, line in enumerate(shaders.load(name).splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{name}:{lineno}: {line.strip()}")
    assert not offenders, (
        "periodic function of the tick counter found:\n  " + "\n  ".join(offenders)
    )


def _pcg(values: np.ndarray) -> np.ndarray:
    """The PRNG from common.wgsl, in numpy."""
    v = values.astype(np.uint64) & MASK32
    state = (v * 747796405 + 2891336453) & MASK32
    shift = ((state >> np.uint64(28)) + 4) & MASK32
    word = (((state >> shift) ^ state) * 277803737) & MASK32
    return ((word >> np.uint64(22)) ^ word) & MASK32


def test_prng_has_no_short_period():
    """A repeating PRNG would make the whole aperiodicity argument moot."""
    counters = np.arange(1_000_000, dtype=np.uint64)
    values = _pcg(counters)
    unique = len(np.unique(values))
    # A 32-bit output over 1e6 draws has ~1.2e5 expected collisions by the
    # birthday bound; far fewer uniques than that would indicate structure.
    assert unique > 950_000, f"only {unique} distinct outputs from 1e6 counters"

    # And consecutive counters must not be correlated, which is the property
    # that matters when hashing (pixel, tick).
    correlation = np.corrcoef(values[:-1].astype(float), values[1:].astype(float))[0, 1]
    assert abs(correlation) < 0.01, f"successive outputs correlate at {correlation:.4f}"


def test_ou_process_is_stationary_and_aperiodic():
    """The climate/psi drift model: mean-reverting, bounded, non-repeating."""
    rng = np.random.default_rng(0)
    theta, sigma = 0.0016, 0.055
    value = 0.0
    series = []
    for _ in range(200_000):
        value = value * (1.0 - theta) + rng.normal() * sigma
        value = float(np.clip(value, -1.0, 1.0))
        series.append(value)
    series = np.asarray(series)

    assert abs(series.mean()) < 0.1, "OU process drifted off its mean"
    assert series.std() > 0.05, "OU process collapsed to its mean"
    assert np.isfinite(series).all()

    # No periodic component: autocorrelation must decay and stay decayed.
    centred = series - series.mean()
    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    spectrum[0] = 0.0
    peak = spectrum.max()
    assert peak < spectrum.sum() * 0.02, (
        "the drift has a dominant frequency component, i.e. it oscillates"
    )


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def test_event_envelope_never_steps():
    params = config.EventParams()
    scheduler = events.EventScheduler(seed=1)
    event = scheduler._spawn(params)

    dt = 1.0 / 20.0
    previous = event.envelope()
    largest = 0.0
    while not event.finished:
        event.elapsed += dt
        current = event.envelope()
        largest = max(largest, abs(current - previous))
        previous = current
    # A raised cosine over tens of seconds cannot move fast per tick.
    assert largest < 0.02, f"event envelope stepped by {largest:.4f} in one tick"


def test_event_envelope_starts_and_ends_at_zero():
    scheduler = events.EventScheduler(seed=2)
    event = scheduler._spawn(config.EventParams())
    assert event.envelope() == pytest.approx(0.0, abs=1e-9)
    event.elapsed = event.duration
    assert event.envelope() == pytest.approx(0.0, abs=1e-6)


def test_events_stay_under_the_flash_area_threshold():
    """Event radius is capped at the WCAG area threshold, with margin."""
    params = config.EventParams()
    scheduler = events.EventScheduler(seed=3)
    for _ in range(200):
        event = scheduler._spawn(params)
        area = np.pi * event.radius**2
        assert area < 0.25, f"event covers {area:.1%} of the screen"


def test_event_arrivals_are_memoryless():
    """Poisson arrivals, so the next event is never predictable from the last."""
    params = config.EventParams(rate_per_hour=60.0, max_concurrent=8)
    scheduler = events.EventScheduler(seed=4)
    dt = 1.0
    spawns = []
    for step in range(20_000):
        before = scheduler.spawned
        scheduler.update(dt, params)
        if scheduler.spawned > before:
            spawns.append(step)
    assert len(spawns) > 50, "too few events to judge"
    gaps = np.diff(spawns)
    # Exponential inter-arrivals have std ~ mean; a fixed cadence would have
    # std near zero.
    assert gaps.std() > gaps.mean() * 0.5, (
        f"arrival gaps look regular: mean {gaps.mean():.1f}, std {gaps.std():.1f}"
    )


def test_events_can_be_disabled():
    params = config.EventParams(enabled=False)
    scheduler = events.EventScheduler(seed=5)
    for _ in range(10_000):
        scheduler.update(1.0, params)
    assert scheduler.spawned == 0


# ---------------------------------------------------------------------------
# GPU soak
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_long_run_stays_finite_and_in_band(gpu_device, offscreen_target):
    """The core long-duration property: it neither dies nor saturates."""
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 1

    width, height = 128, 96
    engine = engine_module.Engine(device, width, height, params, seed=7)
    target, fmt = offscreen_target(width, height)
    scheduler = events.EventScheduler(seed=8)

    history = []
    for _ in range(1200):
        rows, _ = scheduler.pack(8)
        engine.tick(params, rows)
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)
        scheduler.update(1.0 / params.sim_hz, params.events)
        history.append(engine.read_stats()["mean_v"])

    image = engine.read_final_rgba()
    assert np.isfinite(image).all(), "output contains non-finite values"

    stats = engine.read_stats()
    for key in ("mean_v", "var_v", "mean_activity", "exposure"):
        assert np.isfinite(stats[key]), f"{key} is not finite"

    assert stats["mean_v"] > 0.02, f"field died: mean_v {stats['mean_v']:.5f}"
    assert stats["mean_v"] < 0.6, f"field saturated: mean_v {stats['mean_v']:.5f}"
    assert stats["mean_activity"] > 1e-4, (
        f"field settled: activity {stats['mean_activity']:.6f}"
    )

    # The homeostat should have moved the field toward its target, not away.
    target_mass = params.homeostat.target_mass
    start_error = abs(history[50] - target_mass)
    end_error = abs(history[-1] - target_mass)
    assert end_error < start_error, (
        f"homeostat did not converge: error went {start_error:.4f} -> {end_error:.4f}"
    )


@pytest.mark.slow
def test_steady_state_allocates_nothing(gpu_device, offscreen_target):
    """A run of 10^7 frames will find any per-frame allocation.

    Bind groups are created on demand and cached by resource identity, so the
    cache must stop growing once every ping-pong parity has been seen.
    """
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 2

    width, height = 96, 64
    engine = engine_module.Engine(device, width, height, params, seed=9)
    target, fmt = offscreen_target(width, height)

    def step():
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    # The sanitise pass runs every 60 ticks and flips field parity, so the full
    # set of ping-pong combinations only appears after a couple of cycles.
    for _ in range(150):
        step()
    settled = len(engine._bind_cache)

    for _ in range(150):
        step()
    assert len(engine._bind_cache) == settled, (
        f"bind group cache grew from {settled} to {len(engine._bind_cache)}; "
        "something is allocating per frame"
    )


@pytest.mark.slow
def test_recovers_from_a_corrupted_field(gpu_device, offscreen_target):
    """A NaN must not be able to destroy the run permanently.

    This is the failure mode with no natural recovery: one non-finite texel
    propagates through diffusion and takes the whole field with it, and a
    session two days in has no way back.
    """
    device, _ = gpu_device
    params = config.Config().resolve()
    params.render.layers = 1

    width, height = 96, 64
    engine = engine_module.Engine(device, width, height, params, seed=10)
    target, fmt = offscreen_target(width, height)

    for _ in range(20):
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    # Inject NaN and infinity across a band of the reaction field.
    layer = engine.layers[0]
    poison = np.zeros((height, width, 4), dtype=np.float32)
    poison[..., 0] = 1.0
    poison[: height // 4, :, 1] = np.nan
    poison[height // 4 : height // 2, :, 1] = np.inf
    device.queue.write_texture(
        {"texture": layer.reaction.textures[layer.reaction.index],
         "mip_level": 0, "origin": (0, 0, 0)},
        np.ascontiguousarray(poison.astype(np.float16)),
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, 1),
    )

    for _ in range(90):  # past at least one sanitise pass
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    image = engine.read_final_rgba()
    assert np.isfinite(image).all(), "non-finite values reached the output"
    stats = engine.read_stats()
    assert np.isfinite(stats["mean_v"]), "field statistics are still poisoned"
    assert stats["mean_v"] > 0.0, "field did not recover"
