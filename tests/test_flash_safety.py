"""The flash-safety invariant -- the most important test in the suite.

What the shader actually guarantees, stated precisely:

    Per pixel, |dL| per frame is bounded by ``max_luma_delta``, measured
    against the *motion-compensated* previous frame.

The reprojection matters for the phrasing. At a *fixed* screen pixel, a
filament translating past can legitimately produce a larger change than the
limit -- that is honest motion, not a flash, and smearing it out would be the
wrong behaviour. So the guarantee is tested two ways:

* :func:`test_limiter_bound_holds_exactly` removes flow entirely, which makes
  reprojection the identity, and asserts the per-pixel bound directly. This
  isolates the limiter.

* :func:`test_wcag_flash_criterion` runs the system normally and asserts the
  thing photosensitivity guidance actually cares about: that a large,
  *correlated* area never changes fast. WCAG 2.3.1 / PEAT define a flash as a
  pair of opposing relative-luminance changes of >=10% covering >25% of the
  screen. If fewer than 25% of pixels ever change by >=10% in a single frame,
  the general flash threshold cannot be met regardless of timing.

Both are asserted under adversarial parameter changes, because a guarantee that
only holds for sensible settings is not a safety property.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import reference as R
from anastomosis import config, engine as engine_module, events

WIDTH, HEIGHT = 128, 96

# f16 storage on the output texture, plus the Oklab cube root, cost a little
# precision. This is 5% of the limit; it was 20x larger before the gamut-mapping
# fix in common.wgsl, which is what made the residual pure quantisation rather
# than a real leak. Keep it tight -- a loose tolerance here would hide exactly
# the class of bug it caught.
F16_TOLERANCE = 0.0005

# WCAG general flash threshold.
FLASH_LUMINANCE_FRACTION = 0.10
FLASH_AREA_FRACTION = 0.25


def _make_engine(gpu_device, params, seed=4242):
    device, _ = gpu_device
    return engine_module.Engine(device, WIDTH, HEIGHT, params, seed=seed)


def _capture(engine, params, target, fmt, frames, mutate=None):
    """Run `frames` frames, returning the Oklab lightness of each output."""
    scheduler = events.EventScheduler(seed=11)
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


def test_limiter_bound_holds_exactly(gpu_device, offscreen_target):
    """With no flow, reprojection is the identity and the bound is exact."""
    params = config.Config().resolve()
    params.render.layers = 1
    # Zero velocity => history reprojection is a no-op => any per-pixel change
    # is change the limiter was responsible for bounding.
    params.flow.psi_gain = 0.0
    params.flow.field_gain = 0.0

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 40)

    deltas = np.abs(np.diff(frames, axis=0))
    limit = params.safety.max_luma_delta + F16_TOLERANCE
    worst = float(deltas.max())
    assert worst <= limit, (
        f"per-pixel lightness step {worst:.5f} exceeds the limit "
        f"{params.safety.max_luma_delta:.5f} (+{F16_TOLERANCE} tolerance)"
    )


def test_limiter_holds_under_adversarial_parameters(gpu_device, offscreen_target):
    """Slamming parameters between extremes must not defeat the bound.

    Parameters are *stepped*, not ramped, deliberately: the application ramps
    them, but the shader-level guarantee must not depend on that.
    """
    params = config.Config().resolve()
    params.render.layers = 1
    params.flow.psi_gain = 0.0
    params.flow.field_gain = 0.0

    def mutate(index, p):
        extreme = index % 2 == 0
        # Every one of these is a lever that could plausibly produce a jump.
        p.render.filament_luma = 0.9 if extreme else 0.0
        p.render.background_luma = 0.3 if extreme else 0.0
        p.render.l_max = 0.9 if extreme else 0.05
        p.render.c_max = 0.22 if extreme else 0.0
        p.render.extinction = 6.0 if extreme else 0.2
        p.safety.exposure_target = 0.40 if extreme else 0.02

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 40, mutate=mutate)

    deltas = np.abs(np.diff(frames, axis=0))
    limit = params.safety.max_luma_delta + F16_TOLERANCE
    worst = float(deltas.max())
    assert worst <= limit, (
        f"adversarial parameter stepping produced a lightness step of "
        f"{worst:.5f}, above the limit {limit:.5f}"
    )


def test_a_mode_switch_cannot_defeat_the_bound(gpu_device, offscreen_target):
    """Slamming between the two modes' tables must not defeat the bound.

    The application ramps a mode switch like every other change (DESIGN.md
    §14.5), but the shader-level guarantee must not depend on that. Stepped
    here, un-ramped, every frame: each frame takes *everything* either mode's
    curve table drives, resolved at opposite macro extremes under opposite
    modes -- strictly more movement than any real switch, which keeps the
    macros where they are.

    The shipped activation table is still a copy of the regulation one
    (§14.8 step 1), so today the modes contribute the same values and the
    macro extremes do the work; when step 3 retunes the activation endpoints,
    their divergence rides in here with no change to the test.
    """
    from dataclasses import fields

    params = config.Config().resolve()
    params.render.layers = 1
    params.flow.psi_gain = 0.0
    params.flow.field_gain = 0.0

    ends = [
        config.Config(
            macros=config.Macros(**{f.name: value for f in fields(config.Macros)}),
            mode=mode,
        ).resolve()
        for mode, value in (("regulation", 0.0), ("activation", 1.0))
    ]
    paths = {
        path
        for table in config.MODE_CURVES.values()
        for entries in table.values()
        for path, *_ in entries
    }
    paths.add("render.hue_anchor")  # the palette macro's non-lerp effect

    def mutate(index, p):
        source = ends[index % 2]
        for path in paths:
            config.set_path(p, path, config.get_path(source, path))
        # Zero flow keeps reprojection the identity, so the per-pixel bound
        # stays exact -- the same isolation as the adversarial test above.
        p.flow.psi_gain = 0.0
        p.flow.field_gain = 0.0

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 40, mutate=mutate)

    deltas = np.abs(np.diff(frames, axis=0))
    limit = params.safety.max_luma_delta + F16_TOLERANCE
    worst = float(deltas.max())
    assert worst <= limit, (
        f"stepping between the modes' resolutions produced a lightness step "
        f"of {worst:.5f}, above the limit {limit:.5f}"
    )


def test_wcag_flash_criterion(gpu_device, offscreen_target):
    """Under normal operation, no frame may flash a large area at once."""
    params = config.Config().resolve()
    params.render.layers = 2

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 45)

    deltas = np.abs(np.diff(frames, axis=0))
    flashing = (deltas >= FLASH_LUMINANCE_FRACTION).mean(axis=(1, 2))
    worst = float(flashing.max())
    assert worst < FLASH_AREA_FRACTION, (
        f"{worst:.1%} of the screen changed by >={FLASH_LUMINANCE_FRACTION:.0%} "
        f"lightness in a single frame; the WCAG area threshold is "
        f"{FLASH_AREA_FRACTION:.0%}"
    )


def test_wcag_flash_criterion_at_the_activation_top(gpu_device, offscreen_target):
    """The shipped activation endpoints, at the busy corner, under the area
    criterion.

    Step 2's sweep (tests/tempo_sweep.py) certified this axis to 6x the
    regulation tops on a warmed-up field; this pins the *shipped* endpoints
    the way the regulation test above pins the defaults, over the fresh-start
    frames where the most change happens. The macros sit at the corner that
    maximises the metric -- most material, finest features, fastest motion,
    brightest -- not at the defaults.
    """
    params = config.Config(macros=config.Macros(
        intensity=1.0, scale=0.0, tempo=1.0,
        filament_glow=1.0, brightness=1.0,
    ), mode="activation").resolve()
    params.render.layers = 2

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 45)

    deltas = np.abs(np.diff(frames, axis=0))
    flashing = (deltas >= FLASH_LUMINANCE_FRACTION).mean(axis=(1, 2))
    worst = float(flashing.max())
    assert worst < FLASH_AREA_FRACTION, (
        f"{worst:.1%} of the screen changed by >={FLASH_LUMINANCE_FRACTION:.0%} "
        f"lightness in a single frame at the activation top; the WCAG area "
        f"threshold is {FLASH_AREA_FRACTION:.0%}"
    )


def test_global_luminance_is_stable(gpu_device, offscreen_target):
    """Mean screen lightness must not lurch, even as exposure adapts.

    A correlated whole-screen change is the most provocative thing the system
    could do, so it is held far tighter than the per-pixel limit.
    """
    params = config.Config().resolve()
    params.render.layers = 2

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 45)

    means = frames.mean(axis=(1, 2))
    steps = np.abs(np.diff(means))
    worst = float(steps.max())
    assert worst <= params.safety.max_luma_delta, (
        f"mean screen lightness moved {worst:.5f} in one frame"
    )


def test_startup_fades_in_rather_than_appearing(gpu_device, offscreen_target):
    """The first frames must rise gradually from black.

    Not a separate mechanism -- it falls out of the history buffer starting at
    zero and the limiter bounding the approach. Worth asserting because
    "appears abruptly at launch" would be a flash like any other.
    """
    params = config.Config().resolve()
    params.render.layers = 1

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 12)

    first = float(frames[0].max())
    assert first <= params.safety.max_luma_delta + F16_TOLERANCE, (
        f"first frame reached lightness {first:.4f}; it should still be nearly black"
    )
    # And it should be monotonically brightening, not oscillating.
    means = frames.mean(axis=(1, 2))
    assert np.all(np.diff(means) >= -F16_TOLERANCE), "startup fade is not monotonic"


def test_the_bound_holds_from_black_under_chromatic_demand(gpu_device, offscreen_target):
    """A maximal step off a black history must not leak through gamut mapping.

    The regression the mode-switch test surfaced (found while building
    DESIGN.md §14.8 step 1): from a black history the limiter permits a step of
    (+max_luma_delta, +-max_chroma_delta, +-max_chroma_delta), which at that
    lightness is far out of gamut. The bisection in `gamut_map_oklab` accepted
    trial points with channels down to -1e-6, and the final clamp then raised
    them to zero -- which near black *raises* L by ~1e-3 per channel, because
    the cube root's slope is unbounded at zero. Measured before the `in_gamut`
    fix: stored L up to 0.0125 against a 0.010 budget.

    Startup is the honest way to manufacture a black history; the demand is
    manufactured rather than hoped for, because whether the *simulation*
    happens to put strong chroma on a dark pixel is a matter of which
    trajectory a chaotic field wandered into. Every knob here is reachable by
    a user: the chroma floor raised to the chroma ceiling makes every pixel,
    black ones included, ask for full chroma from the first frame; the chroma
    slew limit sits at its user-settable ceiling because the leak grows with
    the chroma step; and the hue anchor flipping by pi each frame keeps the
    demand a *change* in chroma rather than a satisfied one. Under the
    pre-fix tolerance this fails on ~1000 pixel-frames at 0.0107, not on one
    lucky pixel.
    """
    params = config.Config().resolve()
    params.render.layers = 1
    # Zero flow makes reprojection the identity, so the per-pixel bound is
    # exact -- the same isolation as test_limiter_bound_holds_exactly.
    params.flow.psi_gain = 0.0
    params.flow.field_gain = 0.0
    params.render.chroma_floor = 0.22
    params.render.c_max = 0.22
    params.safety.max_chroma_delta = config.SAFETY_CEILINGS[
        "safety.max_chroma_delta"][1]

    def mutate(index, p):
        p.render.hue_anchor = 0.0 if index % 2 == 0 else math.pi

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = _capture(engine, params, target, fmt, 12, mutate=mutate)

    deltas = np.abs(np.diff(frames, axis=0))
    first = float(frames[0].max())
    worst = float(max(first, deltas.max()))
    limit = params.safety.max_luma_delta + F16_TOLERANCE
    assert worst <= limit, (
        f"a chromatic climb out of black produced a lightness step of "
        f"{worst:.5f}, above the limit {limit:.5f}"
    )


@pytest.mark.parametrize("bad_value", [0.5, 10.0, -1.0])
def test_config_cannot_defeat_the_bound(bad_value):
    """No config value may raise the limit past its hard ceiling."""
    resolved = config.Config(overrides={"safety.max_luma_delta": bad_value}).resolve()
    ceiling = config.SAFETY_CEILINGS["safety.max_luma_delta"][1]
    assert resolved.safety.max_luma_delta <= ceiling


def _worst_case_flashes_per_second(luma_delta: float, fps: float = 30.0) -> float:
    """Flash rate under a sustained maximum-rate oscillation.

    The limiter permits at most `luma_delta` of change per frame, so a 10%
    excursion needs 0.10/luma_delta frames and an opposing pair twice that.
    This is the worst case the limiter allows, not a typical one.
    """
    frames_per_excursion = FLASH_LUMINANCE_FRACTION / luma_delta
    return fps / (2.0 * frames_per_excursion)


def test_ceiling_implies_wcag_margin():
    """Even at the maximum *user-settable* value, the standard is unreachable.

    This test found a genuine arithmetic error in the original design: the
    ceiling was set to 0.03, which permits 4.5 flashes/second -- above the
    WCAG limit of 3, not below it as the design claimed.
    """
    ceiling = config.SAFETY_CEILINGS["safety.max_luma_delta"][1]
    rate = _worst_case_flashes_per_second(ceiling)
    assert rate < 3.0, (
        f"at the ceiling {ceiling} the system could reach {rate:.2f} flashes/second"
    )


def test_default_has_comfortable_margin():
    """The shipped default should sit well clear of the threshold, not at it."""
    default = config.SafetyParams().max_luma_delta
    rate = _worst_case_flashes_per_second(default)
    assert rate <= 1.6, f"default {default} gives {rate:.2f} flashes/second"


def test_the_two_ceilings_are_jointly_safe_at_any_frame_rate():
    """The luma-delta and frame-rate ceilings must be safe as a *pair*.

    The flash arithmetic is per-frame times frame rate, and for as long as the
    two ceilings were independent the pair (0.012, 60 FPS) was reachable --
    3.6 flashes/second, above the WCAG limit of 3, found while writing
    DESIGN.md §14.5(2). `validate` now holds the product to
    `MAX_LUMA_PER_SECOND`, so the worst case is 1.8/s at every frame rate the
    table admits, asked for here exactly the way a config file would ask:
    both values at once, as large as they will go.
    """
    fps_lo, fps_hi = config.SAFETY_CEILINGS["max_fps"]
    for fps in sorted({fps_lo, 24, 30, 48, fps_hi}):
        resolved = config.Config(overrides={
            "max_fps": fps,
            "safety.max_luma_delta": 1.0,
        }).resolve()
        rate = _worst_case_flashes_per_second(
            resolved.safety.max_luma_delta, fps=float(resolved.max_fps))
        assert rate <= 1.8 + 1e-9, (
            f"at max_fps = {fps} the ceilings jointly permit "
            f"{rate:.2f} flashes/second"
        )
        # And at the design's 30 FPS the coupling binds at exactly the
        # documented per-frame ceiling -- it must not tighten what §7 promises.
        if fps == 30:
            assert resolved.safety.max_luma_delta == pytest.approx(
                config.SAFETY_CEILINGS["safety.max_luma_delta"][1])
