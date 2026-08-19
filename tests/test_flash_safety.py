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

# The same, for relative luminance -- and it is nearly all headroom. The Oklab
# bound pays for a cube root taken on f16 storage; this one is applied by
# interpolating in the very space the buffer holds, so it lands on its budget
# essentially exactly (measured: 0.009999 against 0.010, with the clamp
# binding). The tolerance is kept because f16 rounding can still go either way.
F16_LUMINANCE_TOLERANCE = 0.0005

# WCAG general flash threshold.
FLASH_LUMINANCE_FRACTION = 0.10
FLASH_AREA_FRACTION = 0.25

# WCAG red flash threshold: a pair of opposing transitions involving a
# saturated red, where the two states differ by more than 0.2 of maximum
# relative luminance. Unlike the general threshold it says nothing about how
# large the *chromatic* change is, so the general bound does not obviously
# cover it -- what covers it is the luminance clause, which is twice as large a
# change as the general threshold needs and therefore takes twice as long.
RED_LUMINANCE_DIFFERENCE = 0.20
# For a pair to exceed 3 per second each pair must fit in 333 ms and each
# transition in 167 ms, which at 30 FPS is five frames. A transition slower
# than that cannot be part of a flash however large it eventually becomes.
RED_WINDOW_FRAMES = 5


def _make_engine(gpu_device, params, seed=4242):
    device, _ = gpu_device
    return engine_module.Engine(device, WIDTH, HEIGHT, params, seed=seed)


def _capture_rgb(engine, params, target, fmt, frames, mutate=None):
    """Run `frames` frames, returning the linear sRGB of each output."""
    scheduler = events.EventScheduler(seed=11)
    captured = []
    for index in range(frames):
        if mutate is not None:
            mutate(index, params)
        rows, _ = scheduler.pack(8)
        engine.tick(params, rows)
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)
        scheduler.update(1.0 / params.sim_hz, params.events)
        captured.append(engine.read_final_rgba()[..., :3])
    return np.stack(captured)


def _capture(engine, params, target, fmt, frames, mutate=None):
    """Run `frames` frames, returning the Oklab lightness of each output."""
    return R.lightness(_capture_rgb(engine, params, target, fmt, frames, mutate))


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


def test_relative_luminance_bound_holds_exactly(gpu_device, offscreen_target):
    """The same bound, in the units WCAG is written in.

    The test above measures Oklab `L`, which is the quantity the image is
    graded in but not the quantity the photosensitivity thresholds are defined
    in. `L` is roughly the cube root of relative luminance, so a step bounded
    in one is bounded in the other only by a factor that grows with lightness
    -- and the chroma limiter, which `L` is blind to by construction, moves
    luminance as well. DESIGN.md 14.3 measures both. This asserts the bound the
    standard actually cares about, directly.
    """
    params = config.Config().resolve()
    params.render.layers = 1
    params.flow.psi_gain = 0.0
    params.flow.field_gain = 0.0

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = R.relative_luminance(
        _capture_rgb(engine, params, target, fmt, 40))

    deltas = np.abs(np.diff(frames, axis=0))
    limit = params.safety.max_luminance_delta + F16_LUMINANCE_TOLERANCE
    worst = float(deltas.max())
    assert worst <= limit, (
        f"per-pixel relative-luminance step {worst:.5f} exceeds the limit "
        f"{params.safety.max_luminance_delta:.5f} "
        f"(+{F16_LUMINANCE_TOLERANCE} tolerance)"
    )


def test_relative_luminance_bound_holds_at_the_oklab_ceilings(
    gpu_device, offscreen_target
):
    """And it holds with every brightness and limit value at its maximum.

    This is the configuration DESIGN.md 14.3 identified. `l_max`,
    `background_luma`, `filament_luma`, `exposure_target` and both Oklab limits
    are at their user-settable ceilings, which puts the image where `dY/dL` is
    largest, and the palette anchor is slammed half a turn every frame at full
    chroma, which drives the chroma limiter -- the one Oklab `L` is blind to.

    Measured with the final clamp removed, this reaches 0.0123 of relative
    luminance in a single frame against a 0.010 budget: a real breach, not a
    theoretical one, and 23% over rather than the 0.0005 that quantisation
    accounts for. It is the test that distinguishes the two bounds, and the
    reason the ceilings on the Oklab limits can stay where they are.
    """
    params = config.Config().resolve()
    params.render.layers = 1
    params.flow.psi_gain = 0.0
    params.flow.field_gain = 0.0
    params.render.l_max = config.SAFETY_CEILINGS["render.l_max"][1]
    params.render.c_max = config.SAFETY_CEILINGS["render.c_max"][1]
    params.render.background_luma = (
        config.SAFETY_CEILINGS["render.background_luma"][1])
    params.render.filament_luma = config.SAFETY_CEILINGS["render.filament_luma"][1]
    params.render.chroma_floor = params.render.c_max
    params.safety.exposure_target = (
        config.SAFETY_CEILINGS["safety.exposure_target"][1])
    params.safety.max_luma_delta = config.SAFETY_CEILINGS["safety.max_luma_delta"][1]
    params.safety.max_chroma_delta = (
        config.SAFETY_CEILINGS["safety.max_chroma_delta"][1])

    def mutate(index, p):
        p.render.hue_anchor = 0.0 if index % 2 == 0 else np.pi

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    frames = R.relative_luminance(
        _capture_rgb(engine, params, target, fmt, 40, mutate=mutate))

    deltas = np.abs(np.diff(frames, axis=0))
    limit = params.safety.max_luminance_delta + F16_LUMINANCE_TOLERANCE
    worst = float(deltas.max())
    assert worst <= limit, (
        f"with every ceiling at its maximum, the relative-luminance step "
        f"reached {worst:.5f} against a limit of {limit:.5f}"
    )


def test_red_flash_criterion(gpu_device, offscreen_target):
    """WCAG's second threshold: no fast transition to or from a saturated red.

    Forced into the red sector at full chroma, because a test of the red
    criterion run on a palette that never produces a red is a test of nothing
    -- so the run is asserted to have actually produced saturated-red pixels
    before the criterion is checked on them.

    Saturated red *is* reachable in this palette and always has been: at the
    default `c_max` such states exist up to Oklab L = 0.415. The criterion is
    not retired by staying out of that part of the gamut; it is retired by the
    rate at which the bound above lets anything move. Measured here, the run
    puts saturated red on up to the whole screen in every frame, and the worst
    five-frame luminance change on any of those pixels is 0.0089 against the
    0.20 the criterion requires -- a factor of 22.
    """
    params = config.Config().resolve()
    params.render.layers = 1
    # All hue from the anchor: no regional spread, no orientation term. The
    # anchor is applied both when pigment adopts a hue and again at composite,
    # so the hue on screen is twice this -- about 29 degrees, which is where
    # `R / (R + G + B)` peaks.
    params.render.hue_anchor = 0.25
    params.render.hue_spread = 0.0
    params.pigment.hue_from_orientation = 0.0
    # Fully saturated everywhere, rather than only where the field is active.
    params.render.c_max = config.SAFETY_CEILINGS["render.c_max"][1]
    params.render.chroma_floor = params.render.c_max

    engine = _make_engine(gpu_device, params)
    target, fmt = offscreen_target(WIDTH, HEIGHT)
    rgb = _capture_rgb(engine, params, target, fmt, 60)

    red = R.saturated_red(rgb)
    assert red.any(), (
        "no saturated-red pixel appeared, so this run did not exercise the "
        "criterion it is asserting"
    )

    luminance = R.relative_luminance(rgb)
    window = RED_WINDOW_FRAMES
    difference = np.abs(luminance[window:] - luminance[:-window])
    involves_red = red[window:] | red[:-window]
    transitions = (difference >= RED_LUMINANCE_DIFFERENCE) & involves_red
    worst_area = float(transitions.mean(axis=(1, 2)).max())
    assert worst_area < FLASH_AREA_FRACTION, (
        f"{worst_area:.1%} of the screen made a >={RED_LUMINANCE_DIFFERENCE} "
        f"luminance transition involving a saturated red within {window} "
        f"frames; the WCAG area threshold is {FLASH_AREA_FRACTION:.0%}"
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
