"""The Gray-Scott regime, and the control decisions that depend on it.

These lock in findings that were *measured*, not assumed, and that the code
comments in reaction.wgsl and homeostat.wgsl refer to. If someone later
"tidies" the defaults back toward the textbook F=0.038/K=0.062, or removes the
kill-follows-feed coupling because it looks redundant, these fail.

Marked slow: each runs a few thousand Gray-Scott steps in numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

import reference as R
from anastomosis import config

pytestmark = pytest.mark.slow

SIZE = 96
TICKS = 1100
WARMUP = 400


def _run(feed, kill, **kwargs):
    return R.run(feed, kill, ticks=TICKS, warmup=WARMUP, size=SIZE, **kwargs)


def test_default_regime_stays_alive():
    """The shipped default must not settle or die."""
    params = config.Config().resolve()
    stats = _run(params.reaction.feed, params.reaction.kill)
    assert stats["mean_v"] > 0.05, "default regime lost its field"
    assert stats["activity_late"] > 5e-4, (
        f"default regime settled: late activity {stats['activity_late']:.6f}"
    )
    assert stats["var_v"] > 2e-3, "default regime has no spatial structure"


def test_default_regime_beats_the_textbook_one():
    """Documents why the default is not the familiar F=0.038/K=0.062."""
    params = config.Config().resolve()
    chosen = _run(params.reaction.feed, params.reaction.kill)
    textbook = _run(0.038, 0.062)
    assert chosen["activity_late"] > textbook["activity_late"] * 2.0, (
        f"chosen regime {chosen['activity_late']:.6f} is not clearly more "
        f"persistent than the textbook one {textbook['activity_late']:.6f}"
    )


def test_kill_is_a_monotonic_lever_for_both_objectives():
    """Why the homeostat steers with kill rather than feed.

    Mean V and activity both rise as kill falls, so one control serves both.
    """
    feed = config.Config().resolve().reaction.feed
    kills = [0.045, 0.047, 0.049, 0.051, 0.053, 0.055]
    results = [_run(feed, k) for k in kills]

    masses = [r["mean_v"] for r in results]
    assert all(a > b for a, b in zip(masses, masses[1:])), (
        f"mean V is not monotonically decreasing in kill: {masses}"
    )
    activities = [r["activity_late"] for r in results]
    assert activities[0] > activities[-1] * 1.5, (
        f"activity does not respond usefully to kill: {activities}"
    )


def test_feed_is_not_a_safe_activity_lever():
    """The counterpart: feed's effect on activity is not monotonic.

    This is the reason the homeostat does not use feed to chase activity. If
    this ever starts passing as monotonic, the controller could be simplified.
    """
    kill = config.Config().resolve().reaction.kill
    feeds = [0.014, 0.018, 0.022, 0.026, 0.030]
    activities = [_run(f, kill)["activity_late"] for f in feeds]
    monotonic = all(a <= b for a, b in zip(activities, activities[1:])) or all(
        a >= b for a, b in zip(activities, activities[1:])
    )
    assert not monotonic, (
        f"feed now looks monotonic for activity ({activities}); the homeostat "
        "routing in homeostat.wgsl could be revisited"
    )


def test_kill_must_follow_feed_or_regions_die():
    """The single most important coupling in the simulation.

    A downward climate excursion in feed, with kill held fixed, drives the
    field to zero and it cannot restart. With kill tracking along the live
    band, the same excursion survives.
    """
    params = config.Config().resolve()
    feed = params.reaction.feed
    kill = params.reaction.kill
    slope = params.reaction.kill_follows_feed
    excursion = -params.climate.range_feed

    fixed = _run(feed + excursion, kill)
    following = _run(feed + excursion, kill + slope * excursion)

    assert fixed["mean_v"] < 0.01, (
        "the uncorrelated excursion no longer kills the field; the sweep this "
        "coupling was derived from may no longer apply"
    )
    assert following["mean_v"] > 0.02, (
        f"kill-follows-feed did not keep the region alive "
        f"(mean_v {following['mean_v']:.4f})"
    )


def test_climate_range_stays_on_the_live_band():
    """Every reachable corner of the parameter space must stay alive.

    "Reachable" means after the shader's clamps, which is the point: the
    climate range alone reaches (F=0.010, K=0.0431), which is dead. The
    liveness floor in `reaction.wgsl` is what prevents that, so the floor is
    applied here too rather than tested around.

    A dead region would not simply be dull -- the climate field advects, so it
    would be carried around the screen and would not recover on its own.
    """
    params = config.Config().resolve()
    reaction = params.reaction
    slope = reaction.kill_follows_feed

    # Worst case includes the homeostat pushing in the same direction.
    homeostat_swing = 0.006

    for feed_dev in (-params.climate.range_feed, 0.0, params.climate.range_feed):
        for kill_dev in (-params.climate.range_kill, 0.0, params.climate.range_kill):
            for correction in (-homeostat_swing, 0.0, homeostat_swing):
                feed, kill = config.clamp_reaction(
                    reaction.feed + feed_dev + correction,
                    reaction.kill + slope * feed_dev + kill_dev + correction,
                    reaction,
                )
                stats = _run(feed, kill)
                assert stats["mean_v"] > 0.01, (
                    f"corner F={feed:.4f} K={kill:.4f} is dead "
                    f"(mean_v {stats['mean_v']:.5f})"
                )


def test_liveness_floor_is_actually_needed():
    """Guards against someone widening the clamps back out.

    Without the floor, the climate range reaches a genuinely dead point.
    """
    params = config.Config().resolve()
    reaction = params.reaction
    unclamped_feed = reaction.feed - params.climate.range_feed
    unclamped_kill = (
        reaction.kill
        - reaction.kill_follows_feed * params.climate.range_feed
        - params.climate.range_kill
    )
    _, clamped_kill = config.clamp_reaction(
        unclamped_feed, unclamped_kill, reaction)
    assert clamped_kill > unclamped_kill, (
        "the climate range no longer reaches below the liveness floor; either "
        "the floor or the range has changed"
    )
    stats = _run(unclamped_feed, unclamped_kill)
    assert stats["mean_v"] < 0.01, (
        f"the unclamped corner F={unclamped_feed:.4f} K={unclamped_kill:.4f} is "
        f"no longer dead (mean_v {stats['mean_v']:.5f}); the floor may be "
        "unnecessary, or the regime has moved"
    )


def test_homeostat_targets_match_the_measured_regime():
    """Targets must be reachable, not aspirational.

    A target the system cannot reach means the integral term saturates and the
    controller sits permanently at its limit, which is both useless and a
    source of coordinated drift.
    """
    params = config.Config().resolve()
    stats = _run(params.reaction.feed, params.reaction.kill)
    homeostat = params.homeostat

    for name, measured, target in (
        ("mass", stats["mean_v"], homeostat.target_mass),
        ("variance", stats["var_v"], homeostat.target_variance),
        ("activity", stats["activity_late"], homeostat.target_activity),
    ):
        ratio = measured / target
        assert 0.3 < ratio < 3.0, (
            f"{name} target {target:g} is far from what the default regime "
            f"actually produces ({measured:g}); ratio {ratio:.2f}"
        )


def test_laplacian_is_conservative_and_isotropic():
    """The stencil must not create or destroy material, or prefer an axis."""
    rng = np.random.default_rng(0)
    field = rng.random((64, 64))
    lap = R.laplacian9(field)
    assert abs(lap.sum()) < 1e-9, "laplacian is not conservative on a torus"

    # A radially symmetric bump must give a radially symmetric response.
    # Odd size so the bump is exactly centred and the mirror test is valid.
    y, x = np.ogrid[-32:33, -32:33]
    bump = np.exp(-(x * x + y * y) / 50.0)
    response = R.laplacian9(bump)
    assert np.allclose(response, response.T, atol=1e-12)
    assert np.allclose(response, response[::-1, :], atol=1e-12)
