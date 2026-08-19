"""The morphology of the field, and the lever that keeps it from freezing.

DESIGN.md §4.7. The soak tests assert the field is *alive*; every measure they
use -- mass, variance, activity -- is invariant under rearrangement, so a field
can pass all of them and still be the same picture it was an hour ago. Worse,
the picture it settles on is a dense lattice of similar-sized round features,
which is a common trypophobia trigger. For a regulation aid that is a
functional defect, not a matter of taste, so it gets asserted rather than
eyeballed.

Two properties are checked, and they are different claims:

* **Sizes are not uniform across the field.** This is the accessibility one.
  Uniformity is the trigger, so a churning field of identically-sized holes
  would not fix anything.
* **The arrangement does not hold still.** This is the monotony one: features
  must merge and split over a long run, not merely jitter.

Plus the constraint that makes the lever usable at all: it must not move the
quantities the homeostat defends, because anything that moves mass reaches the
image as a slow global luminance swing through the exposure governor.

These run the numpy reference rather than the GPU. The shader is held to the
same arithmetic by ``test_reaction_matches_numpy_with_a_varying_du``.
"""

from __future__ import annotations

import numpy as np
import pytest

import morphology as M
from anastomosis import config

pytestmark = pytest.mark.slow

SIZE = 192
# One climate texel per this many simulation cells. About 24 keeps a region
# large enough to establish its own wavelength; the engine's ratio is ~40.
FACTOR = 24
TILE = 32
TICKS = 4000


def _equilibrate(reaction, du, dv, ticks=TICKS, size=SIZE, window=400):
    """Run to a mature field; return it with its late-window activity."""
    u, v = M.seed_field(size, seed=1)
    activity: list[float] = []
    for tick in range(ticks):
        before = v
        for _ in range(2):
            u, v = M.gray_scott_step(
                u, v, reaction.feed, reaction.kill, du=du, dv=dv)
        if tick > ticks - window:
            activity.append(float(np.abs(v - before).mean()))
    return v, float(np.mean(activity))


@pytest.fixture(scope="module")
def fixed_and_varying():
    """A field at constant `du`, and one under a climate `scale` snapshot."""
    resolved = config.Config().resolve()
    reaction = resolved.reaction
    scale = M.climate_scale_field(SIZE, factor=FACTOR, seed=1)
    du_field, dv_field = M.diffusion_from_scale(
        scale, reaction, resolved.climate.range_du)

    fixed_v, fixed_activity = _equilibrate(reaction, reaction.du, reaction.dv)
    varying_v, varying_activity = _equilibrate(reaction, du_field, dv_field)
    return {
        "reaction": reaction,
        "homeostat": resolved.homeostat,
        "du_field": du_field,
        "fixed": (fixed_v, fixed_activity),
        "varying": (varying_v, varying_activity),
    }


def test_feature_size_is_not_uniform_across_the_field(fixed_and_varying):
    """The accessibility claim: coarse and fine regions must coexist.

    Measured per tile, because the global length scale cannot tell a
    polydisperse field from a monodisperse one -- both can have the same mean.
    """
    fixed_v = fixed_and_varying["fixed"][0]
    varying_v = fixed_and_varying["varying"][0]

    fixed_ell = M.local_length_scale(fixed_v, TILE)
    varying_ell = M.local_length_scale(varying_v, TILE)
    fixed_cv = float(fixed_ell.std() / fixed_ell.mean())
    varying_cv = float(varying_ell.std() / varying_ell.mean())

    assert varying_cv > fixed_cv * 1.3, (
        f"feature size is no more varied across the field than at constant du "
        f"(cv {varying_cv:.3f} against a control of {fixed_cv:.3f}); the "
        f"climate scale channel is not reaching the reaction"
    )


def test_local_feature_size_tracks_the_local_diffusion_rate(fixed_and_varying):
    """The mechanism, not just its effect.

    A field can be non-uniform by accident. This asserts that where the climate
    asks for coarse structure, coarse structure is what appears -- which is
    what makes the regions migrate with the climate rather than sit still.
    """
    varying_v = fixed_and_varying["varying"][0]
    ell = M.local_length_scale(varying_v, TILE)
    local_du = M.tile_mean(fixed_and_varying["du_field"], TILE)

    correlation = float(np.corrcoef(ell.ravel(), local_du.ravel())[0, 1])
    assert correlation > 0.5, (
        f"local feature size does not follow local du (r = {correlation:.2f}); "
        f"the regions are averaging back out to a single wavelength"
    )


def test_the_lever_does_not_move_what_the_homeostat_defends(fixed_and_varying):
    """Why `du` and not `kill` or the dv/du ratio.

    Mass and activity must stay inside the homeostat's deadband, and close to
    the fixed control. Anything that moves mass moves the exposure governor
    with it, which turns a morphology change into a slow global brightness
    cycle -- the coordinated global change DESIGN.md §4.3 forbids.
    """
    homeostat = fixed_and_varying["homeostat"]
    fixed_v, fixed_activity = fixed_and_varying["fixed"]
    varying_v, varying_activity = fixed_and_varying["varying"]

    for name, target, value in (
        ("mass", homeostat.target_mass, float(varying_v.mean())),
        ("activity", homeostat.target_activity, varying_activity),
    ):
        lo = target * (1.0 - homeostat.deadband)
        hi = target * (1.0 + homeostat.deadband)
        assert lo < value < hi, (
            f"{name} {value:.5f} left the homeostat deadband "
            f"[{lo:.5f}, {hi:.5f}]; the controller would start fighting the "
            f"morphology drift"
        )

    drift = abs(float(varying_v.mean()) - float(fixed_v.mean()))
    assert drift < 0.1 * float(fixed_v.mean()), (
        f"mass moved {drift:.4f} against the fixed-du control "
        f"({fixed_v.mean():.4f}); this lever is supposed to be nearly "
        f"orthogonal to mass"
    )


def test_a_drifting_climate_makes_the_arrangement_churn():
    """The monotony claim: structures must merge and split, not just jitter.

    The climate is stood in for by a slow crossfade between two independent
    snapshots, which is what its advection and OU drift amount to from the
    reaction's point of view: the local diffusion rate at a given place changes
    smoothly and without repeating.

    The fixed-du control is asserted to *fail* the same threshold. Without
    that, the test would still pass with the whole mechanism disconnected.
    """
    resolved = config.Config().resolve()
    reaction = resolved.reaction
    size, ticks, warmup, period, sample = 160, 4000, 2000, 1400, 200
    first = M.climate_scale_field(size, factor=20, seed=1)
    second = M.climate_scale_field(size, factor=20, seed=2)

    # Both arms warm up identically -- same seed, and the drift only starts
    # after the warmup -- so the warmup is run once and forked.
    u0, v0 = M.seed_field(size, seed=1)
    for _ in range(warmup):
        for _ in range(2):
            u0, v0 = M.gray_scott_step(
                u0, v0, reaction.feed, reaction.kill,
                du=reaction.du, dv=reaction.dv)

    results = {}
    for mode in ("fixed", "drift"):
        u, v = u0.copy(), v0.copy()

        features: list[int] = []
        holes: list[int] = []
        for tick in range(ticks):
            if mode == "drift":
                angle = 2.0 * np.pi * tick / period
                scale = np.cos(angle) * first + np.sin(angle) * second
                du, dv = M.diffusion_from_scale(
                    scale, reaction, resolved.climate.range_du)
            else:
                du, dv = reaction.du, reaction.dv
            for _ in range(2):
                u, v = M.gray_scott_step(
                    u, v, reaction.feed, reaction.kill, du=du, dv=dv)
            if tick % sample == 0:
                mask = v > 0.25
                features.append(M.count_features(mask))
                holes.append(M.count_holes(mask))
        counts = np.asarray(features, dtype=float)
        results[mode] = {
            "cv": float(counts.std() / max(counts.mean(), 1e-9)),
            "holes": np.asarray(holes, dtype=float),
        }

    floor = 0.10
    assert results["drift"]["cv"] > floor, (
        f"feature count barely moved over {ticks} ticks "
        f"(cv {results['drift']['cv']:.3f}); the arrangement is still frozen"
    )
    assert results["fixed"]["cv"] < floor, (
        f"the fixed-du control now churns on its own "
        f"(cv {results['fixed']['cv']:.3f}), so this test no longer "
        f"demonstrates anything about the climate scale channel"
    )

    # Weak on its own -- the shipped regime is a spot field, so the hole count
    # is small and noisy either way -- but a hole count that only ever rises is
    # the specific pathology of anastomosis without autolysis: every fusion
    # adds a cycle and nothing destroys one. Worth asserting before the flux
    # pruning of §4.7 step 3 lands, so a regression is visible when it does.
    deltas = np.diff(results["drift"]["holes"])
    assert (deltas > 0).any() and (deltas < 0).any(), (
        f"hole count is monotone over the run "
        f"({results['drift']['holes'].tolist()}); cells are only ever being "
        f"created or only ever destroyed"
    )
