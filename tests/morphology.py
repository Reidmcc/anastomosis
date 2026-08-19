"""Morphology measurement for the reaction field -- see DESIGN.md §4.7.

:mod:`reference` measures whether the field is *alive*: mass, variance,
activity. Those are all invariant under rearrangement, which is exactly why the
homeostat could hold every one of them in band while the picture stayed the same
picture for hours. This module measures the arrangement instead:

* ``count_features``  -- how many separate structures there are,
* ``length_scale``    -- how big they are,
* ``local_length_scale`` -- how big they are *per region*, which is the measure
  that says whether sizes are uniform across the screen,
* ``count_holes``     -- how many enclosed voids they surround, which is the
  quantity the trypophobia complaint is actually about.

It also carries a stand-in for the ``scale`` channel of the climate field
(:func:`climate_scale_field`), so the offline runs and the tests can drive a
spatially varying diffusion rate without standing up a GPU.

Everything is NumPy-only and toroidal, matching the simulation domain. The Euler
characteristic in particular is a vectorised count of 2x2 pixel patterns rather
than a labelling pass, so it is cheap enough to run every few hundred ticks
inside a soak test.

Run as a script to reproduce the sweeps recorded in DESIGN.md §4.7::

    python tests/morphology.py           # the du/kill/ratio sweeps
    python tests/morphology.py drift     # fixed vs drifting du, globally
    python tests/morphology.py spatial   # fixed vs climate-varying du
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Matching the rest of the suite: conftest puts this directory on the path for
# pytest, and the inserts here make the module work when run as a script too --
# the repo root as well, since the sweeps read the shipped defaults.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reference import gray_scott_step, seed_field  # noqa: E402


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------


def euler_characteristic(mask: np.ndarray) -> float:
    """Euler characteristic of a binary mask on a torus, 8-connected foreground.

    Counted over 2x2 windows: ``chi = (C1 - C3 - 2*Cd) / 4``, where ``C1`` and
    ``C3`` are windows holding exactly one and exactly three foreground pixels
    and ``Cd`` is those holding two on a diagonal. Equivalent to
    ``components - holes``, so pairing it with :func:`count_features` gives the
    hole count without a second labelling pass.

    The sign on the diagonal term is the connectivity convention and must match
    :func:`count_features`: ``-2*Cd`` counts diagonally-touching pixels as one
    object (8-connected foreground, 4-connected background), ``+2*Cd`` counts
    them as two. Mismatching the two gives negative hole counts on any field
    with diagonal contacts, which is every real one.
    """
    a = mask
    b = np.roll(mask, -1, 1)
    c = np.roll(mask, -1, 0)
    d = np.roll(np.roll(mask, -1, 0), -1, 1)
    total = a.astype(np.int16) + b + c + d
    ones = np.count_nonzero(total == 1)
    threes = np.count_nonzero(total == 3)
    diagonal = np.count_nonzero(
        (total == 2) & ((a & d & ~b & ~c) | (b & c & ~a & ~d))
    )
    return (ones - threes - 2 * diagonal) / 4.0


def count_features(mask: np.ndarray, max_iterations: int = 4000) -> int:
    """Number of 8-connected components on a torus.

    Label propagation rather than a union-find: it is a handful of vectorised
    ``roll`` operations, keeps the module dependency-free, and the fields this
    runs on are small enough that the iteration count never matters.
    """
    if not mask.any():
        return 0
    height, width = mask.shape
    labels = np.where(mask, np.arange(height * width).reshape(height, width), -1)
    labels = labels.astype(np.int32)
    for _ in range(max_iterations):
        spread = labels.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    spread = np.maximum(
                        spread, np.roll(np.roll(labels, dy, 0), dx, 1)
                    )
        spread = np.where(mask, spread, -1)
        if np.array_equal(spread, labels):
            break
        labels = spread
    return int(len(np.unique(labels[mask])))


def count_holes(mask: np.ndarray) -> int:
    """Enclosed voids in the structure: ``components - chi``."""
    return int(round(count_features(mask) - euler_characteristic(mask)))


# ---------------------------------------------------------------------------
# Scale
# ---------------------------------------------------------------------------


def gradient_magnitude(field: np.ndarray) -> np.ndarray:
    gx = (np.roll(field, -1, 1) - np.roll(field, 1, 1)) * 0.5
    gy = (np.roll(field, -1, 0) - np.roll(field, 1, 0)) * 0.5
    return np.hypot(gx, gy)


def length_scale(v: np.ndarray) -> float:
    """Characteristic feature size in cells, ``mean V / mean |grad V|``.

    This is the measure DESIGN.md §4.7 proposes adding to the homeostat: it is a
    single reduction over the field, so it costs one extra term in the existing
    per-tile pass rather than a new one.
    """
    return float(v.mean() / max(gradient_magnitude(v).mean(), 1e-9))


def tile_mean(field: np.ndarray, tile: int) -> np.ndarray:
    """Block average, for measuring a quantity per region rather than globally."""
    h, w = field.shape
    h -= h % tile
    w -= w % tile
    return field[:h, :w].reshape(h // tile, tile, w // tile, tile).mean(axis=(1, 3))


def local_length_scale(v: np.ndarray, tile: int = 32) -> np.ndarray:
    """:func:`length_scale` per tile: the spread of feature size across the field.

    The global figure cannot see the difference that matters here. A field where
    every structure is the same size and a field holding coarse and fine regions
    side by side can have identical mean ell, identical mass and identical
    activity -- and it is the *uniformity*, not the size, that makes the first
    one a trypophobia trigger (DESIGN.md 4.7). This is the measure that
    separates them.
    """
    return tile_mean(v, tile) / np.maximum(tile_mean(gradient_magnitude(v), tile), 1e-9)


# ---------------------------------------------------------------------------
# A stand-in for the climate `scale` channel
# ---------------------------------------------------------------------------


def bilinear_upsample(low: np.ndarray, factor: int) -> np.ndarray:
    """Periodic bilinear upsample, matching how the climate field is sampled."""
    h, w = low.shape
    ys = (np.arange(h * factor) + 0.5) / factor - 0.5
    xs = (np.arange(w * factor) + 0.5) / factor - 0.5
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = low[y0 % h][:, x0 % w]
    b = low[y0 % h][:, (x0 + 1) % w]
    c = low[(y0 + 1) % h][:, x0 % w]
    d = low[(y0 + 1) % h][:, (x0 + 1) % w]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx


# Standard deviation the climate field actually settles at, measured off a
# running engine (ticks 1200-3600, every channel). Nothing like the +-1 the
# field is clamped to: the OU drive is spatially white and the per-tick
# diffusion removes almost all of it immediately. Anything measuring the effect
# of a `range_*` parameter has to use this, or it overstates the effect by
# roughly a factor of ten.
CLIMATE_SD = 0.11


# Quantiles of the trail field the running engine actually settles at, measured
# on a mature 160-cell single layer at the default macros -- the same kind of
# constant as CLIMATE_SD above, and recorded for the same reason: the shading
# balance is a statement about where the ceiling falls in this distribution, and
# there is no way to say that from the reaction alone.
#
# They move with the intensity macro, which concentrates the network (DESIGN.md
# 4.7 records the mass-weighted figures shifting by 3x across it), so anything
# asserted against them is an assertion about the default.
TRAIL_QUANTILES = {0.5: 0.009, 0.9: 0.291, 0.99: 0.934}
TRAIL_MAX = 3.01


def climate_scale_field(
    size: int, factor: int = 20, seed: int = 1, smooth: int = 3,
    amplitude: float = CLIMATE_SD,
) -> np.ndarray:
    """One snapshot of ``climate_c.x`` on a `size` square torus.

    Not the real climate pass -- there is no advection and no OU step -- but it
    has the two properties that matter for morphology. It is smooth, and one
    texel of it spans `factor` simulation cells, so a region gets to be large
    enough to establish its own wavelength: in the engine that ratio is about
    40 (a 64x36 field over a 1440p layer), and much below ~20 the regions are
    only a few features across and no local length scale can form. And it is
    scaled to the amplitude the running field actually reaches, not to the
    range it is clamped to.
    """
    rng = np.random.default_rng(seed)
    low = rng.normal(0.0, 1.0, (size // factor, size // factor))
    for _ in range(smooth):
        low = 0.5 * low + 0.125 * (
            np.roll(low, 1, 0) + np.roll(low, -1, 0)
            + np.roll(low, 1, 1) + np.roll(low, -1, 1)
        )
    low *= amplitude / max(low.std(), 1e-9)
    return np.clip(bilinear_upsample(low, factor), -1.0, 1.0)


def diffusion_from_scale(
    scale: np.ndarray, reaction, range_du: float
) -> tuple[np.ndarray, np.ndarray]:
    """(du, dv) fields from a climate `scale` channel, mirroring reaction.wgsl.

    The deviation is geometric and dv rides on du, so the dv/du ratio is held
    exactly -- that is what makes this lever move feature size without moving
    mass. The clamp is :func:`anastomosis.config.clamp_du`, vectorised.
    """
    du = np.clip(
        reaction.du * np.exp(range_du * scale), reaction.du_min, reaction.du_max
    )
    return du, reaction.dv * (du / reaction.du)


def morphology(v: np.ndarray, threshold: float = 0.25) -> dict[str, float]:
    """Every arrangement measure at once, for one snapshot of the V field."""
    mask = v > threshold
    features = count_features(mask)
    return {
        "mean_v": float(v.mean()),
        "var_v": float(v.var()),
        "features": float(features),
        "holes": float(features - euler_characteristic(mask)),
        "length_scale": length_scale(v),
        "area": float(mask.mean()),
    }


# ---------------------------------------------------------------------------
# Reproduction of the DESIGN.md §4.7 measurements
# ---------------------------------------------------------------------------

SIZE = 160
DU_BASE, DV_BASE = 0.2097, 0.1050
RATIO = DV_BASE / DU_BASE


def equilibrate(
    feed: float = 0.018,
    kill: float = 0.051,
    du: float = DU_BASE,
    dv: float = DV_BASE,
    ticks: int = 4000,
    substeps: int = 2,
    size: int = SIZE,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run to a mature field and report the late-window activity with it."""
    u, v = seed_field(size, seed=seed)
    activity: list[float] = []
    for tick in range(ticks):
        before = v
        for _ in range(substeps):
            u, v = gray_scott_step(u, v, feed, kill, du=du, dv=dv)
        if tick > ticks - 400:
            activity.append(float(np.abs(v - before).mean()))
    return u, v, float(np.mean(activity)) if activity else 0.0


def _row(label: str, v: np.ndarray, activity: float) -> None:
    m = morphology(v)
    print(
        f"{label:<22} features={m['features']:5.0f} holes={m['holes']:6.0f} "
        f"ell={m['length_scale']:5.2f} meanV={m['mean_v']:.4f} "
        f"var={m['var_v']:.5f} area={m['area']:.3f} activity={activity:.5f}"
    )


def sweeps() -> None:
    """The three lever sweeps. `du` at fixed ratio is the flat-mass one."""
    print("--- du, ratio fixed (the morphology lever) ---")
    for du in (0.12, DU_BASE, 0.26, 0.32, 0.40, 0.50):
        _, v, activity = equilibrate(du=du, dv=du * RATIO)
        _row(f"du={du:.3f}", v, activity)

    print("\n--- kill (moves mass; the homeostat already owns it) ---")
    for kill in (0.0460, 0.0480, 0.0500, 0.0510, 0.0520, 0.0540, 0.0560):
        _, v, activity = equilibrate(kill=kill)
        _row(f"kill={kill:.4f}", v, activity)

    print("\n--- dv/du ratio (moves mass hardest) ---")
    for dv in (0.075, 0.090, 0.105, 0.120, 0.135):
        _, v, activity = equilibrate(dv=dv)
        _row(f"ratio={dv / DU_BASE:.2f}", v, activity)


def drift(ticks: int = 6000, period: int = 3000, warmup: int = 3000) -> None:
    """Fixed vs drifting `du`: does the arrangement actually churn?

    The drift here is a deterministic triangle so the run reproduces. In the
    engine it must be an OU walk carried by the climate field -- a globally
    coherent breathing of feature size would be exactly the coordinated global
    change DESIGN.md §4.2 exists to prevent.
    """
    for mode in ("fixed", "drift"):
        u, v = seed_field(SIZE, seed=1)
        for _ in range(warmup):
            for _ in range(2):
                u, v = gray_scott_step(u, v, 0.018, 0.051, du=DU_BASE, dv=DV_BASE)

        features, scales, activity, mass = [], [], [], []
        for tick in range(ticks):
            if mode == "drift":
                phase = (tick % period) / period
                du = 0.16 + (0.34 - 0.16) * (1.0 - abs(2.0 * phase - 1.0))
            else:
                du = DU_BASE
            before = v
            for _ in range(2):
                u, v = gray_scott_step(u, v, 0.018, 0.051, du=du, dv=du * RATIO)
            activity.append(float(np.abs(v - before).mean()))
            mass.append(float(v.mean()))
            if tick % 250 == 0:
                features.append(count_features(v > 0.25))
                scales.append(length_scale(v))

        counts = np.asarray(features, dtype=float)
        print(
            f"{mode:>6}: features {counts.min():.0f}-{counts.max():.0f} "
            f"(mean {counts.mean():.0f}, s.d. {counts.std():.0f}, "
            f"cv {counts.std() / max(counts.mean(), 1e-9):.2f})  "
            f"ell {min(scales):.2f}-{max(scales):.2f}  "
            f"meanV {np.mean(mass):.4f}  activity {np.mean(activity):.5f}"
        )


def spatial(
    ticks: int = 4000, size: int = 192, factor: int = 24, tile: int = 32
) -> None:
    """Fixed vs climate-varying `du`: is the feature size still uniform?

    This is the shipped mechanism, unlike :func:`drift`, which walks `du`
    globally. The figure to watch is the *spread* of the per-tile length scale
    and its correlation with the local diffusion rate: a strong correlation is
    what says the regions really did adopt their own wavelength rather than
    averaging back out to one.
    """
    from anastomosis.config import Config

    resolved = Config().resolve()
    reaction = resolved.reaction
    scale = climate_scale_field(size, factor=factor, seed=1)
    du_field, dv_field = diffusion_from_scale(
        scale, reaction, resolved.climate.range_du)
    print(f"climate scale s.d. {scale.std():.2f}, "
          f"du {du_field.min():.3f}-{du_field.max():.3f}")

    for label, du, dv in (
        ("fixed", reaction.du, reaction.dv), ("varying", du_field, dv_field),
    ):
        u, v = seed_field(size, seed=1)
        activity: list[float] = []
        for tick in range(ticks):
            before = v
            for _ in range(2):
                u, v = gray_scott_step(u, v, reaction.feed, reaction.kill, du=du, dv=dv)
            if tick > ticks - 400:
                activity.append(float(np.abs(v - before).mean()))
        ell = local_length_scale(v, tile)
        local_du = tile_mean(np.broadcast_to(du, v.shape), tile)
        correlation = (
            np.corrcoef(ell.ravel(), local_du.ravel())[0, 1]
            if local_du.std() > 0 else float("nan")
        )
        _row(label, v, float(np.mean(activity)))
        print(f"{'':<22} local ell {ell.min():.2f}-{ell.max():.2f} "
              f"cv {ell.std() / ell.mean():.3f} corr(ell, du) {correlation:.2f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "drift":
        drift()
    elif mode == "spatial":
        spatial()
    else:
        sweeps()
