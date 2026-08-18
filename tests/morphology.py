"""Morphology measurement for the reaction field -- see DESIGN.md §4.7.

:mod:`reference` measures whether the field is *alive*: mass, variance,
activity. Those are all invariant under rearrangement, which is exactly why the
homeostat could hold every one of them in band while the picture stayed the same
picture for hours. This module measures the arrangement instead:

* ``count_features``  -- how many separate structures there are,
* ``length_scale``    -- how big they are,
* ``count_holes``     -- how many enclosed voids they surround, which is the
  quantity the trypophobia complaint is actually about.

Everything is NumPy-only and toroidal, matching the simulation domain. The Euler
characteristic in particular is a vectorised count of 2x2 pixel patterns rather
than a labelling pass, so it is cheap enough to run every few hundred ticks
inside a soak test.

Run as a script to reproduce the sweeps recorded in DESIGN.md §4.7::

    python tests/morphology.py           # the du/kill/ratio sweeps
    python tests/morphology.py drift     # fixed vs drifting du
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Matching the rest of the suite: conftest puts this directory on the path for
# pytest, and the insert here makes the module work when run as a script too.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "drift":
        drift()
    else:
        sweeps()
