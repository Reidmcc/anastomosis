"""NumPy reference implementations of the numerical core.

Two jobs:

* parity testing against the WGSL, which catches shader bugs that otherwise
  present only as "it looks a bit wrong";
* fast offline exploration of the Gray-Scott parameter map, which is how the
  defaults in :mod:`anastomosis.config` were chosen rather than guessed.

These must stay in step with ``reaction.wgsl`` and ``advect.wgsl``.
"""

from __future__ import annotations

import numpy as np


def laplacian9(field: np.ndarray) -> np.ndarray:
    """Nine-point Laplacian on a torus, matching ``reaction.wgsl``.

    Weights: 0.2 orthogonal, 0.05 diagonal, -1 centre. More isotropic than the
    five-point stencil, which leaves a visible square bias in large structures.
    """
    ortho = (
        np.roll(field, 1, 0) + np.roll(field, -1, 0)
        + np.roll(field, 1, 1) + np.roll(field, -1, 1)
    )
    diag = (
        np.roll(np.roll(field, 1, 0), 1, 1) + np.roll(np.roll(field, 1, 0), -1, 1)
        + np.roll(np.roll(field, -1, 0), 1, 1) + np.roll(np.roll(field, -1, 0), -1, 1)
    )
    return ortho * 0.2 + diag * 0.05 - field


def gray_scott_step(
    u: np.ndarray,
    v: np.ndarray,
    feed: float | np.ndarray,
    kill: float | np.ndarray,
    du: float = 0.2097,
    dv: float = 0.1050,
    dt: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """One substep, matching ``reaction.wgsl`` including its clamps."""
    rate = u * v * v
    u_next = u + dt * (du * laplacian9(u) - rate + feed * (1.0 - u))
    v_next = v + dt * (dv * laplacian9(v) + rate - (feed + kill) * v)
    return np.clip(u_next, 0.0, 1.5), np.clip(v_next, 0.0, 1.0)


def seed_field(
    size: int = 128, blobs: int | None = None, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """U=1, V=0 with scattered seeds, matching ``Layer._seed_state``."""
    rng = np.random.default_rng(seed)
    u = np.ones((size, size), dtype=np.float64)
    v = np.zeros((size, size), dtype=np.float64)
    if blobs is None:
        blobs = max(8, (size * size) // 12000)
    for _ in range(blobs):
        cx, cy = rng.integers(0, size, 2)
        radius = int(rng.integers(4, 12))
        ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = xs * xs + ys * ys <= radius * radius
        yy = np.arange(cy - radius, cy + radius + 1) % size
        xx = np.arange(cx - radius, cx + radius + 1) % size
        patch = np.ix_(yy, xx)
        v[patch] = np.where(mask, 0.45, v[patch])
        u[patch] = np.where(mask, 0.35, u[patch])
    return u, v


def run(
    feed: float | np.ndarray,
    kill: float | np.ndarray,
    ticks: int = 2000,
    substeps: int = 2,
    size: int = 128,
    seed: int = 0,
    warmup: int = 500,
    **kwargs,
) -> dict[str, float]:
    """Run and report the statistics the homeostat controls.

    ``activity`` is the mean per-tick |dV/dt| measured after ``warmup``, which
    is the quantity that distinguishes a live pattern from a settled one.
    """
    u, v = seed_field(size, seed=seed)
    activities: list[float] = []
    for tick in range(ticks):
        before = v
        for _ in range(substeps):
            u, v = gray_scott_step(u, v, feed, kill, **kwargs)
        if tick >= warmup:
            activities.append(float(np.abs(v - before).mean()))
    return {
        "mean_v": float(v.mean()),
        "var_v": float(v.var()),
        "activity": float(np.mean(activities)) if activities else 0.0,
        # Late-window activity: if this has collapsed relative to the mean, the
        # pattern is still settling and will not stay alive.
        "activity_late": float(np.mean(activities[-200:])) if activities else 0.0,
    }
