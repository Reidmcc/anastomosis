"""NumPy reference implementations of the numerical core.

Two jobs:

* parity testing against the WGSL, which catches shader bugs that otherwise
  present only as "it looks a bit wrong";
* fast offline exploration of the Gray-Scott parameter map, which is how the
  defaults in :mod:`anastomosis.config` were chosen rather than guessed.

These must stay in step with ``reaction.wgsl``, ``trail.wgsl``,
``advect.wgsl`` and ``psi.wgsl``.
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
    du: float | np.ndarray = 0.2097,
    dv: float | np.ndarray = 0.1050,
    dt: float = 0.85,
) -> tuple[np.ndarray, np.ndarray]:
    """One substep, matching ``reaction.wgsl`` including its clamps.

    Every rate may be a scalar or a field. ``feed`` and ``kill`` vary per region
    because the climate does; ``du`` and ``dv`` vary for the same reason and
    additionally carry the morphology drift of DESIGN.md §4.7, which is what
    keeps the feature size from being pinned to a single wavelength. Callers
    are responsible for applying the shader's clamps to the rates they pass --
    :func:`anastomosis.config.clamp_du` and
    :func:`anastomosis.config.clamp_reaction` mirror them.
    """
    rate = u * v * v
    u_next = u + dt * (du * laplacian9(u) - rate + feed * (1.0 - u))
    v_next = v + dt * (dv * laplacian9(v) + rate - (feed + kill) * v)
    return np.clip(u_next, 0.0, 1.5), np.clip(v_next, 0.0, 1.0)


def trail_step(
    trail: np.ndarray,
    income: np.ndarray,
    deposited: np.ndarray,
    decay: float | np.ndarray,
    income_rate: float,
    prune_gain: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One trail update, matching ``trail.wgsl`` including its clamps.

    Returns ``(trail, income, prune_relative)`` -- the three channels the shader
    writes. ``prune_relative`` is the extra decay the flux-pruning term applied,
    as a fraction of the base rate; the reduce pass sums it weighted by trail to
    work out how much mass the agent layer has to hand back.

    The pruning is deliberately one-sided: ``decay_eff >= decay`` everywhere, so
    no texel is ever given a longer memory than the base rate. See DESIGN.md
    §4.7 for why the obvious mass-neutral alternative -- centring the term so
    that well-fed strands decay *slower* -- freezes the field.
    """
    expenditure = decay * trail
    income_next = income + (deposited - income) * income_rate
    deficit = np.clip(1.0 - income_next / (expenditure + 1e-6), 0.0, 1.0)
    prune_relative = prune_gain * deficit
    decay_eff = np.clip(decay * (1.0 + prune_relative), 0.001, 0.5)
    value = np.clip(trail * (1.0 - decay_eff) + deposited, 0.0, 8.0)
    return value, np.clip(income_next, 0.0, 8.0), np.clip(prune_relative, 0.0, 8.0)


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


# ---------------------------------------------------------------------------
# Oklab, matching common.wgsl. Used by the flash-safety tests to measure the
# output in the same space the shader bounds it in.
# ---------------------------------------------------------------------------


def linear_srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """`rgb` is (..., 3) linear sRGB; returns (..., 3) Oklab."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    cbrt = lambda x: np.sign(x) * np.cbrt(np.abs(x) + 1e-12)  # noqa: E731
    l_, m_, s_ = cbrt(l), cbrt(m), cbrt(s)
    return np.stack([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ], axis=-1)


def lightness(rgb: np.ndarray) -> np.ndarray:
    """Oklab L only, which is what the *perceptual* half of the bound uses."""
    return linear_srgb_to_oklab(rgb)[..., 0]


# WCAG relative luminance, matching `relative_luminance` in common.wgsl. This
# is the quantity the photosensitivity thresholds are defined in, and it is not
# Oklab L: L is roughly its cube root, so `dY/dL = 3L^2`, and for a chromatic
# colour L is not luminance at all. DESIGN.md 14.3.
LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722])


def relative_luminance(rgb: np.ndarray) -> np.ndarray:
    """`rgb` is (..., 3) linear sRGB; returns (...) WCAG relative luminance."""
    return rgb @ LUMINANCE_WEIGHTS


def saturated_red(rgb: np.ndarray) -> np.ndarray:
    """WCAG's saturated-red test: `R / (R + G + B) >= 0.8`, on *encoded* sRGB.

    Encoded rather than linear because that is the form the criterion is
    written against, and the two disagree sharply near black -- which is where
    most of this application's pixels live.
    """
    encoded = linear_to_srgb(rgb)
    total = encoded.sum(axis=-1)
    # A pixel with no light in it has no hue to be red; the ratio is 0/0 there,
    # and calling that "saturated red" would make the criterion meaningless on
    # a field whose ground is nearly black.
    return np.where(total > 1e-6, encoded[..., 0] / np.maximum(total, 1e-12), 0.0) >= 0.8


def linear_to_srgb(c: np.ndarray) -> np.ndarray:
    """The sRGB transfer function, matching common.wgsl."""
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


# ---------------------------------------------------------------------------
# The flow's vector potential, matching psi.wgsl and the tiling noise in
# common.wgsl.
#
# Ported because the property that matters here is not visible in any single
# value: the noise has to *close on the domain*, and the cost of it not closing
# is a statistic of the field it drives, measured over hundreds of ticks
# (tests/test_seam.py).
# ---------------------------------------------------------------------------

MASK32 = 0xFFFFFFFF


def pcg(values: np.ndarray) -> np.ndarray:
    """The PRNG from ``common.wgsl``. Carried in uint64 and masked, because
    numpy's uint32 overflow behaviour has varied across versions and the hash
    is only the hash if it wraps."""
    v = np.asarray(values, dtype=np.uint64) & MASK32
    state = (v * np.uint64(747796405) + np.uint64(2891336453)) & MASK32
    shift = (state >> np.uint64(28)) + np.uint64(4)
    word = (((state >> shift) ^ state) * np.uint64(277803737)) & MASK32
    return ((word >> np.uint64(22)) ^ word) & MASK32


def pcg3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return pcg(a ^ pcg(b ^ pcg(c)))


def hash_grid(ix: np.ndarray, iy: np.ndarray, s: int) -> np.ndarray:
    """Lattice value in [-1, 1], matching ``hash_grid`` in common.wgsl."""
    ix = np.asarray(ix, dtype=np.uint64) & MASK32
    iy = np.asarray(iy, dtype=np.uint64) & MASK32
    h = pcg3(
        (ix * np.uint64(374761393)) & MASK32,
        (iy * np.uint64(668265263)) & MASK32,
        np.full(ix.shape, s & MASK32, dtype=np.uint64),
    )
    return h.astype(np.float64) * 2.3283064365386963e-10 * 2.0 - 1.0


def value_noise_tiled(
    px: np.ndarray, py: np.ndarray, period: int, s: int
) -> np.ndarray:
    """``value_noise_tiled`` from common.wgsl: the lattice index wraps at
    ``period``, so the result repeats exactly every ``period`` units."""
    ix, iy = np.floor(px), np.floor(py)
    fx, fy = px - ix, py - iy
    wx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
    wy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
    i, j = ix.astype(np.int64), iy.astype(np.int64)
    lo_x, lo_y = i % period, j % period
    hi_x, hi_y = (i + 1) % period, (j + 1) % period
    a = hash_grid(lo_x, lo_y, s)
    b = hash_grid(hi_x, lo_y, s)
    c = hash_grid(lo_x, hi_y, s)
    d = hash_grid(hi_x, hi_y, s)
    top = a + (b - a) * wx
    bottom = c + (d - c) * wx
    return top + (bottom - top) * wy


# The per-octave offsets and seed mixes of ``value_noise_octaves_tiled``. The
# offsets exist to keep the three lattices off each other's grid lines; being
# constants, they cannot affect the period.
_OCTAVES = (
    (1, 0.37, 0.11, 0x00000000, 0.62),
    (2, 4.19, 7.53, 0x9E3779B9, 0.26),
    (4, 2.71, 5.09, 0x85EBCA6B, 0.12),
)


def value_noise_octaves_tiled(
    u: np.ndarray, v: np.ndarray, period: int, s: int
) -> np.ndarray:
    """``value_noise_octaves_tiled`` from common.wgsl, over uv in [0, 1]."""
    n = max(int(period), 1)
    # The period is folded into the stream, so that two calls at neighbouring
    # periods are independent rather than sharing lattice values.
    stream = (s ^ ((n * 0x27D4EB2F) & MASK32)) & MASK32
    total = np.zeros(np.broadcast(u, v).shape, dtype=np.float64)
    for multiple, off_x, off_y, mix, weight in _OCTAVES:
        cells = n * multiple
        total += value_noise_tiled(
            u * cells + off_x, v * cells + off_y, cells, stream ^ mix
        ) * weight
    return total


def psi_increment(
    u: np.ndarray, v: np.ndarray, noise_scale: float, s: int
) -> np.ndarray:
    """The OU forcing of ``psi.wgsl``: two bracketing integer periods,
    crossfaded, with the weights normalised in quadrature so that a
    non-integer ``psi_noise_scale`` does not thin the forcing."""
    scale = max(float(noise_scale), 1.0)
    period = int(np.floor(scale))
    blend = scale - np.floor(scale)
    norm = 1.0 / np.sqrt(max((1.0 - blend) ** 2 + blend ** 2, 1e-6))
    return norm * (
        value_noise_octaves_tiled(u, v, period, s) * (1.0 - blend)
        + value_noise_octaves_tiled(u, v, period + 1, s) * blend
    )


def psi_run(
    width: int = 64,
    height: int = 48,
    ticks: int = 700,
    theta: float = 0.0022,
    sigma: float = 0.085,
    noise_scale: float = 3.0,
    seed: int = 0x5EED,
    increment=psi_increment,
) -> np.ndarray:
    """Evolve psi for `ticks`, matching ``psi.wgsl``.

    ``increment`` is injectable so that a test can drive the same integrator
    with a forcing that does *not* tile and show that the difference is the
    forcing, not the integrator.
    """
    u, v = np.meshgrid(
        (np.arange(width) + 0.5) / width, (np.arange(height) + 0.5) / height
    )
    psi = np.zeros((height, width), dtype=np.float64)
    for tick in range(ticks):
        neighbours = (
            np.roll(psi, 1, 1) + np.roll(psi, -1, 1)
            + np.roll(psi, 1, 0) + np.roll(psi, -1, 0)
        )
        smoothed = psi + (neighbours * 0.25 - psi) * 0.25
        s = (tick ^ seed ^ 0x5BF03635) & MASK32
        psi = np.clip(
            smoothed * (1.0 - theta) + increment(u, v, noise_scale, s) * sigma,
            -8.0, 8.0,
        )
    return psi


def seam_curvature(field: np.ndarray, margin: int = 4) -> tuple[float, float]:
    """How much more sharply the field bends across each wrap seam than inside.

    Curvature, not gradient, because the artefact is a *kink*. A forcing that
    does not close on the domain leaves the field it drives continuous but bent
    along u=0 and v=0, and a second difference is what a bend shows up in; the
    first difference is dominated by the field's own large-scale slope, which
    varies enough between runs to swamp the thing being measured.

    Returns the ratio for each axis. 1.0 means the seam bends no more than
    anywhere else, which is what a genuinely toroidal field looks like.
    """
    height, width = field.shape
    bend_x = np.abs(np.roll(field, -1, 1) - 2.0 * field + np.roll(field, 1, 1))
    bend_y = np.abs(np.roll(field, -1, 0) - 2.0 * field + np.roll(field, 1, 0))
    across_u = bend_x[:, [width - 1, 0]].mean()
    across_v = bend_y[[height - 1, 0], :].mean()
    inside_u = bend_x[:, margin:width - margin].mean()
    inside_v = bend_y[margin:height - margin, :].mean()
    return float(across_u / inside_u), float(across_v / inside_v)
