"""Blue-noise threshold mask via void-and-cluster.

Needed because of a specific failure mode described in DESIGN.md §6: an 8-bit
display showing a very slowly drifting smooth gradient produces visible banding,
and the band boundaries *crawl* as the gradient moves. A crawling boundary is a
moving hard edge -- visual punctuation, the one thing this application must not
produce. White-noise dither fixes the banding but is visibly grainy; blue noise
has no low-frequency content, so the eye integrates it away.

Ulichney's void-and-cluster algorithm. Generation is a few seconds for a 64x64
mask, so the result is cached on disk.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SIZE = 64
DEFAULT_SIGMA = 1.9


def _wrapped_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Gaussian on a torus, matching the mask's tiling."""
    axis = np.arange(size)
    delta = np.minimum(axis, size - axis).astype(np.float64)
    d2 = delta[:, None] ** 2 + delta[None, :] ** 2
    return np.exp(-d2 / (2.0 * sigma * sigma))


def generate(size: int = DEFAULT_SIZE, sigma: float = DEFAULT_SIGMA, seed: int = 7) -> np.ndarray:
    """Return a ``(size, size)`` float32 array of thresholds in [0, 1)."""
    rng = np.random.default_rng(seed)
    total = size * size
    kernel_freq = np.fft.rfft2(_wrapped_gaussian_kernel(size, sigma))

    def energy(pattern: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft2(pattern.astype(np.float64)) * kernel_freq
        return np.fft.irfft2(spectrum, s=(size, size))

    def tightest_cluster(pattern: np.ndarray) -> tuple[int, int]:
        e = np.where(pattern, energy(pattern), -np.inf)
        return np.unravel_index(int(np.argmax(e)), (size, size))

    def largest_void(pattern: np.ndarray) -> tuple[int, int]:
        e = np.where(pattern, np.inf, energy(pattern))
        return np.unravel_index(int(np.argmin(e)), (size, size))

    # Phase 0: scatter an initial minority pattern, then homogenise it by
    # repeatedly moving the most clustered point into the largest void.
    initial_ones = max(1, total // 10)
    pattern = np.zeros((size, size), dtype=bool)
    pattern.flat[rng.choice(total, initial_ones, replace=False)] = True

    for _ in range(total):
        cluster = tightest_cluster(pattern)
        pattern[cluster] = False
        void = largest_void(pattern)
        if void == cluster:
            pattern[cluster] = True
            break
        pattern[void] = True

    prototype = pattern.copy()
    rank = np.full((size, size), -1, dtype=np.int64)

    # Phase 1: remove points one at a time, ranking downward from the initial
    # population size.
    working = prototype.copy()
    for r in range(initial_ones - 1, -1, -1):
        cluster = tightest_cluster(working)
        working[cluster] = False
        rank[cluster] = r

    # Phase 2: add points one at a time, ranking upward.
    working = prototype.copy()
    for r in range(initial_ones, total):
        void = largest_void(working)
        working[void] = True
        rank[void] = r

    if (rank < 0).any():  # pragma: no cover - defensive
        raise RuntimeError("void-and-cluster left unranked cells")

    return ((rank + 0.5) / total).astype(np.float32)


def load_or_generate(
    cache_dir: Path | None = None,
    size: int = DEFAULT_SIZE,
    sigma: float = DEFAULT_SIGMA,
) -> np.ndarray:
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "anastomosis"
    path = cache_dir / f"bluenoise_{size}_{sigma:g}.npy"

    if path.exists():
        try:
            mask = np.load(path)
            if mask.shape == (size, size):
                return mask.astype(np.float32)
            log.warning("cached blue-noise mask has wrong shape, regenerating")
        except Exception:
            log.warning("could not read cached blue-noise mask, regenerating")

    log.info("generating %dx%d blue-noise mask (one-off, a few seconds)", size, size)
    mask = generate(size, sigma)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, mask)
    except OSError:
        log.debug("could not cache blue-noise mask to %s", path)
    return mask


def radial_spectrum(mask: np.ndarray) -> np.ndarray:
    """Radially averaged power spectrum, for verifying the mask is blue.

    Used by the tests: blue noise must have suppressed low frequencies, which is
    the entire property that makes it preferable to white noise here.
    """
    size = mask.shape[0]
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(mask - mask.mean()))) ** 2
    centre = size // 2
    y, x = np.indices((size, size))
    radius = np.hypot(y - centre, x - centre).astype(int)
    counts = np.bincount(radius.ravel(), minlength=centre + 1)
    totals = np.bincount(radius.ravel(), weights=spectrum.ravel(), minlength=centre + 1)
    return totals[: centre + 1] / np.maximum(counts[: centre + 1], 1)
