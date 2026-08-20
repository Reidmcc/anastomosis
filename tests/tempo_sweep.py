"""Tempo / WCAG-area sweep -- DESIGN.md §14.8 step 2. Measurement, not a test.

The binding constraint on the activation mode's tempo endpoints is not the
per-pixel limiter -- honest motion is exempt from that by design -- but the
WCAG area criterion: the fraction of pixels whose lightness changes by >=10%
in a single frame must stay under 25%, measured at fixed pixels the way WCAG
measures, so translation counts against it even though the limiter rightly
permits it. A translating filament changes pixels at its edges, so the
fraction scales with speed times perimeter density -- which couples this
sweep to intensity and scale, and is why it runs at the busy corner of the
mode (intensity and glow high, scale at its finest) rather than at defaults.

This inverts the usual tuning order, per §14.5(1): the activation tempo tops
are not chosen and then tested, they are *derived* from this map -- set where
the sustained area fraction still has comfortable margin under the 25%
criterion (§14.5 suggests <=15%).

What is swept is a single multiplier on the three motion primitives the tempo
macro drives -- agents.speed, flow.psi_gain, flow.field_gain -- applied on
top of the regulation table's top values, with the simulation at 30 Hz and
one render per tick, so a rendered frame is a display frame at the 30 FPS
cap and per-frame motion is what a viewer would see. The proposed activation
tops (§14.3: speed ~2.2, psi ~4.0, field ~2.4) sit between 1.6x and 1.9x on
this axis.

Caveat, the same one §4.7 records for morphology: a small test field is not a
1440p display. Feature size in *cells* is resolution-independent, so on a
small field each feature covers far more of the screen and its moving edges
weigh more in the area fraction -- the small sizes here overstate the
fraction relative to the real display. The sweep therefore runs at more than
one size to show the direction of that scaling, and the binding numbers for
shipped endpoints want re-measuring on real hardware at real resolution.

Run:  python tests/tempo_sweep.py             # the full table (some minutes)
      python tests/tempo_sweep.py --quick     # fewer points, smaller frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import reference as R  # noqa: E402
from anastomosis import config, engine as engine_module, events  # noqa: E402

# WCAG general flash threshold, as in test_flash_safety.py.
FLASH_LUMINANCE_FRACTION = 0.10

# The three primitives the multiplier scales, at the regulation table's top.
MOTION_TOPS = {
    "agents.speed": None,       # filled from MODE_CURVES below
    "flow.psi_gain": None,
    "flow.field_gain": None,
}


def busy_corner_params() -> config.Params:
    """The mode's busy corner: everything that adds material, contrast or
    perimeter at its top, features at their finest, simulation at the 30 Hz
    the display cap implies. Regulation table -- the activation one is either
    a copy (before §14.8 step 3) or derived from this measurement (after)."""
    params = config.Config(macros=config.Macros(
        intensity=1.0, scale=0.0, tempo=1.0,
        filament_glow=1.0, brightness=1.0,
    ), mode="regulation").resolve()
    params.sim_hz = 30.0
    params.render.layers = 2
    return params


def motion_tops() -> dict[str, float]:
    table = config.MODE_CURVES["regulation"]["tempo"]
    tops = {}
    for path, _lo, hi, _gamma in table:
        if path in MOTION_TOPS:
            tops[path] = hi
    assert set(tops) == set(MOTION_TOPS), "the tempo curve moved; update this sweep"
    return tops


def sweep_point(
    device, width: int, height: int, multiplier: float,
    warmup: int, measure: int, seed: int,
) -> dict[str, float]:
    """One engine, grown fresh at this multiplier, measured after warm-up."""
    import wgpu

    params = busy_corner_params()
    for path, top in motion_tops().items():
        config.set_path(params, path, top * multiplier)

    engine = engine_module.Engine(device, width, height, params, seed=seed)
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    target, fmt = texture.create_view(), "rgba8unorm"
    scheduler = events.EventScheduler(seed=seed)

    def frame() -> None:
        rows, _ = scheduler.pack(8)
        engine.tick(params, rows)
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)
        scheduler.update(1.0 / params.sim_hz, params.events)

    # Warm-up renders as well as ticks: the exposure governor and the
    # limiter's history are part of the steady state being measured.
    for _ in range(warmup):
        frame()

    previous = R.lightness(engine.read_final_rgba()[..., :3])
    areas = np.empty(measure)
    areas_half = np.empty(measure)  # at half the threshold: the approach
    peak = np.empty(measure)
    for i in range(measure):
        frame()
        current = R.lightness(engine.read_final_rgba()[..., :3])
        delta = np.abs(current - previous)
        areas[i] = float((delta >= FLASH_LUMINANCE_FRACTION).mean())
        areas_half[i] = float((delta >= FLASH_LUMINANCE_FRACTION / 2).mean())
        peak[i] = float(delta.max())
        previous = current

    return {
        "median": float(np.median(areas)),
        "p95": float(np.quantile(areas, 0.95)),
        "max": float(areas.max()),
        "half_p95": float(np.quantile(areas_half, 0.95)),
        "peak_dl": float(peak.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="fewer multipliers, shorter runs, one size")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from anastomosis import device as device_module

    device, _info = device_module.request_device()

    if args.quick:
        sizes = [(128, 96)]
        multipliers = [1.0, 2.0, 4.0]
        warmup, measure = 240, 60
    else:
        sizes = [(128, 96), (224, 126)]
        multipliers = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        warmup, measure = 300, 90

    print("tempo sweep at the busy corner (intensity/glow/brightness 1.0, "
          "scale 0.0, sim 30 Hz, layers 2)")
    print(f"warm-up {warmup} frames, measured {measure}; "
          f"area = fraction of pixels with |dL| >= "
          f"{FLASH_LUMINANCE_FRACTION:.0%} per frame; WCAG threshold 25%")
    print("half = the same area at half the threshold (the approach); "
          "peak = worst single-pixel per-frame |dL|")
    for width, height in sizes:
        print(f"\n  {width}x{height}")
        print("  mult   median     p95     max   half(p95)  peak|dL|")
        for m in multipliers:
            stats = sweep_point(
                device, width, height, m, warmup, measure, args.seed)
            print(f"  {m:4.2f}   {stats['median']:6.1%}  {stats['p95']:6.1%}"
                  f"  {stats['max']:6.1%}    {stats['half_p95']:6.1%}"
                  f"    {stats['peak_dl']:6.3f}")


if __name__ == "__main__":
    main()
