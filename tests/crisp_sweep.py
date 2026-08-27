"""Root crispness / tip-speed sweep -- DESIGN.md §15.7(2). Measurement, not a
test.

The §14 step 2 lesson said the WCAG area criterion leaves the fungal tempo
axis free *because* that mode's fields are soft: a moving edge spends many
frames crossing any one pixel. The rhizotron sharpens edges on purpose --
`root_edge` is the transfer softness, and the crisp look is the mode's
identity -- and a growing tip is a bright front arriving at new pixels. This
sweep measures what that actually costs, and its output is what licenses the
shipped `root_edge` floor and the elongation ceilings in `config.validate`:
a retune past either re-runs this script first.

The binding period is the *growth burst*, not the steady state: step 3's
structure does not decay, so a mature plant is a still image and its area
fraction is zero by construction. The sweep therefore warms up only briefly
-- the plant mid-window, laterals and fines spawning en masse -- and measures
through the expansion, with the descent at its config ceiling underneath so
the scroll's contribution is in the number too. Branching is made eager and
spacing tight, so the measured corner is busier than any default.

Run:  python tests/crisp_sweep.py             # the full table
      python tests/crisp_sweep.py --quick     # fewer points, one size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import reference as R  # noqa: E402
from anastomosis import config  # noqa: E402
from anastomosis.rhizotron import RhizotronEngine  # noqa: E402

FLASH_LUMINANCE_FRACTION = 0.10

# The shipped elongation rates the multiplier scales.
BASE_ELONG = {
    "rhizotron.elong_axis": None,
    "rhizotron.elong_lateral": None,
    "rhizotron.elong_fine": None,
}


def busy_corner_params(multiplier: float, edge: float) -> config.Params:
    """Everything that adds bright moving edges, at its eager end."""
    # Resolved as the rhizotron so backend-conditional resolution -- the
    # §17.5 exposure lift above all -- is inside the measurement.
    params = config.Config(backend="rhizotron", macros=config.Macros(
        intensity=1.0, filament_glow=1.0, brightness=1.0,
    )).resolve()
    params.sim_hz = 30.0
    rhiz = params.rhizotron
    for path in BASE_ELONG:
        config.set_path(params, path, config.get_path(params, path) * multiplier)
    rhiz.root_edge = edge
    # Eager branching: tight spacing, high probability, a full pool.
    rhiz.spacing_axis = 6.0
    rhiz.spacing_lateral = 4.0
    rhiz.branch_prob = 6.0
    rhiz.axes_per_plant = rhiz.max_axes
    # The descent at its ceiling underneath it all.
    rhiz.descent_rate = 60.0
    return config.validate(params)


def sweep_point(
    device, width: int, height: int, multiplier: float, edge: float,
    warmup: int, measure: int, seed: int,
) -> dict[str, float]:
    import wgpu

    params = busy_corner_params(multiplier, edge)
    engine = RhizotronEngine(device, width, height, params, seed=seed)
    texture = device.create_texture(
        size=(width, height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    target, fmt = texture.create_view(), "rgba8unorm"

    def frame() -> None:
        engine.tick(params, [])
        engine.render(params, frac=0.5, target_view=target, target_format=fmt)

    # Enough for the image to fade in and the plant to be growing hard, not
    # enough for it to finish: the expansion is what is being measured.
    for _ in range(warmup):
        frame()

    previous = R.lightness(engine.read_final_rgba()[..., :3])
    areas = np.empty(measure)
    areas_half = np.empty(measure)
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
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from anastomosis import device as device_module

    device, _info = device_module.request_device()

    if args.quick:
        sizes = [(128, 96)]
        multipliers = [1.0, 4.0]
        edges = [0.5, 0.18, 0.02]
        warmup, measure = 90, 90
    else:
        sizes = [(128, 96), (224, 126)]
        multipliers = [1.0, 2.0, 4.0]
        edges = [0.5, 0.18, 0.05, 0.02]
        warmup, measure = 90, 120

    print("root crispness sweep at the eager corner (full axis pool, tight "
          "spacing, brisk branching, descent at its ceiling, sim 30 Hz)")
    print(f"warm-up {warmup} frames, measured {measure} over the growth "
          f"burst; area = fraction of pixels with |dL| >= "
          f"{FLASH_LUMINANCE_FRACTION:.0%} per frame; WCAG threshold 25%")
    for width, height in sizes:
        print(f"\n  {width}x{height}")
        print("  mult  edge   median     p95     max   half(p95)  peak|dL|")
        for m in multipliers:
            for e in edges:
                stats = sweep_point(
                    device, width, height, m, e, warmup, measure, args.seed)
                print(f"  {m:4.1f}  {e:4.2f}   {stats['median']:6.1%}"
                      f"  {stats['p95']:6.1%}  {stats['max']:6.1%}"
                      f"    {stats['half_p95']:6.1%}    {stats['peak_dl']:6.3f}")


if __name__ == "__main__":
    main()
