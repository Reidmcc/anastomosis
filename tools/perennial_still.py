"""Headless stills of the Perennial pane, for the eyes-pass and the count test.

The §17 instrument, kept as a tool: run the real rhizotron backend at
shipped defaults (or resumed from a saved column -- a live pane many seasons
deep is the emergent signal every renderer here is tuned against), render
often enough that the output chain's history is converged, and write a
still at each requested moment. What comes out is what the screen would
show, minus only the dither.

Two things this tool does that a screenshot cannot:

* **Ghost-zeroed A/B.** With ``--ab`` every still is written twice, once as
  rendered and once with the strata's darkening zeroed, and the pixel
  difference between the pair is reported -- the fraction of the pane the
  buried seasons change by at least one 8-bit step, and by at least four.
  This is the argument-ender from the second long watch (docs/perennial.md):
  strata that pass every test and never reach the screen show up here as a
  zero.
* **Seasons at real pacing.** ``--seasons N`` runs until N more seasons
  have turned, writing a still at each fossil moment and one at the end,
  and ``--save`` writes the finished column as a checkpoint so a long run
  can be chained or handed to the application.

Usage:
    python tools/perennial_still.py --out stills/ --ticks 3000 12000
    python tools/perennial_still.py --out stills/ --resume ~/.local/state/anastomosis/checkpoint-rhizotron.npz --ab
    python tools/perennial_still.py --out stills/ --resume col.npz --seasons 3 --save col-3.npz

Each still is named by seed, season and tick. An existing file is never
overwritten; delete a run to retake it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# The repo this file sits in, ahead of any installed copy: the stills must
# come from the code under review.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import wgpu

STRATA_KNOBS = ("strata_crisp", "strata_soft", "strata_bedrock")


def to_srgb8(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear.astype(np.float32), 0.0, 1.0)
    srgb = np.where(
        x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)
    return np.clip(srgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)


def lightness(rgb8: np.ndarray) -> np.ndarray:
    """Rec. 709 luma of an 8-bit image, in 8-bit steps -- the coarse
    instrument a viewer's display actually quantises to."""
    r, g, b = (rgb8[..., i].astype(np.float32) for i in range(3))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("stills"))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="a rhizotron checkpoint to start from (built at its geometry)")
    parser.add_argument(
        "--save", type=Path, default=None,
        help="write the finished column as a checkpoint here")
    parser.add_argument(
        "--ticks", type=int, nargs="*", default=[],
        help="absolute tick counts to save stills at (ascending)")
    parser.add_argument(
        "--seasons", type=int, default=0,
        help="run until this many more seasons have turned, with a still "
             "at each fossil moment and one at the end")
    parser.add_argument(
        "--max-ticks", type=int, default=0,
        help="stop after this many ticks whatever the seasons did")
    parser.add_argument(
        "--every", type=int, default=32,
        help="render one frame every this many ticks while running, so the "
             "exposure governor and the slew limiter stay converged")
    parser.add_argument(
        "--settle", type=int, default=240,
        help="frames rendered before each still (hundreds, not thirty)")
    parser.add_argument(
        "--override", action="append", default=[],
        help="config override, path=value (e.g. rhizotron.strata_step=0.7)")
    parser.add_argument(
        "--ab", action="store_true",
        help="also write each still with the strata zeroed, and report the "
             "pixel difference between the pair")
    parser.add_argument(
        "--crop", type=float, nargs=4, default=None,
        metavar=("CX", "CY", "W", "H"),
        help="also save a close-crop of each still, centred at the "
             "window-fraction point (CX, CY) with the given pixel size")
    parser.add_argument("--prefix", default="perennial")
    args = parser.parse_args()

    overrides = {}
    for item in args.override:
        path, _, value = item.partition("=")
        overrides[path] = float(value)

    from anastomosis import checkpoint, config, device as device_module
    from anastomosis.app import _write_png
    from anastomosis.rhizotron import RhizotronEngine

    device, info = device_module.request_device()
    print(f"device: {info.describe()}")
    params = config.Config(backend="rhizotron", overrides=overrides).resolve()

    saved = None
    geometry = None
    if args.resume is not None:
        saved = checkpoint.load(args.resume)
        if saved is None:
            print(f"could not read {args.resume}")
            return 1
        geometry = checkpoint.required_geometry(saved, backend="rhizotron")
        if geometry is None:
            print(f"{args.resume} is not a usable rhizotron checkpoint")
            return 1
        sim_hz = float(saved.meta.get("sim_hz") or 0.0)
        if sim_hz > 0.0:
            params.sim_hz = sim_hz
    engine = RhizotronEngine(
        device, args.width, args.height, params, seed=args.seed,
        geometry=geometry)
    if saved is not None:
        if not checkpoint.restore(engine, saved):
            print("the checkpoint did not restore")
            return 1
        print(f"resumed: {saved.describe()} -- season {engine._season}")
    print(f"column: {engine.geometry.describe()} at {params.sim_hz:g} Hz")

    target = device.create_texture(
        size=(args.width, args.height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
               | wgpu.TextureUsage.COPY_SRC),
    ).create_view()

    def render(frames: int) -> None:
        for _ in range(frames):
            engine.render(
                params, frac=1.0, target_view=target,
                target_format="rgba8unorm")

    def grab() -> np.ndarray:
        return to_srgb8(engine.read_final_rgba()[..., :3])

    args.out.mkdir(parents=True, exist_ok=True)

    def still(tag: str) -> None:
        stem = (
            f"{args.prefix}-seed{engine.seed & 0xFFFFFFFF:08x}"
            f"-season{engine._season:03d}-tick{engine.tick_count}{tag}")
        path = args.out / f"{stem}.png"
        if path.exists():
            print(f"kept existing {path}")
            return
        render(args.settle)
        rows = grab()
        _write_png(path, rows)
        print(f"saved {path}")
        if args.crop is not None:
            cx, cy, cw, ch = args.crop
            half_w, half_h = int(cw) // 2, int(ch) // 2
            px = min(max(int(cx * args.width), half_w), args.width - half_w)
            py = min(max(int(cy * args.height), half_h), args.height - half_h)
            crop = rows[py - half_h:py + half_h, px - half_w:px + half_w]
            _write_png(args.out / f"{stem}-crop.png", np.ascontiguousarray(crop))
        if args.ab:
            rhiz = params.rhizotron
            kept = {k: getattr(rhiz, k) for k in STRATA_KNOBS}
            for k in STRATA_KNOBS:
                setattr(rhiz, k, 0.0)
            render(args.settle)
            bare = grab()
            for k, v in kept.items():
                setattr(rhiz, k, v)
            render(args.settle)
            _write_png(args.out / f"{stem}-nostrata.png", bare)
            delta = np.abs(lightness(rows) - lightness(bare))
            print(
                f"  strata A/B: {100.0 * (delta >= 1.0).mean():.1f}% of the "
                f"pane differs by >=1 step, {100.0 * (delta >= 4.0).mean():.1f}% "
                f"by >=4, mean |dL| {delta.mean():.2f} steps, "
                f"max {delta.max():.0f}")

    marks = sorted(set(int(t) for t in args.ticks))
    seasons_wanted = engine._season + max(args.seasons, 0)
    start_tick = engine.tick_count
    last_report = time.time()
    last_report_tick = engine.tick_count

    def report() -> None:
        nonlocal last_report, last_report_tick
        now = time.time()
        if now - last_report >= 30.0:
            rate = (engine.tick_count - last_report_tick) / (now - last_report)
            sim_min = (engine.tick_count - start_tick) / params.sim_hz / 60.0
            print(
                f"  tick {engine.tick_count} ({sim_min:.1f} sim-min in, "
                f"{rate:.0f} ticks/s), season {engine._season}, drive "
                f"{engine._intern:.3f}, germ {engine._germ_ease:.3f}, "
                f"wood {engine._wood_mass:.0f}, living {engine._living_mass:.0f}")
            last_report, last_report_tick = now, engine.tick_count

    def step() -> None:
        engine.tick(params)
        if engine.tick_count % max(args.every, 1) == 0:
            render(1)
        if engine.fossil_due:
            engine.fossil_due = False
            print(f"  fossil moment at tick {engine.tick_count}, "
                  f"season {engine._season}")
            if args.seasons > 0:
                still("-fossil")
        report()

    for mark in marks:
        while engine.tick_count < mark:
            step()
        still("")

    if args.seasons > 0:
        while engine._season < seasons_wanted and (
            args.max_ticks <= 0
            or engine.tick_count - start_tick < args.max_ticks
        ):
            step()
        still("-end")
    elif not marks:
        still("")

    if args.save is not None:
        snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)
        checkpoint.save(args.save, snapshot)
        print(f"saved column to {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
