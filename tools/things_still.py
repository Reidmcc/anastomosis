"""Headless stills of the Small Strange Things, for the eyes-pass.

The §17 lesson, kept as this build's own instrument (DESIGN.md §18.6 step
3): grow a world from its founding five on the real backend at shipped
defaults, render every tick so the output chain's history is always
converged, and write a still at each requested age. What comes out is what
the screen would show, minus only the dither.

Usage:
    python tools/things_still.py --out stills/ --ticks 600 1800 3600
    python tools/things_still.py --out stills/ --seed 7 --width 960 --height 540

Each still is named by seed and tick. An existing file is never
overwritten; delete a run to retake it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The repo this file sits in, ahead of any installed copy: the stills must
# come from the code under review.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import wgpu


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("stills"))
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument(
        "--ticks", type=int, nargs="+", default=[300, 1200, 3600],
        help="tick counts to save stills at, in ascending order",
    )
    parser.add_argument(
        "--override", action="append", default=[],
        help="config override, path=value (e.g. things.sparkle_rate=2.4)",
    )
    args = parser.parse_args()

    from anastomosis import config, device as device_module
    from anastomosis.app import _write_png
    from anastomosis.things import ThingsEngine

    overrides = {}
    for item in args.override:
        path, _, value = item.partition("=")
        overrides[path] = float(value)

    device, info = device_module.request_device()
    print(f"device: {info.describe()}")
    params = config.Config(backend="things", overrides=overrides).resolve()
    engine = ThingsEngine(
        device, args.width, args.height, params, seed=args.seed)
    print(f"world: {engine.geometry.describe()}")

    target = device.create_texture(
        size=(args.width, args.height, 1),
        format=wgpu.TextureFormat.rgba8unorm,
        usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
               | wgpu.TextureUsage.COPY_SRC),
    ).create_view()

    args.out.mkdir(parents=True, exist_ok=True)
    marks = sorted(set(int(t) for t in args.ticks))
    for mark in marks:
        while engine.tick_count < mark:
            engine.tick(params)
            engine.render(
                params, frac=1.0, target_view=target,
                target_format="rgba8unorm")
        linear = np.clip(
            engine.read_final_rgba()[..., :3].astype(np.float32), 0.0, 1.0)
        srgb = np.where(
            linear <= 0.0031308,
            linear * 12.92,
            1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
        )
        rows = np.clip(srgb * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        path = args.out / (
            f"things-seed{args.seed:08x}-tick{engine.tick_count}.png")
        if path.exists():
            print(f"kept existing {path}")
            continue
        _write_png(path, rows)
        alive = "?"
        try:
            from anastomosis.things import THING_ALIVE, THING_DTYPE
            raw = device.queue.read_buffer(engine.things.cur)
            pop = np.frombuffer(raw, dtype=THING_DTYPE)
            alive = int(((pop["flags"] & THING_ALIVE) != 0).sum())
        except Exception:
            pass
        print(f"saved {path}  (population {alive})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
