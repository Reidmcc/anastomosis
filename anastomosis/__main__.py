"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anastomosis",
        description=(
            "A long-running generative visual field for self-regulation and "
            "stimming."
        ),
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fps", type=int, default=30,
        help="frame rate cap (default: 30)",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--no-vsync", action="store_true",
        help="disable vsync; normally leave this on",
    )
    parser.add_argument(
        "--no-ui", action="store_true",
        help="run without the control panel",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="config file (default: ~/.config/anastomosis/config.toml)",
    )
    parser.add_argument(
        "--preset", type=str, default=None,
        help="start from a named preset",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="fix the random seed; omit for a different world each launch",
    )
    parser.add_argument(
        "--write-config", action="store_true",
        help="write a default config file and exit",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help=(
            "start from a fresh field instead of resuming the saved state "
            "(the control panel has a button for this too)"
        ),
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true",
        help="neither save nor resume simulation state",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=None, metavar="PATH",
        help=(
            "checkpoint file "
            "(default: ~/.local/state/anastomosis/checkpoint.npz)"
        ),
    )
    parser.add_argument(
        "--checkpoint-interval", type=float, default=None, metavar="SECONDS",
        help="how often to save simulation state (default: 300)",
    )
    parser.add_argument("--list-presets", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from . import config as config_module
    from . import presets as presets_module

    if args.list_presets:
        for name in presets_module.names():
            print(name)
        return 0

    config_path = args.config or config_module.default_config_path()

    if args.write_config:
        cfg = config_module.Config()
        if args.preset:
            cfg.macros = presets_module.get(args.preset)
            cfg.preset_name = args.preset
        config_module.save(cfg, config_path)
        print(f"wrote {config_path}")
        return 0

    if args.preset:
        cfg = config_module.load(config_path)
        try:
            cfg.macros = presets_module.get(args.preset)
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 2
        cfg.preset_name = args.preset
        config_module.save(cfg, config_path)

    from .app import AppOptions, Application

    options = AppOptions(
        width=args.width,
        height=args.height,
        max_fps=args.fps,
        vsync=not args.no_vsync,
        fullscreen=args.fullscreen,
        config_path=config_path,
        seed=args.seed,
        ui=not args.no_ui,
        checkpoint=not args.no_checkpoint,
        resume=not args.reset,
        checkpoint_path=args.checkpoint,
    )
    # Left unset unless asked for, so the default interval has one home.
    if args.checkpoint_interval is not None:
        options.checkpoint_seconds = args.checkpoint_interval

    try:
        Application(options).run()
    except KeyboardInterrupt:
        print()
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
