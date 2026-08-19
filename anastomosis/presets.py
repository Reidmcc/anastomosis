"""Named macro settings.

Presets matter more here than in a typical creative tool. This is a regulation
aid, so *quickly getting back to the one that worked* is worth more than
fine-grained tweaking, and someone reaching for it may not be in a state to
adjust eight sliders.

Every preset names every macro, `event_rate` and `parallax` included. The
arrival rate used to be part of `intensity`, so each preset already implied one;
the values here are the rates these presets have always produced, carried across
unchanged when the rate became a knob of its own.

`parallax` is the exception to that carrying-across, because there was nothing
to carry: it used to be part of `depth`, over a range that drove a mechanism
which did not work (see `Backend._update_parallax`), so what each preset implied
about the viewpoint's travel was a description of nothing happening. These are
new values, set by what each preset is *for* -- `deep` swings furthest, `quiet`
and `ember` least, since a preset for a dark room at 2 a.m. is not the one that
should be moving the most.
"""

from __future__ import annotations

from .config import Macros

PRESETS: dict[str, Macros] = {
    # The shipped default: dark ground, moderate luminous filaments.
    "default": Macros(
        intensity=0.50, scale=0.50, tempo=0.45, palette=0.30,
        brightness=0.35, filament_glow=0.45, depth=0.60,
        parallax=0.60,
        event_rate=0.50,
    ),
    # Sparser, slower, dimmer. For background presence rather than attention.
    "quiet": Macros(
        intensity=0.24, scale=0.62, tempo=0.22, palette=0.30,
        brightness=0.22, filament_glow=0.30, depth=0.72,
        parallax=0.42,
        event_rate=0.34,
    ),
    # Denser network, more visible structure, still slow.
    "dense": Macros(
        intensity=0.78, scale=0.34, tempo=0.42, palette=0.44,
        brightness=0.38, filament_glow=0.55, depth=0.50,
        parallax=0.55,
        event_rate=0.65,
    ),
    # Large, slow, strongly layered -- the most "deep water" of the set.
    "deep": Macros(
        intensity=0.42, scale=0.80, tempo=0.28, palette=0.62,
        brightness=0.30, filament_glow=0.40, depth=0.92,
        parallax=0.88,
        event_rate=0.44,
    ),
    # Near-monochrome and very dim, for a dark room or late at night.
    "ember": Macros(
        intensity=0.38, scale=0.55, tempo=0.30, palette=0.06,
        brightness=0.16, filament_glow=0.34, depth=0.66,
        parallax=0.48,
        event_rate=0.42,
    ),
    # Brighter filaments against the same dark ground.
    "luminous": Macros(
        intensity=0.55, scale=0.45, tempo=0.50, palette=0.52,
        brightness=0.34, filament_glow=0.74, depth=0.58,
        parallax=0.58,
        event_rate=0.52,
    ),
    # Faster drift and hue rotation. Still nowhere near "energetic" -- tempo is
    # the macro most likely to be over-set, so even the top preset is gentle.
    "current": Macros(
        intensity=0.52, scale=0.40, tempo=0.74, palette=0.36,
        brightness=0.36, filament_glow=0.48, depth=0.54,
        parallax=0.72,
        event_rate=0.50,
    ),
}

DEFAULT_PRESET = "default"


def get(name: str) -> Macros:
    from dataclasses import replace

    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return replace(PRESETS[name])


def names() -> list[str]:
    return list(PRESETS)
