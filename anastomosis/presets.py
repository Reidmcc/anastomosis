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

`stability` is newer still and sits at 0.0 in every preset, and that zero is
a carrying-across rather than an omission: 0.0 *is* the volatile character
every one of these was judged on (DESIGN.md §9), so any other value would
change a picture somebody chose by name.

Every preset also carries a *mode* (DESIGN.md §14): a preset is a bundle of
macro positions, and a macro position only means something through the curve
table of the mode it was tuned in -- `dense` at `intensity=0.78` describes a
regulation field, and the same numbers through the activation table would
describe a different picture nobody has judged. The seven originals are
regulation presets; `prism`, `cascade` and `spark` are activation's
(:data:`ACTIVATION_PRESETS`); `pulse` is resonance's
(:data:`RESONANCE_PRESETS`, DESIGN.md §16). That is why :func:`names` can
filter by mode and the panel shows the active mode's list.
"""

from __future__ import annotations

from .config import MODES, Macros

PRESETS: dict[str, Macros] = {
    # The shipped default: dark ground, moderate luminous filaments.
    "default": Macros(
        intensity=0.50, scale=0.50, tempo=0.45, palette=0.30,
        brightness=0.35, filament_glow=0.45, depth=0.60, stability=0.0,
        parallax=0.60,
        event_rate=0.50,
    ),
    # Sparser, slower, dimmer. For background presence rather than attention.
    "quiet": Macros(
        intensity=0.24, scale=0.62, tempo=0.22, palette=0.30,
        brightness=0.22, filament_glow=0.30, depth=0.72, stability=0.0,
        parallax=0.42,
        event_rate=0.34,
    ),
    # Denser network, more visible structure, still slow.
    "dense": Macros(
        intensity=0.78, scale=0.34, tempo=0.42, palette=0.44,
        brightness=0.38, filament_glow=0.55, depth=0.50, stability=0.0,
        parallax=0.55,
        event_rate=0.65,
    ),
    # Large, slow, strongly layered -- the most "deep water" of the set.
    "deep": Macros(
        intensity=0.42, scale=0.80, tempo=0.28, palette=0.62,
        brightness=0.30, filament_glow=0.40, depth=0.92, stability=0.0,
        parallax=0.88,
        event_rate=0.44,
    ),
    # Near-monochrome and very dim, for a dark room or late at night.
    "ember": Macros(
        intensity=0.38, scale=0.55, tempo=0.30, palette=0.06,
        brightness=0.16, filament_glow=0.34, depth=0.66, stability=0.0,
        parallax=0.48,
        event_rate=0.42,
    ),
    # Brighter filaments against the same dark ground.
    "luminous": Macros(
        intensity=0.55, scale=0.45, tempo=0.50, palette=0.52,
        brightness=0.34, filament_glow=0.74, depth=0.58, stability=0.0,
        parallax=0.58,
        event_rate=0.52,
    ),
    # Faster drift and hue rotation. Still nowhere near "energetic" -- tempo is
    # the macro most likely to be over-set, so even the top preset is gentle.
    "current": Macros(
        intensity=0.52, scale=0.40, tempo=0.74, palette=0.36,
        brightness=0.36, filament_glow=0.48, depth=0.54, stability=0.0,
        parallax=0.72,
        event_rate=0.50,
    ),
}

# The activation presets (DESIGN.md §14.8 step 4). First cuts: the endpoints
# they sit inside are measured (§14.8 step 2), but where inside them each of
# these *belongs* is a judgement for real eyes, exactly as §13 says of the
# defaults -- these are starting points for that judgement, not its record.
# All three keep the dark ground; the mode's energy is chroma and motion.
ACTIVATION_PRESETS: dict[str, Macros] = {
    # Colour variety first: the polychrome triad strong, everything else
    # moderate, so the families are what the eye is given to follow.
    "prism": Macros(
        intensity=0.72, scale=0.45, tempo=0.48, palette=0.50,
        brightness=0.40, filament_glow=0.55, depth=0.55, stability=0.0,
        parallax=0.60,
        event_rate=0.55,
    ),
    # Motion first: fast flow, brisk viewpoint, events arriving every few
    # minutes -- the field spends most of its time inside something happening.
    "cascade": Macros(
        intensity=0.55, scale=0.35, tempo=0.85, palette=0.30,
        brightness=0.38, filament_glow=0.50, depth=0.50, stability=0.0,
        parallax=0.78,
        event_rate=0.85,
    ),
    # Fine, dense and brisk: the busiest texture of the set, a step short of
    # the measured corner on every axis.
    "spark": Macros(
        intensity=0.82, scale=0.20, tempo=0.65, palette=0.68,
        brightness=0.42, filament_glow=0.65, depth=0.42, stability=0.0,
        parallax=0.52,
        event_rate=0.70,
    ),
}

PRESETS.update(ACTIVATION_PRESETS)

# The resonance preset (DESIGN.md §16). One to start: the mode's character
# comes from the audio drive rather than from the knobs, so what a preset
# contributes here is a sensible *ground state* for the drive to modulate --
# motion-forward, since the ~100 ms path that reads as "with the music" is
# the flow, with tempo short of the top so the drive has headroom to surge
# into rather than a ceiling to press against. A first cut for §16.8 step 5's
# judgement by eyes, like activation's three were for §14's.
RESONANCE_PRESETS: dict[str, Macros] = {
    "pulse": Macros(
        intensity=0.60, scale=0.40, tempo=0.65, palette=0.45,
        brightness=0.38, filament_glow=0.50, depth=0.50, stability=0.0,
        parallax=0.65,
        event_rate=0.55,
    ),
}

PRESETS.update(RESONANCE_PRESETS)

# The rhizotron's presets (DESIGN.md §15). A preset here is macro positions
# through the *regulation* table -- the root world has one tuning -- plus the
# world it describes: `meadow` at scale 0.25 is a fibrous root community, and
# the same numbers under a fungal backend would describe a picture nobody has
# judged, which is why the panel's list follows the backend as well as the
# mode. First cuts, for the judgement §13 defers to eyes.
RHIZOTRON_PRESETS: dict[str, Macros] = {
    # The balanced default: the shipped look, exactly.
    "loam": Macros(
        intensity=0.50, scale=0.50, tempo=0.45, palette=0.30,
        brightness=0.35, filament_glow=0.45, depth=0.60, stability=0.0,
        parallax=0.60,
        event_rate=0.50,
    ),
    # Fine and fibrous: tight spacing, hairline widths, an eager busy
    # community -- grass country.
    "meadow": Macros(
        intensity=0.68, scale=0.25, tempo=0.55, palette=0.22,
        brightness=0.36, filament_glow=0.50, depth=0.60, stability=0.0,
        parallax=0.60,
        event_rate=0.55,
    ),
    # Sparse, deep and austere: a few bold plunging axes in rusty ground,
    # rain rare, everything slower.
    "taproot": Macros(
        intensity=0.30, scale=0.85, tempo=0.35, palette=0.42,
        brightness=0.30, filament_glow=0.42, depth=0.60, stability=0.0,
        parallax=0.60,
        event_rate=0.40,
    ),
}

PRESETS.update(RHIZOTRON_PRESETS)

DEFAULT_PRESET = "default"

# Which mode each preset was tuned in. A separate table rather than a field on
# `Macros`, because a mode is not a knob: `Macros` is the shape of the eight
# sliders and travels through ramping and TOML as exactly that. Every preset
# must appear here; `mode_of` treats absence as an error rather than guessing.
# The rhizotron presets are regulation presets: that world has one tuning.
PRESET_MODES: dict[str, str] = {
    name: (
        "activation" if name in ACTIVATION_PRESETS
        else "resonance" if name in RESONANCE_PRESETS
        else "regulation"
    )
    for name in PRESETS
}

# Which *world* each preset describes: the fungal field (either depth
# backend) or the rhizotron. Same reasoning as the mode table -- macro
# positions only mean something through the world that renders them -- and
# the same discipline: every preset appears, absence is an error.
PRESET_WORLDS: dict[str, str] = {
    name: ("rhizotron" if name in RHIZOTRON_PRESETS else "fungal")
    for name in PRESETS
}


def world_for_backend(backend: str) -> str:
    """The preset world a backend draws from."""
    return "rhizotron" if backend == "rhizotron" else "fungal"


def get(name: str) -> Macros:
    from dataclasses import replace

    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; available: {sorted(PRESETS)}")
    return replace(PRESETS[name])


def mode_of(name: str) -> str:
    """The mode a preset was tuned in, and therefore belongs to."""
    if name not in PRESET_MODES:
        raise KeyError(f"preset {name!r} has no mode; add it to PRESET_MODES")
    mode = PRESET_MODES[name]
    if mode not in MODES:
        raise KeyError(f"preset {name!r} names unknown mode {mode!r}")
    return mode


def world_of(name: str) -> str:
    """The world a preset describes -- ``"fungal"`` or ``"rhizotron"``."""
    if name not in PRESET_WORLDS:
        raise KeyError(f"preset {name!r} has no world; add it to PRESET_WORLDS")
    return PRESET_WORLDS[name]


def names(mode: str | None = None, world: str | None = None) -> list[str]:
    """Preset names, optionally filtered by mode and by world.

    Unfiltered by default, which is what every pre-mode caller meant.
    """
    return [
        name for name in PRESETS
        if (mode is None or mode_of(name) == mode)
        and (world is None or world_of(name) == world)
    ]
