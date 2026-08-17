"""Parameter model, macro mapping, TOML persistence, and safety validation.

Two tiers, per DESIGN.md §9:

* **Macros** — seven knobs in 0..1, the normal interface.
* **Primitives** — the ~50 values the shaders actually read.

Macros drive primitives through the curve table in :data:`MACRO_CURVES`. The
config file may also pin individual primitives, which override the macro result.

Every value that could affect flash safety is clamped to a hard ceiling here
(:data:`SAFETY_CEILINGS`) before it can reach the GPU. That clamp is the last
line of defence for the invariant in DESIGN.md §7, so it lives in the
parameter layer rather than in the renderer, where a future refactor might
route around it.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TAU = math.tau


# --------------------------------------------------------------------------
# Primitive parameter groups
# --------------------------------------------------------------------------


@dataclass
class SafetyParams:
    """Output-stage limits. See DESIGN.md §7.

    ``max_luma_delta`` is the load-bearing one: at 30 FPS a value of 0.01 means
    a 10% luminance excursion takes >=333 ms, capping the system at 1.5
    flashes/second against the WCAG threshold of 3.
    """

    max_luma_delta: float = 0.010
    max_chroma_delta: float = 0.030
    iir_alpha: float = 0.20
    dither_amount: float = 1.0
    exposure_target: float = 0.16
    # Asymmetric: darkening is allowed to act faster than brightening, since
    # the unsafe direction is always "gets brighter".
    exposure_attack: float = 0.0035
    exposure_release: float = 0.0090


@dataclass
class AgentParams:
    """Physarum layer. Deposits are soft splats; see DESIGN.md §2."""

    density: float = 0.27  # agents per simulation cell
    speed: float = 0.90  # cells per tick
    sensor_angle: float = 0.42  # radians, off-axis sensors
    sensor_distance: float = 7.0  # cells
    turn_rate: float = 0.32  # radians per tick
    jitter: float = 0.10  # radians, stochastic steering
    deposit: float = 0.018  # per tick, kept far below trail_decay
    fusion_bias: float = 0.55  # commitment to sensed junctions, 0..1
    trail_decay: float = 0.055
    trail_diffuse: float = 1.15  # gaussian sigma in cells
    starve_threshold: float = 0.004
    max_age: float = 2400.0  # ticks before forced respawn


@dataclass
class ReactionParams:
    """Gray-Scott, coupled to the trail field."""

    # Chosen by sweeping the Gray-Scott map (see tests/reference.py and
    # test_regime.py). This point sits on the persistently-live ridge: it holds
    # mean V ~0.12 with variance ~0.009 and, critically, does not settle. The
    # more familiar F=0.038/K=0.062 looks similar for a few minutes and then
    # goes static, which is useless here.
    feed: float = 0.0180
    kill: float = 0.0510
    du: float = 0.2097
    dv: float = 0.1050
    dt: float = 0.85
    substeps: int = 2
    # Trail raises local feed through a saturating curve, so a heavily
    # trafficked texel cannot drive feed off the map. This is the coupling that
    # makes filaments nucleate reaction structure.
    trail_feed_gain: float = 0.012
    # Kill tracks feed along the live band rather than staying fixed. Without
    # this, a downward climate excursion in feed drives whole regions to zero
    # and they never recover -- verified in test_regime.py.
    kill_follows_feed: float = 0.55


@dataclass
class FlowParams:
    """Divergence-free velocity field, ``v = curl(psi)``. DESIGN.md §2.

    ``psi`` is a *stateful* field evolved by a spatial OU process, never a
    function of wall-clock time -- see DESIGN.md §3 for why that distinction
    matters over multi-day runs.
    """

    psi_gain: float = 1.30  # weight of the imposed-weather component
    field_gain: float = 0.85  # weight of the structure-following component
    psi_theta: float = 0.0022  # OU mean reversion per tick
    psi_sigma: float = 0.085  # OU noise amplitude per tick
    # Structural: psi texture size divisor. Fixed at startup, since changing
    # it would mean reallocating textures mid-session.
    psi_scale: int = 4
    # The `scale` macro drives this instead -- same perceptual effect on
    # feature size, no reallocation.
    psi_noise_scale: float = 3.0
    advect_dt: float = 1.0


@dataclass
class PigmentParams:
    """The advected field that is actually shaded. DESIGN.md §2."""

    inject_rate: float = 0.055  # how fast pigment adopts local structure
    density_from_v: float = 2.9
    density_from_trail: float = 0.85
    activity_rate: float = 0.020  # lowpass on |dV/dt|; deliberately very slow
    activity_gain: float = 26.0
    # Material keeps the hue it was born with and carries it along the flow.
    # Low values make structures of different ages chromatically distinct, so
    # the field marbles instead of shifting as one; high values make the whole
    # field track the drifting anchor together.
    hue_inject_mix: float = 0.010
    hue_from_orientation: float = 0.55


@dataclass
class ClimateParams:
    """Slowly drifting field of local parameter values. DESIGN.md §4.1."""

    width: int = 64
    height: int = 36
    theta: float = 0.0016  # OU mean reversion per tick
    sigma: float = 0.055  # OU noise per tick
    advect_gain: float = 0.22  # how fast regimes migrate
    diffuse: float = 0.30
    # Deviation amplitudes, applied as base + range * climate_texel.
    range_feed: float = 0.0080
    range_kill: float = 0.0035
    range_sensor_angle: float = 0.22
    range_sensor_distance: float = 3.0
    range_deposit: float = 0.008
    range_decay: float = 0.018
    range_flow: float = 0.55
    range_hue: float = 1.15  # radians


@dataclass
class HomeostatParams:
    """Loose, slow regulation that keeps the system alive. DESIGN.md §4.2.

    Deliberately wide deadband and long time constant: a tight controller makes
    the output feel regulated and monotonous, and becomes itself a source of
    coordinated global change (i.e. punctuation).
    """

    target_mass: float = 0.118  # mean V
    target_variance: float = 0.0090  # var V, proxy for "has structure"
    target_activity: float = 0.0012  # mean |dV/dt|, measured not guessed
    deadband: float = 0.30  # fractional, +-30%
    gain_p: float = 0.010
    gain_i: float = 0.0009
    integral_limit: float = 0.35
    tau_seconds: float = 120.0


@dataclass
class EventParams:
    """Poisson-arrival localised perturbations. DESIGN.md §4.3.

    Events are applied to *climate*, never to pigment or luminance directly, so
    their effect reaches the image only after several stages of diffusion.
    """

    enabled: bool = True
    rate_per_hour: float = 7.5
    attack_seconds: float = 45.0
    hold_seconds: float = 60.0
    release_seconds: float = 90.0
    strength: float = 0.85
    max_radius_frac: float = 0.24  # <= 0.25, the WCAG flash-area threshold
    max_concurrent: int = 4


@dataclass
class RenderParams:
    """Compositing, depth, and the Oklab colour mapping. DESIGN.md §5-6."""

    layers: int = 3
    base_scale: float = 1.0  # front layer, fraction of display resolution
    scale_falloff: float = 0.5  # each layer back is this much smaller
    # Back layers get larger on-screen features for free by being simulated at
    # lower resolution, so this is a fine-tuning knob rather than the main
    # mechanism; >1 exaggerates the difference.
    feature_falloff: float = 1.0
    # Screen-relative speed of each layer back. Combined with scale_falloff in
    # the engine, since a cell on a half-resolution layer covers twice the
    # screen distance.
    tempo_falloff: float = 0.70

    parallax: float = 0.020
    parallax_drift: float = 0.00035
    dof_radius: float = 3.2  # cells, at the backmost layer
    fog_amount: float = 0.42  # atmospheric attenuation at the backmost layer
    depth_dim: float = 0.55  # luminance retained at the backmost layer
    depth_desat: float = 0.45  # chroma retained at the backmost layer
    extinction: float = 2.6  # Beer-Lambert coefficient

    # Luminance. Defaults sit in the "dark ground, moderate luminous filament"
    # register.
    background_luma: float = 0.030
    filament_luma: float = 0.360  # user-adjustable filament brightness
    glow_gamma: float = 0.78  # <1 lifts faint structure without raising peaks
    l_max: float = 0.620  # hard ceiling on Oklab L

    # Chroma and hue.
    c_max: float = 0.145
    chroma_activity_gain: float = 5.5
    chroma_floor: float = 0.012
    hue_turns_per_hour: float = 1.33  # ~45 min per full rotation
    hue_anchor: float = 0.0  # radians; set by the palette macro
    hue_spread: float = 0.85  # spatial hue variation, radians


@dataclass
class Params:
    """Complete primitive parameter set."""

    sim_hz: float = 20.0
    max_fps: int = 30
    agents: AgentParams = field(default_factory=AgentParams)
    reaction: ReactionParams = field(default_factory=ReactionParams)
    flow: FlowParams = field(default_factory=FlowParams)
    pigment: PigmentParams = field(default_factory=PigmentParams)
    climate: ClimateParams = field(default_factory=ClimateParams)
    homeostat: HomeostatParams = field(default_factory=HomeostatParams)
    events: EventParams = field(default_factory=EventParams)
    render: RenderParams = field(default_factory=RenderParams)
    safety: SafetyParams = field(default_factory=SafetyParams)


@dataclass
class Macros:
    """The seven knobs the control panel exposes. All in 0..1."""

    intensity: float = 0.50
    scale: float = 0.50
    tempo: float = 0.45
    palette: float = 0.30
    brightness: float = 0.35
    filament_glow: float = 0.45
    depth: float = 0.60


# --------------------------------------------------------------------------
# Macro -> primitive curves
# --------------------------------------------------------------------------

# path, low value, high value, gamma. The macro is raised to ``gamma`` before
# lerping, so gamma > 1 gives finer control at the low end.
MACRO_CURVES: dict[str, list[tuple[str, float, float, float]]] = {
    "intensity": [
        ("agents.density", 0.10, 0.44, 1.0),
        ("agents.deposit", 0.009, 0.028, 1.0),
        ("agents.fusion_bias", 0.35, 0.72, 1.0),
        ("reaction.trail_feed_gain", 0.012, 0.034, 1.0),
        ("events.rate_per_hour", 2.5, 14.0, 1.3),
        ("render.chroma_activity_gain", 3.5, 8.0, 1.0),
        ("pigment.inject_rate", 0.032, 0.085, 1.0),
    ],
    "scale": [
        # Larger scale == coarser features: slower agents, longer sensors,
        # faster diffusion.
        ("agents.sensor_distance", 4.0, 12.0, 1.0),
        ("agents.trail_diffuse", 0.85, 1.90, 1.0),
        ("reaction.du", 0.16, 0.26, 1.0),
        ("reaction.dv", 0.080, 0.130, 1.0),
        ("flow.psi_noise_scale", 2.0, 5.0, 1.0),
    ],
    "tempo": [
        ("sim_hz", 12.0, 26.0, 1.0),
        ("agents.speed", 0.55, 1.35, 1.0),
        ("flow.psi_gain", 0.70, 2.10, 1.0),
        ("flow.field_gain", 0.45, 1.30, 1.0),
        ("flow.psi_theta", 0.0012, 0.0038, 1.0),
        ("climate.advect_gain", 0.12, 0.38, 1.0),
        ("render.hue_turns_per_hour", 0.55, 2.60, 1.2),
    ],
    "palette": [
        # Palette selects a hue anchor; the spatial spread widens slightly at
        # the extremes so the ends of the range are not flat monochrome.
        ("render.hue_spread", 0.55, 1.25, 1.0),
    ],
    "brightness": [
        ("render.background_luma", 0.012, 0.075, 1.0),
        ("render.l_max", 0.44, 0.78, 1.0),
        ("safety.exposure_target", 0.10, 0.26, 1.0),
    ],
    "filament_glow": [
        # The user-facing "how luminous are the filaments" control.
        ("render.filament_luma", 0.16, 0.62, 1.0),
        ("render.glow_gamma", 0.92, 0.62, 1.0),
        ("render.extinction", 1.9, 3.6, 1.0),
    ],
    "depth": [
        ("render.parallax", 0.006, 0.038, 1.0),
        ("render.dof_radius", 1.2, 5.4, 1.0),
        ("render.fog_amount", 0.18, 0.62, 1.0),
        ("render.depth_dim", 0.78, 0.38, 1.0),
        ("render.depth_desat", 0.72, 0.30, 1.0),
    ],
}

# Macro values that are not a simple lerp.
def _palette_hue_anchor(v: float) -> float:
    """Palette macro 0..1 maps to a full hue circle."""
    return v * TAU


# --------------------------------------------------------------------------
# Hard safety ceilings -- see DESIGN.md §7
# --------------------------------------------------------------------------

SAFETY_CEILINGS: dict[str, tuple[float, float]] = {
    # path: (minimum, maximum)
    "safety.max_luma_delta": (0.0005, 0.030),
    "safety.max_chroma_delta": (0.0005, 0.100),
    "safety.iir_alpha": (0.02, 1.000),
    "safety.exposure_attack": (0.0, 0.050),
    "safety.exposure_release": (0.0, 0.050),
    "safety.exposure_target": (0.02, 0.400),
    "render.l_max": (0.05, 0.900),
    "render.c_max": (0.0, 0.220),
    "render.filament_luma": (0.0, 0.900),
    "render.background_luma": (0.0, 0.300),
    "events.max_radius_frac": (0.0, 0.250),
    "sim_hz": (4.0, 60.0),
    "max_fps": (5, 60),
}


# --------------------------------------------------------------------------
# Dataclass path helpers
# --------------------------------------------------------------------------


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    current = getattr(obj, parts[-1])
    # Preserve integer-typed primitives (substeps, psi_scale, layers, ...).
    if isinstance(current, bool):
        setattr(obj, parts[-1], bool(value))
    elif isinstance(current, int):
        setattr(obj, parts[-1], int(round(value)))
    else:
        setattr(obj, parts[-1], float(value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class Config:
    """Macros plus explicit primitive overrides.

    ``resolve()`` produces the :class:`Params` the engine consumes.
    """

    macros: Macros = field(default_factory=Macros)
    overrides: dict[str, float] = field(default_factory=dict)
    preset_name: str = "default"

    def resolve(self) -> Params:
        params = Params()

        for macro_name, curves in MACRO_CURVES.items():
            value = getattr(self.macros, macro_name)
            value = min(1.0, max(0.0, float(value)))
            for path, lo, hi, gamma in curves:
                t = value**gamma if gamma != 1.0 else value
                set_path(params, path, _lerp(lo, hi, t))

        # Non-lerp macro effects.
        params.render.hue_anchor = _palette_hue_anchor(
            min(1.0, max(0.0, self.macros.palette))
        )

        # Explicit overrides win over macros.
        for path, value in self.overrides.items():
            try:
                set_path(params, path, value)
            except AttributeError:
                log.warning("unknown parameter override %r, ignoring", path)

        validate(params)
        return params


def validate(params: Params) -> Params:
    """Clamp every safety-relevant value to its hard ceiling.

    Clamps rather than raises: this runs on hot-reload of a file the user is
    editing by hand during a multi-hour session, and killing the session over a
    typo would be a worse outcome than silently correcting it.
    """
    for path, (lo, hi) in SAFETY_CEILINGS.items():
        value = get_path(params, path)
        clamped = min(hi, max(lo, value))
        if clamped != value:
            log.warning(
                "parameter %s = %g is outside the permitted range [%g, %g]; "
                "clamped to %g",
                path,
                value,
                lo,
                hi,
                clamped,
            )
            set_path(params, path, clamped)

    # Structural values that are not floats.
    params.render.layers = max(1, min(5, int(params.render.layers)))
    params.reaction.substeps = max(1, min(8, int(params.reaction.substeps)))
    params.flow.psi_scale = max(1, min(16, int(params.flow.psi_scale)))
    params.events.max_concurrent = max(0, min(8, int(params.events.max_concurrent)))
    return params


# --------------------------------------------------------------------------
# Parameter ramping -- no parameter change is ever a step. DESIGN.md §9.
# --------------------------------------------------------------------------

# Per-path time constants in seconds; anything unlisted uses the default.
RAMP_TAU_DEFAULT = 1.5
RAMP_TAU: dict[str, float] = {
    "render.background_luma": 4.0,
    "render.filament_luma": 4.0,
    "render.l_max": 4.0,
    "render.c_max": 4.0,
    "safety.exposure_target": 6.0,
    "render.hue_turns_per_hour": 8.0,
    "render.fog_amount": 3.0,
    "render.depth_dim": 3.0,
    "render.depth_desat": 3.0,
    # Structural / integer values snap immediately; ramping them is meaningless.
    "sim_hz": 0.5,
    "render.hue_anchor": 12.0,
}

# Paths whose value is an angle in radians and must ramp along the shortest
# arc. Lerping these linearly would send the palette knob the long way round
# the colour circle -- a slow but very visible global hue sweep.
CIRCULAR_PATHS = frozenset({"render.hue_anchor"})


class ParamRamp:
    """Exponentially approaches a target parameter set.

    Adjusting a control must not itself cause visual punctuation, so every
    float reaching the GPU is smoothed. Integers and bools snap, since they are
    structural (layer count, substeps) rather than perceptual.
    """

    def __init__(self, params: Params) -> None:
        self.current = copy.deepcopy(params)
        self.target = copy.deepcopy(params)

    def set_target(self, params: Params) -> None:
        self.target = copy.deepcopy(params)

    def update(self, dt: float) -> Params:
        _ramp_dataclass(self.current, self.target, dt, prefix="")
        return self.current

    def snap(self, params: Params) -> None:
        self.current = copy.deepcopy(params)
        self.target = copy.deepcopy(params)


def _ramp_dataclass(cur: Any, tgt: Any, dt: float, prefix: str) -> None:
    for f in fields(cur):
        name = f.name
        path = f"{prefix}{name}"
        cur_value = getattr(cur, name)
        tgt_value = getattr(tgt, name)

        if is_dataclass(cur_value):
            _ramp_dataclass(cur_value, tgt_value, dt, prefix=f"{path}.")
        elif isinstance(cur_value, bool) or isinstance(cur_value, int):
            setattr(cur, name, tgt_value)
        elif isinstance(cur_value, float):
            tau = RAMP_TAU.get(path, RAMP_TAU_DEFAULT)
            alpha = 1.0 - math.exp(-dt / max(tau, 1e-4))
            if path in CIRCULAR_PATHS:
                delta = (tgt_value - cur_value + math.pi) % TAU - math.pi
                setattr(cur, name, (cur_value + delta * alpha) % TAU)
            else:
                setattr(cur, name, cur_value + (tgt_value - cur_value) * alpha)


# --------------------------------------------------------------------------
# TOML persistence
# --------------------------------------------------------------------------

_HEADER = """\
# Anastomosis configuration.
#
# Edit and save: changes are hot-reloaded, and every parameter is ramped
# smoothly rather than stepped, so it is safe to adjust while running.
#
# [macros] are the normal interface -- seven knobs, all 0..1.
# [overrides] pins individual primitive parameters by dotted path, e.g.
#   "render.filament_luma" = 0.42
# Overrides take precedence over macros.
#
# Safety-relevant values are clamped to hard ceilings on load (see
# DESIGN.md §7); an out-of-range value is corrected with a warning rather
# than rejected.
"""


def load(path: str | Path) -> Config:
    import tomllib

    path = Path(path)
    if not path.exists():
        log.info("no config at %s, using defaults", path)
        return Config()

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    macros = Macros()
    for key, value in (data.get("macros") or {}).items():
        if hasattr(macros, key):
            setattr(macros, key, min(1.0, max(0.0, float(value))))
        else:
            log.warning("unknown macro %r in %s, ignoring", key, path)

    overrides = {str(k): v for k, v in (data.get("overrides") or {}).items()}
    return Config(
        macros=macros,
        overrides=overrides,
        preset_name=str(data.get("preset_name", "default")),
    )


def save(config: Config, path: str | Path) -> None:
    import tomlkit

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = tomlkit.document()
    for line in _HEADER.splitlines():
        doc.add(tomlkit.comment(line.lstrip("#").rstrip()))
    doc.add(tomlkit.nl())

    doc.add("preset_name", config.preset_name)

    macros = tomlkit.table()
    for f in fields(config.macros):
        macros.add(f.name, round(float(getattr(config.macros, f.name)), 4))
    doc.add("macros", macros)

    overrides = tomlkit.table()
    for key, value in sorted(config.overrides.items()):
        overrides.add(key, value)
    doc.add("overrides", overrides)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    tmp.replace(path)  # atomic, so a hot-reload never sees a half-written file
    log.info("wrote config to %s", path)


def default_config_path() -> Path:
    import os

    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "anastomosis" / "config.toml"
