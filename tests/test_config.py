"""Parameter model: macros, safety clamping, ramping, persistence."""

from __future__ import annotations

import math

import pytest

from dataclasses import fields

from anastomosis import config, presets


def test_macros_span_their_declared_range():
    """Each macro must actually move every primitive it claims to drive."""
    for macro_name, curves in config.MACRO_CURVES.items():
        low = config.Config(macros=config.Macros(**{macro_name: 0.0})).resolve()
        high = config.Config(macros=config.Macros(**{macro_name: 1.0})).resolve()
        for path, lo, hi, _gamma in curves:
            if lo == hi:
                continue
            got_low = config.get_path(low, path)
            got_high = config.get_path(high, path)
            assert got_low != got_high, (
                f"macro {macro_name!r} does not move {path}"
            )


def test_every_macro_drives_something():
    """A knob on the panel that reaches no primitive is a knob that lies."""
    from dataclasses import fields

    for f in fields(config.Macros):
        assert f.name in config.MACRO_CURVES, (
            f"macro {f.name!r} has no curve, so moving it would do nothing"
        )


def test_curve_value_agrees_with_resolve():
    """The panel reads slider positions through this; it must not drift.

    `curve_value` exists so the control panel can say what a position means
    without copying the curve's constants, which is only worth anything if it
    stays the same function `resolve` applies.
    """
    for macro_name, curves in config.MACRO_CURVES.items():
        for path, _lo, _hi, _gamma in curves:
            for step in range(0, 11):
                value = step / 10.0
                resolved = config.Config(
                    macros=config.Macros(**{macro_name: value})
                ).resolve()
                assert config.curve_value(macro_name, path, value) == pytest.approx(
                    config.get_path(resolved, path)
                ), f"{macro_name} -> {path} at {value}"


def test_curve_value_rejects_a_path_its_macro_does_not_drive():
    with pytest.raises(KeyError):
        config.curve_value("event_rate", "render.l_max", 0.5)


# --- event rate ------------------------------------------------------------


def test_the_event_rate_is_its_own_knob():
    """`intensity` no longer moves it; `event_rate` does, and monotonically.

    The two were one control, which meant asking for a denser network also
    asked for more interruptions. Separating them is only real if intensity
    has genuinely let go of the rate.
    """
    quiet = config.Config(macros=config.Macros(intensity=0.0)).resolve()
    busy = config.Config(macros=config.Macros(intensity=1.0)).resolve()
    assert quiet.events.rate_per_hour == pytest.approx(busy.events.rate_per_hour)

    rates = [
        config.Config(macros=config.Macros(event_rate=step / 20.0))
        .resolve().events.rate_per_hour
        for step in range(21)
    ]
    assert rates == sorted(rates)
    assert rates[0] < rates[-1]


def test_the_event_rate_knob_spans_a_useful_range_of_intervals():
    """Rare enough to be a background, frequent enough to be company."""
    low = config.Config(macros=config.Macros(event_rate=0.0)).resolve()
    high = config.Config(macros=config.Macros(event_rate=1.0)).resolve()

    slowest_minutes = 60.0 / low.events.rate_per_hour
    fastest_minutes = 60.0 / high.events.rate_per_hour
    assert 60.0 <= slowest_minutes <= 240.0, (
        f"the slow end is one every {slowest_minutes:.0f} min, which is either "
        "not restful or not distinguishable from off"
    )
    assert 2.0 <= fastest_minutes <= 6.0, (
        f"the fast end is one every {fastest_minutes:.0f} min"
    )


def test_the_centre_of_the_event_rate_knob_is_the_designed_interval():
    """DESIGN.md 4.3 specifies a mean inter-arrival of about eight minutes.

    That is the setting the field was tuned at, so it has to be what an
    untouched slider and an untouched config give -- and where the hand lands
    when someone drags the knob back to the middle.
    """
    for macros in (config.Macros(), presets.get("default")):
        params = config.Config(macros=macros).resolve()
        minutes = 60.0 / params.events.rate_per_hour
        assert 7.0 <= minutes <= 9.0, f"{minutes:.1f} min at the centre detent"


def test_the_event_rate_moves_nothing_but_the_rate():
    """Frequency is not amplitude. Nothing that shapes an event may move.

    This is what makes the knob safe to expose without a ceiling of its own:
    at the top of its travel the field spends more of its time inside an
    event, but no event is bigger, stronger or faster than it was.
    """
    from dataclasses import fields as dataclass_fields

    low = config.Config(macros=config.Macros(event_rate=0.0)).resolve()
    high = config.Config(macros=config.Macros(event_rate=1.0)).resolve()
    for f in dataclass_fields(config.EventParams):
        if f.name == "rate_per_hour":
            continue
        assert getattr(low.events, f.name) == getattr(high.events, f.name), (
            f"the event rate knob moved events.{f.name}"
        )
    assert low.render == high.render and low.agents == high.agents


def test_presets_keep_the_arrival_rate_they_had():
    """The rate used to come from `intensity`; presets must not have shifted.

    Each preset implied an arrival rate through its intensity, and someone
    returning to `quiet` after this change should find the same one. These are
    the rates the old curve produced -- lerp(2.5, 14.0, intensity ** 1.3).
    """
    for name in presets.names():
        macros = presets.get(name)
        was = 2.5 + 11.5 * (macros.intensity**1.3)
        now = config.Config(macros=macros).resolve().events.rate_per_hour
        assert now == pytest.approx(was, rel=0.05), (
            f"preset {name} went from {was:.2f} to {now:.2f} events/hour"
        )


def test_overrides_beat_macros():
    cfg = config.Config(
        macros=config.Macros(filament_glow=1.0),
        overrides={"render.filament_luma": 0.123},
    )
    assert cfg.resolve().render.filament_luma == pytest.approx(0.123)


def test_unknown_override_is_ignored_not_fatal():
    """A typo in a hand-edited config must not kill a multi-day session."""
    cfg = config.Config(overrides={"render.no_such_parameter": 1.0})
    cfg.resolve()  # must not raise


@pytest.mark.parametrize("path", sorted(config.SAFETY_CEILINGS))
def test_every_safety_ceiling_is_enforced(path):
    lo, hi = config.SAFETY_CEILINGS[path]
    above = config.Config(overrides={path: hi * 10 + 1}).resolve()
    below = config.Config(overrides={path: lo - abs(lo) - 1000}).resolve()
    assert config.get_path(above, path) <= hi
    assert config.get_path(below, path) >= lo


def test_validate_clamps_in_place():
    params = config.Params()
    params.safety.max_luma_delta = 99.0
    config.validate(params)
    assert params.safety.max_luma_delta == config.SAFETY_CEILINGS[
        "safety.max_luma_delta"][1]


def test_diffusion_band_is_bounded_for_stability():
    """A hand-edited du_max must not be able to blow up the explicit scheme."""
    cfg = config.Config(overrides={"reaction.du_max": 40.0, "reaction.du_min": 9.0})
    params = cfg.resolve()
    assert params.reaction.du_max * params.reaction.dt < 1.0, (
        "du_max was accepted past the explicit-diffusion stability limit"
    )
    assert params.reaction.du_min <= params.reaction.du_max, (
        "the du band was left inverted, so clamp_du would collapse it"
    )


def test_structural_values_stay_integers():
    params = config.Config(overrides={
        "render.layers": 99, "reaction.substeps": 0, "flow.psi_scale": 100,
    }).resolve()
    assert isinstance(params.render.layers, int) and 1 <= params.render.layers <= 5
    assert params.reaction.substeps >= 1
    assert 1 <= params.flow.psi_scale <= 16


# --- ramping ---------------------------------------------------------------


def test_ramp_never_steps():
    start = config.Config(macros=config.Macros(brightness=0.0)).resolve()
    end = config.Config(macros=config.Macros(brightness=1.0)).resolve()
    ramp = config.ParamRamp(start)
    ramp.set_target(end)

    previous = start.render.background_luma
    for _ in range(20):
        current = ramp.update(1 / 30).render.background_luma
        # One frame must never cover most of the distance.
        assert abs(current - previous) < abs(
            end.render.background_luma - start.render.background_luma) * 0.5
        previous = current


def test_ramp_converges():
    start = config.Config(macros=config.Macros(intensity=0.0)).resolve()
    end = config.Config(macros=config.Macros(intensity=1.0)).resolve()
    ramp = config.ParamRamp(start)
    ramp.set_target(end)
    for _ in range(2000):
        current = ramp.update(1 / 30)
    assert current.agents.density == pytest.approx(end.agents.density, abs=1e-4)


def test_hue_ramps_the_short_way_round():
    """A palette change must not sweep the long way through the colour circle."""
    params = config.Config().resolve()
    ramp = config.ParamRamp(params)
    ramp.current.render.hue_anchor = 0.1
    target = config.Config().resolve()
    target.render.hue_anchor = config.TAU - 0.1
    ramp.set_target(target)

    first = ramp.update(0.5).render.hue_anchor
    # Going the short way means decreasing through the 0/TAU wrap, so the value
    # either drops slightly or wraps to just under TAU. It must not climb.
    assert first < 0.1 or first > config.TAU - 0.5


def test_integers_snap_rather_than_ramping():
    start = config.Config().resolve()
    end = config.Config().resolve()
    end.render.layers = 1
    ramp = config.ParamRamp(start)
    ramp.set_target(end)
    assert ramp.update(1 / 30).render.layers == 1


# --- persistence -----------------------------------------------------------


def test_toml_roundtrip(tmp_path):
    cfg = config.Config(
        macros=presets.get("deep"),
        overrides={"render.filament_luma": 0.42},
        preset_name="deep",
    )
    path = tmp_path / "config.toml"
    config.save(cfg, path)
    loaded = config.load(path)
    assert loaded.macros == cfg.macros
    assert loaded.preset_name == "deep"
    assert loaded.overrides["render.filament_luma"] == pytest.approx(0.42)


def test_missing_file_yields_defaults(tmp_path):
    loaded = config.load(tmp_path / "absent.toml")
    assert loaded.macros == config.Macros()


def test_out_of_range_macro_in_file_is_clamped(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[macros]\nintensity = 5.0\nbrightness = -2.0\n")
    loaded = config.load(path)
    assert loaded.macros.intensity == 1.0
    assert loaded.macros.brightness == 0.0


def test_a_file_predating_the_event_rate_keeps_the_rate_it_implied(tmp_path):
    """Upgrading must not change how often an existing setup is interrupted.

    Before the split, `intensity` drove the arrival rate; a file that turned
    the intensity down was asking to be left alone, and reading it as the
    default event rate would nearly double the interruptions on a setup
    someone had already settled on.
    """
    path = tmp_path / "legacy.toml"
    path.write_text("[macros]\nintensity = 0.24\nbrightness = 0.22\n")
    params = config.Config(macros=config.load(path).macros).resolve()
    was = 2.5 + 11.5 * (0.24**1.3)
    assert params.events.rate_per_hour == pytest.approx(was, rel=0.02)


def test_an_explicit_event_rate_is_never_second_guessed(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[macros]\nintensity = 0.24\nevent_rate = 0.9\n")
    assert config.load(path).macros.event_rate == pytest.approx(0.9)


def test_a_file_that_sets_no_intensity_gets_the_default_event_rate(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text("[macros]\nbrightness = 0.5\n")
    assert config.load(path).macros.event_rate == config.Macros().event_rate


def test_save_is_atomic(tmp_path):
    """Hot-reload must never observe a half-written file."""
    path = tmp_path / "c.toml"
    config.save(config.Config(), path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


# --- presets ---------------------------------------------------------------


@pytest.mark.parametrize("name", presets.names())
def test_presets_resolve_and_stay_safe(name):
    params = config.Config(macros=presets.get(name)).resolve()
    for path, (lo, hi) in config.SAFETY_CEILINGS.items():
        value = config.get_path(params, path)
        assert lo <= value <= hi, f"preset {name} put {path} out of range"


@pytest.mark.parametrize("name", presets.names())
def test_presets_are_in_the_dark_register(name):
    """All presets must keep a dark ground; that was an explicit requirement."""
    params = config.Config(macros=presets.get(name)).resolve()
    assert params.render.background_luma < 0.09, (
        f"preset {name} has a background lightness of "
        f"{params.render.background_luma:.3f}, which is not a dark ground"
    )
    assert params.render.filament_luma > params.render.background_luma


def test_preset_get_returns_a_copy():
    a = presets.get("quiet")
    a.intensity = 0.99
    assert presets.get("quiet").intensity != 0.99


def test_hue_anchor_covers_the_circle():
    low = config.Config(macros=config.Macros(palette=0.0)).resolve()
    high = config.Config(macros=config.Macros(palette=1.0)).resolve()
    assert low.render.hue_anchor == pytest.approx(0.0)
    assert high.render.hue_anchor == pytest.approx(math.tau)


def test_the_sensing_reach_stays_inside_the_band_it_is_stable_in():
    """DESIGN.md §4.9: sensing reach is bounded by the width of what it senses.

    An agent that senses much further than a strand is wide cuts corners hard
    enough to straighten the strand it is following, and a straight strand on a
    torus closes on itself after one lap and is then reinforced every lap. Past
    a ratio of about four the whole population ends up on one such strand.

    The bound has to hold in three places at once, which is why this asserts all
    three: the base value, the `scale` macro that overrides it -- the macro used
    to sweep from 4.7 to 6.3 and so was over the threshold along its entire
    length, default included -- and the climate deviation on top of that, which
    reaches its own clamp in the tails. The last of those is the shader's job;
    :func:`config.clamp_sensor_distance` mirrors it.
    """
    for step in range(21):
        cfg = config.Config()
        cfg.macros.scale = step / 20.0
        params = cfg.resolve()
        agents = params.agents
        ceiling = agents.sensor_reach_max * agents.trail_diffuse

        ratio = agents.sensor_distance / agents.trail_diffuse
        assert ratio < 2.8, (
            f"scale={cfg.macros.scale:.2f} sets a sensing reach of "
            f"{ratio:.2f}x the trail width; measured, 2.5 holds a network and "
            f"3.2 was still half-condensed on a small field"
        )

        spread = params.climate.range_sensor_distance
        for deviation in (-1.0, -0.5, 0.5, 1.0):
            reach = config.clamp_sensor_distance(
                agents.sensor_distance + spread * deviation, agents)
            assert 1.0 <= reach <= ceiling + 1e-9, (
                f"a climate deviation of {deviation:+.1f} reaches {reach:.2f}, "
                f"outside the clamp's own band"
            )

    # And the ceiling itself must stay under what was measured to survive.
    assert config.Config().resolve().agents.sensor_reach_max <= 3.7


# ---------------------------------------------------------------------------
# The parallax split
# ---------------------------------------------------------------------------


def test_the_depth_macro_no_longer_moves_the_viewpoint():
    """`depth` and `parallax` answer different questions and must not overlap.

    Everything left under `depth` is a shading trick applied to a *normalised*
    depth -- how much the far face is fogged, dimmed, desaturated and blurred --
    and says the same thing about that face however far away it is. The
    viewpoint's travel is the one cue that comes from the scene moving, and it
    is the one somebody turns up when the shading is not enough on its own. Two
    paths driven by two macros would leave whichever resolved last silently
    winning.
    """
    paths = {path for path, *_ in config.MACRO_CURVES["depth"]}
    assert "render.parallax" not in paths
    assert "render.parallax_tau" not in paths

    parallax_paths = {path for path, *_ in config.MACRO_CURVES["parallax"]}
    assert parallax_paths == {"render.parallax", "render.parallax_tau"}

    # Moving `depth` across its whole travel must leave the viewpoint alone.
    reaches = {
        config.Config(macros=config.Macros(depth=v)).resolve().render.parallax
        for v in (0.0, 0.5, 1.0)
    }
    assert len(reaches) == 1


def test_the_parallax_macro_spans_still_to_unmistakable():
    """A knob that cannot reach far enough to settle the question is not much
    use, and this one exists because the question was unsettled."""
    off = config.Config(macros=config.Macros(parallax=0.0)).resolve().render
    full = config.Config(macros=config.Macros(parallax=1.0)).resolve().render
    assert off.parallax == 0.0, "the bottom of the travel must be a still camera"
    # A quarter of the screen's width between the near and far material.
    assert full.parallax >= 0.2
    # And more travel comes with more speed, since a knob that moved only the
    # travel would take four times as long to show twice as much.
    assert full.parallax_tau < off.parallax_tau


def test_a_config_from_before_the_split_takes_the_new_default(tmp_path):
    """There is nothing to carry across from a pre-split file.

    `depth` did drive `render.parallax`, but over a range chosen against a walk
    that never moved, so what the old file says about parallax describes a
    setting that did nothing. Inheriting it would preserve a bug's
    configuration; the new default stands instead.
    """
    path = tmp_path / "config.toml"
    path.write_text(
        "preset_name = \"default\"\n"
        "[macros]\ndepth = 0.9\n[overrides]\n",
        encoding="utf-8",
    )
    loaded = config.load(path)
    assert loaded.macros.depth == 0.9
    assert loaded.macros.parallax == config.Macros().parallax

    # An explicit value is of course kept.
    path.write_text(
        "preset_name = \"default\"\n"
        "[macros]\ndepth = 0.9\nparallax = 0.2\n[overrides]\n",
        encoding="utf-8",
    )
    assert config.load(path).macros.parallax == 0.2


@pytest.mark.parametrize("name", presets.names())
def test_every_preset_names_every_macro(name):
    """The module says it does, and a macro a preset forgets is one that snaps
    to its default the moment somebody reaches for that preset."""
    import ast
    import inspect

    source = inspect.getsource(presets)
    tree = ast.parse(source)
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(value, ast.Call):
                    calls[key.value] = {kw.arg for kw in value.keywords}
    named = calls.get(name, set())
    for field in fields(config.Macros):
        assert field.name in named, (
            f"preset {name} does not name {field.name}, so choosing it would "
            "silently reset that knob")
