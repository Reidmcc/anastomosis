"""Parameter model: macros, safety clamping, ramping, persistence."""

from __future__ import annotations

import math

import pytest

from dataclasses import fields

from anastomosis import config, presets


@pytest.mark.parametrize("mode", config.MODES)
def test_macros_span_their_declared_range(mode):
    """Each macro must actually move every primitive it claims to drive."""
    for macro_name, curves in config.MODE_CURVES[mode].items():
        low = config.Config(
            macros=config.Macros(**{macro_name: 0.0}), mode=mode).resolve()
        high = config.Config(
            macros=config.Macros(**{macro_name: 1.0}), mode=mode).resolve()
        for path, lo, hi, _gamma in curves:
            if lo == hi:
                continue
            got_low = config.get_path(low, path)
            got_high = config.get_path(high, path)
            assert got_low != got_high, (
                f"macro {macro_name!r} does not move {path} in {mode} mode"
            )


@pytest.mark.parametrize("mode", config.MODES)
def test_every_macro_drives_something(mode):
    """A knob on the panel that reaches no primitive is a knob that lies."""
    from dataclasses import fields

    for f in fields(config.Macros):
        assert f.name in config.MODE_CURVES[mode], (
            f"macro {f.name!r} has no {mode} curve, so moving it would do "
            "nothing in that mode"
        )


@pytest.mark.parametrize("mode", config.MODES)
def test_curve_value_agrees_with_resolve(mode):
    """The panel reads slider positions through this; it must not drift.

    `curve_value` exists so the control panel can say what a position means
    without copying the curve's constants, which is only worth anything if it
    stays the same function `resolve` applies -- per mode, since the readout
    quotes the table the mode is actually driving through.
    """
    for macro_name, curves in config.MODE_CURVES[mode].items():
        for path, _lo, _hi, _gamma in curves:
            for step in range(0, 11):
                value = step / 10.0
                resolved = config.Config(
                    macros=config.Macros(**{macro_name: value}), mode=mode
                ).resolve()
                assert config.curve_value(
                    macro_name, path, value, mode=mode
                ) == pytest.approx(
                    config.get_path(resolved, path)
                ), f"{macro_name} -> {path} at {value} in {mode} mode"


def test_curve_value_rejects_a_path_its_macro_does_not_drive():
    with pytest.raises(KeyError):
        config.curve_value("event_rate", "render.l_max", 0.5)


# --- modes -----------------------------------------------------------------


def test_the_mode_picks_the_curve_table(monkeypatch):
    """`resolve` reads the active mode's table and no other.

    Shown with a deliberately divergent table rather than with the shipped
    values, so that what is asserted is the *wiring* rather than today's
    endpoints: this must keep failing if the mode is wired to nothing, however
    far the two tables happen to have diverged (DESIGN.md §14.8 step 3).
    """
    tweaked = {
        macro: list(entries)
        for macro, entries in config.MODE_CURVES["activation"].items()
    }
    tweaked["palette"] = [("render.hue_spread", 2.0, 3.0, 1.0)]
    monkeypatch.setitem(config.MODE_CURVES, "activation", tweaked)

    macros = config.Macros(palette=0.5)
    regulation = config.Config(macros=macros, mode="regulation").resolve()
    activation = config.Config(macros=macros, mode="activation").resolve()

    assert config.get_path(activation, "render.hue_spread") == pytest.approx(2.5)
    # And the regulation table is untouched by the divergence.
    assert config.get_path(regulation, "render.hue_spread") == pytest.approx(
        config.curve_value("palette", "render.hue_spread", 0.5, mode="regulation")
    )


def test_the_modes_share_their_structure():
    """Both tables drive the same macros, and the same paths under each.

    The activation endpoints are free to move -- that is the whole point of
    the second table (DESIGN.md §14.3) -- but its *shape* is not: a macro that
    drove a primitive in one mode and not the other would mean a slider going
    dead, or gaining a hidden effect, when the mode changes.
    """
    regulation = config.MODE_CURVES["regulation"]
    activation = config.MODE_CURVES["activation"]
    assert set(regulation) == set(activation)
    for macro in regulation:
        assert [p for p, *_ in regulation[macro]] == [
            p for p, *_ in activation[macro]
        ], f"macro {macro!r} drives different paths in the two modes"


def test_the_regulation_table_is_the_shipped_one():
    """`MACRO_CURVES` and the regulation entry must stay the same object.

    Everything historical -- the legacy event-rate inversion, every measured
    value in DESIGN.md -- means that table specifically, and a copy would let
    the two drift apart silently.
    """
    assert config.MODE_CURVES["regulation"] is config.MACRO_CURVES


def test_the_safety_ceilings_are_mode_blind():
    """One ceiling table serves both modes -- DESIGN.md §14.7.

    Trivially true today, because `validate` never looks at the mode; this
    holds the door shut on a future where activation is given its own, looser
    validation path.
    """
    for path, (lo, hi) in config.SAFETY_CEILINGS.items():
        above = [
            config.get_path(
                config.Config(mode=mode, overrides={path: hi * 10 + 1}).resolve(),
                path,
            )
            for mode in config.MODES
        ]
        below = [
            config.get_path(
                config.Config(
                    mode=mode, overrides={path: lo - abs(lo) - 1000}
                ).resolve(),
                path,
            )
            for mode in config.MODES
        ]
        assert above[0] == above[1] == pytest.approx(hi), path
        assert below[0] == below[1] == pytest.approx(lo), path


def test_an_unknown_mode_is_normalised_rather_than_trusted(caplog):
    """A typo in the config is a warning and the default, never a crash."""
    import logging

    with caplog.at_level(logging.WARNING):
        assert config.normalise_mode("acivation") == config.DEFAULT_MODE
    assert any("unknown mode" in r.message for r in caplog.records)
    # And the accepted names pass through untouched, whitespace and case aside.
    for mode in config.MODES:
        assert config.normalise_mode(mode.upper() + " ") == mode


def test_the_mode_survives_a_toml_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    config.save(config.Config(mode="activation"), path)
    assert config.load(path).mode == "activation"


def test_a_file_predating_the_mode_is_a_regulation_file(tmp_path):
    """Every config written before §14 was written for the regulation mode."""
    path = tmp_path / "config.toml"
    path.write_text("[macros]\nintensity = 0.5\n")
    assert config.load(path).mode == "regulation"


def test_the_activation_table_is_a_tuning_not_a_copy():
    """Step 3 landed: the two modes genuinely differ.

    A refactor that quietly reverted the activation table to a copy would
    leave every mode test green -- the structure tests are *meant* to pass on
    a copy -- and the second mode a placebo.
    """
    assert config.ACTIVATION_CURVES != config.MACRO_CURVES


def test_the_activation_tempo_tops_stay_inside_the_swept_certificate():
    """The motion endpoints must not outrun their measurement.

    Step 2's sweep certified the WCAG area criterion out to 6x the regulation
    tops on the three motion primitives (DESIGN.md §14.8) -- and to nothing
    beyond that. A future retune past the certificate needs a new sweep, and
    this is the test that says so; `tests/tempo_sweep.py` is how.
    """
    motion = {"agents.speed", "flow.psi_gain", "flow.field_gain"}
    reg = {p: hi for p, _lo, hi, _g in config.MACRO_CURVES["tempo"]}
    act = {p: hi for p, _lo, hi, _g in config.ACTIVATION_CURVES["tempo"]}
    for path in motion:
        assert act[path] <= 6.0 * reg[path], (
            f"{path} tops out at {act[path]}, beyond the swept 6x of "
            f"{reg[path]} -- re-run the sweep before shipping this"
        )


def test_the_activation_scale_respects_the_measured_reaction_floor():
    """§4.7: du below ~0.17 walks activity toward the homeostat's floor, and
    0.16 -- the shipped low end -- is already at the edge. Activation biases
    the knob finer at the *top*; its floor must not dig below regulation's."""
    reg = {p: (lo, hi) for p, lo, hi, _g in config.MACRO_CURVES["scale"]}
    act = {p: (lo, hi) for p, lo, hi, _g in config.ACTIVATION_CURVES["scale"]}
    assert act["reaction.du"][0] >= reg["reaction.du"][0]
    # And the dv/du ratio §4.7 requires (moving it moves mass, which drags
    # the exposure governor into a global luminance swing) holds at both ends.
    for lo_hi in (0, 1):
        ratio = act["reaction.dv"][lo_hi] / act["reaction.du"][lo_hi]
        assert ratio == pytest.approx(0.50, abs=0.01)


# --- the polychrome palette -- DESIGN.md §14.4 ------------------------------


def test_polychrome_is_identically_zero_at_gain_zero():
    """Gain zero is the regulation mapping, bit for bit, not merely nearby."""
    import reference

    import numpy as np

    c = np.linspace(-1.0, 1.0, 2001)
    assert np.all(reference.polychrome_offset(c, 0.0, 0.06) == 0.0)


def test_polychrome_reaches_three_distinct_families():
    """Plateaus at -120, 0 and +120 degrees, reachable at the climate's own
    realised extremes (~+-0.44 -- §4.1), not just at the [-1, 1] clamp."""
    import reference

    import numpy as np

    well = 2.0 * math.pi / 3.0
    off = reference.polychrome_offset(np.array([-0.44, 0.0, 0.44]), 1.0, 0.06)
    assert off[0] == pytest.approx(-well, abs=1e-3)
    assert off[1] == pytest.approx(0.0, abs=1e-12)
    assert off[2] == pytest.approx(well, abs=1e-3)
    # Half gain, half separation: the gain scales the triad rather than
    # gating it, so the ramp on the parameter is a smooth widening.
    assert reference.polychrome_offset(
        np.array([0.44]), 0.5, 0.06)[0] == pytest.approx(well / 2, abs=1e-3)


def test_polychrome_is_a_smooth_staircase_not_a_threshold():
    """"No thresholds anywhere in shading" (§1) applies to the warp.

    Monotone, and with its slope bounded by the analytic maximum
    gain * well * k -- a jump would show up as a slope far past it. The
    climate's bilinear smoothness then makes the spatial transition at least
    a climate texel wide on top of this.
    """
    import reference

    import numpy as np

    c = np.linspace(-1.0, 1.0, 40001)
    off = reference.polychrome_offset(c, 1.0, 0.06)
    slopes = np.diff(off) / np.diff(c)
    assert np.all(slopes >= -1e-12), "the staircase must be monotone"
    k = 2.5 / 0.06
    ceiling = (2.0 * math.pi / 3.0) * k
    assert slopes.max() <= ceiling * 1.01, "slope beyond the analytic bound"


def test_regulation_cannot_reach_the_polychrome_and_activation_can():
    for value in (0.0, 0.5, 1.0):
        macros = config.Macros(intensity=value)
        reg = config.Config(macros=macros, mode="regulation").resolve()
        assert reg.render.polychrome == 0.0
    act = config.Config(
        macros=config.Macros(intensity=1.0), mode="activation").resolve()
    assert act.render.polychrome == pytest.approx(1.0)


def test_a_preset_asked_for_at_launch_brings_its_mode(tmp_path):
    """`--preset spark` must not resolve spark's macros through the
    regulation table: a preset is positions plus the table they were tuned
    against, so the CLI carries the mode into the config it writes."""
    import subprocess
    import sys

    for name, mode in (("spark", "activation"), ("quiet", "regulation")):
        path = tmp_path / f"{name}.toml"
        result = subprocess.run(
            [sys.executable, "-m", "anastomosis", "--write-config",
             "--preset", name, "--config", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        loaded = config.load(path)
        assert loaded.mode == mode
        assert loaded.preset_name == name


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


@pytest.mark.parametrize("mode", config.MODES)
def test_the_event_rate_moves_nothing_but_the_rate(mode):
    """Frequency is not amplitude. Nothing that shapes an event may move.

    This is what makes the knob safe to expose without a ceiling of its own:
    at the top of its travel the field spends more of its time inside an
    event, but no event is bigger, stronger or faster than it was. Per mode,
    because the activation table carries the concurrency cap on this macro --
    as a *constant*, which this test is what keeps honest.
    """
    from dataclasses import fields as dataclass_fields

    low = config.Config(macros=config.Macros(event_rate=0.0), mode=mode).resolve()
    high = config.Config(macros=config.Macros(event_rate=1.0), mode=mode).resolve()
    for f in dataclass_fields(config.EventParams):
        if f.name == "rate_per_hour":
            continue
        assert getattr(low.events, f.name) == getattr(high.events, f.name), (
            f"the event rate knob moved events.{f.name}"
        )
    assert low.render == high.render and low.agents == high.agents


def test_only_activation_shortens_the_event_envelope():
    """§14.6: the envelope rides the *tempo* macro, in activation only.

    Regulation events take the §4.3 minute-or-two to come up at any tempo;
    activation's build to ~15 s at the fast end, with the slow end shared, so
    the bottom of the travel is the same instrument. The concurrency cap is
    per mode and off every knob's travel.
    """
    for tempo in (0.0, 1.0):
        reg = config.Config(
            macros=config.Macros(tempo=tempo), mode="regulation").resolve()
        assert reg.events.attack_seconds == pytest.approx(45.0)
        assert reg.events.release_seconds == pytest.approx(90.0)
        assert reg.events.max_concurrent == 4

    act_slow = config.Config(
        macros=config.Macros(tempo=0.0), mode="activation").resolve()
    act_fast = config.Config(
        macros=config.Macros(tempo=1.0), mode="activation").resolve()
    assert act_slow.events.attack_seconds == pytest.approx(45.0)
    assert act_fast.events.attack_seconds == pytest.approx(15.0)
    assert act_fast.events.release_seconds == pytest.approx(40.0)
    assert act_slow.events.max_concurrent == act_fast.events.max_concurrent == 6


def test_activation_shears_the_structure_harder_without_disabling_it():
    """Activation carries the trail faster; regulation keeps the shipped rate.

    Trail advection (§4.7 step 6) landed on main at 0.5 while this branch was
    in flight, as the fix that dissolves the trail hubs -- so the regulation
    curve's job here is to hold it *at its default*, not to switch it off. An
    earlier draft of this branch pinned it to zero, which would have disabled
    a shipped fix for a real visual defect in the mode that is the
    application's default; this test is what would now catch that.

    Below 1.0 at the top on purpose: the shear that stretches filaments is the
    difference between the trail's rate and the pigment's, so parity would
    quietly remove the effect it is asking for.
    """
    default = config.AgentParams().trail_advect
    assert default > 0.0, "main ships trail advection on; this test assumes it"

    for tempo in (0.0, 0.5, 1.0):
        reg = config.Config(
            macros=config.Macros(tempo=tempo), mode="regulation").resolve()
        assert reg.agents.trail_advect == pytest.approx(default)

    act_slow = config.Config(
        macros=config.Macros(tempo=0.0), mode="activation").resolve()
    act_fast = config.Config(
        macros=config.Macros(tempo=1.0), mode="activation").resolve()
    assert act_slow.agents.trail_advect == pytest.approx(default)
    assert act_fast.agents.trail_advect > default
    assert act_fast.agents.trail_advect < 1.0


def test_presets_keep_the_arrival_rate_they_had():
    """The rate used to come from `intensity`; presets must not have shifted.

    Each preset implied an arrival rate through its intensity, and someone
    returning to `quiet` after this change should find the same one. These are
    the rates the old curve produced -- lerp(2.5, 14.0, intensity ** 1.3).

    Regulation *fungal* presets only: the activation set and the rhizotron's
    both postdate the split, so there is no old rate for them to keep.
    """
    for name in presets.names("regulation", "fungal"):
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


def test_sensing_cap_ratio_is_bounded_or_off():
    """The cap is a ratio to the equilibrium mean trail; its own sanity here.

    Zero or negative must mean cleanly disabled, like `deposit_cap`. A huge
    ratio is indistinguishable from disabled but would still pack a live
    clamp, so it is bounded. The absolute value and its liveness floor -- a
    cap near the starve threshold reads the whole population as starving --
    are applied where the value is packed, in `Backend._physics_values`.
    """
    off = config.Config(overrides={"agents.sense_cap": -3.0}).resolve().agents
    assert off.sense_cap == 0.0
    huge = config.Config(overrides={"agents.sense_cap": 1e9}).resolve().agents
    assert huge.sense_cap <= 100.0


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
    params = config.Config(
        macros=presets.get(name), mode=presets.mode_of(name)).resolve()
    for path, (lo, hi) in config.SAFETY_CEILINGS.items():
        value = config.get_path(params, path)
        assert lo <= value <= hi, f"preset {name} put {path} out of range"


@pytest.mark.parametrize("name", presets.names())
def test_presets_are_in_the_dark_register(name):
    """All presets must keep a dark ground; that was an explicit requirement."""
    params = config.Config(
        macros=presets.get(name), mode=presets.mode_of(name)).resolve()
    assert params.render.background_luma < 0.09, (
        f"preset {name} has a background lightness of "
        f"{params.render.background_luma:.3f}, which is not a dark ground"
    )
    assert params.render.filament_luma > params.render.background_luma


def test_preset_get_returns_a_copy():
    a = presets.get("quiet")
    a.intensity = 0.99
    assert presets.get("quiet").intensity != 0.99


@pytest.mark.parametrize("name", presets.names())
def test_every_preset_carries_a_mode(name):
    """A preset is macro positions plus the table they were tuned against.

    `mode_of` raises on an untagged or mistagged preset, so calling it for
    every name is the whole assertion: adding a preset without saying which
    mode judged it must fail here rather than resolve through the wrong table.
    """
    assert presets.mode_of(name) in config.MODES


def test_the_preset_lists_partition_by_mode():
    """Filtering by mode loses nothing and mixes nothing.

    Written structurally rather than against today's counts, because the
    counts are meant to change: activation has no presets until §14.8 step 4,
    and this must keep holding when it gains them.
    """
    per_mode = [presets.names(mode) for mode in config.MODES]
    flattened = [name for names in per_mode for name in names]
    assert sorted(flattened) == sorted(presets.names())
    assert len(flattened) == len(set(flattened)), "a preset appears in two modes"


def test_hue_anchor_covers_the_circle():
    low = config.Config(macros=config.Macros(palette=0.0)).resolve()
    high = config.Config(macros=config.Macros(palette=1.0)).resolve()
    assert low.render.hue_anchor == pytest.approx(0.0)
    assert high.render.hue_anchor == pytest.approx(math.tau)


@pytest.mark.parametrize("mode", config.MODES)
def test_the_sensing_reach_stays_inside_the_band_it_is_stable_in(mode):
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

    Per mode, because each mode's `scale` curve moves the reach and the
    diffusion over its own range, and §4.9's bifurcation does not care which
    tuning walked over it.
    """
    for step in range(21):
        cfg = config.Config(mode=mode)
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


# --------------------------------------------------------------------------
# The volumetric slab's size -- the three named widths of `VOLUME_DETAIL`
# --------------------------------------------------------------------------


def test_each_named_slab_size_reaches_the_width_it_names():
    """The name is the interface; `volume.width` is what the geometry reads.

    A tier that resolved to the wrong width would be invisible until somebody
    grew a field at it and wondered why it looked the same.
    """
    for name, width in config.VOLUME_DETAIL.items():
        params = config.Config(volume_detail=name).resolve()
        assert params.volume.width == width, (
            f"volume_detail {name!r} resolved to {params.volume.width}, "
            f"not the {width} it names"
        )


def test_the_named_sizes_are_the_three_that_were_costed():
    """Sizes are a promise about GPU cost, so adding one is a deliberate act.

    Each was chosen against a measured budget (DESIGN.md §8.1) and against the
    others; a fourth appearing here without that work is the thing this
    catches.
    """
    assert config.VOLUME_DETAIL == {
        "standard": 512, "fine": 768, "finest": 1024,
    }
    assert config.DEFAULT_VOLUME_DETAIL == "standard"
    # The default must be what `VolumeParams` already documents, or a config
    # written before this setting existed would silently change size.
    assert (config.VOLUME_DETAIL[config.DEFAULT_VOLUME_DETAIL]
            == config.VolumeParams().width)


def test_every_named_size_is_buildable():
    """Within the clamps, and within core WebGPU's 3D texture limit.

    `validate` bounds `volume.width` to 2048 -- the guaranteed
    `maxTextureDimension3D`. A tier above that would be silently shrunk, which
    is the one failure mode a name cannot make visible.
    """
    for name, width in config.VOLUME_DETAIL.items():
        params = config.Config(volume_detail=name).resolve()
        assert params.volume.width == width <= 2048
        # Multiples of 32, so `VolumeGeometry.derive`'s rounding is a no-op and
        # the width asked for is the width allocated.
        assert width % 32 == 0


@pytest.mark.parametrize(
    "given,expected",
    [
        ("fine", "fine"),
        ("FINEST", "finest"),
        ("  standard  ", "standard"),
        # A bare width, since that is the obvious thing to write next to a key
        # whose neighbours are numbers of voxels.
        ("768", "fine"),
        ("1024", "finest"),
        # Junk, absence, and a size that is not on offer all fall back.
        ("enormous", "standard"),
        ("640", "standard"),
        ("", "standard"),
        (None, "standard"),
    ],
)
def test_a_slab_size_off_disk_is_normalised_rather_than_trusted(given, expected):
    assert config.normalise_volume_detail(given) == expected


def test_an_unknown_slab_size_warns_rather_than_failing_the_launch(caplog):
    """Same reasoning as `normalise_backend`: a typo must not stop the day."""
    with caplog.at_level("WARNING"):
        assert config.normalise_volume_detail("gigantic") == "standard"
    assert "gigantic" in caplog.text


def test_an_explicit_width_override_beats_the_named_size():
    """`[overrides]` is the escape hatch for a size that is not one of three.

    Overrides beat macros everywhere else, and the named size is applied on the
    macro side of that line deliberately -- so a hand-written `volume.width`
    still wins.
    """
    cfg = config.Config(
        volume_detail="finest", overrides={"volume.width": 640},
    )
    assert cfg.resolve().volume.width == 640


def test_the_slab_size_survives_a_toml_roundtrip(tmp_path):
    """It is structural, so losing it on save would change the next field."""
    cfg = config.Config(volume_detail="fine", backend="volumetric")
    path = tmp_path / "config.toml"
    config.save(cfg, path)
    loaded = config.load(path)
    assert loaded.volume_detail == "fine"
    assert loaded.resolve().volume.width == 768


def test_a_config_predating_the_setting_keeps_the_original_size(tmp_path):
    """An upgrade must not silently quadruple somebody's GPU load."""
    path = tmp_path / "config.toml"
    path.write_text('backend = "volumetric"\n[macros]\nintensity = 0.5\n')
    loaded = config.load(path)
    assert loaded.volume_detail == config.DEFAULT_VOLUME_DETAIL
    assert loaded.resolve().volume.width == config.VolumeParams().width
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
