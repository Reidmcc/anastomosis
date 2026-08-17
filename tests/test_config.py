"""Parameter model: macros, safety clamping, ramping, persistence."""

from __future__ import annotations

import math

import pytest

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
