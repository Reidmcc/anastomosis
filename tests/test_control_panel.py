"""The panel's side of the parameter model -- DESIGN.md §9.

Two things are worth pinning without standing up a window for them: that every
macro is reachable from the panel, and that the one macro shown as something
other than a bare number says the right thing. Both are the kind of mistake
that is invisible until someone opens the panel and finds a knob missing or a
readout that disagrees with the simulation.

These import the panel module, which needs PySide6, but never construct a
widget -- so they run headless and cost nothing.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

pytest.importorskip("PySide6", reason="the control panel needs PySide6")

from anastomosis import config  # noqa: E402
from anastomosis.ui import control_panel  # noqa: E402


def test_every_macro_has_a_control():
    """A macro with no control is a parameter the user cannot reach.

    The panel builds its sliders from `MACRO_LABELS` plus the event-rate row,
    so a macro added to the dataclass without an entry in either is one that
    exists in the config file and nowhere else.
    """
    labelled = {name for name, _label, _tip in control_panel.MACRO_LABELS}
    labelled.add(control_panel.EVENT_RATE_MACRO)
    for f in fields(config.Macros):
        assert f.name in labelled, (
            f"macro {f.name!r} has no control, so it can only be set by hand"
        )


def test_no_control_names_a_macro_that_does_not_exist():
    names = {f.name for f in fields(config.Macros)}
    for name, _label, _tip in control_panel.MACRO_LABELS:
        assert name in names
    assert control_panel.EVENT_RATE_MACRO in names


def test_the_event_rate_control_drives_the_arrival_rate():
    """The readout resolves the path the row claims to move."""
    assert control_panel.EVENT_RATE_PATH == "events.rate_per_hour"
    config.curve_value(
        control_panel.EVENT_RATE_MACRO, control_panel.EVENT_RATE_PATH, 0.5
    )  # must not raise: the macro must actually drive that path


@pytest.mark.parametrize(
    "rate_per_hour, expected",
    [
        (0.5, "about one every 2.0 h"),
        (2.0, "about one every 30 min"),
        (7.5, "about one every 8 min"),
        (20.0, "about one every 3 min"),
        (60.0, "about one a minute"),
    ],
)
def test_the_interval_readout_reads_as_english(rate_per_hour, expected):
    assert control_panel.describe_interval(rate_per_hour) == expected


def test_the_interval_readout_survives_a_rate_of_zero():
    """`validate` has no floor for this path, so zero can reach the readout."""
    assert control_panel.describe_interval(0.0)  # must not divide by zero


def test_the_readout_agrees_with_the_simulation_across_the_whole_travel():
    """What the label says and what the scheduler is given must be one number.

    The panel reads the curve directly rather than the live parameters, which
    are mid-ramp; the risk in doing that is the two drifting apart.
    """
    for step in range(21):
        value = step / 20.0
        resolved = config.Config(
            macros=config.Macros(event_rate=value)
        ).resolve().events.rate_per_hour
        from_curve = config.curve_value(
            control_panel.EVENT_RATE_MACRO, control_panel.EVENT_RATE_PATH, value
        )
        assert from_curve == pytest.approx(resolved)
        assert control_panel.describe_interval(from_curve) == (
            control_panel.describe_interval(resolved)
        )
