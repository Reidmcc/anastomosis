"""The panel's side of the parameter model -- DESIGN.md §9.

Three things are worth pinning without standing up a window for them: that
every macro is reachable from the panel, that the controls shown as something
other than a bare number say the right thing, and that the slab's thickness
slider can only propose thicknesses the geometry would actually build. All are
the kind of mistake that is invisible until someone opens the panel and finds a
knob missing, a readout that disagrees with the simulation, or a slider that
does not land where it says it does.

The thickness row is then exercised as a widget, offscreen and against the
shared stand-in application, because what it does is destructive: it discards a
field that may have been growing for days. That it asks first, that declining
really does leave the field alone, and that it is dead under a backend with no
slab to reshape are properties of the wiring rather than of any function, so
they are checked through the wiring.

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


# ---------------------------------------------------------------------------
# The slab's thickness
# ---------------------------------------------------------------------------


def test_the_thickness_slider_counts_in_steps_the_geometry_honours():
    """A slider position that rounds to a different slab is a control that
    does not land where it says it does. The geometry rounds the thickness up
    to a multiple of eight, so the slider counts in eights."""
    from anastomosis import volume as volume_module

    params = config.Config().resolve()
    low, high = volume_module.depth_limits(2560, 1440, params)
    assert low % control_panel.DEPTH_STEP == 0
    assert high % control_panel.DEPTH_STEP == 0

    for position in range(low // control_panel.DEPTH_STEP,
                          high // control_panel.DEPTH_STEP + 1):
        wanted = position * control_panel.DEPTH_STEP
        params.volume.depth = wanted
        built = volume_module.VolumeGeometry.derive(2560, 1440, params)
        assert built.depth == wanted, (
            f"the slider can ask for {wanted} and would get {built.depth}")


def test_the_thickness_readout_prices_the_slab_it_describes():
    from anastomosis import volume as volume_module

    params = config.Config().resolve()
    geometry = volume_module.VolumeGeometry.derive(2560, 1440, params)
    line = control_panel.describe_slab(geometry)
    assert "512 x 288 x 48" in line
    assert control_panel.describe_bytes(geometry.field_bytes) in line


@pytest.mark.parametrize(
    "count, expected",
    [
        (650 << 20, "650 MB"),
        (1 << 30, "1.0 GB"),
        ((1 << 30) * 39 // 10, "3.9 GB"),
    ],
)
def test_the_memory_readout_reads_as_english(count, expected):
    assert control_panel.describe_bytes(count) == expected


# ---------------------------------------------------------------------------
# The thickness row, as a widget
# ---------------------------------------------------------------------------


def _thickness_panel(monkeypatch, answer, backend="volumetric"):
    """The panel offscreen, with the confirmation answered rather than shown."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import panelstub

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: answer)
    )

    class App(panelstub.PanelApp):
        def __init__(self):
            super().__init__(backend)
            self.asked: list[int] = []

        def set_volume_depth(self, depth: int) -> int:
            self.asked.append(int(depth))
            return super().set_volume_depth(depth)

    app = App()
    return app, control_panel.ControlPanel(app)


def test_the_thickness_row_offers_the_range_the_geometry_allows(monkeypatch):
    """The slider spans the whole travel and starts where the field is."""
    app, panel = _thickness_panel(monkeypatch, None)
    low, high = app.volume_depth_limits()
    assert panel.thickness_slider.minimum() * control_panel.DEPTH_STEP == low
    assert panel.thickness_slider.maximum() * control_panel.DEPTH_STEP == high
    assert (panel.thickness_slider.value() * control_panel.DEPTH_STEP
            == app.volume_slab().depth)
    # Nothing to do until the slider has been moved off where the field is.
    assert not panel.thickness_apply.isEnabled()
    assert control_panel.describe_slab(app.volume_slab()) == (
        panel.thickness_note.text())
    panel.close()


def test_confirming_a_new_thickness_grows_a_new_slab(monkeypatch):
    from PySide6 import QtWidgets

    app, panel = _thickness_panel(monkeypatch, QtWidgets.QMessageBox.Yes)
    wanted = 144
    panel.thickness_slider.setValue(wanted // control_panel.DEPTH_STEP)
    assert panel.thickness_apply.isEnabled(), (
        "a thickness other than the running one and nothing to press")
    assert str(wanted) in panel.thickness_note.text()

    panel.thickness_apply.click()
    assert app.asked == [wanted]
    # And the row settles onto the new thickness, with nothing left to apply.
    assert (panel.thickness_slider.value() * control_panel.DEPTH_STEP) == wanted
    assert not panel.thickness_apply.isEnabled()
    panel.close()


def test_declining_leaves_the_field_at_the_thickness_it_had(monkeypatch):
    """Hours of accumulated field, so "no" has to mean nothing happened."""
    from PySide6 import QtWidgets

    app, panel = _thickness_panel(monkeypatch, QtWidgets.QMessageBox.No)
    was = panel.thickness_slider.value()
    panel.thickness_slider.setValue(144 // control_panel.DEPTH_STEP)
    panel.thickness_apply.click()
    assert app.asked == []
    assert panel.thickness_slider.value() == was, "the slider kept the refusal"
    panel.close()


def test_the_thickness_row_is_dead_under_the_layered_backend(monkeypatch):
    """There is no slab to reshape, and a control that silently does nothing is
    worse than one that says why it cannot."""
    app, panel = _thickness_panel(monkeypatch, None, backend="layered")
    assert not panel.thickness_slider.isEnabled()
    assert not panel.thickness_apply.isEnabled()
    assert panel.thickness_note.text() == control_panel.THICKNESS_FOR_LAYERED
    panel.close()
