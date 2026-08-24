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

Most import the panel module, which needs PySide6, but never construct a
widget -- so they run headless and cost nothing. The few that do build a panel
hand it `panelstub.PanelApp`, which is the one stand-in every panel test
shares: what is being checked there is the panel's own wiring, and a real
application would drag a GPU device in behind it.
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
# The mode selector -- DESIGN.md §14
# ---------------------------------------------------------------------------


def _mode_panel(monkeypatch):
    """The panel offscreen, with any confirmation dialog turned into a failure.

    A mode switch is ramped and loses nothing, so it must never ask -- a
    dialog appearing here is a regression, and the test should say so rather
    than hang on it.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import panelstub

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def refuse(*_a, **_k):
        raise AssertionError("a mode switch asked a question; it must not")

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", staticmethod(refuse))
    app = panelstub.PanelApp()
    return app, control_panel.ControlPanel(app)


def test_the_mode_selector_lists_both_modes():
    """One entry per mode, keyed by the names the config understands."""
    listed = [name for name, _label, _tip in control_panel.MODE_LABELS]
    assert listed == list(config.MODES)


def test_switching_the_mode_reaches_the_app_and_refilters_the_presets(monkeypatch):
    app, panel = _mode_panel(monkeypatch)
    from anastomosis import presets

    # The list is filtered by *world* as well as by mode (DESIGN.md §15), so
    # the expectation is asked of the backend this panel is actually driving
    # rather than of the whole preset table -- offering `meadow` under a
    # fungal view would offer a picture nobody has judged.
    world = presets.world_for_backend(app.backend)

    assert panel.mode_combo.currentData() == "regulation"
    assert panel.preset_combo.count() == len(presets.names("regulation", world))

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("activation"))
    panel._on_mode()
    assert app.config.mode == "activation"
    # The preset list is the new mode's -- asserted against whatever that
    # mode actually has rather than against a count, so it holds both before
    # and after a mode gains presets. A mode with none says so rather than
    # offering regulation presets through the wrong table.
    names = presets.names("activation", world)
    assert panel.preset_combo.count() == len(names)
    assert panel.preset_combo.isEnabled() == bool(names)

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("regulation"))
    panel._on_mode()
    assert app.config.mode == "regulation"
    assert panel.preset_combo.count() == len(presets.names("regulation", world))
    assert panel.preset_combo.isEnabled()
    panel.close()


def test_the_rhizotron_offers_its_own_presets_and_no_mode(monkeypatch):
    """DESIGN.md §15: the root world has one tuning and its own presets.

    Both halves matter to the panel. The preset list follows the *world*, so
    `meadow` is offered here and `dense` is not -- a preset is macro
    positions plus the table they were tuned against plus the world that
    renders them. And the Mode selector, which under this backend would
    change nothing at all (`Config.resolve` pins the rhizotron to the
    regulation table), is disabled and says why rather than sitting there
    inviting a click that does nothing.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import panelstub
    from anastomosis import presets

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app = panelstub.PanelApp(backend="rhizotron")
    panel = control_panel.ControlPanel(app)

    listed = [panel.preset_combo.itemText(i)
              for i in range(panel.preset_combo.count())]
    assert listed == presets.names("regulation", "rhizotron")
    assert "meadow" in listed
    assert "dense" not in listed
    assert panel.preset_combo.isEnabled()

    assert not panel.mode_combo.isEnabled()
    assert "one tuning" in panel.mode_combo.toolTip()
    panel.close()


def test_the_readouts_quote_the_active_modes_curve(monkeypatch):
    """A slider readout must describe the table that is actually driving.

    The divergence is injected rather than taken from the shipped tables, so
    that the assertion is about the readout following the mode at all rather
    than about today's endpoints: with the activation event-rate curve pinned
    to a known value, the same slider position must read differently the
    moment the mode changes.
    """
    tweaked = {
        macro: list(entries)
        for macro, entries in config.MODE_CURVES["activation"].items()
    }
    tweaked["event_rate"] = [("events.rate_per_hour", 60.0, 60.0, 1.0)]
    monkeypatch.setitem(config.MODE_CURVES, "activation", tweaked)

    app, panel = _mode_panel(monkeypatch)
    slider = panel.sliders[control_panel.EVENT_RATE_MACRO]
    slider.setValue(control_panel.SLIDER_STEPS // 2)
    before = panel.values[control_panel.EVENT_RATE_MACRO].text()

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("activation"))
    panel._on_mode()
    after = panel.values[control_panel.EVENT_RATE_MACRO].text()
    assert before != after
    assert after == control_panel.describe_interval(60.0)
    panel.close()


def test_every_slab_size_has_a_control():
    """A size the config offers and the panel does not is a size nobody finds.

    The panel builds its list from `VOLUME_DETAIL_LABELS` and the application
    reads `config.VOLUME_DETAIL`, so the two drifting apart is invisible until
    somebody opens the panel looking for a size that is missing -- or picks
    one the application does not know.
    """
    labelled = [name for name, _label, _tip in control_panel.VOLUME_DETAIL_LABELS]
    assert labelled == list(config.VOLUME_DETAIL), (
        "the panel's slab sizes and the config's have drifted apart"
    )
    for name, label, tip in control_panel.VOLUME_DETAIL_LABELS:
        # The voxel count belongs in the label: it is the number anyone
        # comparing these against a GPU budget will be looking for.
        assert str(config.VOLUME_DETAIL[name]) in label, label
        assert tip.strip(), f"slab size {name!r} has no explanation"


def test_the_slab_size_control_follows_the_backend(monkeypatch):
    """It is greyed out rather than hidden under the layered view.

    There is no slab to size there, but a control that vanishes is harder to
    find again than one that is visible and says why it is unavailable.
    """
    app, panel = _detail_panel(monkeypatch, None)
    assert panel.detail_combo.isEnabled()
    assert panel.detail_combo.currentData() == "fine"

    app.backend = "layered"
    panel._load_from_app()
    assert not panel.detail_combo.isEnabled()
    assert "volumetric" in panel.detail_combo.toolTip()

    app.backend = "volumetric"
    panel._load_from_app()
    assert panel.detail_combo.isEnabled()
    panel.close()


def test_the_status_line_says_when_a_saved_field_is_not_the_chosen_size(
    monkeypatch,
):
    """A resumed slab keeps the size it grew at, so the combo can lead.

    Without this the user picks a size in the config, relaunches, sees the old
    picture and the new name, and has no way to tell that the setting landed
    and is simply waiting for a reset.
    """
    import types

    import panelstub

    app, panel = _detail_panel(monkeypatch, None)
    app.engine = types.SimpleNamespace(
        tick_count=10,
        geometry=types.SimpleNamespace(
            width=512, describe=lambda: "512x288x48 voxels (7.1 M)"),
        read_stats=lambda: panelstub.stats(mean_v=0.1, mean_activity=0.001),
    )

    # Chosen "fine" (768), running a saved 512 field: the panel must say so.
    panel._refresh_status()
    text = panel.status_labels["depth"].text()
    assert "512x288x48" in text
    assert "768" in text, f"no hint that the chosen size is waiting: {text!r}"

    # Once they agree there is nothing to explain, and the row stays quiet.
    app.volume_detail = "standard"
    panel._refresh_status()
    assert "reset" not in panel.status_labels["depth"].text()
    panel.close()


def test_a_wider_slab_reopens_the_thickness_travel(monkeypatch):
    """The two slab controls meet here, and the panel has to follow.

    The thickness ceiling is the shorter lateral axis, so choosing a wider
    slab raises it. A slider whose range were fixed when the panel was built
    would go on refusing a thickness the geometry would now allow.
    """
    from PySide6 import QtWidgets

    app, panel = _detail_panel(
        monkeypatch, QtWidgets.QMessageBox.Yes, volume_detail="standard")
    narrow = panel.thickness_slider.maximum()

    panel.detail_combo.setCurrentIndex(panel.detail_combo.findData("finest"))
    panel._on_volume_detail()

    assert app.volume_detail == "finest"
    assert panel.thickness_slider.maximum() > narrow, (
        "the thickness travel did not follow the wider slab"
    )
    low, high = app.volume_depth_limits()
    assert panel.thickness_slider.maximum() * control_panel.DEPTH_STEP == high
    panel.close()


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


def _detail_panel(monkeypatch, answer, backend="volumetric",
                  volume_detail="fine"):
    """The panel offscreen, with the confirmation answered rather than shown."""
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import panelstub

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: answer)
    )
    app = panelstub.PanelApp(backend, volume_detail=volume_detail)
    return app, control_panel.ControlPanel(app)


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


# ---------------------------------------------------------------------------
# The parallax readout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reach, expected",
    [
        (0.0, "still"),
        (0.002, "still"),
        (0.075, "8% of width"),
        (0.25, "25% of width"),
    ],
)
def test_the_parallax_readout_reads_as_english(reach, expected):
    """A share of the screen's width, because that is what is on offer: the
    near material slides this far against the far material."""
    assert control_panel.describe_parallax(reach) == expected


def test_the_parallax_readout_stops_where_the_slab_does(monkeypatch):
    """The readout is the effective travel, not the requested one.

    Under the volumetric backend a thin slab holds the viewpoint down, and a
    control that kept climbing while nothing on screen changed would be lying.
    Stopping is the panel saying "the Thickness slider is what is holding this",
    which it says better than a tooltip can.
    """
    app, panel = _thickness_panel(monkeypatch, None, backend="volumetric")
    slider = panel.sliders[control_panel.PARALLAX_MACRO]

    slider.setValue(control_panel.SLIDER_STEPS)
    thin = panel._parallax_reach(1.0)
    asked = config.curve_value(
        control_panel.PARALLAX_MACRO, control_panel.PARALLAX_PATH, 1.0)
    assert thin < asked, "a 48-deep slab let the whole swing through"

    app.set_volume_depth(288)
    assert panel._parallax_reach(1.0) > thin, (
        "a deeper slab did not unlock more of the travel")
    assert panel._parallax_reach(1.0) == pytest.approx(asked)

    # The layered backend has no slab to be held by.
    other, other_panel = _thickness_panel(monkeypatch, None, backend="layered")
    assert other_panel._parallax_reach(1.0) == pytest.approx(asked)
    panel.close()
    other_panel.close()


# ---------------------------------------------------------------------------
# The resonance extras -- DESIGN.md §16
# ---------------------------------------------------------------------------


def test_the_resonance_extras_appear_only_under_resonance(monkeypatch):
    """The filament option and the capture status are not settings the other
    modes have, so they are hidden -- not greyed -- outside resonance, and
    under the rhizotron, which resolves through regulation whatever the mode
    key says."""
    app, panel = _mode_panel(monkeypatch)
    assert panel.filaments_check.isHidden()
    assert panel.audio_status.isHidden()

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("resonance"))
    panel._on_mode()
    assert app.config.mode == "resonance"
    assert not panel.filaments_check.isHidden()
    assert not panel.audio_status.isHidden()
    # The status line says something true about the drive rather than
    # sitting empty -- here, that it has not been started.
    assert panel.audio_status.text() == app.audio.describe()
    assert panel.audio_status.text()

    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("regulation"))
    panel._on_mode()
    assert panel.filaments_check.isHidden()
    assert panel.audio_status.isHidden()
    panel.close()


def test_the_filament_checkbox_reaches_the_app_and_reads_back(monkeypatch):
    app, panel = _mode_panel(monkeypatch)
    panel.mode_combo.setCurrentIndex(panel.mode_combo.findData("resonance"))
    panel._on_mode()

    assert app.config.filaments is True
    assert panel.filaments_check.isChecked()
    panel.filaments_check.setChecked(False)
    assert app.config.filaments is False
    panel.filaments_check.setChecked(True)
    assert app.config.filaments is True

    # And the sync direction: state arriving from the app (a config reload)
    # lands on the checkbox without echoing back as a second toggle.
    app.config.filaments = False
    calls = []
    original = app.set_filaments
    app.set_filaments = lambda on: calls.append(on) or original(on)
    panel._load_from_app()
    assert not panel.filaments_check.isChecked()
    assert calls == [], "syncing from the app echoed back into it"
    panel.close()


def test_the_rhizotron_hides_the_resonance_extras_too(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    import panelstub

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app = panelstub.PanelApp(backend="rhizotron")
    app.config.mode = "resonance"
    panel = control_panel.ControlPanel(app)
    assert panel.filaments_check.isHidden()
    assert panel.audio_status.isHidden()
    panel.close()
