"""The panel's side of the parameter model -- DESIGN.md §9.

Two things are worth pinning without standing up a window for them: that every
macro is reachable from the panel, and that the one macro shown as something
other than a bare number says the right thing. Both are the kind of mistake
that is invisible until someone opens the panel and finds a knob missing or a
readout that disagrees with the simulation.

Most import the panel module, which needs PySide6, but never construct a
widget -- so they run headless and cost nothing. The few that do build a panel
use the `qt_panel` fixture, which stands a plain object in for the application:
what is being checked there is the panel's own wiring, and a real application
would drag a GPU device in behind it.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

pytest.importorskip("PySide6", reason="the control panel needs PySide6")

from anastomosis import config  # noqa: E402
from anastomosis.ui import control_panel  # noqa: E402


@pytest.fixture
def qt_panel():
    """A control panel over a stand-in application, built offscreen.

    The stand-in carries exactly the attributes the panel reads, which is the
    point: a panel that reached for anything else would fail here rather than
    in front of a user.
    """
    import types

    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from anastomosis import events as events_module
    from anastomosis.ui.control_panel import ControlPanel

    qapp = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    cfg = config.Config(backend="volumetric", volume_detail="fine")
    app = types.SimpleNamespace(
        config=cfg,
        backend="volumetric",
        volume_detail="fine",
        params=cfg.resolve(),
        scheduler=events_module.EventScheduler(seed=1),
        engine=None,
        _frame_times=[],
        _sim_hz_scale=1.0,
        checkpoint_status=lambda: "off",
    )
    panel = ControlPanel(app)
    yield panel, app
    panel._timer.stop()
    panel.deleteLater()
    qapp.processEvents()


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


def test_the_slab_size_control_follows_the_backend(qt_panel):
    """It is greyed out rather than hidden under the layered view.

    There is no slab to size there, but a control that vanishes is harder to
    find again than one that is visible and says why it is unavailable.
    """
    panel, app = qt_panel
    assert panel.detail_combo.isEnabled()
    assert panel.detail_combo.currentData() == "fine"

    app.backend = "layered"
    panel._load_from_app()
    assert not panel.detail_combo.isEnabled()
    assert "volumetric" in panel.detail_combo.toolTip()

    app.backend = "volumetric"
    panel._load_from_app()
    assert panel.detail_combo.isEnabled()


def test_the_status_line_says_when_a_saved_field_is_not_the_chosen_size(qt_panel):
    """A resumed slab keeps the size it grew at, so the combo can lead.

    Without this the user picks a size in the config, relaunches, sees the old
    picture and the new name, and has no way to tell that the setting landed
    and is simply waiting for a reset.
    """
    import types

    panel, app = qt_panel
    app.engine = types.SimpleNamespace(
        tick_count=10,
        geometry=types.SimpleNamespace(
            width=512, describe=lambda: "512x288x48 voxels (7.1 M)"),
        read_stats=lambda: {"mean_v": 0.1, "mean_activity": 0.001},
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
