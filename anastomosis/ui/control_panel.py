"""Qt control panel.

Deliberately an ordinary window: **not** always-on-top, and freely minimizable.
The visual lives on a secondary display while the user works on the main one,
so the panel must be able to get out of the way completely and come back when
wanted. An always-on-top palette would be exactly wrong for that.

It shares the render window's Qt event loop (``rendercanvas.qt``), so there is
no second process, no IPC, and no chance of the two disagreeing about
parameter state.

Every change routes through the same ramping path as a config file edit, so
dragging a slider can never itself produce visual punctuation.
"""

from __future__ import annotations

import logging
from dataclasses import fields

from PySide6 import QtCore, QtWidgets

from .. import presets as presets_module
from ..config import Macros

log = logging.getLogger(__name__)

SLIDER_STEPS = 1000

# Ordered for the panel, with a plain-language description of what each does.
MACRO_LABELS: list[tuple[str, str, str]] = [
    ("intensity", "Intensity", "How much is happening: density, event rate"),
    ("scale", "Scale", "Feature size, from fine filaments to broad forms"),
    ("tempo", "Tempo", "Speed of flow, drift, and colour rotation"),
    ("palette", "Palette", "Where the colour range sits on the hue circle"),
    ("brightness", "Brightness", "Overall level and the background"),
    ("filament_glow", "Filament glow", "How luminous the filaments are"),
    ("depth", "Depth", "Parallax, focus falloff, and atmosphere"),
]


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._updating = False

        self.setWindowTitle("anastomosis — controls")
        # No always-on-top flag: this window must be able to disappear behind
        # whatever the user is actually working on.
        self.setMinimumWidth(360)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_presets())
        layout.addWidget(self._build_macros())
        layout.addWidget(self._build_status())
        layout.addStretch(1)
        layout.addLayout(self._build_buttons())

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(1000)

        self._load_from_app()

    # -- construction -------------------------------------------------------

    def _build_presets(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Preset")
        row = QtWidgets.QHBoxLayout(box)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(presets_module.names())
        self.preset_combo.activated.connect(self._on_preset)
        row.addWidget(self.preset_combo, 1)
        return box

    def _build_macros(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Adjust")
        grid = QtWidgets.QGridLayout(box)
        grid.setVerticalSpacing(6)
        self.sliders: dict[str, QtWidgets.QSlider] = {}
        self.values: dict[str, QtWidgets.QLabel] = {}

        for row, (name, label, tip) in enumerate(MACRO_LABELS):
            caption = QtWidgets.QLabel(label)
            caption.setToolTip(tip)

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, SLIDER_STEPS)
            slider.setToolTip(tip)
            slider.valueChanged.connect(self._on_slider)
            slider.setTracking(True)

            value = QtWidgets.QLabel("0.00")
            value.setMinimumWidth(38)
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

            grid.addWidget(caption, row, 0)
            grid.addWidget(slider, row, 1)
            grid.addWidget(value, row, 2)
            self.sliders[name] = slider
            self.values[name] = value

        return box

    def _build_status(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Status")
        form = QtWidgets.QFormLayout(box)
        self.status_labels = {}
        for key, label in (
            ("runtime", "Running for"),
            ("field", "Field"),
            ("rate", "Sim / frame"),
            ("events", "Events"),
            ("checkpoint", "Saved state"),
        ):
            widget = QtWidgets.QLabel("—")
            widget.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            form.addRow(label, widget)
            self.status_labels[key] = widget
        return box

    def _build_buttons(self) -> QtWidgets.QLayout:
        row = QtWidgets.QHBoxLayout()
        save = QtWidgets.QPushButton("Save as default")
        save.setToolTip("Write the current settings to the config file")
        save.clicked.connect(self._on_save)
        row.addWidget(save)

        # The counterpart to resuming: sessions continue by default, so there has
        # to be an explicit way to say "start again", and it belongs here rather
        # than only on the command line.
        reset = QtWidgets.QPushButton("Reset simulation")
        reset.setToolTip(
            "Discard the current field and its saved state, and grow a new one "
            "from seeds"
        )
        reset.clicked.connect(self._on_reset)
        row.addWidget(reset)

        hide = QtWidgets.QPushButton("Minimise")
        hide.setToolTip("Tuck the panel away; reopen it from the taskbar")
        hide.clicked.connect(self.showMinimized)
        row.addWidget(hide)
        return row

    # -- state --------------------------------------------------------------

    def _load_from_app(self) -> None:
        self._updating = True
        macros = self.app.config.macros
        for name, slider in self.sliders.items():
            value = float(getattr(macros, name))
            slider.setValue(int(round(value * SLIDER_STEPS)))
            self.values[name].setText(f"{value:.2f}")
        index = self.preset_combo.findText(self.app.config.preset_name)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        self._updating = False

    def _current_macros(self) -> Macros:
        macros = Macros()
        for field in fields(macros):
            slider = self.sliders.get(field.name)
            if slider is not None:
                setattr(macros, field.name, slider.value() / SLIDER_STEPS)
        return macros

    # -- handlers -----------------------------------------------------------

    def _on_slider(self) -> None:
        if self._updating:
            return
        macros = self._current_macros()
        for name, slider in self.sliders.items():
            self.values[name].setText(f"{slider.value() / SLIDER_STEPS:.2f}")
        self.app.apply_macros(macros)

    def _on_preset(self) -> None:
        name = self.preset_combo.currentText()
        try:
            macros = presets_module.get(name)
        except KeyError:
            return
        self._updating = True
        for field_name, slider in self.sliders.items():
            slider.setValue(int(round(float(getattr(macros, field_name)) * SLIDER_STEPS)))
            self.values[field_name].setText(f"{getattr(macros, field_name):.2f}")
        self._updating = False
        self.app.config.preset_name = name
        # Ramped like any other change, so switching presets is a slow
        # transition rather than a cut.
        self.app.apply_macros(macros)

    def _on_save(self) -> None:
        self.app.config.macros = self._current_macros()
        try:
            self.app.save_config()
            self.status_labels["runtime"].setText("saved")
        except Exception as exc:
            log.error("could not save config: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Could not save", str(exc))

    def _on_reset(self) -> None:
        # Hours of accumulated field state, and nothing brings it back once the
        # checkpoint is gone, so this asks first.
        answer = QtWidgets.QMessageBox.question(
            self,
            "Reset simulation",
            "Start a new simulation from seeds?\n\n"
            "The current field and the saved state are discarded. The image "
            "settles down and grows back over a few minutes.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            self.app.reset_simulation()
        except Exception as exc:
            log.error("could not reset the simulation: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Could not reset", str(exc))

    def _refresh_status(self) -> None:
        engine = self.app.engine
        if engine is None:
            return
        try:
            stats = engine.read_stats()
        except Exception:
            return

        ticks = engine.tick_count
        seconds = ticks / max(self.app.params.sim_hz, 1e-3)
        hours, rest = divmod(int(seconds), 3600)
        minutes, secs = divmod(rest, 60)
        self.status_labels["runtime"].setText(
            f"{hours}h {minutes:02d}m {secs:02d}s   ({ticks:,} ticks)"
        )
        self.status_labels["field"].setText(
            f"density {stats['mean_v']:.3f}   activity {stats['mean_activity']:.5f}"
        )
        window = self.app._frame_times[-30:] or [0.0]
        frame_ms = 1000.0 * sum(window) / len(window)
        self.status_labels["rate"].setText(
            f"{self.app.params.sim_hz * self.app._sim_hz_scale:.1f} Hz   "
            f"{frame_ms:.1f} ms/frame"
        )
        self.status_labels["events"].setText(self.app.scheduler.describe())
        self.status_labels["checkpoint"].setText(self.app.checkpoint_status())

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        self._timer.stop()
        super().closeEvent(event)
