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
import time
from dataclasses import fields

from PySide6 import QtCore, QtGui, QtWidgets

from .. import config as config_module
from .. import events as events_module
from .. import presets as presets_module
from .. import window as window_module
from ..config import Macros

log = logging.getLogger(__name__)

SLIDER_STEPS = 1000

# How long the line under the event buttons keeps its message. Long enough to
# read without hunting for it, short enough that a stale one is never mistaken
# for a report on the event now running.
NOTE_SECONDS = 6.0

# What that line says when it has nothing more particular to report.
NOTE_IDLE = "Ask for a perturbation. It builds over a minute or two."
NOTE_AT_CAP = (
    "As many events are running as the settings allow; the buttons come back "
    "as they fade."
)

# The one macro that does not live in the "Adjust" group: how often events
# arrive belongs beside the buttons that ask for one, not in a column of
# sliders about how the image looks. It is an ordinary macro in every other
# respect -- saved, ramped, carried by presets -- and `_current_macros` finds
# it through `self.sliders` like any other.
EVENT_RATE_MACRO = "event_rate"
EVENT_RATE_PATH = "events.rate_per_hour"

# The two depth backends (DESIGN.md §5), with the difference stated in terms of
# what the user will see rather than of how it is drawn.
BACKEND_LABELS: list[tuple[str, str, str]] = [
    (
        "layered",
        "Layered",
        "Three sheets at different depths, the finest detail. The default.",
    ),
    (
        "volumetric",
        "Volumetric",
        "One continuous volume: material genuinely passes in front of and "
        "behind other material, and casts shade into it. Coarser, and asks "
        "more of the graphics card.",
    ),
]

# Ordered for the panel, with a plain-language description of what each does.
MACRO_LABELS: list[tuple[str, str, str]] = [
    ("intensity", "Intensity", "How much is happening: density, contrast, colour"),
    ("scale", "Scale", "Feature size, from fine filaments to broad forms"),
    ("tempo", "Tempo", "Speed of flow, drift, and colour rotation"),
    ("palette", "Palette", "Where the colour range sits on the hue circle"),
    ("brightness", "Brightness", "Overall level and the background"),
    ("filament_glow", "Filament glow", "How luminous the filaments are"),
    ("depth", "Depth", "Parallax, focus falloff, and atmosphere"),
]

# One entry per kind in ``events.EVENT_KINDS``, in the order the buttons should
# read. A kind with no entry here still gets a button -- built from its own name
# -- so adding one to the simulation cannot silently leave it unreachable from
# the panel; the label is what is missing, not the control.
EVENT_LABELS: dict[str, str] = {
    "bloom": "A productive region: structure densifies and fills in",
    "dieback": "The opposite: material thins out and dissolves",
    "current": "A shift in flow, carrying and stretching what is there",
    "tint": "A regional colour shift, with little structural effect",
    "rift": "The network comes apart across the region, rather than thinning",
}


def describe_interval(rate_per_hour: float) -> str:
    """Plain language for a mean arrival rate.

    Events per hour is the wrong unit for the question the slider answers,
    which is how long the field goes between things happening. The mean is the
    only honest summary of that: arrivals are Poisson, so any particular gap
    can be a good deal shorter or longer than this, and "about" is carrying
    that rather than being polite.
    """
    minutes = 60.0 / max(rate_per_hour, 1e-6)
    if minutes < 1.5:
        return "about one a minute"
    if minutes < 90.0:
        return f"about one every {minutes:.0f} min"
    return f"about one every {minutes / 60.0:.1f} h"


class ControlPanel(QtWidgets.QWidget):
    def __init__(self, app) -> None:
        super().__init__()
        self.app = app
        self._updating = False
        # When the note under the event buttons stops being current.
        self._note_expires = 0.0

        self.setWindowTitle("anastomosis — controls")
        # No always-on-top flag: this window must be able to disappear behind
        # whatever the user is actually working on.
        self.setMinimumWidth(360)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        layout.addWidget(self._build_presets())
        layout.addWidget(self._build_backend())
        layout.addWidget(self._build_macros())
        layout.addWidget(self._build_events())
        layout.addWidget(self._build_status())
        layout.addStretch(1)
        layout.addLayout(self._build_buttons())

        self._bind_fullscreen_key()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh_events)
        self._timer.timeout.connect(self._refresh_status)
        self._timer.start(1000)

        self._load_from_app()
        self._refresh_events()

    # -- construction -------------------------------------------------------

    def _build_presets(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Preset")
        row = QtWidgets.QHBoxLayout(box)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(presets_module.names())
        self.preset_combo.activated.connect(self._on_preset)
        row.addWidget(self.preset_combo, 1)
        return box

    def _build_backend(self) -> QtWidgets.QWidget:
        """How depth is drawn.

        Structural rather than perceptual, so it does not belong among the
        sliders: nothing about it can be ramped, and choosing it grows a new
        field rather than adjusting the one on screen. It asks before doing
        that, for the same reason the reset button does -- except that here the
        field being left is kept, so the answer is much less costly than it
        looks.
        """
        box = QtWidgets.QGroupBox("Depth")
        row = QtWidgets.QHBoxLayout(box)
        self.backend_combo = QtWidgets.QComboBox()
        for name, label, tip in BACKEND_LABELS:
            self.backend_combo.addItem(label, name)
            self.backend_combo.setItemData(
                self.backend_combo.count() - 1, tip, QtCore.Qt.ToolTipRole)
        self.backend_combo.activated.connect(self._on_backend)
        row.addWidget(self.backend_combo, 1)
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

    def _build_events(self) -> QtWidgets.QWidget:
        """How often events arrive, and buttons that ask for one now.

        The buttons do not bypass anything. The request goes to the same
        scheduler the simulation uses, which builds the event exactly as it
        would have built its own -- same envelope, same size cap, same
        concurrency limit -- so pressing one chooses *when* an event arrives and
        *which* kind, and nothing else. The result is a perturbation that takes
        a minute or two to come up, not a button that does something to the
        picture.

        The slider above them is the same restraint applied to the automatic
        stream: it moves the mean interval between arrivals and reaches nothing
        else. Every property that keeps an event from reading as punctuation --
        the raised-cosine envelope, the radius cap, the amplitude, the limit on
        how many run at once -- is untouched by it, so the fast end of the
        travel is a field that spends more of its time inside an event, never
        one that gets hit harder. That is why this one needs no ceiling of its
        own, where a control over an event's *size* would have to answer to
        `SAFETY_CEILINGS` before it could be exposed at all.
        """
        box = QtWidgets.QGroupBox("Events")
        column = QtWidgets.QVBoxLayout(box)
        column.setSpacing(6)

        column.addLayout(self._build_event_rate())

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        self.event_buttons: dict[str, QtWidgets.QPushButton] = {}

        columns = 3
        for index, kind in enumerate(events_module.EVENT_KINDS):
            button = QtWidgets.QPushButton(kind.capitalize())
            description = EVENT_LABELS.get(kind, f"A {kind} event")
            button.setToolTip(
                f"{description}.\nIt arrives slowly, in one region, and fades "
                "out again over a few minutes."
            )
            # Default-bound `kind`, or every button would trigger the last one.
            # `clicked` passes a checked flag that we have no use for.
            button.clicked.connect(
                lambda _checked=False, kind=kind: self._on_trigger(kind)
            )
            grid.addWidget(button, index // columns, index % columns)
            self.event_buttons[kind] = button
        column.addLayout(grid)

        self.event_note = QtWidgets.QLabel(NOTE_IDLE)
        self.event_note.setWordWrap(True)
        column.addWidget(self.event_note)
        return box

    def _build_event_rate(self) -> QtWidgets.QLayout:
        """The "how often" row: caption, slider, and what it currently means.

        The readout is an interval in plain language rather than the 0..1 the
        other macros show, because a bare number cannot answer the question
        being asked here. "0.34" is not something anyone can want; "about one
        every 14 min" is.
        """
        tip = (
            "How often events arrive on their own.\n"
            "This changes only the timing -- an event is the same size, "
            "strength and length whatever this is set to."
        )
        row = QtWidgets.QHBoxLayout()
        caption = QtWidgets.QLabel("How often")
        caption.setToolTip(tip)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, SLIDER_STEPS)
        slider.setToolTip(tip)
        slider.valueChanged.connect(self._on_slider)
        slider.setTracking(True)

        value = QtWidgets.QLabel("—")
        value.setMinimumWidth(132)
        value.setToolTip(tip)
        value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        row.addWidget(caption)
        row.addWidget(slider, 1)
        row.addWidget(value)
        self.sliders[EVENT_RATE_MACRO] = slider
        self.values[EVENT_RATE_MACRO] = value
        return row

    def _build_status(self) -> QtWidgets.QWidget:
        box = QtWidgets.QGroupBox("Status")
        form = QtWidgets.QFormLayout(box)
        self.status_labels = {}
        for key, label in (
            ("runtime", "Running for"),
            ("depth", "Depth"),
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

    def _bind_fullscreen_key(self) -> None:
        """F11 here too, because this is where the keyboard usually is.

        The render window binds F11 itself, but it is the window the user is
        deliberately *not* interacting with -- it sits on the other display and
        never takes focus. Pressing F11 while adjusting a slider should still
        mean what it means everywhere else, so the panel forwards it to the
        same toggle rather than swallowing it.
        """
        key = QtGui.QKeySequence(window_module.FULLSCREEN_KEY)
        shortcut = QtGui.QShortcut(key, self)
        shortcut.activated.connect(self._on_fullscreen)
        self._fullscreen_shortcut = shortcut

    def _on_fullscreen(self) -> None:
        self.app.toggle_fullscreen()

    # -- state --------------------------------------------------------------

    def _format_macro(self, name: str, value: float) -> str:
        """What the label beside a slider says at this position.

        The event rate is asked what it resolves to rather than being shown
        raw, and asked of `config` rather than of the live parameters: those are
        mid-ramp and would show the number crawling towards the one the hand
        just set, which reads as lag in the control rather than as the smooth
        transition it actually is.
        """
        if name == EVENT_RATE_MACRO:
            return describe_interval(
                config_module.curve_value(EVENT_RATE_MACRO, EVENT_RATE_PATH, value)
            )
        return f"{value:.2f}"

    def _load_from_app(self) -> None:
        self._updating = True
        macros = self.app.config.macros
        for name, slider in self.sliders.items():
            value = float(getattr(macros, name))
            slider.setValue(int(round(value * SLIDER_STEPS)))
            self.values[name].setText(self._format_macro(name, value))
        index = self.preset_combo.findText(self.app.config.preset_name)
        if index >= 0:
            self.preset_combo.setCurrentIndex(index)
        index = self.backend_combo.findData(self.app.backend)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)
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
            self.values[name].setText(
                self._format_macro(name, slider.value() / SLIDER_STEPS)
            )
        self.app.apply_macros(macros)

    def _on_preset(self) -> None:
        name = self.preset_combo.currentText()
        try:
            macros = presets_module.get(name)
        except KeyError:
            return
        self._updating = True
        for field_name, slider in self.sliders.items():
            value = float(getattr(macros, field_name))
            slider.setValue(int(round(value * SLIDER_STEPS)))
            self.values[field_name].setText(self._format_macro(field_name, value))
        self._updating = False
        self.app.config.preset_name = name
        # Ramped like any other change, so switching presets is a slow
        # transition rather than a cut.
        self.app.apply_macros(macros)

    def _on_backend(self) -> None:
        if self._updating:
            return
        wanted = self.backend_combo.currentData()
        if wanted == self.app.backend:
            return
        label = self.backend_combo.currentText()
        answer = QtWidgets.QMessageBox.question(
            self,
            "Change how depth is drawn",
            f"Switch to the {label.lower()} view?\n\n"
            "The picture fades down and comes back up over a few seconds. "
            "The field you are leaving is saved, so switching back later "
            "finds it where it was.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            self._load_from_app()  # put the combo back
            return
        try:
            self.app.switch_backend(wanted)
            self.app.save_config()
        except Exception as exc:
            log.error("could not switch the depth backend: %s", exc)
            QtWidgets.QMessageBox.warning(self, "Could not switch", str(exc))
            self._load_from_app()

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

    def _on_trigger(self, kind: str) -> None:
        try:
            started = self.app.trigger_event(kind)
        except Exception as exc:
            log.error("could not trigger a %s event: %s", kind, exc)
            self._set_note(f"Could not start a {kind}: {exc}")
            return
        if started:
            self._set_note(
                f"{kind.capitalize()} started — it comes up over the next "
                "minute or two."
            )
        else:
            # Refused, not lost: queueing it would mean an event arriving long
            # after the press that asked for it, which is worse than saying no.
            self._set_note(
                f"Not now — {self._active_events()} events are already "
                "running, which is as many as the settings allow. Try again "
                "when one has faded."
            )
        self._refresh_events()

    def _set_note(self, text: str) -> None:
        self.event_note.setText(text)
        self._note_expires = time.monotonic() + NOTE_SECONDS

    def _active_events(self) -> int:
        return len(self.app.scheduler.active)

    def _refresh_events(self) -> None:
        """Keep the buttons honest about whether there is room for another.

        Deliberately separate from `_refresh_status`, which gives up early when
        the engine is not there or a readback fails; whether the scheduler can
        take another event does not depend on either.
        """
        room = self._active_events() < self.app.params.events.max_concurrent
        for button in self.event_buttons.values():
            button.setEnabled(room)
        if self._note_expires and time.monotonic() < self._note_expires:
            # A reply to something the user just pressed outranks the standing
            # text, including when their own event was the one that filled the
            # last slot.
            return
        self.event_note.setText(NOTE_IDLE if room else NOTE_AT_CAP)
        self._note_expires = 0.0

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
        self.status_labels["depth"].setText(
            f"{self.app.backend}   {engine.geometry.describe()}"
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
