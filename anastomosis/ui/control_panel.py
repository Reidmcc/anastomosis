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
from .. import volume as volume_module
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

# The viewpoint drift, which is shown as the travel it produces rather than as
# a bare 0..1. What the number means is "how far the near material slides
# against the far material, across the width of the screen", which is a thing
# somebody can want; "0.60" is not.
#
# It is also the readout that tells you when the slab is the limit rather than
# the knob: under the volumetric backend the drift is held to what the
# thickness justifies (`volume.PARALLAX_MAX_TANGENT`), so on a thin slab the
# percentage stops climbing part-way along the travel. That is the control
# saying "make the slab deeper", and it says it better than a tooltip can.
PARALLAX_MACRO = "parallax"
PARALLAX_PATH = "render.parallax"

PARALLAX_TIP = (
    "How far the viewpoint drifts, and how briskly.\n"
    "The only depth cue here that comes from the scene moving rather than "
    "from how it is shaded -- everything under Depth describes the far "
    "material, where this one lets you see past the near material to it.\n"
    "Under the volumetric view the slab's thickness caps it: parallax is "
    "thickness times the viewing angle, and a thin slab seen from far enough "
    "off-axis is just a slab seen edge-on. If the readout stops rising as you "
    "drag, the Thickness slider above is what is holding it."
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
    ("depth", "Depth", "Focus falloff, atmosphere, and how far the back fades"),
    ("parallax", "Parallax", PARALLAX_TIP),
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


# The thinnest and thickest a slab can be are decided by the window and the
# config (`volume.depth_limits`), but the step is fixed: the geometry rounds the
# thickness up to a multiple of eight so the 4-deep workgroups tile it, so the
# slider counts in eights and never proposes a number it would not get.
DEPTH_STEP = 8

THICKNESS_TIP = (
    "How deep the volumetric slab is, in voxels.\n"
    "Deeper means more material between you and the far face, so occlusion, "
    "shading and atmosphere all have more to work with. It is also what this "
    "view's memory is spent on, which is why the cost is written beside it, "
    "and the returns flatten before the top of the travel does -- past a "
    "point the near structure hides the far face entirely.\n"
    "Structural: it grows a new field rather than adjusting this one."
)

THICKNESS_FOR_LAYERED = (
    "The layered view has no thickness to set. Switch to the volumetric one "
    "above."
)


def describe_bytes(count: int) -> str:
    """Plain language for a quantity of graphics memory.

    Two significant figures and never more: this is an estimate of what a field
    will occupy, offered so that someone moving the slider can see the cost
    climbing, and a fourth digit would be claiming a precision it has not got.
    """
    if count >= 1 << 30:
        return f"{count / (1 << 30):.1f} GB"
    return f"{count / (1 << 20):.0f} MB"


def describe_slab(geometry) -> str:
    """The one line under the thickness slider: what it would build, and what
    that costs. Shaped as a fact about a field rather than as a warning, since
    every value the slider can reach is one the user is entitled to choose."""
    return (
        f"{geometry.width} x {geometry.height} x {geometry.depth} voxels, "
        f"about {describe_bytes(geometry.field_bytes)}"
    )


def describe_parallax(reach: float) -> str:
    """Plain language for how far the viewpoint travels.

    As a share of the screen's width, because that is the thing the eye is
    actually being offered: the near material slides this far against the far
    material, and everything else about the number is arithmetic.
    """
    if reach < 0.005:
        return "still"
    return f"{reach * 100:.0f}% of width"


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
        self._timer.timeout.connect(self._refresh_thickness_range)
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
        """How depth is drawn, and -- for the volumetric view -- how much of it.

        Structural rather than perceptual, so neither control belongs among the
        sliders: nothing about them can be ramped, and either one grows a new
        field rather than adjusting the one on screen. Both ask before doing
        that, for the same reason the reset button does. The two answers are not
        equally costly, though, and the wording says so: switching backend keeps
        the field it leaves, where changing the thickness cannot -- a slab of a
        different depth is a differently shaped field and nothing resamples one
        into the other.

        The thickness is a slider and a separate button rather than a slider
        that acts on release, because dragging it has to be free: the line under
        it prices every position, and reading that is most of what the control
        is for.
        """
        box = QtWidgets.QGroupBox("Depth")
        grid = QtWidgets.QGridLayout(box)
        grid.setVerticalSpacing(6)

        self.backend_combo = QtWidgets.QComboBox()
        for name, label, tip in BACKEND_LABELS:
            self.backend_combo.addItem(label, name)
            self.backend_combo.setItemData(
                self.backend_combo.count() - 1, tip, QtCore.Qt.ToolTipRole)
        self.backend_combo.activated.connect(self._on_backend)
        grid.addWidget(self.backend_combo, 0, 0, 1, 3)

        self.thickness_caption = QtWidgets.QLabel("Thickness")
        self.thickness_caption.setToolTip(THICKNESS_TIP)
        self.thickness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.thickness_slider.setToolTip(THICKNESS_TIP)
        self.thickness_slider.setTracking(True)
        self.thickness_slider.valueChanged.connect(self._on_thickness)
        self.thickness_value = QtWidgets.QLabel("—")
        self.thickness_value.setMinimumWidth(64)
        self.thickness_value.setAlignment(
            QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        grid.addWidget(self.thickness_caption, 1, 0)
        grid.addWidget(self.thickness_slider, 1, 1)
        grid.addWidget(self.thickness_value, 1, 2)

        self.thickness_note = QtWidgets.QLabel("—")
        self.thickness_note.setWordWrap(True)
        grid.addWidget(self.thickness_note, 2, 0, 1, 2)

        self.thickness_apply = QtWidgets.QPushButton("Grow a new slab")
        self.thickness_apply.setToolTip(
            "Build the slab at this thickness. The current field and its saved "
            "state are discarded."
        )
        self.thickness_apply.clicked.connect(self._on_thickness_apply)
        grid.addWidget(self.thickness_apply, 2, 2)

        grid.setColumnStretch(1, 1)
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
            # Wide enough for the longest readout in the column, so that
            # dragging one slider does not shift the others sideways.
            value.setMinimumWidth(88)
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
        if name == PARALLAX_MACRO:
            return describe_parallax(self._parallax_reach(value))
        return f"{value:.2f}"

    def _parallax_reach(self, value: float) -> float:
        """How far the viewpoint will actually travel at this slider position.

        The curve says what was asked for; the slab may say less. Reporting the
        second is the point -- a control whose readout stops moving is telling
        the user something true about why, and one that keeps climbing while
        nothing on screen changes is lying to them.
        """
        asked = config_module.curve_value(PARALLAX_MACRO, PARALLAX_PATH, value)
        if self.app.backend != "volumetric":
            return asked
        slab = self.app.volume_slab()
        thickness = slab.depth / max(slab.width, 1)
        return min(asked, volume_module.PARALLAX_MAX_TANGENT * thickness)

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
        self._sync_thickness()

    # -- the slab's thickness ----------------------------------------------

    def _sync_thickness(self) -> None:
        """Put the thickness row back where the application actually is.

        The travel is asked for rather than remembered: its ceiling is the
        shorter lateral axis of the slab, which follows the window's aspect, so
        moving the window to a differently shaped display moves it.
        """
        low, high = self.app.volume_depth_limits()
        was, self._updating = self._updating, True
        self.thickness_slider.setRange(
            max(low // DEPTH_STEP, 1), max(high // DEPTH_STEP, 1))
        self.thickness_slider.setPageStep(4)
        self.thickness_slider.setValue(
            self.app.volume_slab().depth // DEPTH_STEP)
        self._updating = was
        self._refresh_thickness()

    def _refresh_thickness(self) -> None:
        """What the row says, and whether its button has anything to do."""
        volumetric = self.app.backend == "volumetric"
        proposed = self.app.volume_slab(self.thickness_slider.value() * DEPTH_STEP)
        self.thickness_value.setText(f"{proposed.depth} voxels")
        self.thickness_caption.setEnabled(volumetric)
        self.thickness_slider.setEnabled(volumetric)
        if not volumetric:
            # The knob belongs to a field this backend does not have. Saying so
            # is better than a control that silently changes nothing visible.
            self.thickness_note.setText(THICKNESS_FOR_LAYERED)
            self.thickness_apply.setEnabled(False)
            return
        self.thickness_note.setText(describe_slab(proposed))
        self.thickness_apply.setEnabled(
            proposed.depth != self.app.volume_slab().depth)

    def _refresh_thickness_range(self) -> None:
        """Follow a window reshape, without disturbing a change being composed.

        Only the travel is touched, and only while the slider is not being
        dragged; if the new ceiling is below where the slider sits, Qt brings
        the value down with it, which is the right answer -- that thickness is
        no longer one this window can be given.
        """
        if self.thickness_slider.isSliderDown():
            return
        low, high = self.app.volume_depth_limits()
        wanted = (max(low // DEPTH_STEP, 1), max(high // DEPTH_STEP, 1))
        current = (self.thickness_slider.minimum(), self.thickness_slider.maximum())
        if wanted != current:
            was, self._updating = self._updating, True
            self.thickness_slider.setRange(*wanted)
            self._updating = was
            self._refresh_thickness()

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

    def _on_thickness(self) -> None:
        if self._updating:
            return
        # Dragging prices the position; it does not build anything. The button
        # beside it is the one that touches the field.
        self._refresh_thickness()

    def _on_thickness_apply(self) -> None:
        wanted = self.thickness_slider.value() * DEPTH_STEP
        proposed = self.app.volume_slab(wanted)
        answer = QtWidgets.QMessageBox.question(
            self,
            "Change how deep the slab is",
            f"Grow a new slab {proposed.depth} voxels deep?\n\n"
            "A slab of a different thickness is a differently shaped field, so "
            "the current one and its saved state are discarded, exactly as a "
            "reset discards them. The image settles down and grows back over a "
            f"few minutes, and the new field will hold about "
            f"{describe_bytes(proposed.field_bytes)} on the graphics card.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            self._sync_thickness()  # put the slider back
            return
        try:
            self.app.set_volume_depth(wanted)
            self.app.save_config()
        except Exception as exc:
            # Nothing was discarded: the new field is built before the old one
            # is let go, so the session is still running on the old thickness.
            log.error("could not change the slab's thickness: %s", exc)
            QtWidgets.QMessageBox.warning(
                self,
                "Could not change the thickness",
                f"The field is still running as it was.\n\n{exc}",
            )
        self._sync_thickness()

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
        else:
            # The thickness row belongs to the volumetric field, so it becomes
            # live or dead with this choice.
            self._sync_thickness()

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
        else:
            self._sync_thickness()

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
