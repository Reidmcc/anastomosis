"""Borderless fullscreen, and the F11 that toggles it.

The bug these cover: the application asked the canvas for ``set_fullscreen``
and, when that was missing, logged "fullscreen not supported by this backend"
and stayed windowed. ``rendercanvas`` has never had such a method on any
backend, so ``--fullscreen`` could not work anywhere -- the probe was looking
for an API that does not exist rather than for the window underneath.

So the tests are mostly about the *shape* of what each backend offers: that the
strategies key off window API those backends really have, that entering is
exactly reversible, and that the glfw path never picks the primary monitor when
the window is somewhere else -- that being the one display the visual must not
jump to.
"""

from __future__ import annotations

import panelstub
import pytest

from anastomosis import window as window_module
from anastomosis.app import AppOptions, Application


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeWidget:
    """A canvas shaped like ``rendercanvas.qt.QRenderCanvas``: a top-level widget."""

    def __init__(self, maximized: bool = False) -> None:
        self.calls: list[str] = []
        self._fullscreen = False
        self._maximized = maximized
        self._subwidget = FakeSubwidget()

    def isFullScreen(self):  # noqa: N802 - Qt naming
        return self._fullscreen

    def isMaximized(self):  # noqa: N802
        return self._maximized

    def showFullScreen(self):  # noqa: N802
        self.calls.append("showFullScreen")
        self._fullscreen = True
        self._maximized = False

    def showNormal(self):  # noqa: N802
        self.calls.append("showNormal")
        self._fullscreen = False
        self._maximized = False

    def showMaximized(self):  # noqa: N802
        self.calls.append("showMaximized")
        self._fullscreen = False
        self._maximized = True


class FakeSubwidget:
    def __init__(self) -> None:
        self.focused = 0

    def setFocus(self):  # noqa: N802
        self.focused += 1


class FakeMonitor:
    def __init__(self, x, y, width, height):
        self.pos = (x, y)
        self.size = (width, height)


class FakeVideoMode:
    def __init__(self, width, height):
        self.size = type("Size", (), {"width": width, "height": height})()


class FakeGlfw:
    """Just enough of the glfw module, with the window state it would own."""

    DECORATED = 0x00020005
    TRUE = 1
    FALSE = 0

    def __init__(self, monitors, position=(0, 0), size=(1280, 720)):
        self.monitors = monitors
        self.position = position
        self.size = size
        self.decorated = True

    def get_monitors(self):
        return self.monitors

    def get_monitor_pos(self, monitor):
        return monitor.pos

    def get_video_mode(self, monitor):
        return FakeVideoMode(*monitor.size)

    def get_window_pos(self, window):
        return self.position

    def get_window_size(self, window):
        return self.size

    def set_window_pos(self, window, x, y):
        self.position = (x, y)

    def set_window_size(self, window, width, height):
        self.size = (width, height)

    def set_window_attrib(self, window, attrib, value):
        assert attrib == self.DECORATED, f"unexpected window attribute {attrib}"
        self.decorated = bool(value)


class FakeGlfwCanvas:
    def __init__(self) -> None:
        self._window = object()


@pytest.fixture
def glfw_canvas(monkeypatch):
    """A glfw-shaped canvas on a second monitor, to the right of the primary."""
    glfw = FakeGlfw(
        monitors=[FakeMonitor(0, 0, 1920, 1080), FakeMonitor(1920, 0, 2560, 1440)],
        position=(2100, 200),
        size=(1280, 720),
    )
    monkeypatch.setattr(window_module, "_import_glfw", lambda: glfw)
    return FakeGlfwCanvas(), glfw


# ---------------------------------------------------------------------------
# Choosing a strategy
# ---------------------------------------------------------------------------


def test_a_widget_canvas_gets_the_qt_strategy():
    controller = window_module.controller_for(FakeWidget())
    assert isinstance(controller, window_module.QtFullscreen)


def test_a_glfw_canvas_gets_the_glfw_strategy(glfw_canvas):
    canvas, _ = glfw_canvas
    controller = window_module.controller_for(canvas)
    assert isinstance(controller, window_module.GlfwFullscreen)


def test_a_windowless_canvas_gets_no_strategy():
    """Offscreen and notebook canvases have no window; None is the right answer."""

    class Offscreen:
        def get_physical_size(self):
            return (64, 64)

    assert window_module.controller_for(Offscreen()) is None


def test_the_qt_probe_names_api_the_real_backend_has():
    """The regression guard: probe for methods that exist, not invented ones.

    The old code looked for ``set_fullscreen``/``_set_fullscreen``, which no
    ``rendercanvas`` backend has ever defined, so the fallback branch was the
    only one reachable. Checked against the class rather than an instance, so
    this needs no display.
    """
    # The Qt *widgets* module specifically: PySide6 imports fine on a machine
    # with no Qt platform libraries, and only fails when a widget module is
    # pulled in.
    pytest.importorskip(
        "PySide6.QtWidgets", reason="the Qt backend needs PySide6"
    )
    from rendercanvas.qt import QRenderCanvas

    for name in window_module.QtFullscreen._METHODS:
        assert callable(getattr(QRenderCanvas, name, None)), (
            f"the Qt canvas has no {name}; the strategy would never match"
        )
    assert not hasattr(QRenderCanvas, "set_fullscreen"), (
        "rendercanvas grew a fullscreen API; prefer it over the window API here"
    )


# ---------------------------------------------------------------------------
# Qt
# ---------------------------------------------------------------------------


def test_qt_enters_borderless_fullscreen():
    """``showFullScreen`` and nothing else -- Qt never asks for exclusive mode."""
    canvas = FakeWidget()
    controller = window_module.QtFullscreen.for_canvas(canvas)

    controller.set(True)

    assert controller.is_fullscreen()
    assert canvas.calls == ["showFullScreen"]


def test_qt_leaving_returns_to_a_normal_window():
    canvas = FakeWidget()
    controller = window_module.QtFullscreen.for_canvas(canvas)

    controller.set(True)
    controller.set(False)

    assert not controller.is_fullscreen()
    assert canvas.calls == ["showFullScreen", "showNormal"]


def test_qt_leaving_returns_to_maximised_if_that_is_where_it_came_from():
    canvas = FakeWidget(maximized=True)
    controller = window_module.QtFullscreen.for_canvas(canvas)

    controller.set(True)
    controller.set(False)

    assert canvas.calls == ["showFullScreen", "showMaximized"]
    assert canvas.isMaximized()


def test_qt_setting_the_state_it_is_already_in_does_nothing():
    canvas = FakeWidget()
    controller = window_module.QtFullscreen.for_canvas(canvas)

    controller.set(False)
    controller.set(True)
    controller.set(True)

    assert canvas.calls == ["showFullScreen"]


def test_qt_keeps_the_render_widget_focused():
    """Without focus the widget stops seeing keys, and F11 could not get back out."""
    canvas = FakeWidget()
    controller = window_module.QtFullscreen.for_canvas(canvas)

    controller.set(True)
    controller.set(False)

    assert canvas._subwidget.focused == 2


# ---------------------------------------------------------------------------
# glfw
# ---------------------------------------------------------------------------


def test_glfw_undecorates_and_fills_the_monitor_it_is_on(glfw_canvas):
    canvas, glfw = glfw_canvas
    controller = window_module.GlfwFullscreen.for_canvas(canvas)

    controller.set(True)

    assert controller.is_fullscreen()
    assert not glfw.decorated, "the window still has a frame"
    assert glfw.position == (1920, 0), (
        "the window moved to a monitor it was not on; the visual must stay "
        "on the display the user put it on"
    )
    assert glfw.size == (2560, 1440)


def test_glfw_never_uses_exclusive_fullscreen(glfw_canvas):
    """``set_window_monitor`` is the exclusive path, and is deliberately unused."""
    canvas, glfw = glfw_canvas

    def forbidden(*args, **kwargs):  # pragma: no cover - only runs on failure
        raise AssertionError("exclusive fullscreen was requested")

    glfw.set_window_monitor = forbidden
    window_module.GlfwFullscreen.for_canvas(canvas).set(True)


def test_glfw_leaving_restores_the_exact_frame(glfw_canvas):
    canvas, glfw = glfw_canvas
    controller = window_module.GlfwFullscreen.for_canvas(canvas)
    before = (glfw.position, glfw.size, glfw.decorated)

    controller.set(True)
    controller.set(False)

    assert not controller.is_fullscreen()
    assert (glfw.position, glfw.size, glfw.decorated) == before


def test_glfw_picks_the_monitor_with_the_most_overlap(glfw_canvas):
    """A window straddling two monitors goes fullscreen on its larger half."""
    canvas, glfw = glfw_canvas
    glfw.position = (1720, 100)  # 200px on the primary, 1080px on the second
    glfw.size = (1280, 720)

    window_module.GlfwFullscreen.for_canvas(canvas).set(True)

    assert glfw.position == (1920, 0)


def test_glfw_survives_the_window_going_away(glfw_canvas):
    """A toggle racing the window's close must not raise out of the loop."""
    canvas, _ = glfw_canvas
    controller = window_module.GlfwFullscreen.for_canvas(canvas)
    canvas._window = None

    controller.set(True)

    assert not controller.is_fullscreen()


# ---------------------------------------------------------------------------
# The application
# ---------------------------------------------------------------------------


def _app(tmp_path, canvas, **options):
    app = Application(
        AppOptions(config_path=tmp_path / "config.toml", ui=False, **options)
    )
    app.canvas = canvas
    return app


class EventCanvas(FakeWidget):
    """A canvas that records handlers the way ``rendercanvas`` dispatches them."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.handlers: dict[str, list] = {}

    def add_event_handler(self, callback, *types):
        for event_type in types:
            self.handlers.setdefault(event_type, []).append(callback)

    def press(self, key):
        for callback in self.handlers.get("key_down", []):
            callback({"event_type": "key_down", "key": key, "modifiers": ()})


def test_launching_with_fullscreen_actually_goes_fullscreen(tmp_path):
    canvas = EventCanvas()
    app = _app(tmp_path, canvas, fullscreen=True)

    app._install_fullscreen()

    assert app.fullscreen
    assert canvas.calls == ["showFullScreen"]


def test_launching_without_fullscreen_stays_windowed(tmp_path):
    canvas = EventCanvas()
    app = _app(tmp_path, canvas)

    app._install_fullscreen()

    assert not app.fullscreen
    assert canvas.calls == []


def test_f11_toggles_fullscreen(tmp_path):
    canvas = EventCanvas()
    app = _app(tmp_path, canvas)
    app._install_fullscreen()

    canvas.press("F11")
    assert app.fullscreen, "F11 did not enter fullscreen"

    canvas.press("F11")
    assert not app.fullscreen, "F11 did not come back out of fullscreen"


def test_other_keys_are_left_alone(tmp_path):
    canvas = EventCanvas()
    app = _app(tmp_path, canvas)
    app._install_fullscreen()

    for key in ("F10", "F12", "Escape", "f"):
        canvas.press(key)

    assert not app.fullscreen
    assert canvas.calls == []


def test_a_backend_without_fullscreen_says_so_and_keeps_running(tmp_path, caplog):
    class Windowless:
        def __init__(self):
            self.handlers = {}

        def add_event_handler(self, callback, *types):
            for event_type in types:
                self.handlers.setdefault(event_type, []).append(callback)

    canvas = Windowless()
    app = _app(tmp_path, canvas, fullscreen=True)

    with caplog.at_level("INFO", logger="anastomosis.app"):
        app._install_fullscreen()

    assert not app.fullscreen
    assert app.toggle_fullscreen() is False
    assert any("fullscreen is not available" in r.message for r in caplog.records)


def test_a_refused_state_change_does_not_reach_the_loop(tmp_path):
    """A window manager that says no must not end a days-long session."""

    class Refusing(EventCanvas):
        def showFullScreen(self):  # noqa: N802
            raise RuntimeError("the window manager said no")

    canvas = Refusing()
    app = _app(tmp_path, canvas)
    app._install_fullscreen()

    canvas.press("F11")

    assert not app.fullscreen


# ---------------------------------------------------------------------------
# The control panel
# ---------------------------------------------------------------------------


class PanelStubApp(panelstub.PanelApp):
    """The shared panel stand-in, counting the toggles this module watches."""

    def __init__(self) -> None:
        super().__init__()
        self.toggles = 0

    def toggle_fullscreen(self) -> bool:
        self.toggles += 1
        return False


def test_the_panel_forwards_f11_to_the_same_toggle(monkeypatch):
    """The panel is where the keyboard is; F11 there must mean what it means.

    The shortcut is fired directly rather than by sending a key event, because
    a ``QShortcut`` only matches while its window is active and there is no
    window manager here to make one active.
    """
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip(
        "PySide6.QtWidgets", reason="the control panel needs the [ui] extra"
    )
    from PySide6 import QtGui, QtWidgets

    from anastomosis.ui.control_panel import ControlPanel

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app = PanelStubApp()
    panel = ControlPanel(app)

    assert panel._fullscreen_shortcut.key() == QtGui.QKeySequence("F11")

    panel._fullscreen_shortcut.activated.emit()
    assert app.toggles == 1, "F11 in the panel did not reach the fullscreen toggle"
