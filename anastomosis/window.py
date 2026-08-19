"""Borderless windowed fullscreen, per windowing backend.

``rendercanvas`` deliberately has no fullscreen API -- it owns the surface, not
the window manager -- so this is the layer that reaches past it to the native
window. The application asks for "fullscreen" and gets whichever of the
strategies below the live canvas can actually carry out; there is no generic
one, because the two backends in use have nothing in common at this level.

Borderless windowed in both cases, never exclusive. Exclusive fullscreen asks
the driver for the display, which on a multi-monitor desktop can stall the
compositor driving the *other* screen and pull focus across to this one. The
whole point of this program is that it sits on a second display while the user
works on the first, so a mode that interrupts the first display is not a
fullscreen worth having. Qt's ``showFullScreen`` is already borderless
windowed; glfw's ``set_window_monitor`` is not, so the glfw strategy undecorates
the window and sizes it to the monitor by hand rather than using it.

Entering and leaving are strictly reversible: each strategy records what the
window looked like before, and restores exactly that. A fullscreen toggle is a
change to the presentation only -- the canvas reports a new size, the
application's resize path follows it, and the simulation never learns that
anything happened.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# The key that toggles borderless fullscreen, spelled the way both windowing
# backends report it. F11 is what every other fullscreen window on the desktop
# uses, and this one should not be the exception. Named here rather than in
# either window that binds it, so the render canvas and the control panel
# cannot drift apart about which key it is.
FULLSCREEN_KEY = "F11"


def _import_glfw():
    """The glfw module, or None.

    Its own function so that the glfw strategy can be exercised without a
    display attached, and so that a canvas that merely looks glfw-shaped cannot
    make the import a hard requirement.
    """
    try:
        import glfw
    except ImportError:
        return None
    return glfw


class QtFullscreen:
    """Fullscreen for the Qt backend, where the canvas *is* a top-level widget.

    ``rendercanvas.qt`` builds the window as a ``QWidget`` wrapping the render
    widget, so the ordinary Qt window API is right there on the canvas object.
    ``showFullScreen`` is borderless windowed on every platform Qt supports --
    it sets a window state, and never asks the driver for a mode change -- so
    it is exactly the mode wanted here.
    """

    name = "qt"

    # The window API this strategy needs. Probed rather than type-checked, so
    # that importing PySide6 is not a precondition for asking the question.
    _METHODS = ("showFullScreen", "showNormal", "showMaximized", "isFullScreen")

    @classmethod
    def for_canvas(cls, canvas) -> "QtFullscreen | None":
        if all(callable(getattr(canvas, name, None)) for name in cls._METHODS):
            return cls(canvas)
        return None

    def __init__(self, canvas) -> None:
        self._canvas = canvas
        # Maximised is a distinct state from normal, and leaving fullscreen has
        # to land back on whichever one the window came from.
        self._was_maximized = False

    def is_fullscreen(self) -> bool:
        return bool(self._canvas.isFullScreen())

    def set(self, enabled: bool) -> None:
        if enabled == self.is_fullscreen():
            return
        if enabled:
            self._was_maximized = bool(self._canvas.isMaximized())
            self._canvas.showFullScreen()
        elif self._was_maximized:
            self._canvas.showMaximized()
        else:
            self._canvas.showNormal()
        self._restore_key_focus()

    def _restore_key_focus(self) -> None:
        """Keep the render widget as the window's focus widget.

        Changing window state can make Qt drop and rebuild the native window,
        and the widget that was holding keyboard focus inside it does not
        always survive that. Without focus the render widget stops receiving
        key events, which would leave a window that F11 could enter and not
        leave.

        This is focus *within* the window; it is not ``activateWindow``, and it
        does not pull focus away from whatever the user is typing into on
        their other display.
        """
        widget = getattr(self._canvas, "_subwidget", None)
        setter = getattr(widget, "setFocus", None)
        if callable(setter):
            try:
                setter()
            except Exception as exc:  # pragma: no cover - Qt focus quirks
                log.debug("could not restore focus to the render widget: %s", exc)


class GlfwFullscreen:
    """Fullscreen for the glfw backend, built by hand out of window attributes.

    glfw's own fullscreen -- ``set_window_monitor`` with a monitor -- is the
    exclusive kind, which is the one mode ruled out here. So this undecorates
    the window and places it over the whole of the monitor it is currently on,
    which is borderless windowed fullscreen in the only sense that matters: no
    frame, no titlebar, exactly the monitor's area, and the compositor still in
    charge of every other display.
    """

    name = "glfw"

    @classmethod
    def for_canvas(cls, canvas) -> "GlfwFullscreen | None":
        if getattr(canvas, "_window", None) is None:
            return None
        glfw = _import_glfw()
        if glfw is None:  # pragma: no cover - a glfw canvas without glfw
            return None
        return cls(canvas, glfw)

    def __init__(self, canvas, glfw) -> None:
        self._canvas = canvas
        self._glfw = glfw
        # The frame to come back to: position and size as they were before.
        # Also this strategy's record of *being* fullscreen, since an
        # undecorated window at monitor size is not distinguishable from one
        # the user happened to place that way.
        self._restore: tuple[tuple[int, int], tuple[int, int]] | None = None

    def is_fullscreen(self) -> bool:
        return self._restore is not None

    def set(self, enabled: bool) -> None:
        if enabled == self.is_fullscreen():
            return
        window = getattr(self._canvas, "_window", None)
        if window is None:
            # The window is gone; the canvas is closing. Nothing to do, and
            # nothing worth raising over.
            self._restore = None
            return
        if enabled:
            self._enter(window)
        else:
            self._leave(window)

    def _enter(self, window) -> None:
        glfw = self._glfw
        position = tuple(glfw.get_window_pos(window))
        size = tuple(glfw.get_window_size(window))
        x, y, width, height = self._monitor_area(window)
        glfw.set_window_attrib(window, glfw.DECORATED, glfw.FALSE)
        glfw.set_window_pos(window, x, y)
        glfw.set_window_size(window, width, height)
        # Recorded last: if any of the above raised, the window is still in a
        # state the caller can describe as "not fullscreen" and retry from.
        self._restore = (position, size)

    def _leave(self, window) -> None:
        glfw = self._glfw
        (x, y), (width, height) = self._restore
        self._restore = None
        glfw.set_window_attrib(window, glfw.DECORATED, glfw.TRUE)
        glfw.set_window_size(window, width, height)
        glfw.set_window_pos(window, x, y)

    def _monitor_area(self, window) -> tuple[int, int, int, int]:
        """The monitor the window is mostly on, as ``(x, y, width, height)``.

        glfw will not answer "which monitor is this window on" for a windowed
        window, so it is worked out from the geometry: the monitor sharing the
        most area with the window wins. Picking the primary monitor instead
        would send the visual back to the display the user is working on, every
        time, which is the one place it must not go.
        """
        glfw = self._glfw
        wx, wy = glfw.get_window_pos(window)
        ww, wh = glfw.get_window_size(window)

        best: tuple[int, int, int, int] | None = None
        best_overlap = -1
        for monitor in glfw.get_monitors():
            mx, my = glfw.get_monitor_pos(monitor)
            mode = glfw.get_video_mode(monitor)
            mw, mh = mode.size.width, mode.size.height
            overlap = (
                max(0, min(wx + ww, mx + mw) - max(wx, mx))
                * max(0, min(wy + wh, my + mh) - max(wy, my))
            )
            if overlap > best_overlap:
                best, best_overlap = (mx, my, mw, mh), overlap

        if best is None:  # pragma: no cover - a desktop with no monitors
            raise RuntimeError("glfw reports no monitors")
        return best


# Ordered by specificity: the Qt canvas is a widget and answers to the widget
# probe, and nothing else does, so the two never both match.
STRATEGIES = (QtFullscreen, GlfwFullscreen)


def controller_for(canvas):
    """The fullscreen strategy for this canvas, or None if there isn't one.

    None is a real answer, not a failure: the offscreen and Jupyter canvases
    have no window to make fullscreen. The caller decides what to say about it.
    """
    for strategy in STRATEGIES:
        controller = strategy.for_canvas(canvas)
        if controller is not None:
            return controller
    return None
