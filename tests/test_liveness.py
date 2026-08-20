"""The frame loop must not depend on being asked nicely.

The failure this guards against left no traceback and no log line, and looked
exactly like a healthy session: a stall report written by a watchdog whose
every thread was idle, a main thread parked in Qt's event loop holding nothing,
a GPU with no work outstanding, and a window still showing its last frame at
full size. Nothing was stuck. ``rendercanvas``' scheduler had simply stopped
asking for frames -- and because the two facts it decides that on (is the
window on screen, how big is it) are each written from a single window event
and never re-derived, nothing was ever going to make it start again.

So the application asks the window itself, on a timer of its own, and pushes
the answer back into the canvas whether or not it has changed. What is asserted
here is that level-triggered property, in both directions: a canvas that has
been told something false about the window is corrected on the next poll, and a
loop that has stopped being driven gets a frame forced into it -- while a loop
that is merely slow, and a window that really is off screen, are left alone.
"""

from __future__ import annotations

import pytest

from anastomosis import diagnostics
from anastomosis.app import AppOptions, Application


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class FakeSizeInfo(dict):
    """The canvas' cached size, written where ``rendercanvas`` writes it."""

    def __init__(self, width: int, height: int) -> None:
        super().__init__()
        self["physical_size"] = (width, height)
        self.writes: list[tuple[int, int, float]] = []

    def set_physical_size(self, width: int, height: int, ratio: float) -> None:
        self["physical_size"] = (int(width), int(height))
        self.writes.append((int(width), int(height), float(ratio)))


class FakeSubwidget:
    """The inner render widget: Qt's geometry, and the canvas' idea of it.

    The two are separate on purpose. That they can disagree -- the widget
    knowing its real size while the canvas holds whatever the last resize event
    left behind -- is the whole subject of half these tests.
    """

    def __init__(self, width: int, height: int, ratio: float = 1.0) -> None:
        self._width = width
        self._height = height
        self._ratio = ratio
        self._size_info = FakeSizeInfo(width, height)
        # What the scheduler would do with it: enabled draws, disabled does not.
        self.visible = True
        self.visibility_calls: list[bool] = []

    # -- Qt's side, which stays right whether or not events were delivered
    def width(self):
        return self._width

    def height(self):
        return self._height

    def devicePixelRatioF(self):  # noqa: N802 - Qt naming
        return self._ratio

    def resize(self, width: int, height: int, *, tell_the_canvas: bool) -> None:
        """Resize the window, optionally dropping the event on the floor."""
        self._width, self._height = width, height
        if tell_the_canvas:
            self._size_info.set_physical_size(
                round(width * self._ratio + 0.01),
                round(height * self._ratio + 0.01),
                self._ratio,
            )

    # -- the canvas' side
    def _set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        self.visibility_calls.append(bool(visible))


class FakeCanvas:
    """A canvas shaped like ``rendercanvas.qt.QRenderCanvas``.

    ``force_draw`` calls the draw function, because that is the property the
    kick depends on: it draws without going through the scheduler, so a loop
    the scheduler has stopped driving still advances.
    """

    def __init__(self, width: int = 1920, height: int = 1080, ratio: float = 1.0):
        self._subwidget = FakeSubwidget(width, height, ratio)
        self._draw = None
        self._shown = True
        self._minimised = False
        self.forced = 0
        self.requests = 0
        self.drawing = False

    # -- window state
    def isVisible(self):  # noqa: N802 - Qt naming
        return self._shown

    def isMinimized(self):  # noqa: N802
        return self._minimised

    def minimise(self, minimised: bool = True) -> None:
        self._minimised = minimised

    # -- canvas API
    def get_physical_size(self):
        return self._subwidget._size_info["physical_size"]

    def request_draw(self, draw_function=None):
        self.requests += 1
        if draw_function is not None:
            self._draw = draw_function

    def force_draw(self):
        if self.drawing:
            raise RuntimeError("Cannot force a draw while drawing.")
        self.forced += 1
        if self._draw is not None:
            self._draw()


class Clock:
    """A hand-cranked stand-in for ``time.perf_counter``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def app(tmp_path, monkeypatch):
    """An application wired to a fake window, on a clock the test cranks."""
    clock = Clock()
    # The reconciler reads the clock through the module the way the frame loop
    # does, so the module's clock is what the test cranks -- patched on
    # ``app`` alone rather than on ``time`` itself, which everything else in
    # the process is also reading.
    from anastomosis import app as app_module

    monkeypatch.setattr(
        app_module, "time",
        type("Clocked", (), {"perf_counter": staticmethod(clock),
                             "time": staticmethod(clock)}),
    )

    application = Application(AppOptions(
        ui=False,
        config_path=tmp_path / "config.toml",
        checkpoint=False,
        stall_seconds=0.0,  # no thread: `poll` is driven by hand
        diagnostics_dir=tmp_path / "diagnostics",
    ))
    application.watchdog = diagnostics.StallWatchdog(
        report_dir=tmp_path / "diagnostics",
        stall_seconds=10.0,
        clock=clock,
        snapshot=lambda: {"tick": 4242},
    )
    application.canvas = FakeCanvas()
    application._size = (1920, 1080)
    # What the loop does on being drawn, reduced to the part the poll can see.
    application.canvas.request_draw(application.watchdog.frame)
    application.clock = clock
    return application


def settle(app) -> None:
    """One poll with the loop drawing normally, to establish a baseline."""
    app.watchdog.frame()
    app._reconcile_window()


# ---------------------------------------------------------------------------
# Forcing a frame
# ---------------------------------------------------------------------------


def test_a_loop_that_stops_being_asked_gets_a_frame_forced(app):
    """The failure itself: nothing is stuck, and nothing draws again."""
    settle(app)

    # The scheduler stops asking. Time passes; no frames.
    app.clock.advance(5.0)
    app._reconcile_window()

    assert app.canvas.forced == 1, (
        "the loop went five seconds without a frame and nothing forced one; "
        "this is the freeze the poll exists to end"
    )
    assert app.watchdog.frames == 2, "the forced frame never reached the loop"


def test_a_loop_that_is_merely_slow_is_left_alone(app):
    """A frame every poll is a slow session, not a stopped one."""
    settle(app)

    for _ in range(5):
        app.clock.advance(2.5)
        app.watchdog.frame()
        app._reconcile_window()

    assert app.canvas.forced == 0


def test_a_frame_still_in_flight_is_not_kicked(app):
    """A long frame belongs to the watchdog, which knows the phase it is in."""
    settle(app)
    app.canvas.drawing = True

    app.clock.advance(30.0)
    app._reconcile_window()

    assert app._kicks == 0, "a frame in flight was counted as a stopped loop"


def test_a_scheduler_that_never_comes_back_leaves_a_report(app, tmp_path):
    """The loop is alive, so the watchdog will never write one by itself."""
    settle(app)

    for _ in range(4):
        app.clock.advance(5.0)
        app._reconcile_window()

    assert app._kicks >= 3
    reports = list((tmp_path / "diagnostics").glob("dump-*.txt"))
    assert len(reports) == 1, "the session is being carried by the poll unrecorded"
    assert "scheduler" in reports[0].read_text()


# ---------------------------------------------------------------------------
# Telling the canvas what the window is doing
# ---------------------------------------------------------------------------


def test_the_canvas_is_told_the_window_is_up_on_every_poll(app):
    """The level-triggered property, which is the point of polling at all.

    A canvas paused by an event nobody can see -- a state change during a
    fullscreen transition, a spurious minimise -- is never un-paused by
    anything, because the un-pausing is edge-triggered too. Re-asserting it
    every poll costs a bool and bounds that fault at one poll.
    """
    settle(app)
    app.canvas._subwidget.visible = False  # something paused it behind our back

    app.watchdog.frame()
    app._reconcile_window()

    assert app.canvas._subwidget.visible is True
    assert app.canvas._subwidget.visibility_calls == [True, True], (
        "the window's state is only pushed when this application's own view "
        "of it changes, which is the same edge trigger that caused the freeze"
    )


def test_an_off_screen_window_is_left_paused(app):
    """Not being asked to paint is the correct answer to being minimised."""
    settle(app)
    app.canvas.minimise()

    app.clock.advance(30.0)
    app._reconcile_window()

    assert app.canvas.forced == 0, "a minimised window was made to render"
    assert app.canvas._subwidget.visible is False
    assert app.watchdog.paused, "the watchdog was left to call this a stall"


def test_a_window_that_comes_back_is_drawn_again(app):
    settle(app)
    app.canvas.minimise()
    app.clock.advance(30.0)
    app._reconcile_window()

    app.canvas.minimise(False)
    app._reconcile_window()

    assert app.canvas._subwidget.visible is True
    assert app.watchdog.paused is None
    assert app.canvas.forced == 0, (
        "a window that had only just come back was already being kicked; the "
        "time it spent away is not time it was unasked"
    )

    # And if the scheduler does not pick it up again by itself, it is kicked.
    app.clock.advance(5.0)
    app._reconcile_window()

    assert app.canvas.forced == 1


def test_a_canvas_that_cannot_say_is_not_told_anything(app):
    """glfw windows, offscreen canvases, and every stand-in the suite has."""

    class Windowless:
        def get_physical_size(self):
            return (64, 64)

    app.canvas = Windowless()
    settle(app)

    app.clock.advance(30.0)
    app._reconcile_window()  # must not raise

    assert app._window_visible is None


# ---------------------------------------------------------------------------
# Correcting the size
# ---------------------------------------------------------------------------


def test_a_stale_size_is_corrected_from_the_window(app):
    """A resize event that never arrived, which the canvas cannot detect."""
    settle(app)
    app.canvas._subwidget.resize(1280, 720, tell_the_canvas=False)

    app.watchdog.frame()
    app._reconcile_window()

    assert app.canvas.get_physical_size() == (1280, 720)


def test_a_zero_size_is_corrected_rather_than_cancelling_every_frame(app):
    """The worst shape of it: a canvas of no size refuses every draw."""
    settle(app)
    # What a minimise leaves behind when the restore is never delivered.
    app.canvas._subwidget._size_info.set_physical_size(0, 0, 1.0)

    app.clock.advance(5.0)
    app._reconcile_window()

    assert app.canvas.get_physical_size() == (1920, 1080)
    assert app.canvas.forced == 1, "the size was fixed and nothing drew"


def test_a_window_that_really_has_no_size_is_left_alone(app):
    """Zero is never written back, and a zero-sized window is never kicked."""
    settle(app)
    app.canvas._subwidget.resize(0, 0, tell_the_canvas=True)

    app.clock.advance(30.0)
    app._reconcile_window()

    assert app.canvas.get_physical_size() == (0, 0)
    assert app.canvas.forced == 0


def test_a_size_that_agrees_is_never_rewritten(app):
    """Otherwise every poll would report a resize the window never made."""
    settle(app)
    writes = len(app.canvas._subwidget._size_info.writes)

    for _ in range(5):
        app.clock.advance(2.0)
        app.watchdog.frame()
        app._reconcile_window()

    assert len(app.canvas._subwidget._size_info.writes) == writes


# ---------------------------------------------------------------------------
# Surviving its own failures
# ---------------------------------------------------------------------------


def test_a_poll_that_raises_disarms_itself_rather_than_the_session(app, monkeypatch):
    """A poll erroring every two seconds would bury the log it is written to."""

    def explode():
        raise RuntimeError("something under the poll broke")

    monkeypatch.setattr(app, "_reconcile_window", explode)
    app._window_poll = object()

    app._poll_window()  # must not raise

    assert app._window_poll is None


def test_a_window_that_answers_with_an_error_is_treated_as_unknown(app):
    """Asking a window that is halfway through going away must cost nothing."""

    def explode():
        raise RuntimeError("the window went away mid-poll")

    app.canvas.isVisible = explode

    app.clock.advance(30.0)
    app._reconcile_window()  # must not raise

    assert app._window_visible is None
    assert app.canvas.forced == 1, (
        "a window that could not say whether it was up was assumed to be down"
    )


def test_shutdown_stops_the_poll(app):
    app._window_poll = None
    app.shutdown()

    app._poll_window()

    assert app.canvas.forced == 0, "a poll drew a frame after the world was saved"


# ---------------------------------------------------------------------------
# The API this reaches for
# ---------------------------------------------------------------------------


def test_the_canvas_internals_the_recovery_reaches_for_still_exist():
    """The recovery pokes at ``rendercanvas`` internals; say so out loud.

    Correcting a stale size and re-asserting visibility means writing where the
    backend's own event handlers write, because that is the only place those
    facts live. That is a deliberate reach past the public API, and the price
    of it is this test: an upgrade that renames any of it should fail here,
    loudly, rather than leave a recovery that silently never recovers.
    """
    from rendercanvas.base import BaseRenderCanvas
    from rendercanvas.core.scheduler import Scheduler
    from rendercanvas.core.size import SizeInfo

    assert callable(getattr(BaseRenderCanvas, "force_draw", None)), (
        "the only way to draw without the scheduler is gone"
    )
    assert callable(getattr(BaseRenderCanvas, "_set_visible", None)), (
        "the hook that un-pauses a canvas is gone"
    )
    assert callable(getattr(Scheduler, "set_enabled", None)), (
        "the scheduler no longer has the enabled state _set_visible drives"
    )
    assert callable(getattr(SizeInfo, "set_physical_size", None)), (
        "the canvas' cached size is no longer written through SizeInfo"
    )


def test_the_qt_canvas_keeps_its_state_on_the_inner_widget():
    """Which is why both are pushed into ``_subwidget`` and not the wrapper.

    The top-level Qt canvas subclasses the same base as the widget inside it,
    so ``canvas._set_visible(True)`` is not a mistake that raises -- it drives a
    scheduler the wrapper does not have, and does nothing at all.
    """
    pytest.importorskip(
        "PySide6.QtWidgets", reason="the Qt backend needs PySide6"
    )
    from rendercanvas.base import WrapperRenderCanvas
    from rendercanvas.qt import QRenderCanvas, QRenderWidget

    assert issubclass(QRenderCanvas, WrapperRenderCanvas)
    assert "_subwidget" in QRenderCanvas.__init__.__code__.co_names
    for name in ("isVisible", "isMinimized", "devicePixelRatioF", "width"):
        assert callable(getattr(QRenderCanvas, name, None)) or callable(
            getattr(QRenderWidget, name, None)
        ), f"the Qt canvas can no longer be asked {name}"
