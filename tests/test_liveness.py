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
from anastomosis.app import WINDOW_POLL_SECONDS, AppOptions, Application


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


class FakeScheduler:
    """``rendercanvas``' scheduler, reduced to the state that stops a session.

    Enabled is the whole of it. A disabled scheduler goes on ticking -- it
    processes events, it is not stuck, and nothing anywhere reports a fault --
    and simply never asks for a frame again, which is what a freeze of this
    shape actually is. Nothing outside the canvas can un-set it except by
    telling the canvas the window is visible, which is why the re-assertion
    below is a recovery and a forced frame is not.
    """

    def __init__(self) -> None:
        self._enabled = True
        self._mode = "continuous"
        self._ready_for_present = None
        self._just_cancelled_a_frame = False

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)


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
        # Spelled the way the base class' name mangling leaves it, because
        # that is the name the report reads it under.
        self._BaseRenderCanvas__scheduler = FakeScheduler()

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
        self._BaseRenderCanvas__scheduler.set_enabled(visible)


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
        # A canvas cancels every frame it is given -- forced ones included --
        # when it believes it has no size or has been closed. It says nothing
        # about it, which is the point: from outside, a kick that draws and a
        # kick that is thrown away look identical.
        self.cancels = False

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
        if self.cancels:
            return
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


def test_the_report_says_what_the_scheduler_was_doing(app, tmp_path):
    """Which of the three ways it stops is what decides how to fix it."""
    settle(app)
    app.canvas._subwidget._BaseRenderCanvas__scheduler._enabled = False

    assert "paused" in app._scheduler_summary()
    assert "continuous" in app._scheduler_summary()

    app.canvas._subwidget._BaseRenderCanvas__scheduler._enabled = True
    app.canvas._subwidget._BaseRenderCanvas__scheduler._ready_for_present = object()
    assert "waiting for a frame it asked for" in app._scheduler_summary()


def test_a_forced_frame_the_canvas_throws_away_is_not_counted_as_one(app):
    """A canvas that cancels every frame is a different fault, and worse.

    The kick used to count itself rather than the frame, so a session drawing
    nothing whatever was described in the report as one being carried at a
    frame every few seconds -- and the difference between those two is the
    difference between a window that creeps and a window that has stopped.
    """
    settle(app)
    app.canvas.cancels = True

    for _ in range(3):
        app.clock.advance(5.0)
        app._reconcile_window()

    assert app._kicks == 3
    assert app._blank_kicks == 3
    assert "3 of them drew nothing" in app._kick_summary()
    assert not app._carrying, (
        "the poll took over pacing from a canvas that draws nothing at all"
    )


# ---------------------------------------------------------------------------
# Carrying the session
# ---------------------------------------------------------------------------


def test_a_scheduler_that_will_not_come_back_has_its_pacing_taken_over(app):
    """A frame every four seconds is the freeze, not the recovery from it."""
    settle(app)

    for _ in range(3):
        app.clock.advance(5.0)
        app._reconcile_window()

    assert app._carrying, "the session was left frozen with the poll watching it"
    assert app._poll_interval == pytest.approx(1.0 / app.options.max_fps)

    # And now every pass is a frame, without waiting out the unasked leash.
    drawn = app.watchdog.frames
    for _ in range(5):
        app.clock.advance(app._poll_interval)
        app._reconcile_window()

    assert app.watchdog.frames == drawn + 5, (
        "the poll took over the frame clock and then ran at the old interval"
    )


def test_pacing_is_handed_back_the_moment_the_scheduler_asks_again(app):
    settle(app)
    for _ in range(3):
        app.clock.advance(5.0)
        app._reconcile_window()
    assert app._carrying

    # One frame this poll did not force is the whole signal.
    app.watchdog.frame()
    app.clock.advance(app._poll_interval)
    app._reconcile_window()

    assert not app._carrying
    assert app._poll_interval == pytest.approx(WINDOW_POLL_SECONDS)
    assert app._kicks == 0


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


def test_a_canvas_with_no_window_at_all_is_not_told_anything(app):
    """glfw windows, offscreen canvases, and every stand-in the suite has."""

    class Windowless:
        def get_physical_size(self):
            return (64, 64)

    app.canvas = Windowless()
    settle(app)

    app.clock.advance(30.0)
    app._reconcile_window()  # must not raise

    assert app._window_visible is None


def test_a_window_that_will_not_say_still_has_its_canvas_un_paused(app):
    """The second freeze, and the hole this whole mechanism had in it.

    A canvas paused by a minimise event that never had its counterpart is only
    ever un-paused by being told the window is up. The poll used to skip that
    push whenever the window would not answer -- and then force a frame anyway,
    which goes *around* the scheduler and leaves the pause exactly where it
    found it. The two together are a session that is drawn once every few
    seconds forever: nothing is stuck, nothing errors, and nothing will ever
    start again.

    So a window that cannot say is treated as up, which is the answer the kick
    on the next line already gives the same question.
    """
    settle(app)
    # Something paused the canvas -- a spurious minimise, a state change during
    # a fullscreen transition -- and the window stopped answering afterwards.
    app.canvas._subwidget._set_visible(False)
    app.canvas.isVisible = lambda: (_ for _ in ()).throw(
        RuntimeError("the window will not say")
    )

    app.clock.advance(5.0)
    app._reconcile_window()

    scheduler = app.canvas._subwidget._BaseRenderCanvas__scheduler
    assert scheduler._enabled is True, (
        "a canvas paused behind the application's back was left paused, so "
        "the scheduler will never ask for another frame"
    )
    assert app._window_visible is None, "the report must not invent an answer"
    assert app.watchdog.paused is None, (
        "a window that would not answer was recorded as deliberately parked"
    )


def test_a_probe_that_flaps_does_not_disarm_the_kick(app):
    """Losing the answer is not the window coming back from being away.

    A window that comes back from being off screen has its unasked clock
    restarted, because it could not have been drawn while it was away. If a
    probe that merely stops answering did the same, every poll would restart
    that clock and no frame would ever be forced again.
    """
    settle(app)
    answers = iter([True, None, True, None, True, None, True, None])
    app._window_is_up = lambda: next(answers)

    for _ in range(4):
        app.clock.advance(2.0)
        app._reconcile_window()

    assert app.canvas.forced >= 1, (
        "a window flapping between an answer and none disarmed the recovery"
    )


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
    assert app.canvas._subwidget.visibility_calls[-1] is True, (
        "the canvas was left holding whatever it believed, which is the one "
        "thing this poll exists to stop"
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
    # And the state a report reads to say *which* way the scheduler stopped:
    # paused, waiting for a frame it asked for, or asking into a canvas that
    # throws every frame away. Read under the name the base class' mangling
    # leaves behind, since that is how the report has to reach it.
    assert "_BaseRenderCanvas__scheduler" in (
        BaseRenderCanvas.__init__.__code__.co_names
    ), "the canvas no longer keeps its scheduler where the report reads it"
    for owner, name in (
        (Scheduler.set_enabled, "_enabled"),
        (Scheduler.set_update_mode, "_mode"),
        (Scheduler.__init__, "_ready_for_present"),
    ):
        assert name in owner.__code__.co_names, (
            f"the scheduler no longer keeps {name}, which is what a stall "
            f"report reads to say whether it was asking for frames"
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
