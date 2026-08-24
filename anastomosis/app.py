"""Application: window, frame pacing, hot reload, and the budget governor.

The central mechanism is that **simulation and presentation are decoupled**.
Real time accumulates; simulation ticks are consumed from that accumulator at
``sim_hz``; frames are drawn at the canvas rate with a fractional interpolation
between the last two sim states. This is what makes the frame budget adaptive
without visible artefacts -- when a frame runs long the governor lowers the
tick rate, which the interpolator hides completely. It never changes
resolution at runtime, because that would be a visible discontinuity.

The same reasoning governs window resizes. A resize is a change to the
presentation, not to the world: the engine's output targets follow the window
and the simulation is left running exactly as it was. Rebuilding it would
reseed every field and every agent, which is a hard cut in the middle of a
session intended to last for days.

Launching works the same way round. A simulation's geometry belongs to the
field, not to the window it happens to be shown in, so a launch that finds a
saved field builds its engine in *that field's* shape and presents it into
whatever window it was given -- see ``_start_engine``. The alternative, asking
whether the saved field fits the window, threw away hours of maturity every
time a window opened at a different size.
Closing the window, by contrast, *is* the end of the world, and ``shutdown``
is the one path out: checkpoint, stop the watcher, close the panel, stop the
loop. It is idempotent and hooked from every direction the close can arrive
from -- the canvas's close event, Qt's ``aboutToQuit``, a signal, or simply
``loop.run()`` returning -- because the window is the only thing the user
thinks of as "the application", and anything of it left running afterwards is
a process they have to go and kill by hand.
"""

from __future__ import annotations

import dataclasses
import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import wgpu

from . import audio as audio_module
from . import checkpoint as checkpoint_module
from . import config as config_module
from . import device as device_module
from . import diagnostics as diagnostics_module
from . import engine as engine_module
from . import events as events_module
from . import power as power_module
from . import rhizotron as rhizotron_module
from . import volume as volume_module
from . import window as window_module

log = logging.getLogger(__name__)

# The backends -- the two fungal depth backends (DESIGN.md §5) and the
# rhizotron's soil column (§15) -- and everything that differs between them
# from this side: which engine class to build, and which geometry class
# derives a fresh field's shape. All present the same interface -- tick,
# render, resize, read_stats -- so nothing below this table has to know which
# one is running.
#
# Held as module and attribute names rather than as the classes themselves,
# and resolved when an engine is actually built. Binding the classes here would
# capture whatever they were at import time, which is one of those decisions
# that costs nothing until something wants to substitute one -- the suite
# stands a failing engine in for the real one to check that a launch survives a
# device that will not allocate the saved geometry.
BACKEND_CLASSES = {
    "layered": (engine_module, "Engine", "Geometry"),
    "volumetric": (volume_module, "VolumeEngine", "VolumeGeometry"),
    "rhizotron": (rhizotron_module, "RhizotronEngine", "RhizotronGeometry"),
}


def backend_classes(name: str) -> tuple[type, type]:
    """``(engine class, geometry class)`` for a backend name."""
    module, engine_name, geometry_name = BACKEND_CLASSES[name]
    return getattr(module, engine_name), getattr(module, geometry_name)

# How long a new window size must hold before the output targets follow it.
# Dragging a window edge reports a new size every frame; coalescing them keeps
# the drag from reallocating render targets dozens of times.
RESIZE_SETTLE = 0.15

# How often to ask the window what it is actually doing, and how long the frame
# loop may go unasked before a frame is forced.
#
# Two seconds because this is a reconciliation, not a heartbeat: nothing here
# needs to be prompt, it only needs to happen without anybody asking. Three
# before forcing a frame, so that a checkpoint readback, a shader rebuild or a
# compositor that has skipped a beat is never mistaken for a loop that has
# stopped being driven -- those all end well inside a second, and a frame
# forced on top of one costs a frame.
WINDOW_POLL_SECONDS = 2.0
UNASKED_SECONDS = 3.0

# How many forced frames in a row mean the render scheduler is not coming back
# on its own, and the session is being carried by the poll. Worth a report at
# that point: the loop is alive, so the watchdog will never write one, and the
# stacks are the only thing that says why the scheduler stopped.
#
# It is also the point at which the poll stops reconciling and starts pacing.
# Three forced frames at this interval is ten seconds of a window that has not
# moved, which is a freeze however carefully it is being carried; from here the
# poll runs at the frame interval instead and the session keeps its frame rate
# until the scheduler comes back -- see `_start_carrying`.
KICKS_BEFORE_A_REPORT = 3

# How long to leave a lost device alone before asking for another. A driver
# reset takes seconds, and a laptop switching graphics can take longer; what
# this bounds is the retry, not the wait, so the cost of it being generous is
# at most this much extra black screen and the cost of it being small is a log
# nobody can read.
DEVICE_RETRY_SECONDS = 2.0

# --------------------------------------------------------------------------
# The budget governor's two levers. DESIGN.md §8, §8.3.
# --------------------------------------------------------------------------

# The tick rate is the first lever and the cheap one: the motion-compensated
# interpolator hides a change in it completely, so it may be moved freely. The
# floor is where the interpolator starts extrapolating far enough to matter.
SIM_SCALE_FLOOR = 0.35

# The presented frame rate is the second lever, reached only once the first is
# spent. It exists because §8's original governor assumed the simulation was
# what cost -- true on the card of §8.1, and not true on an integrated GPU,
# where the per-frame render work is at least as likely to be what is over
# budget. Lowering the tick rate against a render-bound frame degrades motion
# quality and recovers nothing at all (§8.3).
#
# Unlike the tick rate this one is visible, which is why it is second and why
# it is restored first. It is never a safety question in the other direction:
# the flash bound's per-frame lightness allowance is sized against `max_fps`
# rather than against the rate achieved, so presenting *fewer* frames only
# slows the worst case -- see `config.validate`. Presenting more is forbidden,
# and `_uncapped_fps` is where that is enforced.
FPS_SCALE_FLOOR = 0.5
# Below this a pan reads as a sequence of steps rather than as motion, which is
# the artefact all of §8's pacing exists to avoid. Nothing may present slower.
MIN_PRESENT_FPS = 12

# Signals that mean "stop", handled so that a `kill` or a session logout ends
# the same way closing the window does -- with the field on disk. SIGINT is in
# the list for symmetry; the render loop installs its own handler for that one
# while it runs, which reaches the same shutdown by closing the canvas.
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


@dataclass
class AppOptions:
    width: int = 1280
    height: int = 720
    max_fps: int = 30
    vsync: bool = True
    fullscreen: bool = False
    config_path: Path | None = None
    seed: int | None = None
    # Which depth backend to run. ``None`` takes the config's, which is the
    # normal case; the CLI flag is for trying the other one without editing a
    # file. Structural either way -- it decides which engine class exists, so
    # it takes effect when a field is grown.
    backend: str | None = None
    # How wide the volumetric slab is, from `config.VOLUME_DETAIL`. ``None``
    # takes the config's, as with ``backend``, and it is structural in the same
    # way -- it decides the shape a field is grown in. Ignored under the
    # layered backend, which has no slab.
    volume_detail: str | None = None
    # Which class of GPU to ask the platform for, from `device.GPU_CHOICES`.
    # A property of the launch rather than of the config, like ``backend``:
    # which card a session runs on is a fact about the machine it was started
    # on, and hot-reloading it would mean rebuilding the world.
    gpu: str = device_module.DEFAULT_GPU_CHOICE
    # ``render.base_scale``, pinned from the command line. ``None`` leaves the
    # config's, which is the normal case; the flag is for trying a cheaper
    # simulation without editing a file. Structural, so it takes effect when a
    # field is grown -- and, being an explicit answer to the question the cell
    # ceiling exists to answer, it turns the automatic ceiling off (§8.3).
    scale: float | None = None
    ui: bool = True
    telemetry_seconds: float = 60.0
    # Stall diagnostics. On by default, because the freeze it exists to catch
    # is by definition the one the user cannot reproduce on request -- it has
    # to already be armed the first time it happens. Zero disables the
    # watchdog; the crash handler is free either way and is always installed.
    stall_seconds: float = diagnostics_module.DEFAULT_STALL_SECONDS
    diagnostics_dir: Path | None = None
    # Checkpointing. Resuming is the default: the whole point of a mature field
    # is that it took hours to grow, so throwing it away on every launch would
    # be the surprising behaviour, not the safe one.
    checkpoint: bool = True
    resume: bool = True
    checkpoint_path: Path | None = None
    checkpoint_seconds: float = checkpoint_module.DEFAULT_INTERVAL_SECONDS


class Application:
    # Class-level so a partially built Application -- tests probe single
    # methods on objects made with ``__new__`` -- passes through
    # `_sync_audio` (reached from any `_retarget`) as a no-op instead of
    # tripping on attributes ``__init__`` has not set yet.
    audio: audio_module.AudioDrive | None = None
    _audio_started = False
    # Same reasoning: `_session_limits` is reached from every `_retarget`, and
    # the cell ceiling is settled later than the first one of those (it needs
    # the adapter). Zero is "no ceiling", which is what a session on a discrete
    # card keeps.
    _cell_budget = 0
    device_info: device_module.DeviceInfo | None = None

    def __init__(self, options: AppOptions) -> None:
        self.options = options
        self.config_path = options.config_path or config_module.default_config_path()
        self.config = config_module.load(self.config_path)
        self._pin_volume_detail()
        resolved = self._session_limits(self.config.resolve())

        self.ramp = config_module.ParamRamp(resolved)
        self.params = resolved
        self.scheduler = events_module.EventScheduler(seed=options.seed)

        self.backend = config_module.normalise_backend(
            options.backend or self.config.backend)

        # The resonance mode's ears (DESIGN.md §16). Built unconditionally --
        # like the watchdog, so nothing downstream needs guarding -- but
        # capture only opens while that mode is active; `_sync_audio` is what
        # starts and stops it as the mode comes and goes.
        self.audio = audio_module.AudioDrive(
            device=self.config.audio_device or None)
        self._audio_started = False
        self._sync_audio()
        # One saved field per backend, so switching does not destroy the one
        # switched away from -- see `checkpoint.default_checkpoint_path`.
        self.checkpoint_path = (
            options.checkpoint_path
            or checkpoint_module.default_checkpoint_path(self.backend)
        )
        # Built before anything can stall, and independently of whether it is
        # enabled, so that the frame loop's `mark` calls never need guarding.
        self.watchdog = diagnostics_module.StallWatchdog(
            report_dir=options.diagnostics_dir,
            stall_seconds=options.stall_seconds,
            snapshot=self.diagnostic_snapshot,
        )
        self._started_at = time.time()
        # The pending flag the frame loop acts on, and a record for the stall
        # report: once a rebuild has succeeded the flag is cleared, and a
        # session that survived a device loss should still say that it did.
        self._device_lost: str | None = None
        self._device_losses = 0
        # When a rebuild was last attempted, so a driver that is still coming
        # back is asked again on a human timescale rather than every frame.
        self._device_retry_at = float("-inf")
        self._resume = bool(options.resume)
        self._saver = checkpoint_module.BackgroundSaver()
        self._last_checkpoint = time.perf_counter()
        self._checkpoint_saved_at: float | None = None
        self.resumed_from: str | None = None

        self.canvas = None
        self.device = None
        self.device_info = None
        self.engine = None
        self.present_context = None
        self.target_format = None
        self.panel = None
        self.loop = None
        self.have_qt = False
        # Set up with the window, in ``_install_fullscreen``. None means this
        # backend has no window that can be made fullscreen.
        self._fullscreen = None

        # Shutdown state. Once ``_stopped`` is set the world has been saved and
        # taken down; nothing may tick, draw or checkpoint again.
        self._stopped = False
        self._stop_requested = False
        self._previous_signal_handlers: dict[int, object] = {}
        self._close_filter = None

        # Pacing state.
        self._accumulator = 0.0
        self._last_time = time.perf_counter()
        self._last_frame_time = 0.0
        self._frame_times: list[float] = []
        self._sim_hz_scale = 1.0
        # The governor's second lever, and the rate currently being asked for.
        # `_present_fps` starts at the launch's cap so the first governor
        # window has a real slot to measure against rather than a zero.
        self._fps_scale = 1.0
        self._present_fps = max(int(options.max_fps), 1)
        # Where the electricity is coming from (DESIGN.md §8.3). Built here and
        # started with the window, so a partially built application -- and the
        # whole of the test suite -- can read `on_battery` without a thread
        # ever having run.
        self.power = power_module.PowerSource(
            poll_seconds=self.params.power.poll_seconds)
        self._battery = False
        self._last_telemetry = time.perf_counter()
        self._watcher = None
        self._size = (0, 0)
        self._pending_size: tuple[int, int] | None = None
        self._pending_since = 0.0

        # Window reconciliation. ``_window_visible`` is what the window last
        # said about itself rather than what was last pushed into the canvas,
        # because the push happens every poll -- see `_reconcile_window`.
        self._window_poll = None
        self._window_visible: bool | None = None
        self._paused_since: float | None = None
        # What the last poll saw, for the stall report to read. Plain
        # attributes because the watchdog thread is what reads them, and it
        # may not ask Qt anything -- see `diagnostic_snapshot`.
        self._polled_at: float | None = None
        self._canvas_size: tuple[int, int] = (0, 0)
        # None until a window has been found that can be asked its own size.
        self._window_geometry: tuple[int, int] | None = None
        self._frames_seen = 0
        self._frames_at = time.perf_counter()
        self._kicks = 0
        self._kicked_at: float | None = None
        # Forced frames the canvas threw away without drawing. Counted apart
        # from the kicks because it is a different fault: a scheduler that has
        # stopped asking is recoverable from out here, and a canvas that
        # cancels every frame it is given is not -- see `_kick_the_loop`.
        self._blank_kicks = 0
        # Whether the poll has taken over pacing from the scheduler, and how
        # often it runs while it has. The interval is an attribute rather than
        # the constant so the fallback path re-arms itself at whichever one is
        # in force.
        self._carrying = False
        self._poll_interval = WINDOW_POLL_SECONDS

    # -- setup --------------------------------------------------------------

    @staticmethod
    def _import_qt_backend():
        """Import the Qt render backend, bound to PySide6.

        Deliberately ``rendercanvas.pyside6`` and not ``rendercanvas.qt``:
        the latter binds to whichever Qt library is *already* in
        ``sys.modules`` and raises ImportError when none is, which -- since we
        import it before anything has touched PySide6 -- happens every single
        run. ``rendercanvas.pyside6`` names the binding explicitly, so the only
        ImportError left is the one that genuinely means "PySide6 is missing".
        """
        from rendercanvas.pyside6 import RenderCanvas, loop

        return RenderCanvas, loop

    def _make_canvas(self):
        title = "anastomosis"
        kwargs = dict(
            size=(self.options.width, self.options.height),
            title=title,
            max_fps=self.options.max_fps,
            vsync=self.options.vsync,
            update_mode="continuous",
        )

        if self.options.ui:
            try:
                RenderCanvas, loop = self._import_qt_backend()
            except ImportError:
                log.warning(
                    "PySide6 not installed, falling back to the default backend; "
                    "the control panel will be unavailable. Install with "
                    "'pip install anastomosis[ui]'."
                )
            except Exception as exc:
                # Any other import-time failure -- a broken Qt install, an ABI
                # mismatch. Worth saying out loud with the real reason, because
                # the fallback silently costs the control panel.
                log.warning(
                    "the Qt backend is unavailable (%s), falling back to the "
                    "default backend; the control panel will be unavailable.",
                    exc,
                )
            else:
                canvas = RenderCanvas(**kwargs)
                log.info("using the Qt backend (control panel available)")
                return canvas, loop, True

        from rendercanvas.auto import RenderCanvas, loop

        return RenderCanvas(**kwargs), loop, False

    def _listen_for_device_loss(self) -> None:
        """Install the error and device-lost listeners on the current device.

        The device is captured in the closure rather than read back off `self`
        when the event fires, so `_on_device_lost` can tell an event about the
        device it is holding from one about a device it has already thrown
        away -- see there.
        """
        device = self.device
        device_module.install_error_handler(
            device, lambda event: self._on_device_lost(event, device))

    def _configure_surface(self) -> None:
        """Point the canvas at the current device.

        Its own method because it happens twice: once at startup, and again
        every time a lost device is replaced (`_rebuild_device`). The context
        object survives, but everything it holds is bound to a device that no
        longer exists, so all of it is asked again.
        """
        self.present_context = self.canvas.get_context("wgpu")
        self.target_format = self.present_context.get_preferred_format(
            self.device.adapter
        )
        self.present_context.configure(
            device=self.device, format=self.target_format
        )

    def setup(self) -> None:
        # First, and before the GPU is touched: a driver that crashes on
        # device creation should leave stacks behind too.
        diagnostics_module.install_crash_handler(self.options.diagnostics_dir)

        self.canvas, self.loop, self.have_qt = self._make_canvas()

        self.device, self.device_info = device_module.request_device(
            gpu=self.options.gpu)
        self._listen_for_device_loss()
        self._configure_surface()

        # Between the adapter and the first geometry, because it decides what
        # shape that geometry is.
        self._settle_cell_budget()
        # And before the first frame, so a session started on battery is
        # already backed off rather than backing off a minute in.
        self._start_power_watch()

        width, height = self.canvas.get_physical_size()
        self._start_engine(width, height)
        self._size = (width, height)

        self._start_hot_reload()
        self._watch_for_close()
        self._install_fullscreen()
        # Last, so that building the engine and restoring a large checkpoint --
        # both legitimately slow, and neither of them the frame loop -- cannot
        # be reported as a stall.
        self.watchdog.start()

    # -- fullscreen ---------------------------------------------------------

    def _install_fullscreen(self) -> None:
        """Pick the fullscreen strategy, bind F11, and honour ``--fullscreen``.

        The hotkey is bound whether or not a strategy was found, so that a
        backend without one says so when the key is pressed rather than
        silently doing nothing.
        """
        self._fullscreen = window_module.controller_for(self.canvas)
        if self._fullscreen is None:
            log.info(
                "fullscreen is not available on this backend; resize manually"
            )
        else:
            log.debug("fullscreen via the %s backend", self._fullscreen.name)

        try:
            self.canvas.add_event_handler(self._on_key, "key_down")
        except Exception as exc:  # pragma: no cover - a canvas with no events
            log.debug("could not bind the fullscreen hotkey: %s", exc)

        if self.options.fullscreen:
            self.set_fullscreen(True)

    def _on_key(self, event) -> None:
        """F11 toggles borderless fullscreen.

        Modifiers are not checked: F11 is not a prefix for anything else here,
        and a user reaching for it with a stray Shift held down means the same
        thing by it either way.
        """
        if event.get("key") == window_module.FULLSCREEN_KEY:
            self.toggle_fullscreen()

    @property
    def fullscreen(self) -> bool:
        return self._fullscreen is not None and self._fullscreen.is_fullscreen()

    def toggle_fullscreen(self) -> bool:
        return self.set_fullscreen(not self.fullscreen)

    def set_fullscreen(self, enabled: bool) -> bool:
        """Go borderless fullscreen, or come back. Returns the state reached.

        Only the window changes. The canvas reports its new size on the next
        frame and ``_follow_canvas_size`` rebuilds the presentation chain from
        there, exactly as it does for a dragged window edge -- so a toggle
        costs the field nothing, however long it has been growing.

        Never raises. A window manager that refuses the state change is a
        disappointment, not a reason to end a session that may have been
        running for days.
        """
        if self._fullscreen is None:
            log.info(
                "fullscreen is not available on this backend; resize the "
                "window manually"
            )
            return False
        try:
            self._fullscreen.set(enabled)
        except Exception as exc:
            log.warning(
                "could not %s fullscreen: %s",
                "enter" if enabled else "leave", exc,
            )
            return self.fullscreen
        log.info("fullscreen %s", "on" if self.fullscreen else "off")
        return self.fullscreen

    # -- window state -------------------------------------------------------

    def _install_window_poll(self) -> None:
        """Start the timer that keeps the canvas honest about the window.

        The frame loop does not drive itself. ``rendercanvas`` owns a scheduler
        that asks for each frame, and everything this application does happens
        because that scheduler asked; when it stops asking, the loop is not
        wedged and not slow, it is simply never called again. The window keeps
        showing the last frame it was given -- under the Qt backend the last
        bitmap is re-blitted on every expose -- so a session in that state
        looks completely ordinary and is completely stopped. It stays that way
        until the process is killed.

        Two things can put it there, and both are the same shape: the scheduler
        pauses when the backend reports the window minimised, and it cancels
        every frame when the canvas' cached size is zero. Neither fact is ever
        re-derived. Both are written from a single window event -- a state
        change, a resize -- and a window event that does not arrive, or that
        arrives while the native window is being rebuilt (which is exactly what
        a fullscreen transition can do), leaves the canvas holding a belief
        about the window that nothing will ever correct.

        So this asks. Every couple of seconds it reads what the window itself
        says -- on screen or not, and how big -- and pushes that into the
        canvas whether or not it has changed, which turns an edge-triggered
        fact into a level-triggered one: a missed event, or a spurious one,
        costs one poll instead of the session. If frames have stopped anyway,
        it forces one, which is the only route back from a scheduler that is
        waiting for a frame it will never be told about.

        None of it runs in the frame loop, and none of it touches the GPU.
        """
        if self.canvas is None:
            return
        if self.have_qt:
            try:
                from PySide6 import QtCore
            except Exception as exc:  # pragma: no cover - Qt was just here
                log.debug("could not start the window poll on Qt: %s", exc)
            else:
                timer = QtCore.QTimer()
                timer.timeout.connect(self._poll_window)
                timer.start(int(self._poll_interval * 1000))
                # Held on the application: an unparented QTimer that nothing
                # references is collected, and a collected timer never fires.
                self._window_poll = timer
                log.debug("window poll armed every %.0fs", WINDOW_POLL_SECONDS)
                return

        # Every other backend gets the same poll from the render loop's own
        # timer. Weaker, because it shares the machinery the scheduler runs on
        # -- but the part of that machinery which stops is the scheduler's own
        # coroutine, and a call_later placed beside it keeps running.
        self._arm_loop_poll()

    def _arm_loop_poll(self) -> None:
        call_later = getattr(self.loop, "call_later", None)
        if not callable(call_later):
            log.debug("this loop cannot schedule the window poll; skipping it")
            return
        try:
            call_later(self._poll_interval, self._poll_window)
        except Exception as exc:
            log.debug("could not schedule the window poll: %s", exc)
            return
        self._window_poll = self.loop

    def _set_poll_interval(self, seconds: float) -> None:
        """Change how often the poll runs, on whichever timer is carrying it.

        The Qt timer is re-intervalled in place; the fallback path re-arms
        itself from the attribute at the end of every pass, so setting it is
        all there is to do there.
        """
        if self._poll_interval == seconds:
            return
        self._poll_interval = seconds
        poll = self._window_poll
        if poll is None or poll is self.loop:
            return
        setter = getattr(poll, "setInterval", None)
        if not callable(setter):
            return
        try:
            setter(int(seconds * 1000))
        except Exception as exc:  # pragma: no cover - Qt timer internals
            log.debug("could not change the window poll's interval: %s", exc)

    def _stop_window_poll(self) -> None:
        poll, self._window_poll = self._window_poll, None
        stop = getattr(poll, "stop", None)
        # The loop's own ``stop`` is not this timer's: when the fallback path
        # is in use the poll is re-armed from `_poll_window`, which stops of
        # its own accord once the application is down.
        if poll is self.loop or not callable(stop):
            return
        try:
            stop()
        except Exception as exc:  # pragma: no cover - Qt teardown ordering
            log.debug("could not stop the window poll: %s", exc)

    def _poll_window(self) -> None:
        """One pass, and the re-arm the fallback path needs."""
        if self._stopped or self._stop_requested or self.canvas is None:
            return
        try:
            self._reconcile_window()
        except Exception:
            # Same reasoning as the watchdog's own thread: this is worth one
            # traceback and not one every two seconds, and a session without
            # it is still a session.
            log.exception("the window poll failed; disarming it")
            self._window_poll = None
            return
        if self._window_poll is self.loop:
            self._arm_loop_poll()

    def _reconcile_window(self) -> None:
        """Push the window's real state into the canvas, and kick if needed."""
        self._polled_at = time.perf_counter()
        visible = self._window_is_up()
        self._tell_the_canvas(visible)
        width, height = self._reconcile_size()

        now = time.perf_counter()
        frames = self.watchdog.frames
        if frames != self._frames_seen:
            self._frames_seen = frames
            self._frames_at = now
            self._kicks = self._blank_kicks = 0
            self._stop_carrying()
            return

        # Nothing has been drawn since the last poll. Whether that is a fault
        # depends on whether anything *could* have been drawn: a window that is
        # off screen is not being asked to paint, and that is the one case
        # where a still frame loop is the right answer rather than a bug.
        if visible is False or width <= 0 or height <= 0:
            return
        # While carrying, every pass is a frame: the interval *is* the frame
        # interval, and waiting three seconds between them would be the freeze
        # this exists to end.
        if not self._carrying and now - self._frames_at < UNASKED_SECONDS:
            return
        self._kick_the_loop(now - self._frames_at, width, height)

    def _window_is_up(self) -> bool | None:
        """Whether the window says it is on screen. ``None`` if it cannot say.

        Probed rather than type-checked, for the same reason the fullscreen
        strategies are: this has to answer for a Qt window, a glfw one and the
        canvas doubles the suite hands it, and asking the object is the only
        question all three can answer.
        """
        shown = getattr(self.canvas, "isVisible", None)
        minimised = getattr(self.canvas, "isMinimized", None)
        if not callable(shown) or not callable(minimised):
            return None
        try:
            return bool(shown()) and not bool(minimised())
        except Exception as exc:  # pragma: no cover - a window mid-teardown
            log.debug("could not read the window's state: %s", exc)
            return None

    def _tell_the_canvas(self, visible: bool | None) -> None:
        """Re-assert the window's visibility, every poll, in all three cases.

        Pushed unconditionally rather than on a change, which is the whole
        point of doing it here: if this only spoke up when *its own* view
        changed, a stray event that paused the canvas a moment later would
        never be answered. Logged on a change, though -- the state is worth a
        line in the log, and two lines a second is not.

        The third case is a window that will not say, and it used to return
        here without pushing anything, which left exactly one way for the
        original fault to survive this poll: a canvas paused by a minimise
        event that never had its counterpart, plus a window that stopped
        answering, is a session that is never drawn again. A forced frame does
        not help -- it goes around the scheduler rather than through it, so it
        leaves the pause exactly where it found it, which is why such a session
        can be kicked for hours and never come back.

        So "cannot say" is pushed as "up". That is the same answer the kick
        already gives the same question -- a window that cannot say whether it
        is up is not assumed to be down -- and its worst case is one poll's
        drawing into a window that really was off screen, which the next poll
        that gets an answer undoes. What is *not* assumed is the rest of it:
        the watchdog is only told the loop is parked on purpose when the window
        actually said so, and the report says "cannot say" rather than
        inventing a state the window never claimed.
        """
        # Three-valued in, two-valued out: only a definite "off screen" pauses
        # anything.
        up = visible is not False
        if visible != self._window_visible:
            # A window that has just come back has not been unasked for the
            # time it spent away -- it could not have been drawn -- so the
            # clock starts from the moment it could. Only from a definite
            # absence, though: a probe that flaps between an answer and none
            # would otherwise reset the kick's clock on every poll and disarm
            # the recovery it is part of.
            came_back = up and self._window_visible is False
            self._window_visible = visible
            self._paused_since = None if up else time.perf_counter()
            self.watchdog.set_paused(
                None if up else "the window is not on screen"
            )
            if came_back:
                self._frames_at = time.perf_counter()
                self._kicks = self._blank_kicks = 0
            log.info(
                "render window %s",
                {
                    True: "is on screen again; resuming",
                    False: "is off screen; pausing until it comes back",
                    None: "will not say whether it is on screen; "
                          "assuming it is",
                }[visible],
            )

        subwidget = getattr(self.canvas, "_subwidget", None)
        setter = getattr(subwidget, "_set_visible", None)
        if callable(setter):
            try:
                setter(up)
            except Exception as exc:  # pragma: no cover - backend internals
                log.debug("could not set the canvas' visibility: %s", exc)

    def _reconcile_size(self) -> tuple[int, int]:
        """The canvas' physical size, corrected if the window disagrees.

        The canvas caches this and writes it in exactly one place: the
        backend's resize handler. A resize that never arrives leaves the cache
        saying something the window stopped being long ago, and a cache saying
        zero is worse than stale -- a canvas of no size cancels every frame it
        is asked for, for the rest of the session.

        So the cache is compared against the widget's own geometry, which Qt
        maintains whether or not the event was delivered, and corrected the way
        the backend's handler would have. Zero is never written back: a window
        that really is zero-sized is a window that really has nothing to draw.
        """
        try:
            cached = tuple(self.canvas.get_physical_size())
        except Exception as exc:  # pragma: no cover - a canvas mid-teardown
            log.debug("could not read the canvas' size: %s", exc)
            self._canvas_size = (0, 0)
            return (0, 0)
        self._canvas_size = cached

        subwidget = getattr(self.canvas, "_subwidget", None)
        size_info = getattr(subwidget, "_size_info", None)
        setter = getattr(size_info, "set_physical_size", None)
        if not callable(setter):
            return cached

        try:
            ratio = float(subwidget.devicePixelRatioF())
            # The rounding the backend's own resize handler applies, offset
            # and all: matching it is what keeps this from correcting a size
            # that was right, once per poll, forever.
            real = (
                round(float(subwidget.width()) * ratio + 0.01),
                round(float(subwidget.height()) * ratio + 0.01),
            )
        except Exception as exc:  # pragma: no cover - backend internals
            log.debug("could not read the window's geometry: %s", exc)
            return cached
        self._window_geometry = real

        if real == cached or real[0] <= 0 or real[1] <= 0:
            return cached
        log.warning(
            "the canvas has the window at %sx%s and the window has itself at "
            "%dx%d; correcting it", cached[0], cached[1], *real,
        )
        setter(real[0], real[1], ratio)
        self._canvas_size = real
        return real

    # -- what the poll saw, for a report written by a thread that cannot ask --

    def _window_summary(self) -> str:
        """The window, as of the last poll. Attribute reads only.

        The line a stall report was missing. A loop sitting in `idle` is
        either a window that stopped being asked to paint or one that stopped
        being drawn, and from inside the watchdog those are the same thing;
        this is the difference, written down by the thread that is allowed to
        ask. The two sizes are printed even when they agree, because their
        agreeing is itself the fact worth having.

        "Cannot say" is a state this reports and not a gap in it: what the
        window answered last is what goes here, including its having refused
        to answer. A report that carried the last *definite* answer instead
        would say "on screen" about a window that had stopped speaking, which
        is the one fact that would have explained the last freeze.
        """
        if self._polled_at is None:
            return "not polled yet"
        state = {True: "on screen", False: "off screen", None: "cannot say"}[
            self._window_visible
        ]
        summary = f"{state}, canvas {self._canvas_size[0]}x{self._canvas_size[1]}"
        if self._window_geometry is not None:
            summary += (
                f", window {self._window_geometry[0]}"
                f"x{self._window_geometry[1]}"
            )
        if self._paused_since is not None:
            gone = time.perf_counter() - self._paused_since
            summary += f", frames paused for {gone:.0f}s"
        return summary

    def _poll_summary(self) -> str:
        """Whether the poll itself is still running.

        A poll that has not run for minutes is not a detail: it means the Qt
        event loop stopped dispatching, which is a different fault from the
        one this poll was written for and would otherwise look identical from
        the watchdog's side.
        """
        if self._polled_at is None:
            return "armed, has not run yet" if self._window_poll else "not running"
        summary = f"last ran {time.perf_counter() - self._polled_at:.0f}s ago"
        if self._window_poll is None:
            # It disarmed itself, which it only does after a traceback -- and
            # that traceback is in the log rather than in here.
            summary += " (no longer armed)"
        return summary

    def _kick_summary(self) -> str:
        if not self._kicks:
            return "none"
        last = (
            f", last {time.perf_counter() - self._kicked_at:.0f}s ago"
            if self._kicked_at is not None else ""
        )
        blank = (
            f", {self._blank_kicks} of them drew nothing"
            if self._blank_kicks else ""
        )
        carrying = ", now pacing the session" if self._carrying else ""
        return f"{self._kicks} in a row{last}{blank}{carrying}"

    def _scheduler(self):
        """``rendercanvas``' frame scheduler for this canvas, or None.

        The same deliberate reach past the public API that the visibility push
        is, and for the same reason: what the scheduler believes about this
        window lives nowhere else, and a report that cannot say whether it was
        asking for frames cannot say why the session stopped. Read only from
        here -- the state is *written* through ``_set_visible``, which is the
        backend's own door.

        Under Qt the canvas that owns it is the inner widget; everywhere else
        it is the canvas itself. The attribute is name-mangled because it is
        private to the base class, which is also why it is spelled out rather
        than reached through a helper that might not exist.
        """
        canvas = getattr(self.canvas, "_subwidget", None)
        if canvas is None:
            canvas = self.canvas
        return getattr(canvas, "_BaseRenderCanvas__scheduler", None)

    def _scheduler_summary(self) -> str:
        """What the scheduler was doing, for a report written from outside it.

        The line this report was missing. "The scheduler stopped asking for
        frames" is a symptom with three quite different causes -- it was
        paused, it is waiting for a frame it asked for and was never told
        about, or it is asking and every frame is being thrown away -- and
        which one it is decides whether the session was recoverable and by
        what. Attribute reads only, like everything else the watchdog thread
        is allowed to touch.
        """
        scheduler = self._scheduler()
        if scheduler is None:
            return "not reachable"
        enabled = getattr(scheduler, "_enabled", None)
        mode = getattr(scheduler, "_mode", "unknown")
        summary = {True: "asking", False: "paused", None: "cannot say"}.get(
            enabled, "cannot say"
        )
        if getattr(scheduler, "_ready_for_present", None) is not None:
            summary += ", waiting for a frame it asked for"
        if getattr(scheduler, "_just_cancelled_a_frame", False):
            summary += ", last frame cancelled"
        return f"{summary} ({mode})"

    def _kick_the_loop(self, unasked: float, width: int, height: int) -> None:
        """Force a frame the scheduler did not ask for.

        This is the way back from every shape of the fault, because it does not
        go through the scheduler at all: the canvas draws and presents on the
        spot, and telling the scheduler that a frame is done is part of that --
        which is precisely what a scheduler waiting on a frame it was never
        told about is waiting for. One forced frame and it is running again.

        When it is not, the session is being carried by this poll, at a frame
        every few seconds. That is a bad way to run and a much better way to
        stop: the field keeps its state, the panel keeps working, the user can
        save and quit in their own time rather than killing a wedged process.
        """
        force = getattr(self.canvas, "force_draw", None)
        if not callable(force):
            return
        try:
            self.canvas.request_draw()
        except Exception as exc:  # pragma: no cover - a canvas without one
            log.debug("could not request a draw: %s", exc)
        before = self.watchdog.frames
        try:
            force()
        except RuntimeError:
            # "Cannot force a draw while drawing" -- a frame is in flight after
            # all, so the loop is slow rather than unasked. That case belongs
            # to the watchdog, which reports it against the phase it is really
            # in instead of guessing from out here.
            return
        except Exception as exc:
            log.warning("could not force a frame: %s", exc)
            return

        # Whether the frame this just forced actually happened. A canvas can
        # take a forced draw and quietly throw it away -- it cancels every
        # frame it is given when it believes it has no size, or that it has
        # been closed -- and counting that as a frame is how a session that is
        # drawing nothing at all comes to be described as one being carried at
        # a frame every few seconds.
        drawn = self.watchdog.frames > before

        # The frame just forced is this poll's own doing, and must not read as
        # the scheduler having come back on the next pass -- otherwise the
        # count below can never reach two, and a session being carried by the
        # poll would look like one recovering from a hiccup every time.
        self._frames_seen = self.watchdog.frames
        self._frames_at = self._kicked_at = time.perf_counter()
        self._kicks += 1
        if not drawn:
            self._blank_kicks += 1
        log.warning(
            "no frame was asked for in %.1fs with the window up at %dx%d; "
            "forced one and %s (%d in a row, scheduler %s)",
            unasked, width, height,
            "it was drawn" if drawn else "the canvas threw it away",
            self._kicks, self._scheduler_summary(),
        )
        if self._kicks == KICKS_BEFORE_A_REPORT:
            path = self.watchdog.dump(
                f"the render scheduler stopped asking for frames; "
                f"{self._kicks} forced in a row"
            )
            log.error(
                "the render scheduler is not asking for frames and is not "
                "coming back; this session is being drawn by the window poll. "
                "%s", f"wrote {path}" if path else "no report was written",
            )
        if self._kicks >= KICKS_BEFORE_A_REPORT and drawn:
            self._start_carrying()

    def _start_carrying(self) -> None:
        """Pace the session from this poll, since the scheduler will not.

        A forced frame every few seconds keeps the field alive and the panel
        answering, which was the whole of the claim made for it: a bad way to
        run and a much better way to stop. What the last report made plain is
        that it is not a way to *run* at all -- from the outside it is a frozen
        window, indistinguishable from the freeze it is recovering from, and a
        session nobody can tell is being carried is a session nobody knows they
        should save and restart.

        So once the scheduler has been given three chances and taken none of
        them, the poll stops being a reconciliation and becomes the frame
        clock: same forced frame, at the interval the frames were asked for.
        Nothing else changes -- the simulation is paced off real elapsed time
        and cannot tell the difference -- and the moment a frame arrives that
        this poll did not force, it hands the pacing straight back.
        """
        if self._carrying:
            return
        self._carrying = True
        self._set_poll_interval(1.0 / max(self.options.max_fps, 1))
        log.error(
            "pacing the session from the window poll at %d fps until the "
            "render scheduler comes back", self.options.max_fps,
        )

    def _stop_carrying(self) -> None:
        """Hand pacing back to a scheduler that has started asking again."""
        if not self._carrying:
            return
        self._carrying = False
        self._set_poll_interval(WINDOW_POLL_SECONDS)
        log.info(
            "the render scheduler is asking for frames again; standing down"
        )

    # -- hot reload ---------------------------------------------------------

    def _start_hot_reload(self) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.info("watchdog not installed; config hot-reload disabled")
            return

        path = self.config_path.resolve()
        if not path.parent.exists():
            return

        app = self

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if Path(str(event.src_path)).resolve() == path:
                    app.reload_config()

            on_created = on_modified

        observer = Observer()
        observer.schedule(Handler(), str(path.parent), recursive=False)
        observer.daemon = True
        observer.start()
        self._watcher = observer
        log.info("watching %s for changes", path)

    def _stop_hot_reload(self) -> None:
        """Stop watching the config file, and wait for the thread to notice.

        Joined rather than just stopped: the handler holds a reference back to
        this application, and a shutdown that has already saved the field must
        not have a thread behind it that can still call ``reload_config``.
        """
        watcher, self._watcher = self._watcher, None
        if watcher is None:
            return
        try:
            watcher.stop()
            watcher.join(timeout=2.0)
        except Exception as exc:  # pragma: no cover - platform watcher quirks
            log.debug("could not stop the config watcher: %s", exc)

    def _pin_volume_detail(self) -> None:
        """Settle which slab size is in force, after any load of the config.

        A ``--volume-detail`` on the command line is a property of the launch
        rather than of the file, so a hot reload must not quietly undo it --
        the same reasoning that keeps ``--backend`` on the application rather
        than in the config. Without the flag the file decides, and a reload
        picks up an edit to it.

        Only the *name* settles here. Nothing is rebuilt: the size is
        structural, so it reaches the image when a field is next grown, which
        is ``reset_simulation`` or a relaunch.
        """
        if self.options.volume_detail:
            self.config.volume_detail = self.options.volume_detail
        self.volume_detail = config_module.normalise_volume_detail(
            self.config.volume_detail)
        self.config.volume_detail = self.volume_detail

    def reload_config(self) -> None:
        try:
            self.config = config_module.load(self.config_path)
            self._pin_volume_detail()
            # Ramped, never stepped: adjusting a control must not itself be
            # visual punctuation.
            self._retarget()
            log.info("reloaded %s", self.config_path)
        except Exception as exc:
            # A malformed file must not take down a session that has been
            # running for days.
            log.error("could not reload config: %s", exc)

    def apply_macros(self, macros: config_module.Macros) -> None:
        """Called by the control panel."""
        self.config.macros = macros
        self._retarget()

    def set_mode(self, name: str) -> bool:
        """Switch which curve table the macros resolve through. DESIGN.md §14.

        Called by the control panel. Not structural, and deliberately shaped
        like a preset switch rather than like ``switch_backend``: nothing is
        rebuilt, nothing is lost, and the change reaches the image as a ramp --
        the field on screen keeps everything it has grown and changes character
        over seconds. Saved when the user saves, like every perceptual setting.
        """
        wanted = config_module.normalise_mode(name)
        if wanted == config_module.normalise_mode(self.config.mode):
            return False
        self.config.mode = wanted
        self._retarget()
        log.info("mode -> %s", wanted)
        return True

    def _retarget(self) -> config_module.Params:
        """Re-resolve the config and ramp towards it. Returns what it resolved to.

        Every route by which a setting changes -- a slider, a preset, a file
        edit, the thickness knob -- ends here, so there is one place where the
        macro curves, the overrides and the session's frame cap are combined,
        and one place a change becomes a ramp target rather than a step.
        """
        resolved = self._session_limits(resolved=self.config.resolve())
        self.ramp.set_target(resolved)
        self._sync_audio()
        return resolved

    def _session_limits(self, resolved: config_module.Params):
        """Apply what the *launch* decides, on top of what the file says.

        Three things, and they have the same shape: each is a fact about the
        machine or the command line rather than about the configuration, so
        none of them may be written back to the file, and all of them have to
        be reapplied every time the file is re-read. Putting them here rather
        than at each call site is what keeps a hot reload from quietly undoing
        a flag the session was started with.

        ``max_fps`` is capped, never raised: the flash arithmetic in
        `config.validate` sized the per-frame lightness allowance against the
        cap, so this is a safety-relevant direction.

        ``base_scale`` takes ``--scale`` if the launch gave one.

        ``cell_budget`` takes the integrated-GPU ceiling if this session found
        an integrated GPU and nothing has already answered the question --
        which `_settle_cell_budget` decides once, when the adapter is known.
        """
        options = self.options
        resolved.max_fps = min(resolved.max_fps, options.max_fps)
        if options.scale is not None:
            resolved.render.base_scale = float(options.scale)
        if self._cell_budget > 0 and "render.cell_budget" not in self.config.overrides:
            resolved.render.cell_budget = self._cell_budget
        return config_module.validate(resolved)

    def _settle_cell_budget(self) -> None:
        """Decide whether this session simulates under a ceiling. §8.3.

        Called once, with the adapter in hand. The ceiling exists because an
        integrated GPU's memory bandwidth is the *machine's*, shared with the
        CPU and the compositor, so a fullscreen stack on a modern laptop panel
        would take a share of it that the "leave the machine usable"
        requirement (§1) does not have to give.

        Three ways to opt out, and all of them are somebody having already
        answered the question this would answer for them:

        * ``--scale``, which sets the resolution directly;
        * ``render.base_scale`` pinned in the config's overrides, likewise;
        * ``render.cell_budget`` pinned there, which is the ceiling itself --
          honoured whatever it says, including zero for "no ceiling, I know
          what this card is".

        Nothing here is written to the config file. The ceiling is a property
        of the machine the session is running on, and a config carried to
        another one must not arrive already sized for this one.
        """
        if self.device_info is None or not self.device_info.is_integrated:
            return

        overrides = self.config.overrides
        if self.options.scale is not None:
            answered = "--scale was given"
        elif "render.base_scale" in overrides:
            answered = "render.base_scale is set in the config"
        elif "render.cell_budget" in overrides:
            answered = "render.cell_budget is set in the config"
        else:
            answered = None
        if answered is not None:
            log.info(
                "integrated GPU, but %s; leaving the simulation size alone",
                answered,
            )
            return

        self._cell_budget = config_module.INTEGRATED_CELL_BUDGET
        self.params = self._retarget()
        # Structural and never ramped, so the ramp is snapped to it: the very
        # next thing this launch does is derive a geometry from `self.params`,
        # and deriving it from a value that is still on its way to the target
        # would build the field at a size nobody chose.
        self.ramp.snap(self.params)
        log.info(
            "sizing the simulation for an integrated GPU: at most %.1f M "
            "cells across the stack (DESIGN.md §8.3). Override with --scale, "
            "or with render.cell_budget in the config.",
            self._cell_budget / 1e6,
        )

    def set_filaments(self, on: bool) -> bool:
        """Show or hide the filament network under resonance. DESIGN.md §16.

        Called by the control panel. The simulation is untouched -- the
        agents keep running and the reaction keeps nucleating on their trail
        -- and only the network's rendered contribution goes, through the
        same ramp as everything else, so the change is a fade over seconds
        rather than a cut. Saved when the user saves, like the mode.
        """
        on = bool(on)
        if self.config.filaments == on:
            return False
        self.config.filaments = on
        self._retarget()
        log.info("filaments %s", "shown" if on else "hidden")
        return True

    def _sync_audio(self) -> None:
        """Open or close the capture stream to match the active mode.

        The drive runs exactly while resonance is what the engine is showing:
        not under the rhizotron (which resolves through regulation whatever
        the mode key says -- `config.active_mode`), and not in the other two
        modes. `_audio_started` tracks the *intent* rather than whether the
        open succeeded, so an unavailable backend is a status line the panel
        shows, polled zeros, and no retry storm on every slider move --
        leaving the mode and coming back is the retry.
        """
        if self.audio is None:
            return  # not built yet; __init__ syncs once it is
        want = (
            config_module.normalise_mode(self.config.mode) == "resonance"
            and config_module.normalise_backend(self.backend) != "rhizotron"
        )
        if want and not self._audio_started:
            self._audio_started = True
            self.audio.start()
            log.info("audio drive: %s", self.audio.describe())
        elif not want and self._audio_started:
            self._audio_started = False
            self.audio.stop()
            log.info("audio drive stopped with the mode")

    def trigger_event(self, kind: str) -> bool:
        """Start one event of `kind` now. Called by the control panel.

        The request is handed to the scheduler with the *live* parameters --
        the ramped ones the tick loop is using this frame, not the config file's
        -- so a requested event is shaped by the same intensity the running
        image is, and is subject to the same concurrency cap. Returns False if
        the scheduler had no room for it.

        Safe to call from a Qt slot: the panel shares the render window's event
        loop, so this runs between frames and never inside a tick.
        """
        event = self.scheduler.trigger(kind, self.params.events)
        if event is None:
            return False
        log.info(
            "requested event %s at (%.2f, %.2f) r=%.3f for %.0fs",
            event.kind, event.x, event.y, event.radius, event.duration,
        )
        return True

    def save_config(self) -> None:
        config_module.save(self.config, self.config_path)

    # -- checkpointing ------------------------------------------------------

    def _start_engine(self, width: int, height: int) -> None:
        """Build the engine, in whatever shape the saved field needs.

        The order is the point. The checkpoint is read *before* the engine
        exists, so the engine can be built at the geometry the saved field was
        grown at rather than the one this window and this config would imply.
        Otherwise every launch became a coin toss: a window opened a few pixels
        wider, or a config edit that moved the layer count, and hours of field
        maturity were quietly discarded for a mismatch that only ever concerned
        the presentation.

        Adopting the saved geometry costs nothing visually, because the
        simulation's resolution is already independent of the window's -- a
        resize only rebuilds the presentation chain (``Engine.resize``), and the
        compositor corrects the aspect difference. The config's structural
        values -- layer count, base scale, agent density, climate and psi sizes
        -- therefore take effect when a new field is grown, which is what
        ``reset_simulation`` is for.

        Every failure path ends in "run with a freshly seeded field": a missing,
        foreign or unbuildable checkpoint is a normal thing to find, not a reason
        to refuse to open.
        """
        derived = self._derive_geometry(width, height)
        saved = self._saved_checkpoint()
        geometry = derived

        if saved is not None:
            wanted = checkpoint_module.required_geometry(saved, backend=self.backend)
            if wanted is None:
                saved = None  # unusable; required_geometry has said why
            elif wanted != derived:
                log.info(
                    "launching the simulation at the saved field's geometry "
                    "(%s) rather than the %s this window and config imply; "
                    "reset the simulation to adopt the latter",
                    wanted.describe(), derived.describe(),
                )
                geometry = wanted

        try:
            self.engine = self._make_engine(width, height, geometry)
        except Exception as exc:
            if geometry is derived:
                raise
            # The saved geometry passed its bounds check but this device would
            # not build it -- a file from a machine with more memory, say.
            log.error(
                "could not build the simulation at the saved geometry (%s); "
                "starting from a fresh field", exc,
            )
            saved = None
            self.engine = self._make_engine(width, height, derived)

        if saved is not None:
            self._restore(saved)

    def _derive_geometry(self, width: int, height: int):
        """The shape a fresh field would have, under the backend in use."""
        _, geometry_class = backend_classes(self.backend)
        return geometry_class.derive(width, height, self.params)

    def _make_engine(self, width: int, height: int, geometry):
        engine_class, _ = backend_classes(self.backend)
        return engine_class(
            self.device, width, height, self.params,
            seed=self.options.seed, geometry=geometry,
        )

    def _saved_checkpoint(self) -> checkpoint_module.Checkpoint | None:
        """The checkpoint on disk, if there is one and this launch wants it.

        ``--reset`` is a statement about the *launch* -- start me a new field
        rather than the old one -- and `_resume` is where it stops being one.
        Once this session has written a field of its own, that field is what a
        later rebuild comes back to: a device lost an hour into a `--reset`
        session must not discard the hour, which is exactly what re-reading the
        launch flag would do.
        """
        if not (self.options.checkpoint and self._resume):
            return None
        saved = checkpoint_module.load(self.checkpoint_path)
        if saved is None:
            log.info(
                "no saved state at %s; starting from a fresh field",
                self.checkpoint_path,
            )
        return saved

    def _restore(self, saved: checkpoint_module.Checkpoint) -> None:
        try:
            if checkpoint_module.restore(self.engine, saved, scheduler=self.scheduler):
                self.resumed_from = saved.describe()
        except Exception as exc:
            # A partially applied restore is still a live field -- every value
            # that reaches the GPU is clamped and the sanitise pass runs every 60
            # ticks -- so carrying on beats refusing to start.
            log.error("could not restore the saved state: %s", exc)

    def save_checkpoint(self, blocking: bool = False) -> bool:
        """Read the simulation state back and write it to disk.

        Must be called between ticks, never mid-tick: the readback assumes the
        deposit accumulator has been drained. ``blocking`` is for shutdown, when
        there is no next frame to hand the write off from.
        """
        if not self.options.checkpoint or self.engine is None:
            return False
        self._last_checkpoint = time.perf_counter()
        try:
            snapshot = checkpoint_module.capture(
                self.engine, scheduler=self.scheduler, sim_hz=self.params.sim_hz
            )
            if blocking:
                self._saver.join()
                checkpoint_module.save(self.checkpoint_path, snapshot)
            else:
                self._saver.submit(self.checkpoint_path, snapshot)
        except Exception as exc:
            # Losing a checkpoint costs the user field maturity after a crash.
            # Losing the session costs them the field itself.
            log.error("could not checkpoint: %s", exc)
            return False
        self._checkpoint_saved_at = time.time()
        # There is now a field of this session's own on disk, so anything that
        # rebuilds the engine from here should come back to it -- see
        # `_saved_checkpoint`.
        self._resume = True
        # The readback stalls this frame. Restart the clock so the stall is not
        # charged to the pacing accumulator, which would otherwise produce a
        # burst of catch-up ticks, or to the governor, which would throttle the
        # tick rate over a cost that recurs once every fifteen minutes.
        self._last_time = time.perf_counter()
        return True

    def checkpoint_status(self) -> str:
        """One line for the control panel."""
        if not self.options.checkpoint:
            return "off"
        if self._checkpoint_saved_at is None:
            return "resumed, not saved yet" if self.resumed_from else "not saved yet"
        age = max(time.time() - self._checkpoint_saved_at, 0.0)
        return f"saved {checkpoint_module.describe_age(age)}"

    def reset_simulation(self) -> None:
        """Discard the current world and grow a new one from seeds.

        The new field starts from scattered seeds and the safety stage's history
        is empty, so the image settles down and grows back rather than cutting --
        a reset is the one moment the user has explicitly asked for a change, and
        even then it is not allowed to be a step.

        This is also the moment the structural config values land. A resumed
        field keeps the geometry it was grown at, so changing the layer count,
        the base scale or the slab's thickness does nothing until there is a new
        field to apply it to, and here there is one.
        """
        width, height = self._size
        self._adopt(self._make_engine(
            width, height, self._derive_geometry(width, height)))
        log.info("simulation reset; growing a new field from seeds")

    def _adopt(self, engine) -> None:
        """Put a freshly grown field on screen, discarding what was there.

        The engine is built by the caller and handed over already made, which
        is what lets a failed allocation be a message rather than a lost
        session: nothing here is destroyed until there is something to replace
        it with. The cost is that both fields exist at once for a moment, which
        matters when the new one is several times the size of the old -- and
        which is still the better trade, since the alternative is a card that
        cannot allocate the new field and no longer holds the old one either.
        """
        # Wait for any in-flight write first, or it would land on disk after the
        # file it describes has been deleted.
        self._saver.join()
        checkpoint_module.discard(self.checkpoint_path)

        self.scheduler = events_module.EventScheduler(seed=self.options.seed)
        self.engine = engine
        self._accumulator = 0.0
        self._sim_hz_scale = 1.0
        self._frame_times.clear()
        self._last_time = time.perf_counter()
        self._last_checkpoint = time.perf_counter()
        self._checkpoint_saved_at = None
        self.resumed_from = None

    # -- the slab's thickness -----------------------------------------------

    def volume_slab(self, depth: int | None = None):
        """The slab a fresh volumetric field would be, at this thickness.

        Answers for the volumetric backend whether or not it is the one
        running, because the control panel has to be able to say what a setting
        would cost before the user commits to it -- and, when the layered
        backend is up, what the thickness knob it can see is currently set to.
        """
        params = self.params
        if depth is not None and int(depth) != params.volume.depth:
            params = dataclasses.replace(
                params,
                volume=dataclasses.replace(params.volume, depth=int(depth)),
            )
        width, height = self._size
        return volume_module.VolumeGeometry.derive(
            max(width, 1), max(height, 1), params)

    def volume_depth_limits(self) -> tuple[int, int]:
        """The thinnest and thickest slab this window allows, in voxels."""
        width, height = self._size
        return volume_module.depth_limits(
            max(width, 1), max(height, 1), self.params)

    def set_volume_depth(self, depth: int) -> int:
        """Grow a new slab this many voxels deep. Returns the thickness taken.

        Structural, and unlike the backend switch it is not a change the saved
        field can survive: a slab of a different thickness is a differently
        shaped array, nothing resamples one into the other, and a launch that
        found the old checkpoint would rebuild the old thickness from it. So
        this discards the field and grows a new one, exactly as a reset does,
        and the panel asks before calling it.

        The new field is built before anything is thrown away. Thickness is
        what this backend's memory is spent on -- the ceiling is some six times
        the default -- so "the card would not allocate that" is a normal answer
        to get, and it leaves the running field untouched and raises.
        """
        low, high = self.volume_depth_limits()
        wanted = max(low, min(high, int(depth)))
        # The geometry rounds to a multiple of eight, so ask it what the number
        # actually means rather than storing something it will not honour.
        wanted = self.volume_slab(wanted).depth
        if wanted == self.volume_slab().depth:
            return wanted  # already the thickness a fresh slab would have

        previous = self.config.overrides.get("volume.depth")
        self.config.overrides["volume.depth"] = wanted
        resolved = self._retarget()
        # Structural, and the field is grown from `self.params` in a moment
        # rather than on the next frame. Integers snap through the ramp anyway;
        # this only moves one of them a frame earlier.
        self.params.volume.depth = resolved.volume.depth

        if self.backend != "volumetric":
            # Nothing to rebuild: the setting lands the next time a slab is
            # grown, which is what switching to this backend does.
            log.info("the slab will be %d voxels deep when one is next grown", wanted)
            return wanted

        width, height = self._size
        try:
            engine = self._make_engine(
                width, height, self._derive_geometry(width, height))
        except Exception:
            if previous is None:
                self.config.overrides.pop("volume.depth", None)
            else:
                self.config.overrides["volume.depth"] = previous
            self.params.volume.depth = self._retarget().volume.depth
            raise
        self._adopt(engine)
        log.info(
            "slab is now %s; growing a new field from seeds",
            engine.geometry.describe(),
        )
        return wanted

    def set_volume_detail(self, name: str) -> bool:
        """Change how wide the volumetric slab is, growing a new field.

        The lateral counterpart of :meth:`set_volume_depth`, and structural for
        the same reason: a slab of a different width is a differently shaped
        array, nothing resamples one into the other, and a launch that found
        the old checkpoint would rebuild the old width from it. So this
        discards the field and grows a new one, and the panel asks first.

        Built before anything is thrown away, exactly as the thickness is, and
        for a sharper version of the same reason. Width is the more expensive
        of the two axes -- it enters the voxel count twice, since the height
        follows it -- so the widest size is some four times the memory of the
        standard one, and "the card would not allocate that" is an ordinary
        answer to get. It leaves the running field untouched and raises.

        Under the layered backend the size is recorded and nothing is regrown:
        there is no slab on screen for it to change, and discarding a mature
        layered field to apply a setting that backend never reads would be a
        pure loss. It lands when a slab is next grown.

        Returns False if the named size is already the one running.
        """
        wanted = config_module.normalise_volume_detail(name)
        if wanted == self.volume_detail:
            return False

        previous_name = self.volume_detail
        previous_pin = self.options.volume_detail
        self.volume_detail = wanted
        self.config.volume_detail = wanted
        # A command-line size would otherwise be re-pinned over this one at the
        # next hot reload, and the change the user just asked for would come
        # undone the next time they saved the file.
        self.options.volume_detail = None

        # Structural, and the field is grown from `self.params` in a moment
        # rather than on the next frame. Integers snap through the ramp anyway;
        # this only moves one of them a frame earlier.
        self.params.volume.width = self._retarget().volume.width

        if self.backend != "volumetric":
            log.info(
                "the slab will be %d voxels across when one is next grown",
                config_module.VOLUME_DETAIL[wanted],
            )
            return True

        width, height = self._size
        try:
            engine = self._make_engine(
                width, height, self._derive_geometry(width, height))
        except Exception:
            self.volume_detail = previous_name
            self.config.volume_detail = previous_name
            self.options.volume_detail = previous_pin
            self.params.volume.width = self._retarget().volume.width
            raise
        self._adopt(engine)
        log.info(
            "slab is now %s; growing a new field from seeds",
            engine.geometry.describe(),
        )
        return True

    def switch_backend(self, name: str) -> bool:
        """Change how depth is drawn, without restarting the process.

        Structural, so it cannot be ramped and cannot be a smooth transition:
        the two backends do not hold the same kind of state, and there is
        nothing to interpolate between a stack of sheets and a slab. Like
        ``reset_simulation`` it is a change the user has explicitly asked for,
        and like a reset the image comes back up from black at the slew
        limiter's rate rather than cutting -- the new engine's history buffer
        starts empty and DESIGN.md §7 bounds how fast it can fill.

        The field being left is checkpointed first and the new backend's own
        saved field is resumed if there is one, so this is genuinely a switch
        rather than a discard: going back finds the field where it was left,
        older by however long the detour took.

        Returns False if the named backend is already the one running.
        """
        wanted = config_module.normalise_backend(name)
        if wanted == self.backend:
            return False

        # Save what we are leaving before the engine that holds it goes away.
        self.save_checkpoint(blocking=True)

        self.backend = wanted
        self.config.backend = wanted
        self.checkpoint_path = (
            self.options.checkpoint_path
            or checkpoint_module.default_checkpoint_path(wanted)
        )
        self.scheduler = events_module.EventScheduler(seed=self.options.seed)
        self.resumed_from = None
        self._checkpoint_saved_at = None

        width, height = self._size
        self._start_engine(width, height)

        self._accumulator = 0.0
        self._sim_hz_scale = 1.0
        self._frame_times.clear()
        self._last_time = time.perf_counter()
        self._last_checkpoint = time.perf_counter()
        # The rhizotron cannot hear (§15 has one tuning), so a backend switch
        # is one of the two edges the capture stream follows.
        self._sync_audio()
        log.info("depth backend is now %s", wanted)
        return True

    # -- shutdown -----------------------------------------------------------

    def _watch_for_close(self) -> None:
        """Hook every route the window has out of existence.

        All three hooks land on the same idempotent ``shutdown``, because which
        one fires first depends on the backend and on how the window was closed,
        and none of them covers the others:

        * the canvas's own close event is the ordinary case, and fires while
          the loop is still turning and the device is still healthy, which is
          exactly when the checkpoint readback is safe;
        * the Qt close event is the same moment seen one layer lower, and does
          not depend on the render loop still being in a state to notice;
        * Qt's ``aboutToQuit`` covers being told to quit without the render
          window being closed first -- a session logout, say.
        """
        self.canvas.add_event_handler(self._on_canvas_close, "close")

        if not self.have_qt:
            return
        try:
            from PySide6 import QtCore, QtWidgets
        except Exception as exc:  # pragma: no cover - Qt was importable a moment ago
            log.debug("could not hook the Qt close signals: %s", exc)
            return

        app = self

        class CloseFilter(QtCore.QObject):
            """Sees the window close as Qt delivers it, before the backend does."""

            def eventFilter(self, obj, event):  # noqa: N802 - Qt naming
                if event.type() == QtCore.QEvent.Type.Close:
                    app.request_stop()
                return False

        # Held on the application, or Python would collect the filter and Qt
        # would be left with a dangling one.
        self._close_filter = CloseFilter()
        self.canvas.installEventFilter(self._close_filter)

        qapp = QtWidgets.QApplication.instance()
        if qapp is not None:
            qapp.aboutToQuit.connect(self.shutdown)

    def _on_canvas_close(self, event=None) -> None:
        self.shutdown()

    def _install_signal_handlers(self) -> None:
        """Turn a SIGINT or SIGTERM into the same orderly close.

        Without this, ``kill`` -- which is what a session logout sends -- ends
        the process where it stands and costs the user however much field
        maturity accumulated since the last periodic save.
        """
        def handler(signum, _frame):
            log.info("received %s; closing", signal.Signals(signum).name)
            self.request_stop()

        for sig in STOP_SIGNALS:
            try:
                self._previous_signal_handlers[sig] = signal.signal(sig, handler)
            except ValueError:
                # Not the main thread (embedded, or a test): the loop and the
                # window close still reach shutdown, so this is not fatal.
                log.debug("could not install a handler for %s", sig)

    def _restore_signal_handlers(self) -> None:
        while self._previous_signal_handlers:
            sig, previous = self._previous_signal_handlers.popitem()
            try:
                signal.signal(sig, previous)
            except (ValueError, TypeError):  # pragma: no cover - as above
                pass

    def request_stop(self) -> None:
        """Ask for shutdown at the next safe point.

        A signal arrives between two bytecodes, which can be in the middle of
        building a tick's command buffer, and the checkpoint readback is only
        valid between ticks. So the request is handed to the loop rather than
        acted on where it lands.
        """
        if self._stopped or self._stop_requested:
            return
        self._stop_requested = True
        try:
            self.loop.call_soon(self.shutdown)
        except Exception as exc:
            # No loop to defer to. A slightly-off checkpoint beats none.
            log.debug("could not defer shutdown to the loop: %s", exc)
            self.shutdown()

    def shutdown(self) -> None:
        """Save the world, take the application down, and let the loop end.

        Idempotent, and safe to call from a close event, a signal handler, Qt's
        quit signal, or the end of ``run``. The checkpoint goes first and
        blocks: everything after it can only make the state harder to read
        back, and there is no next frame to hand the write off from.
        """
        if self._stopped:
            return
        self._stopped = True
        log.info("shutting down")

        # Still watched, because teardown wedging is a real way for this to end
        # -- it is what the exit guard in `__main__` exists for -- but under
        # the patient limit, since the blocking write below is allowed to take
        # a while on a slow disk.
        self.watchdog.mark(diagnostics_module.SHUTDOWN)
        self.save_checkpoint(blocking=True)
        self._saver.join()
        if self.audio is not None:
            self.audio.stop()
        self.power.stop()
        self._stop_window_poll()
        self._stop_hot_reload()
        self._close_panel()
        self._stop_loop()
        self.watchdog.stop()

    def _close_panel(self) -> None:
        panel, self.panel = self.panel, None
        if panel is None:
            return
        try:
            panel.close()
        except Exception as exc:  # pragma: no cover - Qt teardown ordering
            log.debug("could not close the control panel: %s", exc)

    def _stop_loop(self) -> None:
        """Ask the event loop to end, and make sure Qt agrees.

        The loop stops itself once no canvases are left, but the control panel
        is a second top-level window, and a Qt application with a window still
        open has no reason of its own to quit. Saying it explicitly is what
        keeps a closed window from leaving a live process behind.
        """
        for label, action in (
            ("close the canvas", getattr(self.canvas, "close", None)),
            ("stop the loop", getattr(self.loop, "stop", None)),
            ("quit the Qt application", self._quit_qt),
        ):
            if action is None:
                continue
            try:
                action()
            except Exception as exc:
                # Any of these can be gone already, or never have existed --
                # which of the close routes got here first decides. None of it
                # is worth raising over once the field is on disk.
                log.debug("could not %s: %s", label, exc)

    def _quit_qt(self) -> None:
        if not self.have_qt:
            return
        from PySide6 import QtWidgets

        qapp = QtWidgets.QApplication.instance()
        if qapp is not None:
            qapp.quit()

    # -- frame --------------------------------------------------------------

    # -- power ---------------------------------------------------------------

    def _start_power_watch(self) -> None:
        """Begin asking where the machine's power is coming from. §8.3.

        Never fatal, and never a reason not to open: a machine that will not
        say reads as mains, which is the state that changes nothing.

        Started whether or not the backoff is switched on, and the config gates
        the *effect* rather than the reading -- `_on_battery` is what consults
        it, live. Two reasons. Switching the backoff on in a running session
        has to work, and a watch that was never started would leave it reading
        mains forever. And the answer is worth having anyway: a session running
        slowly on a laptop is a different report when the stall file can say it
        was unplugged.
        """
        try:
            self.power.start()
        except Exception as exc:  # pragma: no cover - a platform quirk
            log.debug("could not watch the power source: %s", exc)
            return
        self._battery = self.power.on_battery
        log.info("power source: %s", self.power.describe())

    def _on_battery(self) -> bool:
        """Whether to be running in the cheap register right now.

        The config's switch is read live rather than latched, so turning the
        backoff off in a running session takes effect at the next frame -- and
        turning it on while unplugged takes effect just as fast.
        """
        return bool(self.params.power.battery_backoff and self.power.on_battery)

    def _note_power_change(self) -> None:
        """Log the transition, once, when it happens."""
        battery = self._on_battery()
        if battery == self._battery:
            return
        self._battery = battery
        if battery:
            log.info(
                "on battery: simulating at x%.2f and presenting at %d fps "
                "(DESIGN.md §8.3)",
                self.params.power.battery_sim_scale,
                self.params.power.battery_max_fps,
            )
        else:
            log.info("on mains: back to the full rate")

    # -- pacing --------------------------------------------------------------

    def effective_sim_hz(self) -> float:
        """The tick rate actually in force, after both things that lower it.

        The governor's scale and the battery backoff, in one place, because
        two callers need the same answer and would otherwise each apply half
        of it: the frame loop, which paces off it, and the control panel,
        which reports it -- and a panel reading 20 Hz beside a session ticking
        at 12 is a panel saying something untrue about the only number there
        that the user can act on.
        """
        hz = self.params.sim_hz * self._sim_hz_scale
        if self._on_battery():
            hz *= self.params.power.battery_sim_scale
        return max(hz, 2.0)

    def _uncapped_fps(self) -> int:
        """The rate this session would present at with the governor idle.

        The config's cap, lowered on battery and never raised: `max_fps` is the
        number the flash arithmetic was done against (`config.validate`), so
        nothing here may exceed it.
        """
        cap = max(int(self.params.max_fps), 1)
        if self._on_battery():
            cap = min(cap, max(int(self.params.power.battery_max_fps), 1))
        return cap

    def _target_fps(self) -> int:
        """And what it should present at now, with the governor's lever in."""
        return max(
            int(round(self._uncapped_fps() * self._fps_scale)), MIN_PRESENT_FPS
        )

    def _apply_frame_rate(self) -> None:
        """Push the rate at the canvas, when it has changed.

        The scheduler owns the frame clock, so a rate the application has
        decided on is not in force until the canvas has been told. Pushed only
        on a change, because this runs every frame; never raised past
        `_uncapped_fps`, which is the safety-relevant direction.
        """
        wanted = min(self._target_fps(), self._uncapped_fps())
        if wanted == self._present_fps:
            return
        previous, self._present_fps = self._present_fps, wanted
        try:
            self.canvas.set_update_mode("continuous", max_fps=float(wanted))
        except Exception as exc:  # pragma: no cover - a canvas without one
            log.debug("could not change the frame rate: %s", exc)
            return
        log.info("presenting at %d fps (was %d)", wanted, previous)

    def _governor(self, frame_time: float) -> None:
        """Keep the frame inside its budget, with the two levers of §8.3.

        The tick rate first, because the motion-compensated interpolator hides
        it completely; the presented frame rate only once the tick rate is at
        its floor and frames are *still* long. That order is the whole point of
        the second lever: on the card of §8.1 the simulation is what costs and
        the first lever is enough, while on an integrated GPU the per-frame
        render work is at least as likely to be what is over budget -- and
        against a render-bound frame, lowering the tick rate degrades motion
        and recovers nothing.

        Recovery runs in the opposite order, giving back the visible
        degradation before the invisible one, and it is tested against the
        budget at the *full* rate rather than the reduced one. Otherwise the
        two levers would chase each other: dropping to 20 fps makes the slot
        half again as long, which would read as headroom and put the rate
        straight back.

        Resolution is never touched, at any point. That would be a visible
        discontinuity, and is decided once when the field is grown (§8.3).
        """
        self._frame_times.append(frame_time)
        if len(self._frame_times) < 30:
            return
        window = self._frame_times[-30:]
        self._frame_times = window
        median = sorted(window)[len(window) // 2]

        slot = 1.0 / max(self._present_fps, 1)
        full = 1.0 / max(self._uncapped_fps(), 1)

        if median > slot * 0.92:
            if self._sim_hz_scale > SIM_SCALE_FLOOR:
                self._sim_hz_scale = max(
                    SIM_SCALE_FLOOR, self._sim_hz_scale * 0.97)
            else:
                self._fps_scale = max(FPS_SCALE_FLOOR, self._fps_scale * 0.97)
        elif median < full * 0.55:
            if self._fps_scale < 1.0:
                self._fps_scale = min(1.0, self._fps_scale * 1.01)
            else:
                self._sim_hz_scale = min(1.0, self._sim_hz_scale * 1.01)

    def _follow_canvas_size(self, now: float) -> None:
        """Track the window size without ever restarting the simulation.

        A resize -- including maximising, or a monitor's scale factor changing
        -- only rebuilds the engine's presentation chain. The world keeps
        running: same fields, same agents, same tick counter. Until a new size
        settles the previous output is simply presented into the new window,
        which is a scale, not a discontinuity.
        """
        width, height = self.canvas.get_physical_size()
        if width <= 0 or height <= 0 or (width, height) == self._size:
            self._pending_size = None
            return

        if (width, height) != self._pending_size:
            self._pending_size = (width, height)
            self._pending_since = now
            return
        if now - self._pending_since < RESIZE_SETTLE:
            return

        self.engine.resize(width, height)
        self._size = (width, height)
        self._pending_size = None

    def draw_frame(self) -> None:
        # A paint request can still be in flight when the window goes away, and
        # the state it would tick has already been read back and written out.
        if self._stopped:
            return
        # A frame boundary is a safe point, so a stop asked for mid-tick by a
        # signal is honoured here at the latest.
        if self._stop_requested:
            self.shutdown()
            return
        # Before the watchdog is told a frame happened, and before anything
        # touches the engine: on a lost device every call below would fail or
        # quietly do nothing, and this is the one thread allowed to replace it.
        # A rebuild that does not take leaves the flag set and is retried on
        # the next frame; nothing is drawn in between, which is honest -- there
        # is no device to draw with.
        if self._device_lost is not None and not self._rebuild_device():
            return
        self.watchdog.frame()
        now = time.perf_counter()
        frame_dt = min(now - self._last_time, 0.25)  # clamp after a stall
        self._last_time = now

        self.params = self.ramp.update(frame_dt)
        self._follow_canvas_size(now)
        # Both cheap and both level-triggered: the power watch's answer is a
        # plain attribute written by another thread, and the rate is only
        # pushed at the canvas when it has actually changed.
        self._note_power_change()
        self._apply_frame_rate()

        # The audio drive (DESIGN.md §16.4): features in, an effective
        # parameter set out. Applied after the ramp -- the drive is allowed
        # to be quicker than the 8 s ramp, and its own speed limit is the
        # front end's followers -- and only to this frame's local view, so
        # the ramp's state is never written through. With the drive off, or
        # the room silent, `modulate` is the identity and `params` *is*
        # `self.params`.
        params = self.params
        scheduler_events = params.events
        if self._audio_started:
            features = self.audio.poll()
            params = audio_module.modulate(params, features)
            # While the room is playing, the arrivals are the music's: the
            # scheduler's own drawing pauses (in-flight events finish) and
            # returns in silence -- see `audio.autonomous_arrivals`.
            scheduler_events = audio_module.autonomous_arrivals(
                params.events, features)
            # The second door: an onset may ask for an event, through the
            # same `trigger` the panel's buttons use -- same caps, no
            # privileged path -- shaped by the music inside the ranges the
            # RNG already samples (§16.4): a harder hit is a stronger event,
            # a faster track a brisker envelope. Checked against the cap
            # here first so a busy track reads as a full sky, not a log of
            # refusals.
            ask = self.audio.event_request(params.audio)
            if (
                ask is not None
                and len(self.scheduler.active) < params.events.max_concurrent
            ):
                event = self.scheduler.trigger(
                    ask.kind, params.events, vigor=ask.vigor, pace=ask.pace)
                if event is not None:
                    log.debug(
                        "onset -> %s event (vigor %.2f, pace %.2f) at "
                        "(%.2f, %.2f)",
                        event.kind, ask.vigor, ask.pace, event.x, event.y)

        sim_hz = self.effective_sim_hz()
        tick_interval = 1.0 / sim_hz

        self._accumulator += frame_dt
        # Bound the catch-up so a long stall cannot produce a burst of ticks
        # that would look like a jump.
        max_ticks = 3
        ticks = 0
        while self._accumulator >= tick_interval and ticks < max_ticks:
            # Marked per tick rather than once for the loop: three ticks that
            # are each merely slow should not add up to something that reads
            # as one tick that never returned.
            self.watchdog.mark("tick")
            active = self.scheduler.update(tick_interval, scheduler_events)
            rows, _ = self.scheduler.pack(8)
            self.engine.tick(params, rows)
            self._accumulator -= tick_interval
            ticks += 1
        if ticks == max_ticks:
            self._accumulator = 0.0

        frac = min(self._accumulator / tick_interval, 1.0)

        # Its own phase because it is the one call here that blocks by design:
        # it waits on the presentation queue, so a compositor that stops
        # retiring frames wedges the loop precisely here.
        self.watchdog.mark("acquire")
        texture = self.present_context.get_current_texture()
        self.watchdog.mark("render")
        self.engine.render(
            params,
            frac=frac,
            target_view=texture.create_view(),
            target_format=self.target_format,
            frame_dt=frame_dt,
        )

        self._governor(time.perf_counter() - now)

        if now - self._last_telemetry >= self.options.telemetry_seconds:
            self._last_telemetry = now
            self.watchdog.mark("telemetry")
            self._log_telemetry()

        # After the governor, so the readback stall is never charged to the
        # frame-time window that decides the tick rate.
        if (
            self.options.checkpoint
            and self.options.checkpoint_seconds > 0.0
            and now - self._last_checkpoint >= self.options.checkpoint_seconds
        ):
            self.watchdog.mark("checkpoint")
            self.save_checkpoint()

        # Between frames now, where waiting is ordinary and the watchdog is
        # correspondingly patient about it.
        self.watchdog.mark(diagnostics_module.IDLE)

    def _log_telemetry(self) -> None:
        stats = self.engine.read_stats()
        window = self._frame_times[-30:] or [0.0]
        log.info(
            "tick=%d  mean_v=%.4f var=%.5f activity=%.5f  ell=%.2f (%+.3f)  "
            "cap=%.2f  exposure=%.2f  sim=%.1fHz (x%.2f)  %dfps%s  "
            "frame=%.1fms  events=[%s]",
            self.engine.tick_count,
            stats["mean_v"], stats["var_v"], stats["mean_activity"],
            # Feature size and the correction the loop is holding to get it.
            # Logged together because neither says much alone: an ell drifting
            # with a correction near zero is the field moving on its own, and a
            # steady ell with the correction parked at its bound is the loop
            # asking for something the field will not give (DESIGN.md 4.7).
            stats["ell"], stats["corr_du"],
            # The capacity return must sit clear of its clamp (3.0) or the
            # capacity has become a deposit sink; see AgentParams.deposit_cap.
            stats["cap_return"],
            stats["exposure"],
            self.effective_sim_hz(), self._sim_hz_scale,
            # The presented rate is here because it is now a thing the
            # governor moves (§8.3): a session running at 20 fps because the
            # frames were long looks, in a log that only quotes the tick rate,
            # exactly like one running at 30.
            self._present_fps, " on battery" if self._on_battery() else "",
            1000.0 * (sum(window) / len(window)),
            self.scheduler.describe(),
        )
        if self._audio_started:
            # One line, only while the drive is meant to be listening: a dead
            # stream in a running resonance session should be in the log, not
            # a mystery of a suddenly still field (DESIGN.md §16.6).
            features = self.audio.features
            log.info(
                "audio: %s  level=%.2f bass=%.2f mid=%.2f treble=%.2f "
                "onsets=%d%s",
                self.audio.describe(),
                features.level, features.bass, features.mid, features.treble,
                self.audio.extractor.onsets,
                " (silent)" if features.silent else "",
            )

    # -- diagnostics --------------------------------------------------------

    def diagnostic_snapshot(self) -> dict[str, object]:
        """What the simulation was doing, for a stall report.

        Read from the watchdog thread while the frame loop is wedged, which
        constrains it absolutely: attribute reads and arithmetic only. Nothing
        here may touch the GPU -- the readback it would queue is very possibly
        the thing that is stuck -- take a lock, or call into wgpu or Qt.

        The calls that do run code, rather than read an attribute, are each
        contained: losing one line of context is a much smaller loss than
        losing the report, which is what an exception here would cost.
        """
        def safe(label, fn):
            try:
                return fn()
            except Exception as exc:
                return f"<{label} unavailable: {exc!r}>"

        engine = self.engine
        window = self._frame_times[-30:]
        return {
            "uptime": f"{time.time() - self._started_at:.0f}s",
            "backend": self.backend,
            "device": safe("device", lambda: self.device_info.describe()),
            "device lost": self._device_lost or (
                f"no (recovered {self._device_losses}x)"
                if self._device_losses else "no"
            ),
            # Two plain attribute reads, within this method's constraints, and
            # the pair that separates "this laptop is slow" from "this laptop
            # is unplugged and doing exactly what it was told to" (§8.3).
            "power": safe("power", self.power.describe),
            "rates": safe("rates", lambda: (
                f"{self.effective_sim_hz():.1f} Hz sim, "
                f"{self._present_fps} fps presented"
            )),
            "tick": getattr(engine, "tick_count", "<no engine>"),
            "geometry": safe(
                "geometry",
                lambda: getattr(engine.geometry, "describe", repr)(),
            ),
            "window size": self._size,
            "pending resize": self._pending_size,
            "window": safe("window", self._window_summary),
            "window poll": safe("window poll", self._poll_summary),
            "scheduler": safe("scheduler", self._scheduler_summary),
            "forced frames": safe("forced frames", self._kick_summary),
            "sim_hz": (
                f"{self.params.sim_hz * self._sim_hz_scale:.1f} "
                f"(x{self._sim_hz_scale:.2f})"
            ),
            "accumulator": f"{self._accumulator:.4f}s",
            "frame times": (
                f"{1000.0 * (sum(window) / len(window)):.1f}ms mean of "
                f"{len(window)}, {1000.0 * max(window):.1f}ms worst"
                if window else "none recorded"
            ),
            "checkpoint": safe("checkpoint", self.checkpoint_status),
            "checkpoint writer": "busy" if self._saver.busy else "idle",
            "since last checkpoint": (
                f"{time.perf_counter() - self._last_checkpoint:.0f}s"
            ),
            "events": safe("events", self.scheduler.describe),
            # The drive's one status line (DESIGN.md §16.6(5)): a resonance
            # session whose stream died should say so in a stall report
            # rather than present as a mysteriously still field. An attribute
            # read plus string formatting, within this method's constraints.
            "audio": (
                safe("audio", self.audio.describe)
                if self._audio_started else "off"
            ),
            "preset": self.config.preset_name,
            "config": str(self.config_path),
            "stopping": self._stop_requested,
            "stopped": self._stopped,
            "panel open": self.panel is not None,
            "hot reload": self._watcher is not None,
        }

    def _on_device_lost(self, event, device=None) -> None:
        """The GPU went away underneath a running session.

        Worth a report of its own rather than only a log line. A lost device
        does not stop the frame loop -- every subsequent call simply fails or
        does nothing -- so what the user sees is a window that has stopped
        moving, which is indistinguishable from the freeze the watchdog is
        looking for and is not one. The report says which it was.

        Nothing is rebuilt here. This can be called from a driver's own thread,
        and rebuilding a device from one is how a bad afternoon starts; the
        frame loop notices the flag on its next pass and does the work there,
        where it already owns everything it would have to replace.

        ``device`` is which device the listener was installed on, bound at
        install time, so an event arriving from one that has already been
        replaced can be recognised as such and dropped.
        """
        if device is not None and device is not self.device:
            # A device we have already replaced, saying goodbye. Dropping the
            # old one can itself raise this, and acting on it would take the
            # session straight back into the rebuild it just came out of.
            log.debug("ignoring a lost event from a replaced device")
            return
        reason = str(getattr(event, "reason", None) or event)
        self._device_lost = reason
        if self._stopped or self._stop_requested:
            # Dropping the device is how a session *ends*. Reporting that as a
            # fault would mean a diagnostic file after every clean exit, which
            # is the surest way to make the ones that matter unreadable.
            log.debug("device lost during shutdown: %s", reason)
            return
        log.error("GPU device lost: %s", reason)
        path = self.watchdog.dump(f"device lost: {reason}")
        if path is not None:
            log.error("wrote %s", path)

    def _rebuild_device(self) -> bool:
        """Replace a lost device and everything that was built on it. §8.3.

        A device loss used to be scaffolding: noticed, reported, and then
        nothing, which left a live process showing a still image. That was a
        defensible place to stop while the target was one desktop card, where
        losing a device means a driver reset somebody is already looking at.
        It is not defensible on a laptop, where the same event is a lid
        closing, a dock being pulled, or a hybrid-graphics switch -- routine
        things, happening nightly, to a session explicitly meant to survive
        days.

        Everything on the far side of the device goes: the engine and every
        texture and buffer in it, the bind-group caches, the surface
        configuration. So everything is asked for again, in the same order the
        launch asks for it, and the field comes back from the checkpoint on
        disk. That last part is the cost and it is not recoverable: the field
        in memory died with the device, and reading it back was never possible
        -- the readback would have needed the device that went away. What
        returns is the last save, which at the default interval is up to
        fifteen minutes old, and the log says so rather than letting a
        silently younger field look like a successful recovery.

        Returns True if the session is running again.
        """
        now = time.perf_counter()
        if now - self._device_retry_at < DEVICE_RETRY_SECONDS:
            # A driver that is still resetting will refuse for as long as it
            # takes, and asking it thirty times a second produces a log nobody
            # can read and a window that is busy rather than waiting.
            return False
        self._device_retry_at = now
        reason, self._device_lost = self._device_lost, None
        log.warning("rebuilding after device loss (%s)", reason)

        engine, self.engine = self.engine, None
        self.device = None
        self.device_info = None
        self.present_context = None
        del engine  # its resources belong to a device that no longer exists

        try:
            self.device, self.device_info = device_module.request_device(
                gpu=self.options.gpu)
            self._listen_for_device_loss()
            self._configure_surface()
        except Exception as exc:
            # The GPU is not back yet -- a driver still resetting, a laptop
            # mid-switch. Put the flag back so the next frame tries again, and
            # say so once per attempt rather than thirty times a second.
            self._device_lost = reason
            log.error("could not reacquire a GPU device: %s", exc)
            return False

        width, height = self._size
        if width <= 0 or height <= 0:
            width, height = self.options.width, self.options.height
        self.resumed_from = None
        try:
            # Exactly the launch path, including its own degrade-to-fresh:
            # a checkpoint that cannot be built at is a fresh field, not a
            # dead session.
            self._start_engine(width, height)
        except Exception as exc:
            self._device_lost = reason
            log.error("could not rebuild the simulation: %s", exc)
            return False

        self._size = (width, height)
        # Pacing state describes a device that is gone: the frame times were
        # measured against it, and the accumulator holds however long the loss
        # took to notice.
        self._accumulator = 0.0
        self._sim_hz_scale = 1.0
        self._fps_scale = 1.0
        self._frame_times.clear()
        self._last_time = time.perf_counter()
        self._last_checkpoint = time.perf_counter()
        self._device_losses += 1
        log.warning(
            "recovered on %s; the field is %s",
            self.device_info.describe(),
            f"the one saved at {self.resumed_from}" if self.resumed_from
            else "freshly seeded (there was no usable checkpoint to return to)",
        )
        return True

    # -- run ----------------------------------------------------------------

    def _open_control_panel(self) -> None:
        if not self.options.ui:
            return
        if not self.have_qt:
            log.warning(
                "the control panel needs the Qt backend, which is not in use"
            )
            return
        try:
            from .ui.control_panel import ControlPanel

            self.panel = ControlPanel(self)
            self.panel.show()
        except Exception as exc:
            log.warning("could not open the control panel: %s", exc, exc_info=True)

    def run(self) -> None:
        self.setup()
        self._open_control_panel()

        self.canvas.request_draw(self.draw_frame)
        # After the draw function is set, because the poll can force a frame.
        self._install_window_poll()
        self._install_signal_handlers()
        try:
            self.loop.run()
        finally:
            # Closing the window is the commonest way a session ends, so it has
            # to checkpoint just like an interval tick would. By this point
            # that has usually already happened, from the close event; this is
            # the backstop for the loop ending some other way.
            self.shutdown()
            self._restore_signal_handlers()
