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

import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import wgpu

from . import checkpoint as checkpoint_module
from . import config as config_module
from . import device as device_module
from . import engine as engine_module
from . import events as events_module
from . import volume as volume_module
from . import window as window_module

log = logging.getLogger(__name__)

# The two depth backends (DESIGN.md §5), and everything that differs between
# them from this side: which engine class to build, and which geometry class
# derives a fresh field's shape. Both present the same interface -- tick,
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
}


def backend_classes(name: str) -> tuple[type, type]:
    """``(engine class, geometry class)`` for a backend name."""
    module, engine_name, geometry_name = BACKEND_CLASSES[name]
    return getattr(module, engine_name), getattr(module, geometry_name)

# How long a new window size must hold before the output targets follow it.
# Dragging a window edge reports a new size every frame; coalescing them keeps
# the drag from reallocating render targets dozens of times.
RESIZE_SETTLE = 0.15

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
    ui: bool = True
    telemetry_seconds: float = 60.0
    # Checkpointing. Resuming is the default: the whole point of a mature field
    # is that it took hours to grow, so throwing it away on every launch would
    # be the surprising behaviour, not the safe one.
    checkpoint: bool = True
    resume: bool = True
    checkpoint_path: Path | None = None
    checkpoint_seconds: float = checkpoint_module.DEFAULT_INTERVAL_SECONDS


class Application:
    def __init__(self, options: AppOptions) -> None:
        self.options = options
        self.config_path = options.config_path or config_module.default_config_path()
        self.config = config_module.load(self.config_path)
        resolved = self.config.resolve()
        # Cap the canvas rate from the config, but never above the requested max.
        resolved.max_fps = min(resolved.max_fps, options.max_fps)

        self.ramp = config_module.ParamRamp(resolved)
        self.params = resolved
        self.scheduler = events_module.EventScheduler(seed=options.seed)

        self.backend = config_module.normalise_backend(
            options.backend or self.config.backend)
        # One saved field per backend, so switching does not destroy the one
        # switched away from -- see `checkpoint.default_checkpoint_path`.
        self.checkpoint_path = (
            options.checkpoint_path
            or checkpoint_module.default_checkpoint_path(self.backend)
        )
        self._saver = checkpoint_module.BackgroundSaver()
        self._last_checkpoint = time.perf_counter()
        self._checkpoint_saved_at: float | None = None
        self.resumed_from: str | None = None

        self.canvas = None
        self.device = None
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
        self._last_telemetry = time.perf_counter()
        self._watcher = None
        self._size = (0, 0)
        self._pending_size: tuple[int, int] | None = None
        self._pending_since = 0.0

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

    def setup(self) -> None:
        self.canvas, self.loop, self.have_qt = self._make_canvas()

        self.device, self.device_info = device_module.request_device()
        self.present_context = self.canvas.get_context("wgpu")
        self.target_format = self.present_context.get_preferred_format(
            self.device.adapter
        )
        self.present_context.configure(
            device=self.device, format=self.target_format
        )

        width, height = self.canvas.get_physical_size()
        self._start_engine(width, height)
        self._size = (width, height)

        self._start_hot_reload()
        self._watch_for_close()
        self._install_fullscreen()

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

    def reload_config(self) -> None:
        try:
            self.config = config_module.load(self.config_path)
            resolved = self.config.resolve()
            resolved.max_fps = min(resolved.max_fps, self.options.max_fps)
            # Ramped, never stepped: adjusting a control must not itself be
            # visual punctuation.
            self.ramp.set_target(resolved)
            log.info("reloaded %s", self.config_path)
        except Exception as exc:
            # A malformed file must not take down a session that has been
            # running for days.
            log.error("could not reload config: %s", exc)

    def apply_macros(self, macros: config_module.Macros) -> None:
        """Called by the control panel."""
        self.config.macros = macros
        resolved = self.config.resolve()
        resolved.max_fps = min(resolved.max_fps, self.options.max_fps)
        self.ramp.set_target(resolved)

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
        """The checkpoint on disk, if there is one and this launch wants it."""
        if not (self.options.checkpoint and self.options.resume):
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
        # The readback stalls this frame. Restart the clock so the stall is not
        # charged to the pacing accumulator, which would otherwise produce a
        # burst of catch-up ticks, or to the governor, which would throttle the
        # tick rate over a cost that recurs once every five minutes.
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
        field keeps the geometry it was grown at, so changing the layer count or
        the base scale does nothing until there is a new field to apply it to,
        and here there is one.
        """
        width, height = self._size
        # Wait for any in-flight write first, or it would land on disk after the
        # file it describes has been deleted.
        self._saver.join()
        checkpoint_module.discard(self.checkpoint_path)

        self.scheduler = events_module.EventScheduler(seed=self.options.seed)
        self.engine = self._make_engine(
            width, height, self._derive_geometry(width, height))
        self._accumulator = 0.0
        self._sim_hz_scale = 1.0
        self._frame_times.clear()
        self._last_time = time.perf_counter()
        self._last_checkpoint = time.perf_counter()
        self._checkpoint_saved_at = None
        self.resumed_from = None
        log.info("simulation reset; growing a new field from seeds")

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

        self.save_checkpoint(blocking=True)
        self._saver.join()
        self._stop_hot_reload()
        self._close_panel()
        self._stop_loop()

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

    def _governor(self, frame_time: float) -> None:
        """Throttle the sim tick rate if frames are running long.

        Only the tick rate is adjusted. The interpolator hides that completely,
        whereas changing resolution would be plainly visible.
        """
        self._frame_times.append(frame_time)
        if len(self._frame_times) < 30:
            return
        window = self._frame_times[-30:]
        self._frame_times = window
        median = sorted(window)[len(window) // 2]
        budget = 1.0 / max(self.params.max_fps, 1)

        if median > budget * 0.92:
            self._sim_hz_scale = max(0.35, self._sim_hz_scale * 0.97)
        elif median < budget * 0.55:
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
        now = time.perf_counter()
        frame_dt = min(now - self._last_time, 0.25)  # clamp after a stall
        self._last_time = now

        self.params = self.ramp.update(frame_dt)
        self._follow_canvas_size(now)

        sim_hz = max(self.params.sim_hz * self._sim_hz_scale, 2.0)
        tick_interval = 1.0 / sim_hz

        self._accumulator += frame_dt
        # Bound the catch-up so a long stall cannot produce a burst of ticks
        # that would look like a jump.
        max_ticks = 3
        ticks = 0
        while self._accumulator >= tick_interval and ticks < max_ticks:
            active = self.scheduler.update(tick_interval, self.params.events)
            rows, _ = self.scheduler.pack(8)
            self.engine.tick(self.params, rows)
            self._accumulator -= tick_interval
            ticks += 1
        if ticks == max_ticks:
            self._accumulator = 0.0

        frac = min(self._accumulator / tick_interval, 1.0)

        texture = self.present_context.get_current_texture()
        self.engine.render(
            self.params,
            frac=frac,
            target_view=texture.create_view(),
            target_format=self.target_format,
            frame_dt=frame_dt,
        )

        self._governor(time.perf_counter() - now)

        if now - self._last_telemetry >= self.options.telemetry_seconds:
            self._last_telemetry = now
            self._log_telemetry()

        # After the governor, so the readback stall is never charged to the
        # frame-time window that decides the tick rate.
        if (
            self.options.checkpoint
            and self.options.checkpoint_seconds > 0.0
            and now - self._last_checkpoint >= self.options.checkpoint_seconds
        ):
            self.save_checkpoint()

    def _log_telemetry(self) -> None:
        stats = self.engine.read_stats()
        window = self._frame_times[-30:] or [0.0]
        log.info(
            "tick=%d  mean_v=%.4f var=%.5f activity=%.5f  ell=%.2f (%+.3f)  "
            "exposure=%.2f  sim=%.1fHz (x%.2f)  frame=%.1fms  events=[%s]",
            self.engine.tick_count,
            stats["mean_v"], stats["var_v"], stats["mean_activity"],
            # Feature size and the correction the loop is holding to get it.
            # Logged together because neither says much alone: an ell drifting
            # with a correction near zero is the field moving on its own, and a
            # steady ell with the correction parked at its bound is the loop
            # asking for something the field will not give (DESIGN.md 4.7).
            stats["ell"], stats["corr_du"],
            stats["exposure"],
            self.params.sim_hz * self._sim_hz_scale, self._sim_hz_scale,
            1000.0 * (sum(window) / len(window)),
            self.scheduler.describe(),
        )

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
        self._install_signal_handlers()
        try:
            self.loop.run()
        finally:
            # Closing the window is the commonest way a session ends, so it has
            # to checkpoint just like a five-minute tick would. By this point
            # that has usually already happened, from the close event; this is
            # the backstop for the loop ending some other way.
            self.shutdown()
            self._restore_signal_handlers()
