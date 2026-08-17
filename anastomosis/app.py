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
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import wgpu

from . import config as config_module
from . import device as device_module
from . import engine as engine_module
from . import events as events_module

log = logging.getLogger(__name__)

# How long a new window size must hold before the output targets follow it.
# Dragging a window edge reports a new size every frame; coalescing them keeps
# the drag from reallocating render targets dozens of times.
RESIZE_SETTLE = 0.15


@dataclass
class AppOptions:
    width: int = 1280
    height: int = 720
    max_fps: int = 30
    vsync: bool = True
    fullscreen: bool = False
    config_path: Path | None = None
    seed: int | None = None
    ui: bool = True
    telemetry_seconds: float = 60.0


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

        self.canvas = None
        self.device = None
        self.engine = None
        self.present_context = None
        self.target_format = None

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
        self.engine = engine_module.Engine(
            self.device, width, height, self.params, seed=self.options.seed
        )
        self._size = (width, height)

        self._start_hot_reload()
        if self.options.fullscreen:
            self._try_fullscreen()

    def _try_fullscreen(self) -> None:
        # Borderless windowed fullscreen only. Exclusive fullscreen can stall the
        # compositor on the *other* display and steal focus, which defeats the
        # whole point of running this on a secondary monitor.
        for name in ("set_fullscreen", "_set_fullscreen"):
            method = getattr(self.canvas, name, None)
            if callable(method):
                try:
                    method(True)
                    return
                except Exception as exc:
                    log.debug("fullscreen via %s failed: %s", name, exc)
        log.info("fullscreen not supported by this backend; resize manually")

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

    def save_config(self) -> None:
        config_module.save(self.config, self.config_path)

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

    def _log_telemetry(self) -> None:
        stats = self.engine.read_stats()
        window = self._frame_times[-30:] or [0.0]
        log.info(
            "tick=%d  mean_v=%.4f var=%.5f activity=%.5f  exposure=%.2f  "
            "sim=%.1fHz (x%.2f)  frame=%.1fms  events=[%s]",
            self.engine.tick_count,
            stats["mean_v"], stats["var_v"], stats["mean_activity"],
            stats["exposure"],
            self.params.sim_hz * self._sim_hz_scale, self._sim_hz_scale,
            1000.0 * (sum(window) / len(window)),
            self.scheduler.describe(),
        )

    # -- run ----------------------------------------------------------------

    def run(self) -> None:
        self.setup()

        panel = None
        if self.have_qt and self.options.ui:
            try:
                from .ui.control_panel import ControlPanel

                panel = ControlPanel(self)
                panel.show()
            except Exception as exc:
                log.warning(
                    "could not open the control panel: %s", exc, exc_info=True
                )
        elif self.options.ui:
            log.warning(
                "the control panel needs the Qt backend, which is not in use"
            )

        self.canvas.request_draw(self.draw_frame)
        try:
            self.loop.run()
        finally:
            if self._watcher is not None:
                self._watcher.stop()
            if panel is not None:
                panel.close()
