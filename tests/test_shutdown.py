"""Closing the window ends the application -- DESIGN.md §4.6.

The failure this guards against is not a crash and leaves no traceback: the
window disappears, the user gets their screen back, and the process carries on
in the terminal they started it from, holding the GPU and a whole simulation
that nobody can see. On a Qt session that is the ordinary case rather than a
corner one, because the control panel is a second top-level window and a Qt
application with a window still open has no reason of its own to quit.

So shutdown is asserted from both ends here: that the *state* is saved and torn
down in the right order, and -- in a real subprocess, with a real Qt event loop
and the control panel open -- that closing the render window actually ends the
process, within a time limit, with the field on disk.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from anastomosis import checkpoint

SIZE = (96, 72)


# ---------------------------------------------------------------------------
# The shutdown path, driven through an offscreen canvas
# ---------------------------------------------------------------------------


class _RecordingLoop:
    """A loop that records what the application asks of it, and runs nothing."""

    def __init__(self) -> None:
        self.stopped = 0
        self.deferred: list = []

    def run(self) -> None:
        pass

    def stop(self) -> None:
        self.stopped += 1

    def call_soon(self, callback) -> None:
        self.deferred.append(callback)


class _RecordingWatcher:
    def __init__(self) -> None:
        self.stopped = 0
        self.joined = 0

    def stop(self) -> None:
        self.stopped += 1

    def join(self, timeout=None) -> None:
        self.joined += 1


class _RecordingPanel:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _app(gpu_device, monkeypatch, tmp_path, loop=None, **options):
    """A real Application on a real device, with an offscreen window."""
    from rendercanvas.offscreen import RenderCanvas
    from anastomosis import app as app_module, device as device_module

    device, info = gpu_device
    monkeypatch.setattr(device_module, "request_device", lambda *a, **k: (device, info))
    # The watcher is a separate concern, and a real one would leave a thread
    # behind; the tests that care about it install a recording stand-in.
    monkeypatch.setattr(app_module.Application, "_start_hot_reload", lambda self: None)
    monkeypatch.setattr(
        app_module.Application,
        "_make_canvas",
        lambda self: (
            RenderCanvas(size=SIZE, update_mode="manual"),
            loop if loop is not None else _RecordingLoop(),
            False,
        ),
    )
    app = app_module.Application(app_module.AppOptions(
        width=SIZE[0], height=SIZE[1], ui=False,
        config_path=tmp_path / "config.toml",
        checkpoint_path=tmp_path / "checkpoint.npz",
        checkpoint_seconds=1e9,  # only the shutdown save may write
        **options,
    ))
    app.setup()
    return app


def test_closing_the_window_saves_the_field_and_ends_the_loop(
    gpu_device, monkeypatch, tmp_path
):
    """The whole point: the window going away is a full, orderly shutdown."""
    loop = _RecordingLoop()
    app = _app(gpu_device, monkeypatch, tmp_path, loop=loop)
    for _ in range(3):
        app.draw_frame()
    ticks = app.engine.tick_count

    # What the backend does when the user closes the window: the canvas is
    # marked closed, and the loop emits the close event.
    app.canvas.close()

    assert app._stopped, "the close event did not reach shutdown"
    assert loop.stopped, "the event loop was never asked to stop"
    saved = checkpoint.load(tmp_path / "checkpoint.npz")
    assert saved is not None, "closing the window lost the field"
    assert saved.tick_count == ticks


def test_shutdown_takes_the_panel_and_the_watcher_with_it(
    gpu_device, monkeypatch, tmp_path
):
    """A leftover window or watcher thread is what keeps the process alive."""
    app = _app(gpu_device, monkeypatch, tmp_path)
    app.panel = _RecordingPanel()
    watcher = _RecordingWatcher()
    app._watcher = watcher

    app.shutdown()

    assert app.panel is None and watcher.stopped == 1
    assert watcher.joined == 1, "the watcher thread was never waited for"


def test_shutdown_is_idempotent(gpu_device, monkeypatch, tmp_path):
    """It is reached from several directions at once, and must run only once."""
    from anastomosis import app as app_module

    app = _app(gpu_device, monkeypatch, tmp_path)
    app.draw_frame()

    saves = []
    monkeypatch.setattr(
        app_module.checkpoint_module, "save", lambda path, snap: saves.append(path)
    )

    app.shutdown()
    app.shutdown()
    app._on_canvas_close()
    assert len(saves) == 1


def test_nothing_ticks_after_shutdown(gpu_device, monkeypatch, tmp_path):
    """The saved state is the last word: a late frame must not move past it."""
    app = _app(gpu_device, monkeypatch, tmp_path)
    app.draw_frame()
    app.shutdown()
    ticks = app.engine.tick_count

    app.draw_frame()
    assert app.engine.tick_count == ticks


def test_a_signal_stops_at_a_frame_boundary_not_where_it_lands(
    gpu_device, monkeypatch, tmp_path
):
    """A signal arrives mid-tick, and a mid-tick readback is not a valid one.

    So ``request_stop`` only records the request; the shutdown itself happens
    from the loop, or at the next frame boundary if the loop cannot take it.
    """
    loop = _RecordingLoop()
    app = _app(gpu_device, monkeypatch, tmp_path, loop=loop)
    app.draw_frame()

    app.request_stop()
    assert not app._stopped, "shut down where the signal landed"
    assert loop.deferred == [app.shutdown]

    ticks = app.engine.tick_count
    app.draw_frame()  # the loop never got to run the deferred call
    assert app._stopped and app.engine.tick_count == ticks
    assert checkpoint.load(tmp_path / "checkpoint.npz") is not None


def test_run_shuts_down_even_when_the_loop_just_returns(
    gpu_device, monkeypatch, tmp_path
):
    """The backstop: a loop that ends without ever closing the canvas."""
    holder: dict = {}

    class QuietLoop(_RecordingLoop):
        def run(self):
            holder["app"].draw_frame()

    app = _app(gpu_device, monkeypatch, tmp_path, loop=QuietLoop())
    holder["app"] = app
    app.run()

    assert app._stopped
    assert checkpoint.load(tmp_path / "checkpoint.npz") is not None


def test_signal_handlers_are_restored_afterwards(gpu_device, monkeypatch, tmp_path):
    """A library user's handlers must survive a run."""
    import signal

    from anastomosis import app as app_module

    before = {sig: signal.getsignal(sig) for sig in app_module.STOP_SIGNALS}
    app = _app(gpu_device, monkeypatch, tmp_path)
    app.run()
    assert {sig: signal.getsignal(sig) for sig in app_module.STOP_SIGNALS} == before


# ---------------------------------------------------------------------------
# The real thing: a Qt event loop, the control panel, and a closed window
#
# In a subprocess, because "did the process exit" is the assertion, and it
# cannot be made about the interpreter running the tests.
# ---------------------------------------------------------------------------


def _run_isolated(source: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_closing_the_qt_window_ends_the_process(tmp_path):
    """The reported bug, end to end.

    The panel is open, as it is in a normal session, so Qt itself has no reason
    to quit when the render window closes. The process must still be gone -- and
    the checkpoint written -- shortly after the window is.
    """
    pytest.importorskip("PySide6", reason="the control panel needs PySide6")

    source = """
        import os, sys
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from pathlib import Path
        state = Path(sys.argv[1])

        from anastomosis.app import AppOptions, Application
        from anastomosis import device as device_module

        try:
            device_module.request_device()
        except Exception as exc:
            print(f"skip: no device ({exc})")
            raise SystemExit(0)

        app = Application(AppOptions(
            width=96, height=72, max_fps=10, vsync=False, ui=True,
            config_path=state / "config.toml",
            checkpoint_path=state / "checkpoint.npz",
            checkpoint_seconds=1e9,
            telemetry_seconds=1e9,
        ))

        setup = app.setup

        def setup_then_close():
            setup()
            from PySide6 import QtCore, QtWidgets
            assert app.have_qt, "the Qt backend was not selected"
            # Closing the render window with the panel still open is the case
            # that used to leave the process running.
            QtCore.QTimer.singleShot(
                3000, lambda: QtWidgets.QWidget.close(app.canvas)
            )

        app.setup = setup_then_close
        app.run()
        assert app.panel is None, "the control panel outlived the window"
        print("returned")
    """
    # Generously more than the ~4s the window is up for, and far less than
    # forever, which is what the bug amounted to.
    result = _run_isolated(source, str(tmp_path), timeout=180.0)

    assert result.returncode == 0, result.stderr
    if "skip:" in result.stdout:  # pragma: no cover - CI without any GPU
        pytest.skip(result.stdout.strip())
    assert "returned" in result.stdout, result.stderr

    saved = checkpoint.load(tmp_path / "checkpoint.npz")
    assert saved is not None, "closing the window lost the field"
    assert saved.tick_count > 0


def test_the_exit_guard_ends_a_teardown_that_will_not_finish():
    """Whatever Qt and wgpu do on the way out, the process still goes away."""
    source = """
        import threading, time

        from anastomosis.__main__ import arm_exit_guard

        # A non-daemon thread that never ends: interpreter shutdown would wait
        # for it forever, which is what a wedged driver teardown looks like.
        threading.Thread(target=lambda: time.sleep(600), daemon=False).start()
        arm_exit_guard(0.5)
        print("armed", flush=True)
    """
    result = _run_isolated(source, timeout=30.0)
    assert result.returncode == 0
    assert "armed" in result.stdout
