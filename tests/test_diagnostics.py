"""A freeze must leave evidence behind -- see `anastomosis/diagnostics.py`.

The failure being guarded against here has already happened twice and produced
nothing at all: the window stops, the process stays up, and the log's last line
is an ordinary telemetry line from before it went wrong. Nobody can debug that.
So what is asserted is not that freezes do not happen -- this suite cannot know
that -- but that when one does, a file appears naming the phase the loop went
into and carrying a stack for every thread in the process.

Most of it is driven against a clock the test controls, because a watchdog
tested by waiting out its own timeouts is a watchdog that gets its timeouts
lowered until the test is fast and the assertion is meaningless. Two tests do
run for real, with real threads and real signals, because the parts that only
work outside the interpreter's control cannot be faked: that the watchdog
thread wakes on its own, and that a wedged process still answers a signal.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from anastomosis import diagnostics


class _Clock:
    """A hand-cranked stand-in for ``time.perf_counter``."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _watchdog(tmp_path, clock=None, **kwargs):
    kwargs.setdefault("stall_seconds", 10.0)
    kwargs.setdefault("snapshot", lambda: {"tick": 4242, "backend": "layered"})
    return diagnostics.StallWatchdog(
        report_dir=tmp_path / "diagnostics", clock=clock or _Clock(), **kwargs
    )


# ---------------------------------------------------------------------------
# Noticing
# ---------------------------------------------------------------------------


def test_a_stalled_phase_writes_a_report_that_names_it(tmp_path):
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark("render")
    clock.advance(4.0)
    dog.poll()
    assert dog.reports == [], "reported a frame that was merely slow"

    clock.advance(7.0)
    dog.poll()

    assert len(dog.reports) == 1
    report = dog.reports[0].read_text()
    assert "'render'" in report
    assert "stalled 11.0s" in report


def test_the_report_carries_the_stack_of_whatever_is_stuck(tmp_path):
    """The half of the report that actually identifies the fault."""
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark("acquire")
    clock.advance(30.0)
    dog.poll()

    report = dog.reports[0].read_text()
    # `poll` runs on this thread, so this frame is in the dump -- which is the
    # same relationship the watchdog thread has to a wedged frame loop.
    assert "test_the_report_carries_the_stack_of_whatever_is_stuck" in report
    assert "-- threads --" in report and "-- stacks --" in report
    assert "MainThread" in report


def test_the_report_carries_the_simulation_state(tmp_path):
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark("tick")
    clock.advance(30.0)
    dog.poll()

    report = dog.reports[0].read_text()
    assert "tick: 4242" in report
    assert "backend: layered" in report


def test_between_frames_gets_a_longer_leash(tmp_path):
    """A minimised window is not a freeze, and must not read as one."""
    clock = _Clock()
    dog = _watchdog(tmp_path, clock, stall_seconds=10.0, idle_seconds=45.0)

    dog.mark(diagnostics.IDLE)
    clock.advance(30.0)
    dog.poll()
    assert dog.reports == [], "reported an idle loop as a stall"

    clock.advance(20.0)
    dog.poll()
    assert len(dog.reports) == 1, "an idle loop never stalls at all"


def test_shutdown_is_given_the_same_patience(tmp_path):
    """Teardown writes tens of megabytes; a slow disk is not a freeze."""
    clock = _Clock()
    dog = _watchdog(tmp_path, clock, stall_seconds=10.0, idle_seconds=45.0)

    dog.mark(diagnostics.SHUTDOWN)
    clock.advance(30.0)
    dog.poll()

    assert dog.reports == []


# ---------------------------------------------------------------------------
# Wedged, or merely slow
# ---------------------------------------------------------------------------


def test_a_stall_is_sampled_again_so_it_can_be_told_from_slowness(tmp_path):
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark("checkpoint")
    clock.advance(11.0)
    dog.poll()
    clock.advance(diagnostics.RESAMPLE_SECONDS + 1.0)
    dog.poll()

    assert len(dog.reports) == 1, "a continuing stall started a second report"
    report = dog.reports[0].read_text()
    assert report.count("===== sample") == 2
    assert "sample 2" in report


def test_sampling_stops_before_it_can_fill_a_disk(tmp_path):
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark("render")
    clock.advance(11.0)
    for _ in range(MAX := diagnostics.MAX_SAMPLES + 5):
        dog.poll()
        clock.advance(diagnostics.RESAMPLE_SECONDS + 1.0)

    report = dog.reports[0].read_text()
    assert report.count("===== sample") == diagnostics.MAX_SAMPLES


def test_recovery_closes_the_episode_and_says_so(tmp_path):
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark("render")
    clock.advance(20.0)
    dog.poll()
    first = dog.reports[0]

    # The loop moves again.
    dog.mark("frame")
    dog.poll()
    assert "recovered" in first.read_text()
    assert "at least 20.0s" in first.read_text()

    # And a later stall is a new episode, in its own file.
    dog.mark("render")
    clock.advance(20.0)
    dog.poll()
    assert len(dog.reports) == 2 and dog.reports[1] != first


# ---------------------------------------------------------------------------
# Told what it cannot see
# ---------------------------------------------------------------------------


def test_a_loop_parked_on_purpose_is_not_a_stall(tmp_path):
    """A minimised window is not asked to paint, and never will be by itself.

    No idle timeout can tell that from a freeze -- one long enough to cover a
    window left minimised over lunch is one that would never report a real
    freeze either. So whoever can see the window says so, and while they do,
    sitting between frames stops counting.
    """
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)
    dog.set_paused("the window is not on screen")

    dog.mark(diagnostics.IDLE)
    clock.advance(diagnostics.IDLE_STALL_SECONDS * 10)
    dog.poll()

    assert not dog.reports, "a window nobody is looking at was called a freeze"


def test_a_frame_that_never_returns_is_a_stall_even_when_parked(tmp_path):
    """Only the patient phases are excused: a wedged frame is a fault anyway."""
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)
    dog.set_paused("the window is not on screen")

    dog.mark("render")
    clock.advance(20.0)
    dog.poll()

    assert dog.reports, "a frame that never returned went unreported"


def test_an_episode_that_turns_out_to_be_a_pause_is_closed(tmp_path):
    """Otherwise the report ends mid-episode, reading like a session that died."""
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    dog.mark(diagnostics.IDLE)
    clock.advance(diagnostics.IDLE_STALL_SECONDS + 1.0)
    dog.poll()
    report = dog.reports[0]

    dog.set_paused("the window is not on screen")
    dog.poll()

    assert "not a stall" in report.read_text()
    assert dog.paused == "the window is not on screen"


def test_the_frame_count_is_readable_from_outside(tmp_path):
    """The one number that says the loop is alive, and what the poll watches."""
    dog = _watchdog(tmp_path)

    dog.frame()
    dog.frame()

    assert dog.frames == 2


# ---------------------------------------------------------------------------
# Robustness -- the report is worth more than any one line in it
# ---------------------------------------------------------------------------


def test_a_failing_snapshot_provider_does_not_cost_the_stacks(tmp_path):
    def explode():
        raise RuntimeError("the engine is halfway through being replaced")

    clock = _Clock()
    dog = _watchdog(tmp_path, clock, snapshot=explode)

    dog.mark("tick")
    clock.advance(20.0)
    dog.poll()

    report = dog.reports[0].read_text()
    assert "snapshot failed" in report
    assert "-- stacks --" in report


def test_old_reports_are_pruned(tmp_path):
    clock = _Clock()
    dog = _watchdog(tmp_path, clock)

    for _ in range(diagnostics.KEEP_REPORTS + 6):
        dog.mark("render")
        clock.advance(20.0)
        dog.poll()
        dog.mark("frame")
        dog.poll()

    kept = list((tmp_path / "diagnostics").glob("*.txt"))
    assert len(kept) <= diagnostics.KEEP_REPORTS


def test_a_disabled_watchdog_starts_nothing_and_still_takes_marks(tmp_path):
    """The loop's marks are unguarded, so they must be safe when it is off."""
    dog = _watchdog(tmp_path, stall_seconds=0.0)
    dog.start()

    dog.frame()
    dog.mark("render")

    assert not dog.enabled and not dog.running
    assert dog.phase == "render"
    dog.stop()


def test_dump_writes_a_report_for_a_fault_noticed_elsewhere(tmp_path):
    dog = _watchdog(tmp_path)
    dog.mark("render")

    path = dog.dump("device lost: it just went away")

    assert path is not None
    report = path.read_text()
    assert "device lost: it just went away" in report
    assert "-- stacks --" in report


# ---------------------------------------------------------------------------
# For real: a thread that wakes on its own, and a process that answers a signal
# ---------------------------------------------------------------------------


def _wedged_in_something_that_never_returns(seconds: float) -> None:
    """Stands in for the driver call that does not come back."""
    time.sleep(seconds)


def test_the_watchdog_thread_catches_a_freeze_on_its_own(tmp_path):
    dog = _watchdog(tmp_path, clock=time.perf_counter, stall_seconds=0.4)
    dog.start()
    assert dog.running

    try:
        dog.frame()
        dog.mark("render")
        _wedged_in_something_that_never_returns(1.6)
    finally:
        dog.stop()

    assert len(dog.reports) == 1, "the watchdog thread never fired"
    report = dog.reports[0].read_text()
    assert "'render'" in report
    # The name of the function this thread was stuck in, taken by another
    # thread while it was still stuck. This is the whole mechanism.
    assert "_wedged_in_something_that_never_returns" in report


@pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="no SIGUSR1 on this platform"
)
def test_a_signal_gets_stacks_out_of_a_process_that_answers_nothing(tmp_path):
    """The case the watchdog thread cannot cover: no Python running at all.

    Also asserts the thing that is easy to get wrong and fatal when it is: the
    process must still be alive afterwards. SIGUSR1 terminates by default, so a
    handler registered to chain to the previous one would kill the session the
    user was trying to look inside.
    """
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(Path(__file__).parent.parent)!r})
        from anastomosis import diagnostics
        diagnostics.install_crash_handler({str(tmp_path)!r})
        def a_call_that_holds_the_gil_and_never_returns():
            time.sleep(30)
        print("ready", flush=True)
        a_call_that_holds_the_gil_and_never_returns()
    """)
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert child.stdout.readline().strip() == "ready"
        crash_log = Path(tmp_path) / "crash.log"
        os.kill(child.pid, signal.SIGUSR1)

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if crash_log.exists() and "a_call_that_holds_the_gil" in crash_log.read_text():
                break
            time.sleep(0.05)
        else:
            pytest.fail(f"no stacks arrived in {crash_log}")

        assert child.poll() is None, "the diagnostic dump killed the process"
    finally:
        child.kill()
        child.wait(timeout=10)


# ---------------------------------------------------------------------------
# The application's side
# ---------------------------------------------------------------------------


def test_the_snapshot_reads_an_application_that_is_not_built_yet(tmp_path):
    """It runs on the watchdog thread, so it may never raise, at any stage."""
    from anastomosis import app as app_module

    app = app_module.Application(app_module.AppOptions(
        ui=False, config_path=tmp_path / "config.toml",
        checkpoint_path=tmp_path / "checkpoint.npz",
    ))

    snapshot = app.diagnostic_snapshot()

    assert snapshot["tick"] == "<no engine>"
    assert "unavailable" in str(snapshot["geometry"])
    assert snapshot["backend"]


class _StandInEngine:
    """Enough engine to be driven through one frame, and no more.

    A stand-in rather than a real engine because what is under test is where
    the marks sit in ``draw_frame``, which has nothing to do with a GPU -- and
    a phase-marking test that only runs on a machine with a graphics adapter is
    one that stops being run.
    """

    def __init__(self, watchdog) -> None:
        self._watchdog = watchdog
        self.tick_count = 0
        self.phases: list[str] = []

    def tick(self, params, rows) -> None:
        self.phases.append(self._watchdog.phase)
        self.tick_count += 1

    def render(self, params, **kwargs) -> None:
        self.phases.append(self._watchdog.phase)

    def resize(self, width, height) -> None:
        pass


class _StandInContext:
    def __init__(self, watchdog) -> None:
        self._watchdog = watchdog
        self.phases: list[str] = []

    def get_current_texture(self):
        self.phases.append(self._watchdog.phase)
        return type("Texture", (), {"create_view": lambda self: None})()


class _StandInCanvas:
    def get_physical_size(self):
        return (96, 72)


def _stubbed_app(tmp_path):
    """An application wired to stand-ins, ready to be driven a frame at a time."""
    from anastomosis import app as app_module

    app = app_module.Application(app_module.AppOptions(
        width=96, height=72, ui=False,
        config_path=tmp_path / "config.toml",
        checkpoint=False,
        stall_seconds=0.0,  # no thread: the marks are what is under test
        diagnostics_dir=tmp_path / "diagnostics",
    ))
    app.canvas = _StandInCanvas()
    app.engine = _StandInEngine(app.watchdog)
    app.present_context = _StandInContext(app.watchdog)
    app.target_format = "rgba8unorm"
    app._size = (96, 72)
    return app


def test_the_frame_loop_marks_the_phase_it_is_in(tmp_path):
    """Without this a report names a phase, and the phase is always wrong."""
    app = _stubbed_app(tmp_path)

    # Two frames: the first has no accumulated time to tick with, the second
    # arrives with the whole first frame's worth of it.
    app.draw_frame()
    app._accumulator = 1.0
    app.draw_frame()

    # Each stand-in call records the phase it was called under, so the phase
    # name appearing at all is the assertion that the mark is in the right place.
    assert app.engine.tick_count >= 1, "the second frame never ticked"
    assert "tick" in app.engine.phases, "the tick phase is not marked"
    assert "render" in app.engine.phases, "the render phase is not marked"
    assert app.present_context.phases == ["acquire", "acquire"], (
        "acquiring the swapchain image -- the one call here that blocks by "
        "design -- is not marked as its own phase"
    )
    assert app.watchdog.phase == diagnostics.IDLE, (
        "the loop does not go back to idle, so every ordinary gap between "
        "frames would be judged against the much shorter in-frame limit"
    )


def test_shutdown_marks_itself_and_stops_the_watchdog(tmp_path):
    app = _stubbed_app(tmp_path)
    app.watchdog.stall_seconds = 5.0
    app.watchdog.start()

    app.shutdown()

    assert app.watchdog.phase == diagnostics.SHUTDOWN
    assert not app.watchdog.running, "the watchdog thread outlived the session"
