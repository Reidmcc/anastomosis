"""Stall diagnostics: notice a freeze while it is happening, and write it down.

A freeze is the one failure this program can have that leaves nothing behind.
Every other way it goes wrong ends in a traceback or a log line, but a wedged
frame loop simply stops: the window keeps showing the last frame it drew, the
process stays alive and answers nothing, and when the user eventually kills it
there is not one line in the log to say what it had been doing. By the time
anybody can look, the evidence is gone.

So the evidence is taken while the loop is still wedged, by a thread that is
not the one that is stuck. The frame loop marks each phase of a frame as it
enters it; the watchdog thread notices those marks stop moving, and once they
have been still for long enough it writes a report -- which phase the loop
entered and never left, what the simulation's state was going into it, and, the
part that actually names the culprit, a stack for every thread in the process.

Three properties matter more than completeness:

* **It must not need the stuck thread.** Nothing on this path takes a lock the
  frame loop could be holding, touches the GPU, or calls into wgpu or Qt. A
  report is assembled from plain attribute reads and from ``faulthandler``,
  which walks other threads' frames without asking those threads for anything.
* **It must survive the interpreter being unable to run Python at all.** If a
  driver call has wedged while holding the GIL then no Python thread runs, this
  one included, and the watchdog is as frozen as everything else. That case is
  covered from outside instead, by ``faulthandler``'s C-level handlers:
  :func:`install_crash_handler` arms one for a hard crash and, on POSIX, one on
  a signal, so ``kill -USR1`` still gets stacks out of a process that is
  otherwise completely unresponsive.
* **It must sample more than once.** One stack cannot tell a deadlock from
  something merely slow. Successive samples into the same report can: identical
  stacks a minute apart mean wedged, and the process CPU counters printed
  beside them say whether the time is being spent or waited.

False alarms are the failure mode to design against, because a report that
fires whenever the window is minimised is a report nobody reads, and one nobody
reads is worth nothing on the day it matters. A loop sitting *between* frames
is therefore given much longer than one inside a frame: a compositor that stops
asking for paints is ordinary behaviour, and a frame that never returns is not.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import platform
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

# How long the loop may spend inside one phase of a frame before it counts as
# stalled. Generous beside a 33ms frame, deliberately: the governor already
# absorbs frames that merely run long, and a checkpoint readback of a large
# field is a real multi-second pause on a card that is busy elsewhere.
DEFAULT_STALL_SECONDS = 10.0

# The same, for the phases where waiting is not in itself a symptom. Not being
# asked to paint is something a healthy application does -- minimised,
# occluded, or on a display the compositor has put to sleep -- and shutdown is
# allowed to spend a while writing tens of megabytes to a slow disk.
IDLE_STALL_SECONDS = 45.0

# Phase names this module reasons about rather than merely reports. Everything
# else is opaque to it: the loop can name its phases whatever is clearest.
IDLE = "idle"
SHUTDOWN = "shutdown"
PATIENT_PHASES = frozenset({IDLE, SHUTDOWN})

# How often to take another stack sample once a stall is under way, and how
# many to take in one episode. Spaced widely enough that a slow-but-moving loop
# visibly moves between samples; capped so a freeze left overnight cannot fill
# a disk.
RESAMPLE_SECONDS = 30.0
MAX_SAMPLES = 8

# How many reports to keep. They are tens of kilobytes each, and the oldest
# stall is the least interesting one.
KEEP_REPORTS = 20


def default_report_dir() -> Path:
    """``$XDG_STATE_HOME/anastomosis/diagnostics``.

    State, next to the checkpoint, for the same reason: not hand-editable, not
    regenerable, and not something to scatter into whatever directory the
    program happened to be started from.
    """
    base = os.environ.get("XDG_STATE_HOME")
    root = Path(base) if base else Path.home() / ".local" / "state"
    return root / "anastomosis" / "diagnostics"


@dataclass
class _Episode:
    """One stall, from the moment it is noticed until the loop moves again."""

    path: Path
    phase: str
    entered_at: float
    sampled_at: float
    samples: int = 0
    peak: float = 0.0


class StallWatchdog:
    """Watches the frame loop's phase marks, and reports when they stop moving.

    The loop calls :meth:`mark` as it enters each phase of a frame. That is one
    tuple build and one clock read, which is what makes it affordable on a path
    that runs thirty times a second; everything else happens on this object's
    own thread.

    Constructed whether or not it is wanted, so the loop's ``mark`` calls never
    need guarding -- a watchdog with ``stall_seconds`` at zero simply never
    starts a thread, and its marks go into an attribute nobody reads.
    """

    def __init__(
        self,
        *,
        report_dir: Path | None = None,
        stall_seconds: float = DEFAULT_STALL_SECONDS,
        idle_seconds: float | None = None,
        snapshot: Callable[[], dict[str, object]] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.report_dir = Path(report_dir) if report_dir else default_report_dir()
        self.stall_seconds = float(stall_seconds)
        self.idle_seconds = (
            float(idle_seconds)
            if idle_seconds is not None
            else max(IDLE_STALL_SECONDS, self.stall_seconds)
        )
        self._snapshot = snapshot
        self._clock = clock
        # One tuple, replaced whole, so the watching thread can never read a
        # new phase name against the previous phase's timestamp. Rebinding an
        # attribute is atomic; a lock here would be a lock the stuck thread
        # could be the one holding.
        self._state: tuple[str, float] = ("starting", clock())
        self._frames = 0
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._episode: _Episode | None = None
        # Every report this watchdog has written, newest last. For the tests,
        # and for telling the user where to look.
        self.reports: list[Path] = []

    @property
    def enabled(self) -> bool:
        return self.stall_seconds > 0.0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- the frame loop's side ----------------------------------------------

    def mark(self, phase: str) -> None:
        """Record that the loop has just entered ``phase``.

        Deliberately the cheapest thing that can carry both facts the watchdog
        needs. It is called several times a frame and must stay that way.
        """
        self._state = (phase, self._clock())

    def frame(self) -> None:
        """Mark the start of a frame. Separate from ``mark`` only to count."""
        self._frames += 1
        self._state = ("frame", self._clock())

    @property
    def phase(self) -> str:
        return self._state[0]

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            log.info("stall watchdog disabled")
            return
        if self.running:
            return
        self.mark("starting")
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._watch, name="anastomosis-watchdog", daemon=True
        )
        self._thread.start()
        log.info(
            "stall watchdog armed: %.0fs inside a frame, %.0fs between frames; "
            "reports go to %s",
            self.stall_seconds, self.idle_seconds, self.report_dir,
        )

    def stop(self) -> None:
        """Stop watching. Idempotent, and safe to call from any thread."""
        thread, self._thread = self._thread, None
        self._wake.set()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    # -- the watching thread ------------------------------------------------

    def _watch(self) -> None:
        # Fine enough that the reported stall duration is close to the limit,
        # coarse enough to be free. A watchdog that wakes constantly is a
        # watchdog that shows up in the frame budget it exists to protect.
        interval = max(0.25, min(1.0, self.stall_seconds / 4.0))
        while not self._wake.wait(interval):
            try:
                self.poll()
            except Exception:
                # Nothing here is worth taking the session down for, but a
                # watchdog erroring once per second would bury the log it is
                # supposed to be the interesting thing in.
                log.exception("stall watchdog failed; disarming it")
                return

    def poll(self) -> None:
        """One pass of the check the watchdog thread makes, on its own.

        Public because a watchdog is only worth having if it has been shown to
        fire, and the way to show that without waiting out real timeouts is to
        drive this against a clock the test controls.
        """
        phase, since = self._state
        now = self._clock()
        waited = now - since
        limit = self.idle_seconds if phase in PATIENT_PHASES else self.stall_seconds
        episode = self._episode

        if waited < limit:
            if episode is not None:
                self._recovered(episode)
            return

        if episode is None:
            episode = self._begin(phase, waited, now)
        elif (
            episode.samples < MAX_SAMPLES
            and now - episode.sampled_at >= RESAMPLE_SECONDS
        ):
            self._write(episode, phase, waited, limit, now)
        episode.peak = max(episode.peak, waited)

    def _begin(self, phase: str, waited: float, now: float) -> _Episode:
        episode = _Episode(
            path=self._report_path("stall"),
            phase=phase,
            entered_at=now - waited,
            sampled_at=now,
        )
        self._episode = episode
        # Report first, log second, which is the opposite of the obvious order
        # and deliberate. Logging takes a lock per handler, and a frame loop
        # that froze *inside* a log call is holding that lock -- writing the
        # report first means a freeze of exactly that shape still produces the
        # file, and only loses the line about it.
        self._write(episode, phase, waited, self.stall_seconds, now, first=True)
        # The log line the user did not get last time.
        log.error(
            "frame loop stalled: %.1fs in phase '%s'; wrote %s",
            waited, phase, episode.path,
        )
        return episode

    def _recovered(self, episode: _Episode) -> None:
        self._episode = None
        self._append(
            episode,
            f"\n===== recovered =====\n"
            f"the loop moved again after at least {episode.peak:.1f}s "
            f"stalled in phase {episode.phase!r}\n",
        )
        log.warning(
            "frame loop recovered after at least %.0fs stalled in phase '%s'; "
            "see %s", episode.peak, episode.phase, episode.path,
        )

    # -- writing ------------------------------------------------------------

    def dump(self, reason: str) -> Path | None:
        """Write a one-off report, for a failure noticed by something else.

        The device-lost callback uses this: the loop is not stalled, but the
        state that produced the loss is about to be replaced, and the same
        collection of stacks and simulation state is what anybody would want.
        Safe to call from any thread, including the frame loop's.
        """
        episode = _Episode(
            path=self._report_path("dump"),
            phase=self.phase,
            entered_at=self._clock(),
            sampled_at=self._clock(),
        )
        if not self._write(
            episode, self.phase, 0.0, self.stall_seconds, self._clock(),
            first=True, reason=reason,
        ):
            return None
        return episode.path

    def _write(
        self,
        episode: _Episode,
        phase: str,
        waited: float,
        limit: float,
        now: float,
        *,
        first: bool = False,
        reason: str | None = None,
    ) -> bool:
        episode.samples += 1
        episode.sampled_at = now
        header = (
            f"\n===== sample {episode.samples} at "
            f"{datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        )
        if reason is not None:
            header += f"reason: {reason}\n"
        else:
            header += (
                f"stalled {waited:.1f}s in phase {phase!r} "
                f"(limit {limit:.0f}s)\n"
            )
        header += f"frames drawn: {self._frames}\n"
        for line in _process_lines():
            header += f"{line}\n"
        return self._append(
            episode, header, preamble=first, stacks=True, snapshot=True
        )

    def _append(
        self,
        episode: _Episode,
        text: str,
        *,
        preamble: bool = False,
        stacks: bool = False,
        snapshot: bool = False,
    ) -> bool:
        try:
            episode.path.parent.mkdir(parents=True, exist_ok=True)
            with open(episode.path, "a", encoding="utf-8") as handle:
                if preamble:
                    handle.write(_preamble())
                handle.write(text)
                if stacks:
                    # The thread table first, because faulthandler identifies
                    # threads by ident alone and a stack belonging to
                    # "anastomosis-checkpoint" means something quite different
                    # from the same stack on the main thread.
                    handle.write("\n-- threads --\n")
                    for line in _thread_inventory():
                        handle.write(f"{line}\n")
                    handle.write("\n-- stacks --\n")
                    # faulthandler writes at the file descriptor and knows
                    # nothing about Python's buffer, so the buffer goes out
                    # first or the report comes back interleaved.
                    handle.flush()
                    faulthandler.dump_traceback(file=handle, all_threads=True)
                if snapshot:
                    handle.write("\n-- simulation --\n")
                    for line in self._snapshot_lines():
                        handle.write(f"{line}\n")
        except OSError as exc:
            log.error("could not write the stall report %s: %s", episode.path, exc)
            return False
        if episode.path not in self.reports:
            self.reports.append(episode.path)
        return True

    def _snapshot_lines(self) -> list[str]:
        """The application's own state, if it offered a way to read it.

        Guarded because this runs on the watchdog thread while the main thread
        is wedged: whatever the provider does, the stacks above it are already
        on disk and are the half of the report that identifies the fault.
        """
        if self._snapshot is None:
            return ["(no snapshot provider)"]
        try:
            data = self._snapshot()
        except Exception as exc:
            return [f"(snapshot failed: {exc!r})"]
        return [f"{name}: {value}" for name, value in data.items()]

    def _report_path(self, kind: str) -> Path:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.report_dir / f"{kind}-{stamp}.txt"
        serial = 1
        while path.exists():
            path = self.report_dir / f"{kind}-{stamp}-{serial}.txt"
            serial += 1
        self._prune()
        return path

    def _prune(self) -> None:
        try:
            # By age rather than by name: the two kinds of report have
            # different prefixes, so sorting by name would throw away every
            # on-demand dump before the oldest stall.
            existing = sorted(
                self.report_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime
            )
        except OSError:  # pragma: no cover - the directory may not exist yet
            return
        for stale in existing[: max(0, len(existing) - KEEP_REPORTS + 1)]:
            try:
                stale.unlink()
            except OSError:  # pragma: no cover - someone else's file
                pass


# ---------------------------------------------------------------------------
# Process-level facts, and the handlers that work when Python cannot run
# ---------------------------------------------------------------------------


def _preamble() -> str:
    return (
        "anastomosis stall report\n"
        f"written {datetime.now():%Y-%m-%d %H:%M:%S}\n"
        f"python {platform.python_version()} on {platform.platform()}\n"
        f"executable {sys.executable}\n"
    )


def _process_lines() -> list[str]:
    """Facts about the process that say whether it is working or waiting.

    The CPU counters are the reason this is here. Two samples thirty seconds
    apart with the same stacks and the same user time is a thread blocked in a
    driver call; the same stacks with the time climbing is one spinning in a
    poll loop, which is a different bug with a different fix.
    """
    lines = [f"pid: {os.getpid()}", f"threads: {threading.active_count()}"]
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return lines
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is kilobytes on Linux and bytes on macOS, which is a trap worth
    # spelling out rather than reporting a peak a thousand times off.
    scale = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    lines.append(
        f"cpu: {usage.ru_utime:.1f}s user, {usage.ru_stime:.1f}s system"
    )
    lines.append(f"peak rss: {usage.ru_maxrss / scale:.0f} MiB")
    return lines


def _thread_inventory() -> list[str]:
    """Every live thread, keyed by the ident faulthandler prints below."""
    lines = []
    for thread in threading.enumerate():
        lines.append(
            f"  0x{(thread.ident or 0):x}  {thread.name}"
            f" (native={getattr(thread, 'native_id', None)},"
            f" daemon={thread.daemon}, alive={thread.is_alive()})"
        )
    return sorted(lines)


# Held for the life of the process. faulthandler writes to this descriptor
# from a signal handler, where opening a file is not something that can be
# done, so it is opened once at startup and never closed.
_crash_log = None


def install_crash_handler(report_dir: Path | None = None) -> Path | None:
    """Arm ``faulthandler`` for the failures no Python thread can report.

    Two of them, and the watchdog above covers neither. A hard crash inside the
    graphics driver or Qt kills the process without unwinding, so no Python
    ever runs again -- but faulthandler's handler is C, and writes stacks for
    every thread on the way down. And a call that wedges while *holding* the
    GIL stops every Python thread, the watchdog's included, which is the one
    freeze it cannot report on; registering a signal leaves a way in from
    outside, so ``kill -USR1 <pid>`` still gets stacks out of a process that is
    otherwise answering nothing at all.

    Idempotent, and never fatal: a session that cannot open its crash log is
    still a session worth running.
    """
    global _crash_log
    if _crash_log is not None:
        return Path(_crash_log.name)

    directory = Path(report_dir) if report_dir else default_report_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        # Line buffered, so that anything written before a crash is already on
        # disk when the crash arrives.
        handle = open(directory / "crash.log", "a", encoding="utf-8", buffering=1)
    except OSError as exc:
        log.warning("could not open the crash log: %s", exc)
        return None

    handle.write(
        f"\n==== anastomosis pid {os.getpid()} started "
        f"{datetime.now():%Y-%m-%d %H:%M:%S} ====\n"
    )
    faulthandler.enable(file=handle, all_threads=True)
    _crash_log = handle

    dump_signal = getattr(signal, "SIGUSR1", None)
    if dump_signal is not None and hasattr(faulthandler, "register"):
        try:
            # Emphatically not chained. Chaining runs the handler that was
            # there before, which for SIGUSR1 is the default one, and the
            # default action for SIGUSR1 is to terminate the process -- so a
            # chained dump would kill the very session the user was trying to
            # get a look inside. Unchained, the dump is written and the
            # program carries on exactly as it was.
            faulthandler.register(
                dump_signal, file=handle, all_threads=True, chain=False
            )
        except (RuntimeError, ValueError, OSError) as exc:  # pragma: no cover
            log.debug("could not register the manual dump signal: %s", exc)
        else:
            log.info(
                "if it ever freezes with nothing in the log: kill -USR1 %d "
                "writes every thread's stack to %s",
                os.getpid(), handle.name,
            )
    return Path(handle.name)
