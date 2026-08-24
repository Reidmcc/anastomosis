"""Whether the machine is on mains power, and how often to ask.

Only reached because of laptops (DESIGN.md §8.3). A field that runs for days
on a desktop costs its owner nothing they notice; the same field on a laptop
holds the GPU out of its idle states for the whole of a battery, and the user
who wanted something calm on a second screen did not ask for that.

Three platforms, three unrelated mechanisms, one answer, and the answer has
three states rather than two:

* ``True``  -- on mains,
* ``False`` -- on battery,
* ``None``  -- cannot say.

"Cannot say" is not folded into either, and it is treated as mains by
everything downstream. That is the same rule the window poll follows for a
window that will not say whether it is on screen (DESIGN.md §8.2): the failure
that matters is the one where a wrong guess degrades a session nobody asked to
degrade, and a desktop that has no battery to report is exactly the machine
that must never be throttled for one.

Reading it is not free everywhere -- macOS has no file to read and needs a
subprocess -- so it is read on a thread of its own and the frame loop only
ever looks at the last answer.
"""

from __future__ import annotations

import logging
import platform
import threading

log = logging.getLogger(__name__)

# How often to ask. Slow on purpose: nothing here is urgent, plugging in is not
# a thing that needs to be noticed within a frame, and on macOS each poll is a
# process. A minute is well inside the time it takes anyone to wonder why the
# fans have not changed.
DEFAULT_POLL_SECONDS = 60.0


def _linux_on_mains() -> bool | None:
    """Read ``/sys/class/power_supply``.

    A machine with no mains supply listed is a machine with no battery to be
    on -- a desktop -- which is "cannot say" rather than "on battery": the
    kernel is not reporting a state, there simply is not one.
    """
    from pathlib import Path

    root = Path("/sys/class/power_supply")
    if not root.is_dir():
        return None
    answer: bool | None = None
    try:
        for supply in sorted(root.iterdir()):
            try:
                if (supply / "type").read_text().strip() != "Mains":
                    continue
                online = (supply / "online").read_text().strip()
            except OSError:
                continue
            # Any mains supply that is online settles it; otherwise keep
            # looking, because a docked laptop can list several and only one
            # of them is carrying the power.
            if online == "1":
                return True
            answer = False
    except OSError:
        return None
    return answer


def _windows_on_mains() -> bool | None:
    """``GetSystemPowerStatus``. No subprocess, no dependency."""
    import ctypes

    class Status(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = Status()
    try:
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    except Exception:
        return None
    if not ok:
        return None
    # 0 offline, 1 online, 255 unknown -- and 255 is what a desktop with no
    # battery driver reports, so it must stay "cannot say".
    return {0: False, 1: True}.get(int(status.ACLineStatus))


def _macos_on_mains() -> bool | None:
    """``pmset -g batt``, which says which source it is drawing from."""
    import subprocess

    try:
        out = subprocess.run(
            ["/usr/bin/pmset", "-g", "batt"],
            capture_output=True, text=True, timeout=5.0, check=False,
        ).stdout.lower()
    except (OSError, subprocess.SubprocessError):
        return None
    if "'ac power'" in out:
        return True
    if "'battery power'" in out:
        return False
    return None


def on_mains() -> bool | None:
    """Ask this machine once. ``None`` means it did not say."""
    try:
        system = platform.system()
        if system == "Linux":
            return _linux_on_mains()
        if system == "Windows":
            return _windows_on_mains()
        if system == "Darwin":
            return _macos_on_mains()
    except Exception as exc:  # pragma: no cover - a platform quirk, not a fault
        log.debug("could not read the power source: %s", exc)
    return None


class PowerSource:
    """The last answer, refreshed on a thread of its own.

    Shaped like the stall watchdog and for the same reason: the thing that
    reads it is the frame loop, and the frame loop may not block on a
    subprocess. Reading :attr:`on_battery` is a plain attribute load, which
    needs no lock -- one thread writes it, one reads it, and a reader that
    catches the previous value is a reader that is one minute out of date
    about something that changes a handful of times a day.
    """

    def __init__(
        self,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        probe=on_mains,
    ) -> None:
        self._poll_seconds = max(float(poll_seconds), 1.0)
        self._probe = probe
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # What the machine last said, and the derived answer the app uses.
        # Starts at "cannot say", which reads as mains, so a session never
        # begins throttled on the strength of a poll that has not happened.
        self.mains: bool | None = None
        self.on_battery = False

    def start(self) -> None:
        if self._thread is not None:
            return
        self.poll()  # so the first frame already has a real answer
        self._thread = threading.Thread(
            target=self._run, name="anastomosis-power", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    def poll(self) -> bool:
        """Ask now, update the answer, and return whether it changed."""
        try:
            mains = self._probe()
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("power probe failed: %s", exc)
            mains = None
        battery = mains is False
        changed = battery != self.on_battery
        self.mains = mains
        self.on_battery = battery
        if changed:
            log.info("power source: %s", self.describe())
        return changed

    def _run(self) -> None:
        while not self._stop.wait(self._poll_seconds):
            self.poll()

    def describe(self) -> str:
        if self.mains is None:
            return "unknown (assuming mains)"
        return "mains" if self.mains else "battery"
