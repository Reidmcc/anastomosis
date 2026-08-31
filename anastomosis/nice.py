"""Being a good GPU neighbor: bounded queue depth and preemption gaps.

The problem (issue #40): Windows preempts GPU work between submissions, not
inside them, and ``tick()`` never waits for anything. A loop that calls it as
fast as Python can -- a soak test, a headless capture -- therefore piles
submissions into the hardware queue without bound. Measured on the RTX 3080,
the volumetric backend's tick costs ~6.6 ms of GPU time but only ~1.2 ms to
encode, so a free-running loop is ~70 ticks deep after its first hundred and
minutes deep after an hour, and the desktop compositor's small raster jobs
wait behind that queue: a one-workgroup probe dispatch that completes in 1 ms
on an idle desktop takes 8 ms at the median and 20 ms at p99 under a
free-running soak (``tools/gpu_probe.py`` measures this). A 60 Hz compositor
has a 16.7 ms budget, so windows visibly stop compositing -- at single-digit
*average* GPU utilisation, because what matters is how continuously the queue
is held, not how much work is in it.

The remedy is a policy applied where every backend submits its tick
(:meth:`~anastomosis.backend.Backend._submit_tick`): wait for the tick's own
GPU work to finish -- the queue this process holds is then never more than
one tick deep -- and yield a few milliseconds before the next tick, a window
in which this process provably has nothing in the queue at all. With both in
place the probe reads at its idle baseline (p99 ~7 ms) while a soak still
runs at ~90 ticks/s; without the yield most of the benefit remains (p99
~8 ms) at ~126 ticks/s. Raw free-running is not a throughput either of these
numbers can honestly be compared to: it "finishes" 27 s of submissions and
then drains queued work for another two minutes.

Who gets which default: paced entry points do not need the policy and
free-running ones do, so :func:`default_policy` -- what a freshly built
backend gets -- resolves ``enabled=None`` ("auto") to *on* for hardware
adapters and *off* for software ones (lavapipe has no compositor to protect,
and CI should not sleep). The interactive application overrides that
explicitly from its config (``gpu_nice``, CLI ``--gpu-nice``, default off):
its frame loop already paces ticks to vsync, so its queue never runs deep,
and it is the one entry point where throughput is latency. Tests and headless
scripts construct backends directly and simply inherit the auto default.

The sleep is injectable (:attr:`GpuNice.sleep`) so tests exercise the policy
without waiting on a wall clock, and the whole policy lives on the backend as
a plain attribute (``engine.nice``) so any entry point can replace it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import wgpu

# The yield between tick submissions, when the policy is on. Chosen on the
# RTX 3080: 3 ms is enough that a compositor-shaped probe reads at its idle
# baseline (its jobs are ~1 ms; the window fits one comfortably), and small
# next to the ~8 ms a volumetric tick costs, so the soak throughput cost is
# modest. There is no cliff on either side of it.
DEFAULT_YIELD_SECONDS = 0.003


def drain(device: wgpu.GPUDevice) -> None:
    """Block until every submission on this device has completed.

    ``wgpuDevicePoll(wait=True)`` through wgpu-py's private ``_poll_wait``,
    because the public ``on_submitted_work_done_sync`` is broken in the
    pinned wgpu-py 0.32 (a callback-signature mismatch against its own
    wgpu-native). The public API is the fallback so that a future wgpu-py
    that drops the private name lands on the path it will by then have fixed.
    """
    poll_wait = getattr(device, "_poll_wait", None)
    if poll_wait is not None:
        poll_wait()
    else:  # pragma: no cover - future wgpu-py without the private poll
        device.queue.on_submitted_work_done_sync()


def _is_software(device: wgpu.GPUDevice) -> bool:
    """Whether the device is a software adapter, read off the device itself.

    The same test :mod:`anastomosis.device` applies to its ``DeviceInfo``,
    but from the adapter wgpu keeps on the device, so the policy needs
    nothing passed to it. Unreadable info counts as hardware: the failure
    mode of being nice to a software adapter is a slower test run, the
    failure mode of free-running on hardware is a starved desktop.
    """
    try:
        kind = str(device.adapter.info.get("adapter_type", "")).lower()
    except Exception:  # pragma: no cover - adapter info is core wgpu-py
        return False
    return kind == "cpu"


@dataclass
class GpuNice:
    """The niceness policy one backend applies after each tick submission.

    ``enabled`` is three-valued: ``True`` and ``False`` mean what they say,
    ``None`` means *auto* -- on for hardware adapters, off for software ones,
    decided once at first use and cached.
    """

    enabled: bool | None = None
    yield_seconds: float = DEFAULT_YIELD_SECONDS
    #: The pacing seam. Tests inject a recorder here; nothing in a timed path
    #: should ever call ``time.sleep`` around this policy's back.
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    _resolved: bool | None = field(default=None, repr=False, init=False)

    def applies_to(self, device: wgpu.GPUDevice) -> bool:
        if self.enabled is not None:
            return self.enabled
        if self._resolved is None:
            self._resolved = not _is_software(device)
        return self._resolved

    def after_submit(self, device: wgpu.GPUDevice) -> None:
        """Bound the queue to this one submission, then leave it empty a while."""
        if not self.applies_to(device):
            return
        drain(device)
        if self.yield_seconds > 0.0:
            self.sleep(self.yield_seconds)


def default_policy() -> GpuNice:
    """A fresh auto-mode policy, one per backend so nothing is shared."""
    return GpuNice()
