"""Adapter and device management, including recovery from device loss.

A session is expected to run for days. Over that span a driver reset, a GPU
hang, a monitor being unplugged, or a sleep/wake cycle are all routine rather
than exceptional, and none of them should end the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import wgpu

log = logging.getLogger(__name__)

# What the ``--gpu`` flag offers, and what each asks the driver for.
#
# ``auto`` is the default and passes no preference at all, which is not the
# same as asking for either one: it leaves the choice to the platform, and on
# a laptop with switchable graphics the platform's answer is the integrated
# GPU unless something says otherwise. That is the right default for an
# ambient visual -- the previous "high-performance" spun up the discrete card
# on every hybrid laptop, for a program explicitly sized to leave the machine
# usable and expected to run for days on battery.
#
# The two explicit choices remain, because "leave it to the platform" is not
# an answer on every machine: a Linux PRIME setup can offer both and default
# to whichever the session was launched under.
GPU_CHOICES = ("auto", "integrated", "discrete")
DEFAULT_GPU_CHOICE = "auto"

_PREFERENCE = {
    "auto": None,
    "integrated": "low-power",
    "discrete": "high-performance",
}


def normalise_gpu_choice(name: str | None) -> str:
    """Fold a spelling of a GPU choice onto one of :data:`GPU_CHOICES`."""
    key = str(name or DEFAULT_GPU_CHOICE).strip().lower()
    aliases = {
        "igpu": "integrated", "low-power": "integrated", "low_power": "integrated",
        "dgpu": "discrete", "high-performance": "discrete",
        "high_performance": "discrete", "default": "auto", "": "auto",
    }
    key = aliases.get(key, key)
    if key not in GPU_CHOICES:
        log.warning("unknown GPU choice %r, using %r", name, DEFAULT_GPU_CHOICE)
        return DEFAULT_GPU_CHOICE
    return key


@dataclass
class DeviceInfo:
    vendor: str
    device: str
    adapter_type: str
    backend: str
    is_software: bool
    is_integrated: bool

    def describe(self) -> str:
        kind = ""
        if self.is_software:
            kind = " (software)"
        elif self.is_integrated:
            kind = " (integrated)"
        return f"{self.vendor} {self.device} [{self.backend}]{kind}"


class NoAdapter(RuntimeError):
    """Raised when no GPU adapter could be acquired at all."""


def request_device(
    gpu: str = DEFAULT_GPU_CHOICE,
    force_fallback: bool = False,
) -> tuple[wgpu.GPUDevice, DeviceInfo]:
    """Acquire a device, from the class of GPU ``gpu`` names.

    Requests no optional features and no raised limits: the whole pipeline is
    deliberately core WebGPU, so it runs identically on Vulkan, Metal and DX12,
    on integrated and discrete GPUs alike, and so the headless software adapter
    used by the test suite exercises the same code paths as the real one. That
    is what makes an integrated GPU a question of *cost* rather than one of
    portability -- see DESIGN.md §8.3.
    """
    choice = normalise_gpu_choice(gpu)
    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference=_PREFERENCE[choice],
            force_fallback_adapter=force_fallback,
        )
    except Exception as exc:
        raise NoAdapter(
            f"no GPU adapter available ({exc}). This needs a GPU with Vulkan, "
            "Metal or DX12 and a working driver"
        ) from exc
    if adapter is None:
        # Distinct from the raising path above, and reachable: a driver that
        # simply has nothing to offer returns None rather than failing, and
        # reading `.info` off it would report an AttributeError instead of the
        # thing that actually went wrong.
        raise NoAdapter(
            f"no GPU adapter matched {choice!r}. This needs a GPU with Vulkan, "
            "Metal or DX12 and a working driver"
        )

    info = adapter.info
    kind = str(info.get("adapter_type", "")).lower()
    device_info = DeviceInfo(
        vendor=str(info.get("vendor", "?")),
        device=str(info.get("device", "?")),
        adapter_type=str(info.get("adapter_type", "?")),
        backend=str(info.get("backend_type", "?")),
        is_software=kind == "cpu",
        is_integrated=kind == "integratedgpu",
    )

    device = adapter.request_device_sync(label="anastomosis")
    log.info("using %s", device_info.describe())
    if device_info.is_software:
        log.warning(
            "no hardware GPU found; running on a software adapter. This is fine "
            "for tests but far too slow for interactive use."
        )
    elif device_info.is_integrated and choice == "auto":
        # Said out loud, once, because on a hybrid laptop this is a choice the
        # platform made rather than one the user did, and the flag that undoes
        # it is not discoverable from a window that is merely running.
        log.info(
            "this is an integrated GPU; the simulation will be sized for it "
            "(DESIGN.md §8.3). Pass --gpu discrete for the other one."
        )
    return device, device_info


class DeviceLost(RuntimeError):
    """Raised when the device is lost and the engine must be rebuilt."""


def install_error_handler(device: wgpu.GPUDevice, on_lost) -> None:
    """Route uncaptured GPU errors to the log rather than to stderr spam.

    Errors are logged rather than raised: a single validation error should not
    take down a session that has been running for two days, and the periodic
    sanitise pass plus the output-stage clamps mean a transiently bad frame is
    recoverable.
    """
    try:
        device.add_event_listener("uncapturederror", lambda event: log.error(
            "uncaptured GPU error: %s", getattr(event, "error", event)
        ))
        device.add_event_listener("devicelost", lambda event: on_lost(event))
    except Exception:  # pragma: no cover - older wgpu-py without the event API
        log.debug("device event listeners unavailable on this wgpu-py build")
