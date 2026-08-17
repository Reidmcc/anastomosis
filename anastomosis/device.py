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


@dataclass
class DeviceInfo:
    vendor: str
    device: str
    adapter_type: str
    backend: str
    is_software: bool

    def describe(self) -> str:
        kind = " (software)" if self.is_software else ""
        return f"{self.vendor} {self.device} [{self.backend}]{kind}"


def request_device(
    power_preference: str = "high-performance",
    force_fallback: bool = False,
) -> tuple[wgpu.GPUDevice, DeviceInfo]:
    """Acquire a device, preferring a discrete GPU.

    Requests no optional features: the whole pipeline is deliberately core
    WebGPU so it runs identically on Vulkan, Metal and DX12, and so the headless
    software adapter used by the test suite exercises the same code paths as the
    real one.
    """
    adapter = wgpu.gpu.request_adapter_sync(
        power_preference=power_preference,
        force_fallback_adapter=force_fallback,
    )
    info = adapter.info
    device_info = DeviceInfo(
        vendor=str(info.get("vendor", "?")),
        device=str(info.get("device", "?")),
        adapter_type=str(info.get("adapter_type", "?")),
        backend=str(info.get("backend_type", "?")),
        is_software=str(info.get("adapter_type", "")).lower() == "cpu",
    )

    device = adapter.request_device_sync(label="anastomosis")
    log.info("using %s", device_info.describe())
    if device_info.is_software:
        log.warning(
            "no hardware GPU found; running on a software adapter. This is fine "
            "for tests but far too slow for interactive use."
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
