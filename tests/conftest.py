"""Shared fixtures.

The GPU device is module-scoped because acquiring one costs real time,
especially on the software adapter used in CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def gpu_device():
    from anastomosis import device as device_module

    try:
        device, info = device_module.request_device()
    except Exception as exc:  # pragma: no cover - no GPU at all
        pytest.skip(f"no WebGPU device available: {exc}")

    # An adapter that cannot hold a 3D field is worse than no adapter: the
    # suite still runs, most of it still passes, and the handful of tests that
    # do notice look like stale ones. Say what is actually wrong instead.
    # See gpucaps.py; CI asserts the same thing so a skipped run cannot pass
    # there.
    import gpucaps

    reason = gpucaps.unsupported_reason(device, info)
    if reason is not None:  # pragma: no cover - depends on the driver
        pytest.skip(reason)
    return device, info


@pytest.fixture(scope="session")
def offscreen_target(gpu_device):
    """A render target and its format, for driving Engine.render headlessly."""
    import wgpu

    device, _ = gpu_device

    def make(width: int, height: int):
        texture = device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        return texture.create_view(), "rgba8unorm"

    return make
