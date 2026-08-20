"""What the suite needs from an adapter before its results mean anything.

Every field in this project is a 3D texture on a torus, and every pass writes
one with `textureStore`. That is the one capability worth checking before
running anything, because there is a backend in common use that does not have
it and does not say so: when no Vulkan driver is installed, wgpu falls back to
OpenGL, and Mesa's GL backend binds a 3D storage texture as a single
non-layered slice. Every write then lands at z = 0. The slab collapses onto
its near face, the depth axis stops wrapping, curl stops cancelling, and the
D threads that should have written D different voxels race for one -- so the
run is not even repeatable.

The suite notices, of course: divergence-free, the three-axis wrap and the
checkpoint-resume comparison all fail. But they fail *numerically*, several
layers away from the cause, and the obvious reading of three unrelated-looking
numeric failures is that the tests have gone stale. They have not. The adapter
has. So probe for it directly, once, and say so in the one place where the
answer is still legible.

The probe deliberately shares no code with the shaders it protects: it is a
question about the driver, not about this project's WGSL.
"""

from __future__ import annotations

import functools

import numpy as np
import wgpu

_PROBE = """
@group(0) @binding(0) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    textureStore(dst, vec3<i32>(gid), vec4<f32>(f32(gid.z) + 1.0, 0.0, 0.0, 1.0));
}
"""

_WIDTH, _HEIGHT, _DEPTH = 4, 4, 4


@functools.lru_cache(maxsize=4)
def _depth_survives_a_store(device) -> bool:
    """Does a `textureStore` into a 3D texture keep its z coordinate?"""
    texture = device.create_texture(
        size=(_WIDTH, _HEIGHT, _DEPTH),
        dimension="3d",
        format=wgpu.TextureFormat.rgba16float,
        usage=wgpu.TextureUsage.STORAGE_BINDING | wgpu.TextureUsage.COPY_SRC,
    )
    pipeline = device.create_compute_pipeline(
        layout=wgpu.enums.AutoLayoutMode.auto,
        compute={"module": device.create_shader_module(code=_PROBE),
                 "entry_point": "main"},
    )
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[{"binding": 0, "resource": texture.create_view()}],
    )
    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(pipeline)
    cpass.set_bind_group(0, bind_group)
    cpass.dispatch_workgroups(1, 1, 1)
    cpass.end()
    device.queue.submit([encoder.finish()])

    raw = device.queue.read_texture(
        {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": _WIDTH * 8, "rows_per_image": _HEIGHT},
        (_WIDTH, _HEIGHT, _DEPTH),
    )
    written = np.frombuffer(raw, dtype=np.float16).reshape(
        _DEPTH, _HEIGHT, _WIDTH, 4)[..., 0].astype(np.float32)
    expected = np.arange(1, _DEPTH + 1, dtype=np.float32)
    return bool(np.array_equal(written, np.broadcast_to(
        expected[:, None, None], written.shape)))


def unsupported_reason(device, info) -> str | None:
    """A sentence naming the problem and its fix, or None if the adapter is fine."""
    if _depth_survives_a_store(device):
        return None
    return (
        f"this adapter ({info.describe()}) drops the z coordinate of every "
        "textureStore into a 3D texture, so the slab collapses onto one slice "
        "and nothing the suite measures would mean anything. It is almost "
        "always the OpenGL fallback, chosen because no Vulkan driver is "
        "installed: `apt-get install mesa-vulkan-drivers` gives you lavapipe, "
        "which is what CI runs on."
    )
