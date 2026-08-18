"""WGSL / NumPy parity for the numerical core.

Shader bugs in a system like this do not announce themselves. There are no
assertions on a GPU, the output is a slowly drifting image, and "it looks a bit
wrong" is not something anyone can act on months later. Running the same maths
in numpy and comparing is the only way to know the reaction is the reaction.

The comparison is done in f32 on both sides where possible; the reaction field
is stored as rgba16float, so agreement is asserted to roughly f16 precision.
"""

from __future__ import annotations

import numpy as np
import pytest
import wgpu

import morphology as M
import reference as R
from anastomosis import config, gpu_params, shaders

SIZE = 64
# rgba16float has a 10-bit mantissa; values here are O(0.1-1), so ~1e-3
# absolute is the realistic floor for a round trip through storage.
F16_ATOL = 2e-3


def _run_reaction_on_gpu(
    device, u, v, feed, kill, params, substeps=1, scale=None, range_du=0.0
):
    """Run reaction.wgsl for `substeps` steps and read back (U, V).

    `scale` is the climate `scale` channel (climate_c.x, in [-1, 1]) that sets
    the local diffusion rate. It is uploaded at the simulation's own resolution,
    so the shader's bilinear sample lands exactly on texel centres and the
    comparison stays a comparison of the arithmetic rather than of two
    interpolators.
    """
    size = u.shape[0]

    module = device.create_shader_module(code=shaders.load("reaction.wgsl"))
    pipeline = device.create_compute_pipeline(
        layout=wgpu.enums.AutoLayoutMode.auto,
        compute={"module": module, "entry_point": "main"},
    )

    usage = (
        wgpu.TextureUsage.TEXTURE_BINDING
        | wgpu.TextureUsage.STORAGE_BINDING
        | wgpu.TextureUsage.COPY_SRC
        | wgpu.TextureUsage.COPY_DST
    )

    def make_texture(array):
        texture = device.create_texture(
            size=(size, size, 1), format=wgpu.TextureFormat.rgba16float, usage=usage
        )
        data = np.ascontiguousarray(array.astype(np.float16))
        device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            data,
            {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
            (size, size, 1),
        )
        return texture

    field = np.zeros((size, size, 4), dtype=np.float32)
    field[..., 0] = u
    field[..., 1] = v
    textures = [make_texture(field), make_texture(np.zeros_like(field))]
    # Zero trail and zero climate_a isolate the pure Gray-Scott step.
    trail = make_texture(np.zeros_like(field))
    climate = make_texture(np.zeros_like(field))
    climate_c = np.zeros_like(field)
    if scale is not None:
        climate_c[..., 0] = scale
    climate_c_tex = make_texture(climate_c)

    values = {
        "dims_x": size, "dims_y": size,
        "feed": feed, "kill": kill,
        "du": params.du, "dv": params.dv, "rdt": params.dt,
        "du_min": params.du_min, "du_max": params.du_max,
        "range_du": range_du,
        "trail_feed_gain": 0.0,
        "kill_follows_feed": params.kill_follows_feed,
        "feed_min": params.feed_min, "feed_max": params.feed_max,
        "kill_band": params.kill_band,
        "kill_min": params.kill_min, "kill_max": params.kill_max,
        "range_feed": 0.0, "range_kill": 0.0,
    }
    params_buf = device.create_buffer_with_data(
        data=gpu_params.pack(gpu_params.SIM_DTYPE, values).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    stats_buf = device.create_buffer_with_data(
        data=np.zeros(1, dtype=gpu_params.STATS_DTYPE).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    sampler = device.create_sampler(
        address_mode_u="repeat", address_mode_v="repeat",
        mag_filter="linear", min_filter="linear",
    )

    index = 0
    for _ in range(substeps):
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params_buf, "offset": 0,
                                            "size": params_buf.size}},
                {"binding": 1, "resource": textures[index].create_view()},
                {"binding": 2, "resource": textures[1 - index].create_view()},
                {"binding": 3, "resource": trail.create_view()},
                {"binding": 4, "resource": climate.create_view()},
                {"binding": 5, "resource": climate_c_tex.create_view()},
                {"binding": 6, "resource": sampler},
                {"binding": 7, "resource": {"buffer": stats_buf, "offset": 0,
                                            "size": stats_buf.size}},
            ],
        )
        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((size + 7) // 8, (size + 7) // 8, 1)
        cpass.end()
        device.queue.submit([encoder.finish()])
        index = 1 - index

    raw = device.queue.read_texture(
        {"texture": textures[index], "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
        (size, size, 1),
    )
    out = np.frombuffer(raw, dtype=np.float16).reshape(size, size, 4).astype(np.float32)
    return out[..., 0], out[..., 1]


@pytest.mark.parametrize("substeps", [1, 3])
def test_reaction_matches_numpy(gpu_device, substeps):
    device, _ = gpu_device
    params = config.Config().resolve().reaction

    u0, v0 = R.seed_field(SIZE, blobs=6, seed=3)
    gpu_u, gpu_v = _run_reaction_on_gpu(
        device, u0, v0, params.feed, params.kill, params, substeps=substeps
    )

    ref_u, ref_v = u0.astype(np.float32), v0.astype(np.float32)
    for _ in range(substeps):
        ref_u, ref_v = R.gray_scott_step(
            ref_u, ref_v, params.feed, params.kill,
            du=params.du, dv=params.dv, dt=params.dt,
        )

    assert np.abs(gpu_u - ref_u).max() < F16_ATOL, (
        f"U diverged: max |delta| = {np.abs(gpu_u - ref_u).max():.5f}"
    )
    assert np.abs(gpu_v - ref_v).max() < F16_ATOL, (
        f"V diverged: max |delta| = {np.abs(gpu_v - ref_v).max():.5f}"
    )


@pytest.mark.parametrize("substeps", [1, 3])
def test_reaction_matches_numpy_with_a_varying_du(gpu_device, substeps):
    """Parity must survive the climate-driven diffusion rate of DESIGN.md 4.7.

    The morphology fix makes `du` a field rather than a scalar, and geometric
    rather than additive, which changes the innermost line of the reaction. A
    scalar-only parity test still passes with the whole mechanism silently
    disconnected, so the varying case gets its own check -- across the full
    range the climate is clamped to, which reaches both `du` bounds.
    """
    device, _ = gpu_device
    resolved = config.Config().resolve()
    params = resolved.reaction
    range_du = resolved.climate.range_du

    # Rounded through f16 first: the climate field is an rgba16float texture,
    # so that is the value the shader actually sees.
    ramp = np.linspace(-1.0, 1.0, SIZE, dtype=np.float32)
    scale = np.broadcast_to(ramp, (SIZE, SIZE)).astype(np.float16).astype(np.float32)

    u0, v0 = R.seed_field(SIZE, blobs=6, seed=3)
    gpu_u, gpu_v = _run_reaction_on_gpu(
        device, u0, v0, params.feed, params.kill, params,
        substeps=substeps, scale=scale, range_du=range_du,
    )

    du = np.clip(
        params.du * np.exp(range_du * scale), params.du_min, params.du_max)
    dv = params.dv * (du / params.du)
    assert du.min() == params.du_min and du.max() == params.du_max, (
        "the ramp no longer reaches both clamps, so the clamp in "
        "reaction.wgsl is untested"
    )
    # ...but a typical region must not be sitting on one. The climate does not
    # reach its own clamp: it settles around s.d. 0.11 (morphology.CLIMATE_SD),
    # and at that amplitude the deviation has to be free to act.
    typical = params.du * np.exp(range_du * np.array([-M.CLIMATE_SD, M.CLIMATE_SD]))
    assert (typical > params.du_min).all() and (typical < params.du_max).all(), (
        f"a one-sigma climate excursion now reaches a du clamp "
        f"({typical[0]:.3f}, {typical[1]:.3f}); most of the field would be "
        f"pinned to a single feature size again"
    )

    ref_u, ref_v = u0.astype(np.float32), v0.astype(np.float32)
    for _ in range(substeps):
        ref_u, ref_v = R.gray_scott_step(
            ref_u, ref_v, params.feed, params.kill, du=du, dv=dv, dt=params.dt,
        )

    assert np.abs(gpu_u - ref_u).max() < F16_ATOL, (
        f"U diverged: max |delta| = {np.abs(gpu_u - ref_u).max():.5f}"
    )
    assert np.abs(gpu_v - ref_v).max() < F16_ATOL, (
        f"V diverged: max |delta| = {np.abs(gpu_v - ref_v).max():.5f}"
    )


def test_reaction_conserves_nothing_but_stays_bounded(gpu_device):
    """The shader's clamps must hold even from a hostile initial state."""
    device, _ = gpu_device
    params = config.Config().resolve().reaction
    rng = np.random.default_rng(5)
    u = rng.random((SIZE, SIZE)) * 1.5
    v = rng.random((SIZE, SIZE))

    gpu_u, gpu_v = _run_reaction_on_gpu(
        device, u, v, params.feed, params.kill, params, substeps=4
    )
    assert np.isfinite(gpu_u).all() and np.isfinite(gpu_v).all()
    assert gpu_u.min() >= -1e-3 and gpu_u.max() <= 1.5 + 1e-2
    assert gpu_v.min() >= -1e-3 and gpu_v.max() <= 1.0 + 1e-2


def test_oklab_roundtrip_matches_reference():
    """The host-side Oklab used by the tests must match the shader's."""
    from anastomosis.engine import Engine

    rng = np.random.default_rng(1)
    for _ in range(200):
        lightness = float(rng.random())
        a = float(rng.random() - 0.5) * 0.3
        b = float(rng.random() - 0.5) * 0.3
        rgb = np.array(Engine._oklab_to_linear(lightness, a, b))
        back = R.linear_srgb_to_oklab(rgb)
        assert back[0] == pytest.approx(lightness, abs=1e-4)
        assert back[1] == pytest.approx(a, abs=1e-4)
        assert back[2] == pytest.approx(b, abs=1e-4)


def test_generated_struct_matches_dtype():
    """The WGSL struct and the numpy dtype come from one declaration."""
    for name, fields in (
        ("SimParams", gpu_params.SIM_FIELDS),
        ("RenderParams", gpu_params.RENDER_FIELDS),
        ("Stats", gpu_params.STATS_FIELDS),
    ):
        source = gpu_params.wgsl_struct(name, fields)
        assert source.count(",") == len(fields)
        dtype = gpu_params._dtype(fields)
        # Every field is a 4-byte scalar, so the packing is exactly one word
        # per field with no padding -- the property the whole approach relies on.
        assert dtype.itemsize == len(fields) * 4


def test_all_shaders_compile(gpu_device):
    device, _ = gpu_device
    for name in shaders.all_shader_names():
        device.create_shader_module(code=shaders.load(name), label=name)


def test_blue_noise_is_blue():
    """White-noise dither would be visibly grainy; the mask must be blue."""
    from anastomosis import bluenoise

    mask = bluenoise.generate(32, seed=2)
    spectrum = bluenoise.radial_spectrum(mask)
    low = spectrum[1:4].mean()
    high = spectrum[8:16].mean()
    assert high > low * 50, (
        f"mask is not blue: high/low frequency power ratio is {high / low:.1f}"
    )
    assert mask.min() >= 0.0 and mask.max() < 1.0
