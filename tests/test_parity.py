"""WGSL / NumPy parity for the numerical core.

Shader bugs in a system like this do not announce themselves. There are no
assertions on a GPU, the output is a slowly drifting image, and "it looks a bit
wrong" is not something anyone can act on months later. Running the same maths
in numpy and comparing is the only way to know the reaction is the reaction.

The comparison is done in f32 on both sides where possible; the reaction field
is stored as rgba16float, so agreement is asserted to roughly f16 precision.
"""

from __future__ import annotations

import math

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


def _run_trail_on_gpu(device, trail, income, deposited, params, prune_gain, prune_climate):
    """Run trail.wgsl once and read back (trail, income, prune_relative)."""
    size = trail.shape[0]
    module = device.create_shader_module(code=shaders.load("trail.wgsl"))
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
        device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(array.astype(np.float16)),
            {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
            (size, size, 1),
        )
        return texture

    field = np.zeros((size, size, 4), dtype=np.float32)
    field[..., 0] = trail
    field[..., 1] = income
    src = make_texture(field)
    dst = make_texture(np.zeros_like(field))
    # Zero climate_b leaves decay at its base; climate_c.y carries the regional
    # pruning strength.
    climate_b = make_texture(np.zeros_like(field))
    climate_c_field = np.zeros_like(field)
    climate_c_field[..., 1] = prune_climate
    climate_c = make_texture(climate_c_field)

    # The deposit accumulator is fixed point, matching agents.wgsl.
    quanta = np.round(deposited * 1048576.0).astype(np.uint32)
    deposit_buf = device.create_buffer_with_data(
        data=np.ascontiguousarray(quanta).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    values = {
        "dims_x": size, "dims_y": size,
        "trail_decay": params.trail_decay,
        "income_rate": params.income_rate,
        "prune_gain": prune_gain,
        "range_decay": 0.0,
        "range_prune": config.Config().resolve().climate.range_prune,
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
    bind_group = device.create_bind_group(
        layout=pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": {"buffer": params_buf, "offset": 0,
                                        "size": params_buf.size}},
            {"binding": 1, "resource": src.create_view()},
            {"binding": 2, "resource": dst.create_view()},
            {"binding": 3, "resource": {"buffer": deposit_buf, "offset": 0,
                                        "size": deposit_buf.size}},
            {"binding": 4, "resource": climate_b.create_view()},
            {"binding": 5, "resource": climate_c.create_view()},
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

    raw = device.queue.read_texture(
        {"texture": dst, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
        (size, size, 1),
    )
    out = np.frombuffer(raw, dtype=np.float16).reshape(size, size, 4).astype(np.float32)
    return out[..., 0], out[..., 1], out[..., 2]


@pytest.mark.parametrize("prune_gain", [0.0, 3.0])
def test_trail_matches_numpy(gpu_device, prune_gain):
    """Flux pruning is arithmetic in a shader with no assertions in it.

    Run with pruning off and on: off must reproduce the plain decay-plus-deposit
    the trail pass has always done, on must reproduce the pruned version. The
    inputs deliberately span the interesting cases -- texels with traffic and no
    trail, trail and no traffic, and every ratio between.
    """
    device, _ = gpu_device
    agents = config.Config().resolve().agents
    rng = np.random.default_rng(11)

    trail = (rng.random((SIZE, SIZE)) ** 2 * 2.0).astype(np.float32)
    income = (rng.random((SIZE, SIZE)) ** 2 * 0.06).astype(np.float32)
    # Poisson-spiky, like real agent arrivals, including plenty of empty texels.
    deposited = (rng.random((SIZE, SIZE)) < 0.3) * rng.random((SIZE, SIZE)) * 0.05
    deposited = deposited.astype(np.float32)
    prune_climate = np.linspace(-1.0, 1.0, SIZE, dtype=np.float32)
    prune_climate = np.broadcast_to(prune_climate[:, None], (SIZE, SIZE))
    prune_climate = prune_climate.astype(np.float16).astype(np.float32)

    gpu_trail, gpu_income, gpu_prune = _run_trail_on_gpu(
        device, trail, income, deposited, agents, prune_gain, prune_climate)

    # The shader reads the deposit accumulator back as fixed point, so the
    # reference has to see the same quantised value.
    quantised = np.round(deposited * 1048576.0) / 1048576.0
    range_prune = config.Config().resolve().climate.range_prune
    local_gain = prune_gain * np.exp(range_prune * prune_climate)
    ref_trail, ref_income, ref_prune = R.trail_step(
        trail.astype(np.float64), income.astype(np.float64),
        quantised.astype(np.float64), agents.trail_decay,
        agents.income_rate, local_gain.astype(np.float64),
    )

    assert np.abs(gpu_trail - ref_trail).max() < F16_ATOL, (
        f"trail diverged: max |delta| = {np.abs(gpu_trail - ref_trail).max():.5f}"
    )
    assert np.abs(gpu_income - ref_income).max() < F16_ATOL, (
        f"income diverged: max |delta| = {np.abs(gpu_income - ref_income).max():.5f}"
    )
    assert np.abs(gpu_prune - ref_prune).max() < 2e-2, (
        f"prune channel diverged: max |delta| = "
        f"{np.abs(gpu_prune - ref_prune).max():.5f}"
    )


def test_pruning_only_ever_raises_decay(gpu_device):
    """One-sidedness is the property that keeps the field from freezing.

    Centring the term so that well-fed strands decay *slower* is mass-neutral
    and was the obvious implementation; it also tripled the trail field's
    autocorrelation at a 1050-tick lag, because established structure got
    several times the memory of everything else. The mass is returned through
    the agent deposit instead. If someone reintroduces the centring here, every
    well-fed texel starts retaining more than the control, and this fails.
    """
    device, _ = gpu_device
    agents = config.Config().resolve().agents
    rng = np.random.default_rng(12)

    trail = (rng.random((SIZE, SIZE)) * 2.0).astype(np.float32)
    income = (rng.random((SIZE, SIZE)) * 0.06).astype(np.float32)
    deposited = (rng.random((SIZE, SIZE)) * 0.02).astype(np.float32)
    flat = np.zeros((SIZE, SIZE), dtype=np.float32)

    kept_off, _, prune_off = _run_trail_on_gpu(
        device, trail, income, deposited, agents, 0.0, flat)
    kept_on, _, prune_on = _run_trail_on_gpu(
        device, trail, income, deposited, agents, 6.0, flat)

    assert (prune_off == 0.0).all(), "the prune channel is not inert at zero gain"
    assert (prune_on > 0.0).any(), "pruning at gain 6 removed nothing anywhere"
    assert (kept_on <= kept_off + F16_ATOL).all(), (
        "pruning left some texel with more trail than the unpruned control, so "
        "it is handing decay back locally rather than through the agent layer"
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


def test_partial_struct_size_matches_the_buffer_stride():
    """The reduce pass writes two vec4s per tile; the engine sizes for that.

    A mismatch here is silent: the shader writes past its tile or reads a
    neighbour's, and the homeostat's statistics come out subtly wrong rather
    than obviously broken.
    """
    source = gpu_params.PARTIAL_WGSL
    members = [line for line in source.splitlines() if "vec4<f32>" in line]
    assert len(members) * 16 == gpu_params.PARTIAL_SIZE, (
        f"Partial declares {len(members)} vec4 members but PARTIAL_SIZE is "
        f"{gpu_params.PARTIAL_SIZE}"
    )


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


# --------------------------------------------------------------------------
# The feature-size measurement -- DESIGN.md §4.7 step 5
# --------------------------------------------------------------------------


def _measure_ell_on_gpu(device, v):
    """`stats.ell` as the reduce and homeostat passes compute it, for a given V.

    Both passes, because the length scale is a ratio of two sums taken in the
    first and divided in the second, and it is the ratio that has to be right.
    The controller around it is switched off with zero gains: this is a test of
    the measurement, and a test of the measurement should not be able to fail
    because of the loop.
    """
    size = v.shape[0]
    tiles = ((size + 15) // 16, (size + 15) // 16)

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
        device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(array.astype(np.float16)),
            {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
            (size, size, 1),
        )
        return texture

    field = np.zeros((size, size, 4), dtype=np.float32)
    field[..., 1] = v
    reaction = make_texture(field)
    trail = make_texture(np.zeros_like(field))

    values = {
        "dims_x": size, "dims_y": size,
        # Zero gains: the corrections stay where they are and only the
        # measurement is written.
        "gain_p": 0.0, "gain_i": 0.0, "homeo_rate": 0.0,
        "ell_rate": 0.0, "ell_ref_rate": 0.0, "ell_corr_limit": 0.0,
        "deadband": 1.0, "target_mass": 1.0,
        "target_variance": 1.0, "target_activity": 1.0,
    }
    params_buf = device.create_buffer_with_data(
        data=gpu_params.pack(gpu_params.SIM_DTYPE, values).tobytes(),
        usage=wgpu.BufferUsage.STORAGE,
    )
    partials_buf = device.create_buffer(
        size=gpu_params.PARTIAL_SIZE * tiles[0] * tiles[1],
        usage=wgpu.BufferUsage.STORAGE,
    )
    stats_buf = device.create_buffer_with_data(
        data=np.zeros(1, dtype=gpu_params.STATS_DTYPE).tobytes(),
        usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC,
    )

    def compute(name, entry_point):
        module = device.create_shader_module(code=shaders.load(name), label=name)
        return device.create_compute_pipeline(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={"module": module, "entry_point": entry_point},
        )

    def buffer(buf):
        return {"buffer": buf, "offset": 0, "size": buf.size}

    reduce_pipeline = compute("reduce.wgsl", "reduce_tiles")
    final_pipeline = compute("homeostat.wgsl", "reduce_final")

    encoder = device.create_command_encoder()
    cpass = encoder.begin_compute_pass()
    cpass.set_pipeline(reduce_pipeline)
    cpass.set_bind_group(0, device.create_bind_group(
        layout=reduce_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": buffer(params_buf)},
            {"binding": 1, "resource": reaction.create_view()},
            {"binding": 2, "resource": reaction.create_view()},
            {"binding": 3, "resource": trail.create_view()},
            {"binding": 4, "resource": buffer(partials_buf)},
        ],
    ))
    cpass.dispatch_workgroups(tiles[0], tiles[1], 1)
    cpass.set_pipeline(final_pipeline)
    cpass.set_bind_group(0, device.create_bind_group(
        layout=final_pipeline.get_bind_group_layout(0),
        entries=[
            {"binding": 0, "resource": buffer(params_buf)},
            {"binding": 1, "resource": buffer(partials_buf)},
            {"binding": 2, "resource": buffer(stats_buf)},
        ],
    ))
    cpass.dispatch_workgroups(1, 1, 1)
    cpass.end()
    device.queue.submit([encoder.finish()])

    return np.frombuffer(
        device.queue.read_buffer(stats_buf), dtype=gpu_params.STATS_DTYPE
    )[0]


@pytest.mark.parametrize("du", [0.14, 0.2097, 0.38])
def test_the_length_scale_matches_numpy(gpu_device, du):
    """The measure the feature-size controller is closed on.

    This is the one homeostat input that is not invariant under rearrangement,
    which is the whole reason it was added -- so it is also the one whose
    arithmetic has to agree with the offline sweeps that chose the du band. Any
    other discretisation of the gradient is a different quantity, and every
    figure in DESIGN.md §4.7 would then be quoting a scale the shader does not
    use.

    Run across the band rather than at one point, because the thing being
    checked is a ratio, and a ratio can agree at one value by luck.
    """
    device, _ = gpu_device
    params = config.Config().resolve().reaction

    u, v = R.seed_field(96, blobs=8, seed=5)
    for _ in range(1200):
        u, v = R.gray_scott_step(
            u, v, params.feed, params.kill,
            du=du, dv=params.dv * du / params.du, dt=params.dt,
        )

    # Rounded to what the texture can hold, so this compares the arithmetic
    # rather than the storage format.
    stored = v.astype(np.float16).astype(np.float32)
    stats = _measure_ell_on_gpu(device, v)
    expected = M.length_scale(stored)

    assert stats["ell"] == pytest.approx(expected, rel=0.02), (
        f"the GPU length scale ({stats['ell']:.4f}) disagrees with "
        f"morphology.length_scale ({expected:.4f}) at du = {du}"
    )
    assert stats["mean_grad_v"] == pytest.approx(
        M.gradient_magnitude(stored).mean(), rel=0.02
    )


def test_the_length_scale_follows_the_diffusion_rate(gpu_device):
    """The plant the controller assumes: ell rises with du, as roughly sqrt.

    The loop's gain is sized against that exponent, and its sign is the whole
    of its stability, so it is worth asserting rather than remembering. A run
    that measured the exponent as negative would make the controller a positive
    feedback that drives `du` to a bound.
    """
    device, _ = gpu_device
    params = config.Config().resolve().reaction

    measured = {}
    for du in (0.14, 0.38):
        u, v = R.seed_field(96, blobs=8, seed=5)
        for _ in range(1200):
            u, v = R.gray_scott_step(
                u, v, params.feed, params.kill,
                du=du, dv=params.dv * du / params.du, dt=params.dt,
            )
        measured[du] = float(_measure_ell_on_gpu(device, v)["ell"])

    exponent = math.log(measured[0.38] / measured[0.14]) / math.log(0.38 / 0.14)
    assert 0.3 < exponent < 0.8, (
        f"ell goes as du^{exponent:.2f} (ell {measured[0.14]:.2f} -> "
        f"{measured[0.38]:.2f}); the feature-size loop's gain is sized for "
        f"about 0.5, and a different exponent changes how fast it settles"
    )
