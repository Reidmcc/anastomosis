"""The volumetric slab backend -- DESIGN.md §5.1.

Four things are worth testing here, and they are not the same four the layered
backend needs.

**That the swap is clean.** §5.1 says the output stages are unchanged between
backends. That is a claim about code, and the way to check it is to assert that
both engines reach the screen through the same objects and that the safety
guarantee holds under the new one exactly as it does under the old.

**That the flow is divergence-free.** In two dimensions this is free -- the
velocity is the curl of a scalar and there is nothing to get wrong. In three it
is the whole reason the flow is built as a stored potential and then
differentiated, rather than assembled from velocities, and if that reasoning is
wrong the symptom is pigment slowly accumulating in some places and draining
from others over hours. That is precisely the kind of failure a long soak would
find and a short test would not, so it is checked directly and numerically.

**That the slab really is a slab.** Toroidal on all three axes, with no
boundary case anywhere, and with a depth axis that carries genuine structure
rather than being a stretched copy of one plane.

**That a resumed field is the same field.** As for the layered backend: the
property is not "a file was written" but that the restored engine evolves
identically to the one it was captured from.

The slab used here is tiny -- the software adapter in CI renders correct pixels
far too slowly for anything else -- but every pass, every wrap and every
reduction is the one the full-size slab runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from anastomosis import (
    backend as backend_module,
    checkpoint,
    config,
    engine as engine_module,
    events,
    shaders,
    volume as volume_module,
)

OUT_SIZE = (96, 64)


def _params(**overrides) -> config.Params:
    """A slab small enough for a software adapter, in every other way default."""
    params = config.Config().resolve()
    vol = params.volume
    vol.width = 64
    vol.depth = 16
    vol.climate_width, vol.climate_height, vol.climate_depth = 8, 6, 4
    vol.psi_scale = 4
    vol.steps = 8
    for key, value in overrides.items():
        setattr(vol, key, value)
    return params


def _engine(gpu_device, params, seed=1234, size=OUT_SIZE):
    device, _ = gpu_device
    return volume_module.VolumeEngine(device, size[0], size[1], params, seed=seed)


def _read_volume(device, texture) -> np.ndarray:
    """One 3D texture as ``(d, h, w, 4)`` float32."""
    return checkpoint._read_texture3(device, texture).astype(np.float32)


def _run(engine, params, ticks: int, scheduler=None) -> None:
    for _ in range(ticks):
        rows: list[dict] = []
        if scheduler is not None:
            scheduler.update(1.0 / params.sim_hz, params.events)
            rows, _ = scheduler.pack(8)
        engine.tick(params, rows)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_the_default_slab_is_the_one_the_design_specifies():
    """512 x 288 x 48, on a 16:9 display. DESIGN.md §5.1 names that shape."""
    params = config.Config().resolve()
    geometry = volume_module.VolumeGeometry.derive(2560, 1440, params)
    assert geometry.dims == (512, 288, 48)
    assert 7.0e6 < geometry.voxels < 7.2e6


def test_voxels_stay_cubic_when_the_window_is_not_16_by_9():
    """The height follows the window's aspect, which is what keeps a voxel a
    cube -- the renderer treats the slab as a box of extent (1, h/w, d/w) and
    the simulation treats every axis alike, and neither is true otherwise."""
    params = config.Config().resolve()
    for width, height in ((2560, 1440), (1920, 1200), (1280, 1024)):
        geometry = volume_module.VolumeGeometry.derive(width, height, params)
        implied = geometry.height / geometry.width
        assert abs(implied - height / width) < 0.05, (width, height)


def test_implausible_geometry_is_refused_rather_than_allocated():
    """Geometry off disk decides how much memory a launch tries to take, so it
    is bounds-checked before anything is built from it."""
    good = volume_module.VolumeGeometry.derive(1280, 720, config.Config().resolve())
    assert good.problems() == []

    import dataclasses

    huge = dataclasses.replace(good, depth=4096)
    assert huge.problems(), "past maxTextureDimension3D and still accepted"
    swarm = dataclasses.replace(good, agent_count=good.voxels * 100)
    assert swarm.problems()
    empty = dataclasses.replace(good, width=0)
    assert empty.problems()


# ---------------------------------------------------------------------------
# The swap is clean
# ---------------------------------------------------------------------------


def test_both_backends_share_one_output_chain():
    """§5.1: "the output stages (§6, §7) are unchanged between backends, so
    this is a clean swap rather than a fork". The flash-safety stage is a
    guarantee enforced by construction (§7), and a second copy of it free to
    drift would be the most expensive duplication in the application."""
    shared = backend_module.Backend
    assert issubclass(engine_module.Engine, shared)
    assert issubclass(volume_module.VolumeEngine, shared)
    # The safety stage, the exposure governor, the dither and the parameter
    # mapping are defined once, on the base, and not overridden by either.
    for name in (
        "_output_stage", "_present", "render", "_physics_values",
        "_common_render_values", "_advance_du_walk",
    ):
        assert name not in vars(engine_module.Engine), name
        assert name not in vars(volume_module.VolumeEngine), name
        assert name in vars(shared), name


def test_every_volumetric_shader_compiles(gpu_device):
    """The whole point of building headless against a software adapter."""
    device, _ = gpu_device
    names = [n for n in shaders.all_shader_names()
             if n.startswith("vol_") or n == "raymarch.wgsl"]
    assert len(names) >= 13, names
    for name in names:
        device.create_shader_module(code=shaders.load(name), label=name)


def test_a_tick_and_a_frame_leave_a_live_field(gpu_device, offscreen_target):
    device, _ = gpu_device
    params = _params()
    engine = _engine(gpu_device, params)
    view, fmt = offscreen_target(*OUT_SIZE)

    scheduler = events.EventScheduler(seed=7)
    for _ in range(24):
        _run(engine, params, 1, scheduler)
        engine.render(params, frac=0.5, target_view=view, target_format=fmt)

    stats = engine.read_stats()
    assert stats["count"] == engine.geometry.voxels, (
        "the reduction did not cover the slab")
    assert stats["mean_v"] > 0.005, "the reaction died"
    assert stats["mean_activity"] > 0.0, "the field stopped moving"

    image = engine.read_final_rgba()
    assert image.shape == (OUT_SIZE[1], OUT_SIZE[0], 4)
    assert np.isfinite(image).all()
    assert image[..., :3].max() > 0.0, "nothing reached the screen"


# ---------------------------------------------------------------------------
# The flow
# ---------------------------------------------------------------------------


def test_the_velocity_field_is_divergence_free(gpu_device):
    """The reason the flow is a stored potential differentiated in a second
    pass rather than a sum of velocities.

    ``div(curl(A))`` cancels term for term under central differences because
    the difference operators commute, so the residual here should be the
    float16 quantisation of the velocity texture and nothing else. A
    divergence the simulation actually had would show up as pigment piling into
    some regions and draining from others over hours -- a failure a soak finds
    and a smoke test does not, which is why it is measured directly.
    """
    params = _params()
    engine = _engine(gpu_device, params)
    _run(engine, params, 20)

    velocity = _read_volume(engine.device, engine.slab.velocity)
    vx, vy, vz = velocity[..., 0], velocity[..., 1], velocity[..., 2]
    # The same stencil the shader uses, on the same torus.
    divergence = (
        0.5 * (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2))
        + 0.5 * (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1))
        + 0.5 * (np.roll(vz, -1, axis=0) - np.roll(vz, 1, axis=0))
    )
    magnitude = np.abs(velocity[..., :3]).mean()
    assert magnitude > 1e-4, "no flow to measure"
    residual = float(np.sqrt((divergence**2).mean()) / magnitude)
    # float16 carries about 3 decimal digits; anything at that level is
    # storage, and anything well above it is a real divergence.
    assert residual < 0.02, f"divergence is {residual:.4f} of the flow"


def test_the_depth_anisotropy_damps_motion_through_the_slab(gpu_device):
    """The slab is a few feature-widths deep, so isotropic flow would carry
    material through the whole thickness in seconds and depth would read as
    churn. `depth_flow` weights the potential's lateral components, which
    scales the vertical velocity while leaving the field a curl -- and so still
    exactly divergence-free."""
    lateral, vertical = [], []
    for depth_flow in (1.0, 0.2):
        params = _params(depth_flow=depth_flow)
        engine = _engine(gpu_device, params, seed=11)
        _run(engine, params, 12)
        velocity = _read_volume(engine.device, engine.slab.velocity)
        lateral.append(float(np.abs(velocity[..., :2]).mean()))
        vertical.append(float(np.abs(velocity[..., 2]).mean()))

    free = vertical[0] / max(lateral[0], 1e-9)
    damped = vertical[1] / max(lateral[1], 1e-9)
    assert damped < free * 0.6, (
        f"vertical/lateral only went from {free:.3f} to {damped:.3f}")


# ---------------------------------------------------------------------------
# The slab really is a slab
# ---------------------------------------------------------------------------


def test_the_domain_wraps_on_all_three_axes(gpu_device):
    """No pass anywhere has a boundary case, because there is no boundary.

    If the depth axis did not wrap, material would accumulate against the two
    faces the way a reflecting boundary always does, and the two extreme slices
    would look nothing like their neighbours.
    """
    params = _params()
    engine = _engine(gpu_device, params)
    _run(engine, params, 40)

    trail = _read_volume(
        engine.device, engine.slab.trail.textures[engine.slab.trail.index])[..., 0]

    # Slice-mean trail across the depth axis: a wall would show as the first
    # and last slices standing well outside the spread of the rest.
    per_slice = trail.mean(axis=(1, 2))
    interior = per_slice[1:-1]
    spread = max(float(interior.std()), 1e-9)
    for label, value in (("near face", per_slice[0]), ("far face", per_slice[-1])):
        deviation = abs(float(value) - float(interior.mean())) / spread
        assert deviation < 6.0, f"{label} is {deviation:.1f} sigma off the interior"


def test_the_field_has_real_structure_through_depth(gpu_device):
    """A volume whose slices were all the same field would be a thick sheet.

    The weather's noise is given a lattice period of at least two on the depth
    axis exactly so the front and back of the slab sit on opposite phases; this
    checks that the resulting field genuinely decorrelates through depth rather
    than being one plane extruded.
    """
    params = _params()
    engine = _engine(gpu_device, params)
    _run(engine, params, 30)

    pigment = _read_volume(
        engine.device,
        engine.slab.pigment.textures[engine.slab.pigment.index])[..., 0]
    depth = pigment.shape[0]
    near = pigment[0].ravel()
    far = pigment[depth // 2].ravel()
    if near.std() < 1e-6 or far.std() < 1e-6:
        pytest.skip("the field has not grown enough to correlate")
    correlation = float(np.corrcoef(near, far)[0, 1])
    assert correlation < 0.9, (
        f"opposite sides of the slab correlate at {correlation:.2f}; "
        "the depth axis is carrying no structure of its own")


# ---------------------------------------------------------------------------
# Safety, under the new backend
# ---------------------------------------------------------------------------


def test_the_flash_safety_bound_holds_under_the_slab(gpu_device, offscreen_target):
    """DESIGN.md §7 is a guarantee about the output, not about a backend.

    Driven with the safety-relevant parameters at their ceilings, exactly as
    test_flash_safety does for the layered stack: whatever the simulation does,
    per-pixel Oklab lightness may not move faster than `max_luma_delta` per
    frame.
    """
    params = _params()
    params.render.filament_luma = 0.9
    params.render.l_max = 0.9
    params.safety.max_luma_delta = 0.012
    params.safety.iir_alpha = 1.0

    engine = _engine(gpu_device, params, seed=5)
    view, fmt = offscreen_target(*OUT_SIZE)

    def lightness(rgb: np.ndarray) -> np.ndarray:
        lms = np.stack([
            0.4122214708 * rgb[..., 0] + 0.5363325363 * rgb[..., 1]
            + 0.0514459929 * rgb[..., 2],
            0.2119034982 * rgb[..., 0] + 0.6806995451 * rgb[..., 1]
            + 0.1073969566 * rgb[..., 2],
            0.0883024619 * rgb[..., 0] + 0.2817188376 * rgb[..., 1]
            + 0.6299787005 * rgb[..., 2],
        ], axis=-1)
        root = np.cbrt(np.maximum(lms, 0.0))
        return (0.2104542553 * root[..., 0] + 0.7936177850 * root[..., 1]
                - 0.0040720468 * root[..., 2])

    previous = None
    worst = 0.0
    for _ in range(20):
        _run(engine, params, 1)
        engine.render(params, frac=0.5, target_view=view, target_format=fmt)
        current = lightness(engine.read_final_rgba()[..., :3])
        if previous is not None:
            worst = max(worst, float(np.abs(current - previous).max()))
        previous = current

    # Headroom for the half-float round trip through the output texture, which
    # is what the layered backend's own flash test allows too.
    assert worst <= params.safety.max_luma_delta + 0.004, worst


# ---------------------------------------------------------------------------
# Resuming
# ---------------------------------------------------------------------------


def test_a_resumed_slab_is_the_same_slab(gpu_device):
    """The property that matters: a restored engine evolves identically."""
    params = _params()
    engine = _engine(gpu_device, params, seed=99)
    scheduler = events.EventScheduler(seed=3)
    _run(engine, params, 12, scheduler)

    snapshot = checkpoint.capture(engine, scheduler=scheduler, sim_hz=params.sim_hz)
    assert snapshot.meta["backend"] == "volumetric"

    resumed = _engine(gpu_device, _params(), seed=1)
    assert checkpoint.restore(
        resumed, snapshot, scheduler=events.EventScheduler(seed=77))
    assert resumed.tick_count == engine.tick_count
    assert resumed.seed == engine.seed

    _run(engine, params, 6)
    _run(resumed, params, 6)

    def state(target) -> dict:
        """Every mutable field, whether or not the snapshot saves it.

        Deliberately a superset: the fields left out of the checkpoint are left
        out because the engine rewrites them before anything reads them, and
        comparing them across a resume is what turns that claim into something
        the suite checks rather than something the module asserts about itself.
        """
        arrays = dict(checkpoint.capture(target).arrays)
        for name in checkpoint.VOLUME_DERIVED_FIELDS:
            arrays[f"slab.{name}"] = _read_volume(
                target.device, getattr(target.slab, name))
        return arrays

    left = state(engine)
    right = state(resumed)
    assert set(left) == set(right)
    for name, values in left.items():
        assert np.array_equal(values, right[name]), f"{name} diverged"


def test_a_checkpoint_is_refused_by_the_other_backend(gpu_device, tmp_path):
    """The two hold incompatible state, and no amount of resampling turns one
    into the other. A file from the wrong backend is an ordinary "cannot use
    this, start fresh", not a crash and not a mangled upload."""
    params = _params()
    engine = _engine(gpu_device, params, seed=4)
    _run(engine, params, 3)
    snapshot = checkpoint.capture(engine, sim_hz=params.sim_hz)

    assert checkpoint.required_geometry(snapshot, backend="volumetric") is not None
    assert checkpoint.required_geometry(snapshot, backend="layered") is None

    device, _ = gpu_device
    layered = engine_module.Engine(device, *OUT_SIZE, config.Config().resolve())
    assert checkpoint.compatibility_problems(layered, snapshot)
    assert not checkpoint.restore(layered, snapshot)


def test_each_backend_keeps_its_own_saved_field(monkeypatch, tmp_path):
    """Switching backend must not destroy the field switched away from."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    layered = checkpoint.default_checkpoint_path("layered")
    volumetric = checkpoint.default_checkpoint_path("volumetric")
    assert layered != volumetric
    # The layered backend keeps the name it has always had, so an existing
    # mature field is still found after this change.
    assert layered.name == "checkpoint.npz"
    assert checkpoint.default_checkpoint_path() == layered


# ---------------------------------------------------------------------------
# Selecting the backend
# ---------------------------------------------------------------------------


def test_the_config_chooses_the_backend(tmp_path):
    from anastomosis import config as config_module

    path = tmp_path / "config.toml"
    cfg = config_module.Config(backend="volumetric")
    config_module.save(cfg, path)
    assert config_module.load(path).backend == "volumetric"

    # Untrusted like every other value off disk.
    path.write_text(path.read_text().replace('"volumetric"', '"holographic"'))
    assert config_module.load(path).backend == "layered"


def test_the_cli_flag_reaches_the_application():
    from anastomosis.__main__ import build_parser

    args = build_parser().parse_args(["--backend", "volumetric"])
    assert args.backend == "volumetric"
    assert build_parser().parse_args([]).backend is None


def test_the_application_builds_the_backend_it_was_asked_for(
    gpu_device, monkeypatch, tmp_path
):
    from rendercanvas.offscreen import RenderCanvas, loop as offscreen_loop
    from anastomosis import app as app_module, device as device_module

    device, info = gpu_device
    monkeypatch.setattr(
        device_module, "request_device", lambda *a, **k: (device, info))
    monkeypatch.setattr(app_module.Application, "_start_hot_reload", lambda self: None)
    monkeypatch.setattr(
        app_module.Application, "_make_canvas",
        lambda self: (
            RenderCanvas(size=OUT_SIZE, update_mode="manual"), offscreen_loop, False),
    )

    config_path = tmp_path / "config.toml"
    from anastomosis import config as config_module

    config_module.save(
        config_module.Config(
            backend="volumetric",
            overrides={
                "volume.width": 64, "volume.depth": 16, "volume.steps": 8,
                "volume.psi_scale": 4, "volume.climate_width": 8,
                "volume.climate_height": 6, "volume.climate_depth": 4,
            },
        ),
        config_path,
    )

    app = app_module.Application(app_module.AppOptions(
        width=OUT_SIZE[0], height=OUT_SIZE[1], ui=False,
        config_path=config_path,
        checkpoint_path=tmp_path / "checkpoint.npz",
        checkpoint_seconds=0.0,
    ))
    app.setup()
    assert app.backend == "volumetric"
    assert isinstance(app.engine, volume_module.VolumeEngine)
    app.draw_frame()
    assert app.engine.tick_count > 0
    app.shutdown()


def test_switching_backend_keeps_both_fields(gpu_device, monkeypatch, tmp_path):
    """A switch saves what it is leaving and resumes what it is returning to."""
    from rendercanvas.offscreen import RenderCanvas, loop as offscreen_loop
    from anastomosis import app as app_module, device as device_module

    device, info = gpu_device
    monkeypatch.setattr(
        device_module, "request_device", lambda *a, **k: (device, info))
    monkeypatch.setattr(app_module.Application, "_start_hot_reload", lambda self: None)
    monkeypatch.setattr(
        app_module.Application, "_make_canvas",
        lambda self: (
            RenderCanvas(size=OUT_SIZE, update_mode="manual"), offscreen_loop, False),
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))

    from anastomosis import config as config_module

    config_path = tmp_path / "config.toml"
    config_module.save(
        config_module.Config(overrides={
            "render.layers": 1,
            "volume.width": 64, "volume.depth": 16, "volume.steps": 8,
            "volume.psi_scale": 4, "volume.climate_width": 8,
            "volume.climate_height": 6, "volume.climate_depth": 4,
        }),
        config_path,
    )
    app = app_module.Application(app_module.AppOptions(
        width=OUT_SIZE[0], height=OUT_SIZE[1], ui=False,
        config_path=config_path, checkpoint_seconds=0.0,
    ))
    app.setup()
    assert app.backend == "layered"
    for _ in range(4):
        app.draw_frame()
    layered_ticks = app.engine.tick_count
    assert layered_ticks > 0

    assert app.switch_backend("volumetric")
    assert isinstance(app.engine, volume_module.VolumeEngine)
    assert app.engine.tick_count == 0, "a fresh slab, not the layers' tick count"
    for _ in range(3):
        app.draw_frame()

    assert app.switch_backend("layered")
    assert isinstance(app.engine, engine_module.Engine)
    assert app.engine.tick_count == layered_ticks, (
        "the layered field was not waiting where it was left")
    assert not app.switch_backend("layered"), "switching to the current one is a no-op"
    app.shutdown()


# ---------------------------------------------------------------------------
# Long duration
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_slab_is_still_alive_after_a_long_run(gpu_device):
    """The soak, in the small: does the volume survive being left alone?

    This application is meant to run for days, and the two ways a generative
    field ends are both slow -- it either falls into the absorbing state and
    goes dark, or it runs away and clips. The homeostat exists to prevent both
    (DESIGN.md §4.2), and it is bound to the slab through the same reduce and
    controller passes the layered backend uses, so what is really being checked
    here is that the 3D reduction feeds it the numbers it expects.

    Deliberately not a check that the field looks a particular way. The mass
    setpoint is the controller's business; what a test can say is that the
    field is neither dead nor diverging, and that nothing has gone non-finite.
    """
    params = _params()
    engine = _engine(gpu_device, params, seed=17)
    scheduler = events.EventScheduler(seed=5)

    history = []
    for block in range(15):
        _run(engine, params, 100, scheduler)
        history.append(engine.read_stats())

    for index, stats in enumerate(history):
        for key in ("mean_v", "var_v", "mean_activity"):
            value = stats[key]
            assert np.isfinite(value), f"{key} went non-finite at block {index}"
        assert 0.0 < stats["mean_v"] < 0.6, (
            f"mean V left the usable band at block {index}: {stats['mean_v']:.4f}")

    late = history[-5:]
    assert min(s["mean_v"] for s in late) > 0.01, (
        "the reaction is on its way to the absorbing state")
    assert min(s["mean_activity"] for s in late) > 0.0, (
        "the field has stopped changing")

    # Nothing anywhere may be non-finite: one bad voxel propagates through
    # diffusion and destroys the whole field within seconds (DESIGN.md §4.4).
    slab = engine.slab
    for name in ("trail", "reaction", "pigment"):
        pair = getattr(slab, name)
        data = _read_volume(engine.device, pair.textures[pair.index])
        assert np.isfinite(data).all(), f"{name} holds a non-finite voxel"


def test_a_resize_rebuilds_only_the_presentation(gpu_device, offscreen_target):
    """A resize is a change to the window, not to the world (DESIGN.md §8).

    The slab keeps the resolution it was grown at, along with every field, the
    agents and the tick count; what follows the window is the HDR target, the
    final ping-pong, the exposure partials -- and, unique to this backend, the
    screen-space velocity the march accumulates for the safety stage. That last
    one is the reason this test exists: a window-sized resource the base class
    does not know about is exactly the kind of thing a resize leaves stale.
    """
    params = _params()
    engine = _engine(gpu_device, params)
    view, fmt = offscreen_target(*OUT_SIZE)
    for _ in range(6):
        _run(engine, params, 1)
        engine.render(params, frac=0.5, target_view=view, target_format=fmt)

    before = engine.geometry
    ticks = engine.tick_count

    engine.resize(160, 128)
    assert engine.geometry is before, "the slab was re-resolved"
    assert engine.tick_count == ticks
    assert (engine.motion.width, engine.motion.height) == (160, 128)

    wide_view, wide_fmt = offscreen_target(160, 128)
    for _ in range(3):
        _run(engine, params, 1)
        engine.render(params, frac=0.5, target_view=wide_view, target_format=wide_fmt)

    image = engine.read_final_rgba()
    assert image.shape == (128, 160, 4)
    assert np.isfinite(image).all()


@pytest.mark.slow
def test_the_exposure_governor_reaches_its_target_through_the_march(
    gpu_device, offscreen_target
):
    """The one perceptual claim about this backend a test can actually make.

    How the slab *looks* needs a real GPU and a pair of eyes (DESIGN.md §13).
    What can be checked is that the ray march hands the output stage something
    the governor can work with at all: a volume too sparse to register would
    show up as the exposure saturating near its ceiling with the image still
    dark, and one too dense would pin it at the floor. Either would mean the
    knobs the two backends share do not mean the same thing under both.

    The governor's time constant is measured in seconds by design, so this
    needs several hundred frames to be a statement about the settled level
    rather than about the fade-in.
    """
    params = _params(width=96, depth=24, climate_width=12, climate_height=8,
                     climate_depth=4, steps=24)
    engine = _engine(gpu_device, params, seed=8, size=(160, 96))
    view, fmt = offscreen_target(160, 96)

    for _ in range(700):
        engine.tick(params)
        engine.render(params, frac=0.5, target_view=view, target_format=fmt)

    stats = engine.read_stats()
    settled = stats["img_sum_l"] / max(stats["img_count"], 1.0)
    target = params.safety.exposure_target
    assert abs(settled - target) < 0.03 * max(target, 1e-6) + 0.02, (
        f"mean image lightness settled at {settled:.4f} against a target of "
        f"{target:.4f}")
    assert 0.05 < stats["exposure"] < 12.0, (
        f"the governor is at {stats['exposure']:.2f}, near one of its bounds; "
        "the march is handing it an image it can only just correct")
