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

import re

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


def _owe_ticks(app) -> None:
    """Back-date the frame clock so the next draw_frame owes ticks.

    The shell consumes ticks from real elapsed time (bounded per frame). On
    the software adapter every draw costs more than a tick interval, so the
    debt accrues on its own; on a fast adapter back-to-back draws elapse
    microseconds, and an assertion on ``tick_count`` would be measuring
    adapter speed rather than behaviour. This makes the elapsed time explicit
    instead -- the clamp in ``draw_frame`` turns any large debt into a full
    catch-up burst.
    """
    app._last_time -= 1.0


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


def test_each_named_slab_size_builds_a_slab_of_that_shape():
    """The three sizes of `config.VOLUME_DETAIL`, end to end.

    The name reaches `volume.width` through `Config.resolve` and the width
    reaches the slab through `VolumeGeometry.derive`, and this is the only
    place both halves are checked together -- that choosing "finest" really
    does grow a 1024-voxel slab, and that nothing in the rounding quietly
    lands somewhere else.
    """
    expected = {"standard": (512, 288, 48),
                "fine": (768, 432, 48),
                "finest": (1024, 576, 48)}
    for name, dims in expected.items():
        params = config.Config(volume_detail=name).resolve()
        geometry = volume_module.VolumeGeometry.derive(2560, 1440, params)
        assert geometry.dims == dims, name
        assert geometry.problems() == [], (
            f"{name} is offered but could not be built: {geometry.problems()}"
        )


def test_a_wider_slab_is_sharper_rather_than_merely_bigger():
    """Why the width is the knob worth having at all.

    Raising it must leave the thickness and the aspect alone, so the voxels
    stay cubic and a feature -- which is a fixed number of voxels across, since
    the reaction's scale is in voxels -- covers proportionally fewer display
    pixels. A size that also grew the thickness would cost more and blur the
    same, and would move the march's step count with it.

    Checked at a raised thickness as well as the default, because the two are
    separate knobs and the width must not quietly reach across to the other.
    """
    for thickness in (48, 96):
        previous = None
        for name in ("standard", "fine", "finest"):
            cfg = config.Config(
                volume_detail=name, overrides={"volume.depth": thickness})
            params = cfg.resolve()
            geometry = volume_module.VolumeGeometry.derive(2560, 1440, params)
            assert geometry.depth == thickness, (
                f"{name} moved the thickness to {geometry.depth}, so the "
                f"march's step count and the cost of drawing a frame moved "
                f"with it"
            )
            assert abs(geometry.height / geometry.width - 1440 / 2560) < 0.02
            if previous is not None:
                assert geometry.width > previous.width
                # Cost goes with the voxel count, which goes with the square of
                # the ratio. Worth asserting: a tier that grew linearly would
                # be a size nobody would be able to see the point of.
                ratio = geometry.width / previous.width
                assert abs(geometry.voxels / previous.voxels - ratio ** 2) < 0.05
            previous = geometry


def test_a_wider_slab_raises_the_thickness_ceiling():
    """The one place the two slab knobs meet, and it is in the good direction.

    The thickness is capped at the shorter lateral axis, so choosing a wider
    slab is also what makes a deeper one available. The panel rebuilds the
    thickness travel on a detail change because of this.
    """
    ceilings = {}
    for name in ("standard", "fine", "finest"):
        params = config.Config(volume_detail=name).resolve()
        _, high = volume_module.depth_limits(2560, 1440, params)
        ceilings[name] = high
    assert ceilings["standard"] < ceilings["fine"] < ceilings["finest"]
    # The shorter lateral axis is the height at 16:9, so the ceiling is it.
    assert ceilings["standard"] == 288 and ceilings["finest"] == 576


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
        "_common_render_values", "_advance_ell_walk",
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
# How thick the slab is
# ---------------------------------------------------------------------------


def test_the_thickness_is_capped_at_the_shorter_lateral_axis():
    """The knob's ceiling is structural, not a fact about the panel.

    Past the shorter lateral axis the shape is not a slab, and the cap lives in
    `derive` so that a hand-edited config is held to the same range the control
    panel offers rather than being allowed to ask for something the panel
    cannot even express.
    """
    params = config.Config().resolve()
    params.volume.depth = 4096
    geometry = volume_module.VolumeGeometry.derive(2560, 1440, params)
    assert geometry.depth == geometry.max_depth == min(
        geometry.width, geometry.height)

    low, high = volume_module.depth_limits(2560, 1440, params)
    assert (low, high) == (volume_module.MIN_DEPTH, geometry.max_depth)


def test_the_thickness_is_the_whole_of_what_the_slab_costs():
    """Memory is linear in the thickness, which is what makes it the knob that
    decides what this backend spends -- and what the panel quotes before the
    user commits to a setting."""
    params = config.Config().resolve()
    thin = volume_module.VolumeGeometry.derive(2560, 1440, params)
    params.volume.depth = 4 * thin.depth
    thick = volume_module.VolumeGeometry.derive(2560, 1440, params)

    assert thick.depth == 4 * thin.depth
    ratio = thick.field_bytes / thin.field_bytes
    assert 3.9 < ratio < 4.1, f"four times the slab cost {ratio:.2f} times as much"
    # The documented figure for the default slab, which is the number the
    # design's memory budget (§5.1, §8.1) is written against.
    assert 0.6e9 < thin.field_bytes < 0.75e9


def _render_values(monkeypatch, gpu_device, params):
    """What `_write_render_params` packs, for a slab of this configuration."""
    captured: dict = {}
    real = volume_module.gpu_params.pack

    def spy(dtype, values):
        if dtype is volume_module.gpu_params.RENDER_DTYPE:
            captured.update(values)
        return real(dtype, values)

    # Scoped, so that a second call wraps the real packer rather than the spy
    # the first one left in place.
    with monkeypatch.context() as patched:
        patched.setattr(volume_module.gpu_params, "pack", spy)
        engine = _engine(gpu_device, params)
        engine._write_render_params(params, frac=0.0, frame_dt=1 / 30.0)
    assert captured, "the render block was not packed"
    return captured, engine.geometry


def test_the_march_stays_calibrated_as_the_slab_thickens(
    gpu_device, monkeypatch
):
    """Moving the thickness must change how much material a ray crosses, and
    nothing else.

    Two of the march's lengths used to be fractions of the slab's depth, which
    was harmless while the depth was fixed and is not once it is a knob: held as
    fractions, a slab six times deeper would fade six times as much of its crisp
    near face and send its shadow rays six times as far on the same six steps.
    Both are now lengths in voxels -- calibrated against a filament, which is
    what they are actually about -- so they come out the same absolute size at
    every thickness.
    """
    vol = config.VolumeParams()
    seen = []
    for depth in (16, 32):
        params = _params(depth=depth, steps=64)
        values, geometry = _render_values(monkeypatch, gpu_device, params)
        assert geometry.depth == depth
        seen.append(values)

        # The window, in voxels of the slab it is fading.
        assert values["depth_window"] * depth == pytest.approx(
            vol.depth_window_voxels, rel=1e-6)
        # The shadow reach, in voxels: a world length over the lateral extent,
        # which spans `width` cubic voxels.
        assert values["shadow_reach"] * geometry.width == pytest.approx(
            vol.shadow_voxels, rel=1e-6)
        # One step per slice, since the ceiling is well above both.
        assert values["march_steps"] == depth

    assert seen[0]["shadow_reach"] == pytest.approx(seen[1]["shadow_reach"])
    assert seen[1]["slab_depth"] == pytest.approx(2.0 * seen[0]["slab_depth"]), (
        "a slab twice as deep must be twice as thick in world units, which is "
        "what puts more material in front of the far face")


def test_the_march_step_count_has_a_ceiling(gpu_device, monkeypatch):
    """One step per slice up to `volume.steps`, and no further: the count is
    what the march costs, so a thick slab must not be able to spend without
    limit."""
    params = _params(depth=32, steps=8)
    values, _geometry = _render_values(monkeypatch, gpu_device, params)
    assert values["march_steps"] == 8


def test_the_slab_holds_the_viewpoint_to_the_swing_its_thickness_earns(
    gpu_device, monkeypatch
):
    """Parallax is thickness times the tangent of the viewing angle.

    Which is the geometry underneath every complaint that this backend looks
    flat: 48 voxels against 512 is a sheet of paper, and swinging the viewpoint
    further does not find more depth in a sheet of paper -- it finds the same
    sheet seen edge-on, with the march tracing a long oblique smear through it.
    So the drift's amplitude is held to what the slab justifies, which is also
    what makes the two knobs compound: a deeper slab does not merely have more
    material in it, it has room for the viewpoint to move.
    """
    reaches = {}
    for depth in (16, 48):
        params = _params(depth=depth)
        params.render.parallax = 1.0  # far past anything a slab could earn
        values, geometry = _render_values(monkeypatch, gpu_device, params)
        thickness = geometry.depth / geometry.width
        limit = volume_module.PARALLAX_MAX_TANGENT * thickness
        # The offset riding on it is bounded by one, so the shear the shader
        # gets can never exceed the amplitude it was given.
        for axis in ("cam_shear_x", "cam_shear_y"):
            assert abs(values[axis]) <= limit + 1e-6, (
                f"a {geometry.depth}-deep slab swung to {values[axis]:.3f}, "
                f"past the {limit:.3f} its thickness earns")
        reaches[geometry.depth] = limit

    assert reaches[48] > reaches[16], (
        "a thicker slab did not earn a wider swing, so the thickness and "
        "parallax knobs do not compound")

    # And a request inside what the slab earns is passed through untouched.
    params = _params(depth=48)
    params.render.parallax = 0.01
    values, geometry = _render_values(monkeypatch, gpu_device, params)
    limit = volume_module.PARALLAX_MAX_TANGENT * geometry.depth / geometry.width
    assert 0.01 < limit, "the fixture needs a request the slab can honour"
    assert abs(values["cam_shear_x"]) <= 0.01 + 1e-6


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


def test_the_cli_flag_reaches_the_application_for_the_slab_size():
    from anastomosis.__main__ import build_parser

    args = build_parser().parse_args(["--volume-detail", "finest"])
    assert args.volume_detail == "finest"
    # Absent means "whatever the config says", not "standard" -- otherwise the
    # flag would silently override a size chosen in the file.
    assert build_parser().parse_args([]).volume_detail is None

    # The parser spells its choices out rather than importing the config, so
    # that `--help` costs nothing at startup -- which means they can drift.
    action = next(
        a for a in build_parser()._actions if a.dest == "volume_detail")
    assert tuple(action.choices) == tuple(config.VOLUME_DETAIL), (
        "the command line offers different sizes from the config"
    )


def test_an_unoffered_slab_size_is_refused_at_the_command_line():
    """argparse rejects it outright here, rather than warning and carrying on.

    The opposite of the config file's treatment, and deliberately: a typo in a
    file edited during a running session must not end the session, but a typo
    in a command that has not started one yet is best reported immediately.
    """
    from anastomosis.__main__ import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--volume-detail", "enormous"])


def test_a_command_line_slab_size_survives_a_config_reload():
    """`--volume-detail` is a property of the launch, not of the file.

    Hot reload replaces the whole config from disk, so without re-pinning, a
    size passed on the command line would come undone the first time the user
    saved an unrelated slider change -- hours into a session, silently, and
    only visible at the next reset.
    """
    from anastomosis import app as app_module, config as config_module

    app = app_module.Application.__new__(app_module.Application)

    # Pinned on the command line: the file loses, every time.
    app.options = app_module.AppOptions(volume_detail="finest")
    app.config = config_module.Config(volume_detail="standard")
    app._pin_volume_detail()
    assert app.volume_detail == "finest"
    assert app.config.volume_detail == "finest"

    app.config = config_module.Config(volume_detail="fine")  # a reload
    app._pin_volume_detail()
    assert app.volume_detail == "finest"

    # Not pinned: the file decides, and a reload picks up an edit to it.
    app.options = app_module.AppOptions(volume_detail=None)
    app.config = config_module.Config(volume_detail="fine")
    app._pin_volume_detail()
    assert app.volume_detail == "fine"

    # And a size off disk is normalised here too, not merely copied.
    app.config = config_module.Config(volume_detail="enormous")
    app._pin_volume_detail()
    assert app.volume_detail == config_module.DEFAULT_VOLUME_DETAIL


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
    _owe_ticks(app)
    app.draw_frame()
    assert app.engine.tick_count > 0
    app.shutdown()


def test_changing_the_slab_size_grows_a_new_field_at_the_new_width(
    gpu_device, monkeypatch, tmp_path
):
    """The lateral counterpart of the thickness test, and the same contract.

    A slab of a different width is a differently shaped field, so the setting
    is written to the config, the saved state goes, and a new field is grown --
    and asking for the size already running does none of that.

    The three real sizes are far too large for a software adapter, so the table
    itself is patched down. Everything under it is the real path: the same
    `resolve`, the same geometry, the same engine, the same discard.
    """
    from rendercanvas.offscreen import RenderCanvas, loop as offscreen_loop
    from anastomosis import app as app_module, device as device_module
    from anastomosis import config as config_module

    device, info = gpu_device
    monkeypatch.setattr(
        device_module, "request_device", lambda *a, **k: (device, info))
    monkeypatch.setattr(app_module.Application, "_start_hot_reload", lambda self: None)
    monkeypatch.setattr(
        app_module.Application, "_make_canvas",
        lambda self: (
            RenderCanvas(size=OUT_SIZE, update_mode="manual"), offscreen_loop, False),
    )
    monkeypatch.setattr(
        config_module, "VOLUME_DETAIL",
        {"standard": 64, "fine": 96, "finest": 128})

    config_path = tmp_path / "config.toml"
    checkpoint_path = tmp_path / "checkpoint.npz"
    config_module.save(
        config_module.Config(
            backend="volumetric",
            volume_detail="standard",
            overrides={
                # Everything except the width, which is what is under test.
                "volume.depth": 16, "volume.steps": 8,
                "volume.psi_scale": 4, "volume.climate_width": 8,
                "volume.climate_height": 6, "volume.climate_depth": 4,
            },
        ),
        config_path,
    )
    app = app_module.Application(app_module.AppOptions(
        width=OUT_SIZE[0], height=OUT_SIZE[1], ui=False,
        config_path=config_path, checkpoint_path=checkpoint_path,
        checkpoint_seconds=0.0,
    ))
    app.setup()
    assert app.engine.geometry.width == 64

    _owe_ticks(app)
    for _ in range(3):
        app.draw_frame()
    assert app.engine.tick_count > 0
    assert app.save_checkpoint(blocking=True) and checkpoint_path.exists()

    # The size it is already running at is not a reset.
    assert app.set_volume_detail("standard") is False
    assert app.engine.tick_count > 0, "an unchanged size discarded the field"

    assert app.set_volume_detail("finest")
    assert app.engine.geometry.width == 128
    assert app.volume_detail == "finest"
    assert app.config.volume_detail == "finest"
    assert app.engine.tick_count == 0, "a fresh field, not the old one reshaped"
    assert not checkpoint_path.exists(), (
        "the saved field is of the old shape and would rebuild it on the next "
        "launch"
    )
    # And it draws. Not "and it ticks": `_adopt` restarts the accumulator the
    # ticks are consumed from, so whether three immediate frames cross a tick
    # interval is a question about wall-clock time rather than about the field.
    for _ in range(3):
        app.draw_frame()
    app.shutdown()


def test_a_slab_size_the_card_refuses_leaves_the_field_alone(
    gpu_device, monkeypatch, tmp_path
):
    """The widest size is four times the memory, so this is a real answer.

    The new engine is built before anything is discarded, so a card that will
    not allocate it leaves the running field, its checkpoint and the setting
    exactly as they were, and raises rather than ending the session with no
    image and nothing saved.
    """
    from rendercanvas.offscreen import RenderCanvas, loop as offscreen_loop
    from anastomosis import app as app_module, device as device_module
    from anastomosis import config as config_module

    device, info = gpu_device
    monkeypatch.setattr(
        device_module, "request_device", lambda *a, **k: (device, info))
    monkeypatch.setattr(app_module.Application, "_start_hot_reload", lambda self: None)
    monkeypatch.setattr(
        app_module.Application, "_make_canvas",
        lambda self: (
            RenderCanvas(size=OUT_SIZE, update_mode="manual"), offscreen_loop, False),
    )
    monkeypatch.setattr(
        config_module, "VOLUME_DETAIL",
        {"standard": 64, "fine": 96, "finest": 128})

    config_path = tmp_path / "config.toml"
    checkpoint_path = tmp_path / "checkpoint.npz"
    config_module.save(
        config_module.Config(
            backend="volumetric",
            volume_detail="standard",
            overrides={
                "volume.depth": 16, "volume.steps": 8,
                "volume.psi_scale": 4, "volume.climate_width": 8,
                "volume.climate_height": 6, "volume.climate_depth": 4,
            },
        ),
        config_path,
    )
    app = app_module.Application(app_module.AppOptions(
        width=OUT_SIZE[0], height=OUT_SIZE[1], ui=False,
        config_path=config_path, checkpoint_path=checkpoint_path,
        checkpoint_seconds=0.0,
    ))
    app.setup()
    _owe_ticks(app)
    for _ in range(3):
        app.draw_frame()
    ticks = app.engine.tick_count
    engine = app.engine
    assert app.save_checkpoint(blocking=True) and checkpoint_path.exists()

    def refuse(*args, **kwargs):
        raise MemoryError("the card would not allocate that")

    monkeypatch.setattr(app_module.Application, "_make_engine", refuse)
    with pytest.raises(MemoryError):
        app.set_volume_detail("finest")

    assert app.engine is engine, "the running field was thrown away anyway"
    assert app.engine.tick_count == ticks
    assert checkpoint_path.exists(), "the saved field was discarded on a failure"
    # The setting is back where it was, so the panel and the file agree with
    # what is actually on screen.
    assert app.volume_detail == "standard"
    assert app.config.volume_detail == "standard"
    assert app.params.volume.width == 64
    app.shutdown()


def test_a_size_chosen_at_runtime_outlives_the_command_line_flag():
    """Otherwise the panel's choice would be undone at the next hot reload.

    A launch pinned to one size, then changed in the panel: the pin has to
    give way, or `_pin_volume_detail` would restore the flag's size the first
    time the config file was touched, and the field would be regrown at a size
    the user had already moved away from.
    """
    from anastomosis import app as app_module, config as config_module

    app = app_module.Application.__new__(app_module.Application)
    app.options = app_module.AppOptions(volume_detail="standard")
    app.config = config_module.Config(
        backend="layered", volume_detail="standard")
    # The layered branch returns before anything is rebuilt, which is what
    # lets this stay a cheap stub: the bookkeeping is what is under test.
    app.backend = "layered"
    app.volume_detail = "standard"
    app.params = app.config.resolve()
    app.ramp = config_module.ParamRamp(app.params)

    assert app.set_volume_detail("fine")
    app._pin_volume_detail()  # as a hot reload would
    assert app.volume_detail == "fine"


def test_choosing_a_slab_size_under_the_layered_backend_keeps_its_field():
    """There is no slab to resize, so there is nothing to justify a reset.

    The panel greys the control out under the layered view, so this is not a
    path a user can walk -- but the method is public, a reset is irreversible,
    and "discarded hours of growth to apply a setting this backend does not
    read" is the kind of loss worth making structurally impossible.
    """
    from anastomosis import app as app_module, config as config_module

    app = app_module.Application.__new__(app_module.Application)
    app.options = app_module.AppOptions()
    app.config = config_module.Config(backend="layered", volume_detail="standard")
    app.backend = "layered"
    app.volume_detail = "standard"
    app.params = app.config.resolve()
    app.ramp = config_module.ParamRamp(app.params)

    resets = []
    app.reset_simulation = lambda: resets.append(1)

    assert app.set_volume_detail("finest")
    assert resets == [], "the layered field was discarded to resize a slab"
    # Recorded all the same, so it lands when a slab is next grown.
    assert app.volume_detail == "finest"
    assert app.config.volume_detail == "finest"
    assert app.ramp.target.volume.width == 1024


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
    _owe_ticks(app)
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


def test_the_thickness_knob_grows_a_new_slab(gpu_device, monkeypatch, tmp_path):
    """Changing the thickness is a reset, and says so.

    A slab of a different depth is a differently shaped field: nothing resamples
    one into the other, and a launch that found the old checkpoint would rebuild
    the old thickness from it. So the setting is written to the config, the
    saved state goes, and a new field is grown -- and asking for the thickness
    already running does none of that.
    """
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

    from anastomosis import config as config_module

    config_path = tmp_path / "config.toml"
    checkpoint_path = tmp_path / "checkpoint.npz"
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
        config_path=config_path, checkpoint_path=checkpoint_path,
        checkpoint_seconds=0.0,
    ))
    app.setup()
    assert app.engine.geometry.depth == 16
    low, high = app.volume_depth_limits()
    assert (low, high) == (volume_module.MIN_DEPTH, app.engine.geometry.max_depth)

    _owe_ticks(app)
    for _ in range(3):
        app.draw_frame()
    assert app.engine.tick_count > 0
    assert app.save_checkpoint(blocking=True) and checkpoint_path.exists()

    # The thickness it is already running at is not a reset.
    assert app.set_volume_depth(16) == 16
    assert app.engine.tick_count > 0, "an unchanged thickness discarded the field"

    assert app.set_volume_depth(32) == 32
    assert app.engine.geometry.depth == 32
    assert app.engine.tick_count == 0, "a fresh field, not the old one reshaped"
    assert app.config.overrides["volume.depth"] == 32
    assert not checkpoint_path.exists(), (
        "the saved field is of the old shape and would rebuild it on the next "
        "launch")
    # A new field resets the pacing clock, so a tick needs a tick's worth of
    # elapsed time to have passed rather than just another frame.
    _owe_ticks(app)
    app.draw_frame()
    assert app.engine.tick_count > 0, "the new slab is not ticking"

    # Past the ceiling is held to the ceiling rather than refused: the slider
    # cannot ask for it, but the method is reachable from elsewhere.
    assert app.set_volume_depth(10_000) == high
    assert app.engine.geometry.depth == high
    app.draw_frame()  # the thickest slab this window allows still renders
    app.shutdown()


def test_the_thickness_knob_survives_the_layered_backend(
    gpu_device, monkeypatch, tmp_path
):
    """The panel can carry a setting for a field that does not exist yet.

    Under the layered backend there is no slab to rebuild, so the thickness is
    recorded and lands the next time one is grown -- which is what switching to
    the volumetric backend does.
    """
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
    layered = app.engine

    assert app.set_volume_depth(32) == 32
    assert app.engine is layered, "the layered field was rebuilt over a slab knob"
    assert app.volume_slab().depth == 32

    assert app.switch_backend("volumetric")
    assert app.engine.geometry.depth == 32
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


def _governor_clamp(params: config.Params) -> float:
    """The ceiling the exposure governor clamps to for this run.

    This used to be a literal in `exposure.wgsl`, read back out of the shader
    text so the test could not drift from it. Since the perennial's
    attenuation-only governor (DESIGN.md §17.6) the shader clamps to
    `render.exposure_max`, fed from `safety.exposure_max` -- the enforcing
    site moved into the params, so the params are what is read. The shader
    keeps only a garbage-value fallback, which is asserted to still be there:
    if that wiring changes shape again, this should fail loudly rather than
    return a ceiling the governor no longer honours.
    """
    assert re.search(r"finite_or\(render\.exposure_max,", shaders.load(
        "exposure.wgsl")), (
        "exposure.wgsl no longer takes its ceiling from render.exposure_max; "
        "_governor_clamp needs re-deriving against the new wiring")
    return params.safety.exposure_max


def _brightest_exposure_target() -> float:
    """`safety.exposure_target` at the top of the brightness macro.

    Across both mode tables, because the ceiling has to hold in whichever of
    them asks the governor for most.
    """
    return max(
        high
        for table in config.MODE_CURVES.values()
        for path, _low, high, _gamma in table.get("brightness", ())
        if path == "safety.exposure_target"
    )


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

    The slab is narrow and it is the shipped *thickness*, which is the one
    dimension this cannot economise on. A ray's optical depth accumulates per
    filament crossed, so it scales with the depth in voxels and with nothing
    else here: the same field measures 15.5 through the 24-voxel slab this
    test used to run and 9.8 through the shipped 48 -- both read the way this
    test used to read them -- and the ceiling below falls between the two.
    Width is free -- a ray crosses the same material at any lateral resolution
    -- and 96 is what a software adapter can afford.

    The governor's time constant is measured in seconds by design, so this
    needs several hundred frames to be a statement about the settled level
    rather than about the fade-in -- and the climb got longer when the shading
    rebalance of DESIGN.md 4.7 step 5 reduced how much of the image arrives
    already bright. Both rates are raised by one factor, and that they are the
    same factor is the point: the governor's asymmetry -- brightening slower
    than darkening, because the unsafe direction is always "gets brighter" --
    is what decides where it settles against a field that fluctuates, so
    raising the attack alone moves the level under test instead of merely
    arriving at it sooner. Scaled together the settled level is the shipped
    one, and measured it settles 4.5% above target where raising the attack
    alone put it 10.6% above.

    Both numbers are read as a mean over the last fifth of the run rather than
    off the final frame, because both fluctuate by a few per cent from frame to
    frame and one frame's reading of a settled level is a reading of the
    fluctuation. The slab is still filling in by then -- run on to 1400 frames
    the multiplier falls from 8.9 to 6.3 -- so what this measures is an upper
    bound on what a mature field asks for, which is the safe direction for a
    ceiling to be wrong in.
    """
    params = _params(width=96, depth=48, climate_width=12, climate_height=8,
                     climate_depth=4, steps=48)
    # 4.3x, which puts the attack at the 0.015 the frame count below was set
    # against, and the release at the same multiple of its own shipped value.
    speedup = 4.3
    params.safety.exposure_attack *= speedup
    params.safety.exposure_release *= speedup
    engine = _engine(gpu_device, params, seed=8, size=(160, 96))
    view, fmt = offscreen_target(160, 96)

    frames = 700
    late: list[tuple[float, float]] = []
    for frame in range(frames):
        engine.tick(params)
        engine.render(params, frac=0.5, target_view=view, target_format=fmt)
        if frame >= frames * 4 // 5 and frame % 20 == 0:
            stats = engine.read_stats()
            late.append((stats["img_sum_l"] / max(stats["img_count"], 1.0),
                         stats["exposure"]))

    assert late, "nothing was sampled from the settled stretch"
    settled = sum(value for value, _ in late) / len(late)
    exposure = sum(value for _, value in late) / len(late)
    target = params.safety.exposure_target
    assert abs(settled - target) < 0.03 * max(target, 1e-6) + 0.02, (
        f"mean image lightness settled at {settled:.4f} against a target of "
        f"{target:.4f}")
    # The multiplier the slab needs is much larger than the stack's: a filament
    # network fills far less of a volume than of a plane, so gating the
    # reaction on the network costs the march more than it costs the compositor
    # (DESIGN.md 4.7 step 5, and 13, which lists this among the slab-specific
    # numbers wanting a viewer). Dimming the trail hubs -- step 6's deposit
    # capacity and shading knee -- cost it again and by more. Measured on the
    # 24-voxel slab this test used to run: 8.3 after step 5, 17.6 after step 6,
    # 15.5 once the sensing saturation handed some of it back.
    #
    # What the ceiling protects is the case where that stops being a correction
    # and becomes a saturation, leaving the image permanently dim. It is
    # derived rather than chosen, because the chosen one was raised twice and
    # each raise recorded only that the number had moved. The governor clamps;
    # the brightness macro's top end asks for `safety.exposure_target` at its
    # own ceiling where this run asks for it here; so a multiplier past
    # `clamp * target / top` is one the top of the brightness knob cannot be
    # reached from -- there the governor pins and the image stays dim however
    # far the knob is turned. Measured at the top of that knob the slab does
    # settle its target, with the multiplier at 15.5 against a clamp of 20.
    clamp = _governor_clamp(params)
    top = _brightest_exposure_target()
    ceiling = clamp * target / top
    assert 0.05 < exposure < ceiling, (
        f"the governor settles at {exposure:.2f} against a ceiling of "
        f"{ceiling:.2f}: the top of the brightness macro would ask it for "
        f"{exposure * top / target:.1f} against its clamp of {clamp:.0f}, so "
        f"the march is handing it an image it can only just correct")
