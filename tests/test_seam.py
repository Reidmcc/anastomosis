"""The wrap seam: what the field looks like where it meets itself.

The simulation domain is a torus, and the compositor leans on that. When the
window's aspect no longer matches the simulation's it samples *past* the edge
of the field and lets the wrap supply the rest (``engine.aspect_correction``),
and every layer behind the front one is sampled slightly wider still. So the
line where the field meets itself is not safely off-screen at the window edge:
it is drawn some way inside the frame, and further in the more the window is
reshaped. Anything visible at that line reads as a vertical or horizontal
structure hanging near the edge of the image.

Something was visible there. ``psi`` -- the vector potential the whole velocity
field is the curl of -- was driven by value noise that had no period on the
domain, so every tick injected a discontinuity along u=0 and v=0. psi
integrates its forcing, so that discontinuity accumulated into a permanent kink,
curl turned the kink into a jet, and pigment, trail and climate all stretched
along it. The lines behaved like part of the simulation because they were.

These tests are in three layers, because the property is enforced in the first
and only *observable* in the third:

* the noise closes on the domain, down to the rounding of the sample
  coordinate -- thirteen orders below the field's own texel-to-texel step;
* psi, driven by it, does not bend at the seam -- with the pre-fix forcing
  included as a positive control, so the measurement is known to have teeth;
* the shader agrees with the reference, and shows the same thing on the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import wgpu

import reference as R
from anastomosis import engine as engine_module, gpu_params, shaders

# psi.wgsl's stream constant, so the tests drive the same noise the shader does.
PSI_STREAM = 0x5BF03635

PSI_W, PSI_H = 96, 64
PSI_TICKS = 400
THETA, SIGMA = 0.0022, 0.085

# rgba16float has a 10-bit mantissa and psi is O(0.5), so this is about the
# floor for a value that has been through storage.
F16_ATOL = 2e-3


def _untiled_octaves(u, v, noise_scale, s):
    """The forcing as it was before the fix: value noise with no period.

    Kept as a positive control. A seam measurement that cannot see the defect
    it was written for is not a measurement, and the only way to know it can is
    to hand it the defect.
    """
    def octave(px, py, seed):
        ix, iy = np.floor(px), np.floor(py)
        fx, fy = px - ix, py - iy
        wx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0)
        wy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0)
        i, j = ix.astype(np.int64), iy.astype(np.int64)
        a = R.hash_grid(i, j, seed)
        b = R.hash_grid(i + 1, j, seed)
        c = R.hash_grid(i, j + 1, seed)
        d = R.hash_grid(i + 1, j + 1, seed)
        top = a + (b - a) * wx
        bottom = c + (d - c) * wx
        return top + (bottom - top) * wy

    return (octave(u * noise_scale, v * noise_scale, s) * 0.62
            + octave(u * noise_scale * 2.03, v * noise_scale * 2.03,
                     s ^ 0x9E3779B9) * 0.26
            + octave(u * noise_scale * 4.11, v * noise_scale * 4.11,
                     s ^ 0x85EBCA6B) * 0.12)


# ---------------------------------------------------------------------------
# The noise itself
# ---------------------------------------------------------------------------


# The gap tolerated at the join. The lattice either side of it is the same
# lattice, so the only difference is how the sample coordinate rounds -- `n +
# 0.37` is not bit-identical to `0.37` -- which leaves ~1e-15 in f64 against
# texel-to-texel steps of ~1e-2 in the field itself. The defect this replaced
# left a gap of order 0.5, thirteen orders up from here, so the threshold has
# room to be this tight without being brittle.
SEAM_GAP = 1e-12


def test_the_flow_noise_closes_on_the_domain():
    """f(0, y) == f(1, y), and f(x, 0) == f(x, 1), to floating-point noise."""
    along = np.linspace(0.0, 1.0, 65)
    zeros, ones = np.zeros_like(along), np.ones_like(along)
    for period in (2, 3, 5, 8):
        across_u = np.abs(
            R.value_noise_octaves_tiled(zeros, along, period, 0x11)
            - R.value_noise_octaves_tiled(ones, along, period, 0x11)).max()
        assert across_u < SEAM_GAP, f"u seam at period {period}: {across_u:.2e}"

        across_v = np.abs(
            R.value_noise_octaves_tiled(along, zeros, period, 0x11)
            - R.value_noise_octaves_tiled(along, ones, period, 0x11)).max()
        assert across_v < SEAM_GAP, f"v seam at period {period}: {across_v:.2e}"


def test_the_period_crossfade_closes_on_the_domain_too():
    """`psi_noise_scale` is continuous, so most of its range is a crossfade.

    Tiling needs a whole number of lattice cells across the domain; the knob
    does not deal in whole numbers. The crossfade is where a fix like this
    quietly stops holding, so it is checked over non-integer scales rather than
    the convenient ones.
    """
    along = np.linspace(0.0, 1.0, 65)
    zeros, ones = np.zeros_like(along), np.ones_like(along)
    for scale in (2.0, 2.4, 3.0, 3.75, 4.9, 5.0):
        across_u = np.abs(R.psi_increment(zeros, along, scale, PSI_STREAM)
                          - R.psi_increment(ones, along, scale, PSI_STREAM)).max()
        assert across_u < SEAM_GAP, f"u seam at scale {scale}: {across_u:.2e}"

        across_v = np.abs(R.psi_increment(along, zeros, scale, PSI_STREAM)
                          - R.psi_increment(along, ones, scale, PSI_STREAM)).max()
        assert across_v < SEAM_GAP, f"v seam at scale {scale}: {across_v:.2e}"


def test_the_forcing_is_no_rougher_at_the_seam_than_anywhere_else():
    """Closing on the domain is not enough on its own; it must close *smoothly*.

    A field can be periodic and still have a crease at the join. This measures
    the increment's own step size across the seam against its step size inside,
    averaged over ticks so that one unlucky realisation cannot carry it.
    """
    u, v = np.meshgrid(
        (np.arange(PSI_W) + 0.5) / PSI_W, (np.arange(PSI_H) + 0.5) / PSI_H)

    def roughness(forcing):
        seam_u = inside_u = seam_v = inside_v = 0.0
        for draw in range(24):
            stream = (draw * 2654435761) & 0xFFFFFFFF
            n = forcing(u, v, 3.0, stream)
            step_u = np.abs(np.roll(n, -1, 1) - n)
            step_v = np.abs(np.roll(n, -1, 0) - n)
            seam_u += step_u[:, -1].mean()
            inside_u += step_u[:, 4:PSI_W - 5].mean()
            seam_v += step_v[-1, :].mean()
            inside_v += step_v[4:PSI_H - 5, :].mean()
        return seam_u / inside_u, seam_v / inside_v

    tiled = roughness(R.psi_increment)
    assert max(tiled) < 2.0, f"the tiling forcing steps at the seam: {tiled}"

    # The control: the same measurement on the forcing this replaced.
    untiled = roughness(_untiled_octaves)
    assert min(untiled) > 6.0, (
        f"the untiled forcing no longer shows its seam ({untiled}), so this "
        "measurement proves nothing"
    )


def test_the_period_crossfade_holds_the_forcing_amplitude():
    """Turning the `scale` macro must not turn the weather up or down.

    The crossfade mixes two incoherent fields, so the weights are normalised in
    quadrature rather than to one. Get that wrong -- or let the two periods
    share a stream, which correlates them -- and the forcing swells or thins
    halfway between periods, which moves the flow speed from a knob that is
    supposed to move only feature size.
    """
    u, v = np.meshgrid(
        (np.arange(PSI_W) + 0.5) / PSI_W, (np.arange(PSI_H) + 0.5) / PSI_H)

    def amplitude(scale):
        draws = [R.psi_increment(u, v, scale, (k * 2654435761) & 0xFFFFFFFF).var()
                 for k in range(24)]
        return float(np.sqrt(np.mean(draws)))

    # Sampled across a whole period, i.e. through a crossfade and out the far
    # side, rather than at the integers where the crossfade is inactive.
    levels = [amplitude(scale) for scale in (3.0, 3.25, 3.5, 3.75, 4.0)]
    spread = max(levels) / min(levels)
    assert spread < 1.15, f"forcing amplitude swings {spread:.2f}x: {levels}"


# ---------------------------------------------------------------------------
# What it does to psi
# ---------------------------------------------------------------------------


def test_psi_does_not_bend_at_the_wrap_seam():
    """The end of the chain that matters: velocity is the curl of this field."""
    psi = R.psi_run(PSI_W, PSI_H, PSI_TICKS, theta=THETA, sigma=SIGMA)
    bend_u, bend_v = R.seam_curvature(psi)
    assert max(bend_u, bend_v) < 3.0, (
        f"psi bends at the seam: u {bend_u:.2f}x, v {bend_v:.2f}x the "
        "curvature it has anywhere else"
    )


def test_the_untiled_forcing_reproduces_the_seam():
    """The regression this guards against, run through the same integrator.

    Same field, same ticks, same measurement -- only the forcing differs. That
    is what makes the previous test's number mean something.
    """
    psi = R.psi_run(
        PSI_W, PSI_H, PSI_TICKS, theta=THETA, sigma=SIGMA,
        increment=_untiled_octaves,
    )
    bend_u, bend_v = R.seam_curvature(psi)
    assert min(bend_u, bend_v) > 5.0, (
        f"the untiled forcing left no measurable seam (u {bend_u:.2f}x, "
        f"v {bend_v:.2f}x), so the test above is not testing anything"
    )


# ---------------------------------------------------------------------------
# Where the seam lands on screen
# ---------------------------------------------------------------------------


def _seam_inset(window: tuple[int, int], sim: tuple[int, int], depth: float):
    """Where the field's u=0 and v=0 lines fall, as a fraction of the window.

    Mirrors the mapping in ``composite.wgsl``: a layer is sampled at
    ``(uv - 0.5) * scale + 0.5``, so the seam sits wherever that expression
    reaches zero.
    """
    aspect_x, aspect_y = engine_module.aspect_correction(*window, *sim)
    scale_x = (1.0 + 0.06 * depth) * aspect_x
    scale_y = (1.0 + 0.06 * depth) * aspect_y
    return 0.5 - 0.5 / scale_x, 0.5 - 0.5 / scale_y


def test_the_compositor_draws_the_seam_inside_the_frame():
    """Why a seam is a visible defect here rather than an off-screen curiosity.

    This is the geometry the artefact was reported through: lines near the
    edges that move *further* in when the window is resized, as if the field
    had zoomed out. It had -- on one axis, by exactly the aspect ratio.
    """
    sim = (1280, 720)

    # Front layer, window still the shape the field was built at: the seam is
    # exactly on the window edge.
    inset_x, inset_y = _seam_inset((1920, 1080), sim, depth=0.0)
    assert inset_x == pytest.approx(0.0) and inset_y == pytest.approx(0.0)

    # Layers behind it are sampled wider to read as further away, which brings
    # their seams inside the frame at any window shape.
    back_x, back_y = _seam_inset((1920, 1080), sim, depth=1.0)
    assert back_x > 0.02 and back_y > 0.02

    # And reshaping the window walks the seam further in, monotonically.
    insets = [_seam_inset((width, 1080), sim, depth=0.0)[0]
              for width in (1920, 2400, 2880, 3440)]
    assert insets == sorted(insets)
    assert insets[0] == pytest.approx(0.0)
    assert insets[-1] > 0.15, (
        f"an ultrawide window should push the seam well inside: {insets}"
    )


# ---------------------------------------------------------------------------
# The shader
# ---------------------------------------------------------------------------


def _run_psi_on_gpu(device, width, height, ticks, noise_scale=3.0, seed=0x5EED):
    """Run psi.wgsl for `ticks` ticks and read the field back."""
    module = device.create_shader_module(code=shaders.load("psi.wgsl"))
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
    textures = [
        device.create_texture(
            size=(width, height, 1),
            format=wgpu.TextureFormat.rgba16float,
            usage=usage,
        )
        for _ in range(2)
    ]
    blank = np.zeros((height, width, 4), dtype=np.float16)
    for texture in textures:
        device.queue.write_texture(
            {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
            blank,
            {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
            (width, height, 1),
        )

    index = 0
    for tick in range(ticks):
        values = {
            "psi_w": width, "psi_h": height,
            "psi_theta": THETA, "psi_sigma": SIGMA,
            "psi_noise_scale": noise_scale,
            "tick": tick, "seed": seed,
        }
        params_buf = device.create_buffer_with_data(
            data=gpu_params.pack(gpu_params.SIM_DTYPE, values).tobytes(),
            usage=wgpu.BufferUsage.STORAGE,
        )
        bind_group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": params_buf, "offset": 0,
                                            "size": params_buf.size}},
                {"binding": 1, "resource": textures[index].create_view()},
                {"binding": 2, "resource": textures[1 - index].create_view()},
            ],
        )
        encoder = device.create_command_encoder()
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(pipeline)
        cpass.set_bind_group(0, bind_group)
        cpass.dispatch_workgroups((width + 7) // 8, (height + 7) // 8, 1)
        cpass.end()
        device.queue.submit([encoder.finish()])
        index = 1 - index

    raw = device.queue.read_texture(
        {"texture": textures[index], "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
        (width, height, 1),
    )
    field = np.frombuffer(raw, dtype=np.float16).reshape(height, width, 4)
    return field[..., 0].astype(np.float64)


@pytest.mark.parametrize("noise_scale", [3.0, 3.6])
def test_psi_shader_matches_numpy(gpu_device, noise_scale):
    """Parity over a few ticks, at an integer period and mid-crossfade.

    Few ticks on purpose: psi is stored as f16 and rounds every tick, so a long
    run diverges from an f64 reference by accumulated rounding rather than by
    disagreement. What the seam depends on is the arithmetic, and three ticks
    pin that.
    """
    device, _ = gpu_device
    width, height, ticks = 48, 32, 3

    on_gpu = _run_psi_on_gpu(device, width, height, ticks, noise_scale=noise_scale)
    reference = R.psi_run(
        width, height, ticks, theta=THETA, sigma=SIGMA,
        noise_scale=noise_scale, seed=0x5EED,
    )
    assert np.abs(on_gpu - reference).max() < F16_ATOL, (
        f"psi diverged: max |delta| = {np.abs(on_gpu - reference).max():.5f}"
    )


def test_psi_shader_has_no_seam(gpu_device):
    """And the same measurement, on the shader that actually runs."""
    device, _ = gpu_device
    psi = _run_psi_on_gpu(device, PSI_W, PSI_H, PSI_TICKS)
    bend_u, bend_v = R.seam_curvature(psi)
    assert max(bend_u, bend_v) < 3.0, (
        f"psi.wgsl bends at the seam: u {bend_u:.2f}x, v {bend_v:.2f}x"
    )
