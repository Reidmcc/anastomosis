"""What the two depth backends share: the device-side plumbing and the output chain.

There are two ways to put depth on the screen (DESIGN.md §5): the layered 2.5D
stack in :mod:`anastomosis.engine`, and the volumetric slab in
:mod:`anastomosis.volume`. They differ entirely in what they simulate and how
they turn it into an image, and they differ in *nothing* after that point --
"the output stages (§6, §7) are unchanged between backends, so this is a clean
swap rather than a fork".

This module is where that sentence is made true rather than merely intended.
Everything below the compositor lives here exactly once:

* the HDR target, the final ping-pong that doubles as the slew limiter's
  history, and the exposure partials -- the complete list of window-sized
  resources, and so the complete content of a resize;
* the exposure governor and the flash-safety stage, which is a safety property
  and must not have two implementations that can drift apart (DESIGN.md §7);
* the blue-noise dither and the present blit;
* the parameter mapping from :class:`~anastomosis.config.Params` to the shader
  block, which is what makes a macro mean the same thing under either backend;
* bind-group caching, pipeline construction, and the hue and feature-size
  walks, all of which are backend-agnostic accumulated state.

What a backend supplies is the part above the line: fields, a tick, and a pass
that writes ``self.hdr`` plus a screen-space velocity for the safety stage to
reproject through. :meth:`Backend.present` does the rest.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np
import wgpu

from . import gpu_params, shaders
from .config import Params

log = logging.getLogger(__name__)

TEX_FORMAT = wgpu.TextureFormat.rgba16float
FIELD_USAGE = (
    wgpu.TextureUsage.TEXTURE_BINDING
    | wgpu.TextureUsage.STORAGE_BINDING
    | wgpu.TextureUsage.COPY_SRC
    | wgpu.TextureUsage.COPY_DST
)

# Bounds for geometry that arrives from outside -- i.e. out of a checkpoint
# file, which decides how much memory a launch allocates before anything has
# validated it. These are core WebGPU's guaranteed maximum texture dimensions;
# the 3D one is much smaller, which is why the slab has its own.
MAX_DIM = 8192
MAX_DIM_3D = 2048

# Gain on the viewpoint's walk before the `tanh` that bounds it, chosen so the
# offset spends most of its time using the middle of its travel and only
# occasionally approaches the ends: at 0.55 the r.m.s. offset is 0.45 and the
# 95th percentile 0.79, against a hard bound of 1. Larger, and the viewpoint
# lives near its limits, where `tanh` is flat and the drift slows to nothing
# just when it is furthest out; smaller, and `render.parallax` stops meaning
# anything like the excursion actually seen.
PARALLAX_SHAPE = 0.55

# The lag the walk reaches the viewpoint through, as a fraction of the walk's
# own time constant, and the renormalisation that lag costs.
#
# An Ornstein-Uhlenbeck process is smooth in its envelope and white in its
# increments, which is fine for an amplitude and disastrous for a position: at
# a 75 s time constant the raw walk moves the image half a pixel per frame, at
# random, which is precisely the per-pixel temporal noise this application
# exists not to produce (see blit.wgsl). One first-order lag makes the position
# C1 -- the whiteness moves into the acceleration, where nothing can see it --
# and at a sixth of the walk's own constant it costs almost none of the travel:
# measured over an hour, the frame-to-frame step falls from 0.54 px to 0.020 px
# at 1440p while the peak excursion stays at 49 px of 50. The remaining motion
# is about 0.6 px/s, which is what the parallax actually is.
PARALLAX_LAG = 1.0 / 6.0
PARALLAX_NORM = math.sqrt(1.0 + PARALLAX_LAG)


def parallax_walk_for(offset: float) -> float:
    """The walk that would have put the viewpoint at this offset.

    `tanh`'s inverse, undoing the shaping in `_update_parallax`. A checkpoint
    saves the offset and not the two states behind it, because the offset is
    the part that was visible and the rest is derivable -- this is the deriving,
    and it is what lets a resumed session pick the drift up where it stopped
    rather than sliding back to the centre over the first few minutes.
    """
    bounded = min(max(offset, -0.999), 0.999)
    return math.atanh(bounded) / (PARALLAX_SHAPE * PARALLAX_NORM)


def round_up(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def plausible(value: int, ceiling: int = MAX_DIM) -> bool:
    return isinstance(value, int) and 1 <= value <= ceiling


def aspect_correction(
    out_w: int, out_h: int, sim_w: int, sim_h: int
) -> tuple[float, float]:
    """UV scale factors that keep features square when the shapes differ.

    The simulation keeps the resolution -- and so the aspect ratio -- it was
    built with, but the window can be reshaped at any time. Stretching the
    field to fit would squash every feature in the image, so instead the axis
    that gained relative to the simulation samples *more* of the field. The
    domain is toroidal and the sampler wraps, so the extra area is seamless.

    Returns 1.0 on both axes when the shapes agree, so the common case costs
    nothing.
    """
    ratio = (out_w / max(out_h, 1)) / (sim_w / max(sim_h, 1))
    return max(ratio, 1.0), max(1.0 / ratio, 1.0)


class PingPong:
    """A pair of textures with an alternating current/next index.

    Two-dimensional by default; passing ``depth`` builds a 3D pair instead, for
    the volumetric backend. The only difference on this side is the view
    dimension, which wgpu infers from the texture -- but a 3D texture one voxel
    deep would still be a ``texture_3d`` in the shader, so the distinction is
    the caller's to make and not something to guess from a size of 1.
    """

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        label: str,
        depth: int | None = None,
    ):
        volumetric = depth is not None
        size = (width, height, depth if volumetric else 1)
        self.textures = [
            device.create_texture(
                size=size,
                dimension="3d" if volumetric else "2d",
                format=TEX_FORMAT,
                usage=FIELD_USAGE,
                label=f"{label}[{i}]",
            )
            for i in range(2)
        ]
        self.views = [t.create_view() for t in self.textures]
        self.index = 0

    @property
    def cur(self) -> wgpu.GPUTextureView:
        return self.views[self.index]

    @property
    def nxt(self) -> wgpu.GPUTextureView:
        return self.views[1 - self.index]

    def flip(self) -> None:
        self.index = 1 - self.index


class Backend:
    """Device-side plumbing and the output chain, for either depth backend.

    Subclasses own the simulation: they build their own fields in
    ``__init__`` after calling this one, implement :meth:`tick`, and implement
    :meth:`_compose`, which must write ``self.hdr_view`` and a screen-space
    velocity texture. Everything from the exposure governor onward is here.
    """

    #: Which backend this is, for checkpoint metadata and log lines.
    name = "abstract"

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        seed: int | None = None,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.tick_count = 0
        self.frame_count = 0
        self.seed = seed if seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        self._bind_cache: dict[tuple, wgpu.GPUBindGroup] = {}
        self._layout_cache: dict[int, wgpu.GPUBindGroupLayout] = {}
        # The setpoint of the feature-size loop, as a unit-variance OU state.
        # See `_advance_ell_walk`.
        self._ell_walk = 0.0
        self._walk_rng = np.random.default_rng(self.seed ^ 0xD1FF)
        self.hue_phase = 0.0
        # Where the viewpoint is: a unit-variance walk, and the bounded offset
        # the compositors read off it. See `_update_parallax`.
        #
        # Its own noise stream, rather than the feature-size walk's. The two
        # advance on different clocks -- that one per tick, this one per frame --
        # so sharing would make the feature size's realised path depend on the
        # frame rate, which is a coupling between two mechanisms that have
        # nothing to do with each other.
        self._parallax_rng = np.random.default_rng(self.seed ^ 0x9A17)
        self._parallax: list[list[float]] | None = None
        self._parallax_walk: list[list[float]] = []
        self._parallax_lag: list[list[float]] = []

        self.sampler = device.create_sampler(
            address_mode_u="repeat", address_mode_v="repeat",
            address_mode_w="repeat",
            mag_filter="linear", min_filter="linear", label="wrap_linear",
        )

        # --- Render targets ------------------------------------------------
        self._make_output_targets(width, height)
        # Written only when a resize has to carry the on-screen frame across.
        self.resample_buf: wgpu.GPUBuffer | None = None

        # --- Shared buffers ------------------------------------------------
        self.render_buf = device.create_buffer(
            size=gpu_params.RENDER_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="render_params")
        self.events_buf = device.create_buffer(
            size=gpu_params.EVENT_DTYPE.itemsize * 8,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="events")
        self.stats_buf = device.create_buffer(
            size=gpu_params.STATS_DTYPE.itemsize,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
            label="stats")

        stats = np.zeros(1, dtype=gpu_params.STATS_DTYPE)
        stats["exposure"] = 1.0
        # Zero is the correct starting value: the field starts empty, so there
        # is nothing being pruned and nothing to hand back. The reduction
        # overwrites it every tick.
        stats["prune_return"] = 0.0
        device.queue.write_buffer(self.stats_buf, 0, stats.tobytes())

        self._make_noise_texture()
        self._build_output_pipelines()

    # -- output targets -----------------------------------------------------

    def _make_output_targets(self, width: int, height: int) -> None:
        """(Re)allocate everything whose size follows the window.

        This is the complete list of window-sized resources: the HDR
        composite target, the final ping-pong -- which is also the slew
        limiter's history -- and the buffer the exposure reduction writes its
        per-tile partials into. Nothing here holds simulation state.
        """
        device = self.device
        self.width = width
        self.height = height

        self.hdr = device.create_texture(
            size=(width, height, 1), format=TEX_FORMAT, usage=FIELD_USAGE, label="hdr")
        self.hdr_view = self.hdr.create_view()
        self.final = PingPong(device, width, height, "final")

        img_tiles_x = math.ceil(width / 16)
        img_tiles_y = math.ceil(height / 16)
        self.img_partials = device.create_buffer(
            size=img_tiles_x * img_tiles_y * 16,
            usage=wgpu.BufferUsage.STORAGE, label="img_partials")
        self.img_tiles = (img_tiles_x, img_tiles_y)

    def _extra_output_targets(self) -> list:
        """Window-sized resources a subclass owns, retired alongside these."""
        return []

    def resize(self, width: int, height: int) -> None:
        """Follow a window resize *without* disturbing the simulation.

        Only the presentation chain depends on the window size, so only the
        presentation chain is rebuilt. The simulation keeps the resolution it
        was created with and, with it, every bit of its state -- fields,
        agents, climate, the tick counter. The compositor samples the
        simulation in normalised coordinates, so it does not care that it no
        longer matches the window; the aspect difference is corrected in the
        render parameters.

        Rebuilding the simulation instead would restart the world, and
        re-resolving a running simulation is itself the kind of visible
        discontinuity this application exists to avoid (DESIGN.md §8).
        """
        if width <= 0 or height <= 0 or (width, height) == (self.width, self.height):
            return

        # Everything retired here may be freed once the caches let go of it.
        # `self.final.cur` is the frame currently on screen, so it is read one
        # last time before it goes.
        on_screen = self.final.cur
        retired = [
            self.hdr, self.hdr_view,
            *self.final.textures, *self.final.views,
            self.img_partials, self._buffer_binding(self.img_partials),
            *self._extra_output_targets(),
        ]

        self._make_output_targets(width, height)
        self._carry_history(on_screen)
        self._retire(retired)

        log.info(
            "output now %dx%d; simulation continues at %s, tick %d",
            width, height, self.geometry.describe(), self.tick_count,
        )

    def _carry_history(self, on_screen: wgpu.GPUTextureView) -> None:
        """Rescale the frame that is on screen into the new history buffer.

        The slew limiter emits `history + bounded step` every frame, so a
        history buffer that started black would make the image climb back out
        of black at the limiter's rate -- roughly a second of fade, which is
        precisely the interruption a resize must not cause. Seeding it with the
        old frame leaves the limiter starting where the eye left off, and the
        exposure governor with a sane frame to measure.

        The blur pipeline is reused with a zero radius, which reduces to one
        bilinear tap per destination pixel -- a resampling copy, no new shader.
        """
        if self.resample_buf is None:
            self.resample_buf = self.device.create_buffer(
                size=gpu_params.SIM_DTYPE.itemsize,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                label="resample_params")

        self.device.queue.write_buffer(
            self.resample_buf, 0,
            gpu_params.pack(gpu_params.SIM_DTYPE, {
                "dims_x": self.width, "dims_y": self.height,
                "blur_radius": 0.0, "blur_dir_x": 0.0, "blur_dir_y": 0.0,
            }).tobytes())

        encoder = self.device.create_command_encoder(label="resize_carry")
        cpass = encoder.begin_compute_pass()
        cpass.set_pipeline(self.p_blur)
        cpass.set_bind_group(0, self._bind(self.p_blur, [
            self._buffer_binding(self.resample_buf), on_screen,
            self.final.cur, self.sampler,
        ]))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))
        cpass.end()
        self.device.queue.submit([encoder.finish()])

    def _retire(self, resources: list) -> None:
        """Forget cached bind groups that refer to replaced resources.

        The caches are keyed by `id()`, which CPython recycles as soon as an
        object is freed. A stale entry would therefore not merely waste memory:
        a later, unrelated object landing on the same address would hit it and
        silently bind a destroyed texture.
        """
        dead = {id(resource) for resource in resources}
        self._bind_cache = {
            key: group for key, group in self._bind_cache.items()
            if dead.isdisjoint(key)
        }
        bindings = getattr(self, "_buf_bindings", None)
        if bindings:
            for key in [k for k in bindings if k in dead]:
                del bindings[key]

    # -- setup --------------------------------------------------------------

    def _make_noise_texture(self) -> None:
        from . import bluenoise

        mask = bluenoise.load_or_generate()
        size = mask.shape[0]
        rgba = np.zeros((size, size, 4), dtype=np.float16)
        rgba[..., 0] = mask
        self.noise_tex = self.device.create_texture(
            size=(size, size, 1), format=TEX_FORMAT,
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            label="bluenoise")
        self.device.queue.write_texture(
            {"texture": self.noise_tex, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(rgba),
            {"offset": 0, "bytes_per_row": size * 8, "rows_per_image": size},
            (size, size, 1),
        )
        self.noise_view = self.noise_tex.create_view()

    def _compute(self, shader: str, entry: str = "main") -> wgpu.GPUComputePipeline:
        module = self.device.create_shader_module(
            code=shaders.load(shader), label=shader)
        return self.device.create_compute_pipeline(
            layout=wgpu.enums.AutoLayoutMode.auto,
            compute={"module": module, "entry_point": entry},
            label=f"{shader}:{entry}",
        )

    def _build_output_pipelines(self) -> None:
        """Everything from the exposure measurement to the present blit.

        `blur` is here rather than with the simulation passes because the
        resize path needs it to carry the on-screen frame across, which is an
        output-side concern; the backends that also use it for diffusion or
        depth of field reuse this one pipeline.
        """
        self.p_blur = self._compute("blur.wgsl")
        self.p_safety = self._compute("safety.wgsl")
        self.p_reduce_image = self._compute("reduce_image.wgsl")
        self.p_exposure = self._compute("exposure.wgsl")
        self.p_homeostat = self._compute("homeostat.wgsl", "reduce_final")

        blit_module = self.device.create_shader_module(
            code=shaders.load("blit.wgsl"), label="blit.wgsl")
        self.blit_module = blit_module
        self.p_blit: wgpu.GPURenderPipeline | None = None
        self.blit_format: str | None = None

    def _blit_pipeline(self, target_format: str) -> wgpu.GPURenderPipeline:
        if self.p_blit is None or self.blit_format != target_format:
            self.p_blit = self.device.create_render_pipeline(
                layout=wgpu.enums.AutoLayoutMode.auto,
                vertex={"module": self.blit_module, "entry_point": "vs_main"},
                fragment={
                    "module": self.blit_module,
                    "entry_point": "fs_main",
                    "targets": [{"format": target_format}],
                },
                primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
                label="blit",
            )
            self.blit_format = target_format
        return self.p_blit

    # -- bind group caching -------------------------------------------------

    def _layout(self, pipeline) -> wgpu.GPUBindGroupLayout:
        key = id(pipeline)
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = pipeline.get_bind_group_layout(0)
            self._layout_cache[key] = layout
        return layout

    def _bind(self, pipeline, resources: list) -> wgpu.GPUBindGroup:
        """Bind group for `resources`, cached by resource identity.

        Steady state is a dict lookup and nothing else, which is what keeps a
        multi-day run allocation-free without hand-enumerating every ping-pong
        parity combination.
        """
        key = (id(pipeline),) + tuple(id(r) for r in resources)
        group = self._bind_cache.get(key)
        if group is None:
            group = self.device.create_bind_group(
                layout=self._layout(pipeline),
                entries=[
                    {"binding": i, "resource": r} for i, r in enumerate(resources)
                ],
            )
            self._bind_cache[key] = group
        return group

    def _buffer_binding(self, buffer: wgpu.GPUBuffer) -> dict:
        """Stable dict per buffer, so identity-keyed caching works."""
        cache = getattr(self, "_buf_bindings", None)
        if cache is None:
            cache = {}
            self._buf_bindings = cache
        binding = cache.get(id(buffer))
        if binding is None:
            binding = {"buffer": buffer, "offset": 0, "size": buffer.size}
            cache[id(buffer)] = binding
        return binding

    @staticmethod
    def _groups(width: int, height: int, size: int = 8) -> tuple[int, int]:
        return math.ceil(width / size), math.ceil(height / size)

    @staticmethod
    def _groups3(
        width: int, height: int, depth: int, size: tuple[int, int, int] = (4, 4, 4)
    ) -> tuple[int, int, int]:
        return (
            math.ceil(width / size[0]),
            math.ceil(height / size[1]),
            math.ceil(depth / size[2]),
        )

    # -- parameter packing --------------------------------------------------

    def _physics_values(
        self, params: Params, feature: float, tempo: float, agent_density: float
    ) -> dict:
        """Every physical parameter the shaders read, at one feature/tempo scale.

        Shared between the backends on purpose. These are the numbers that
        decide what the simulation *is* -- how agents sense and steer, where the
        reaction sits on the Gray-Scott map, how far the climate deviates -- and
        a macro has to mean the same thing whichever way the result is put on
        screen. The two callers add only their own dimensions and identity --
        and their agent density, which is the one physical number they do not
        share (`agents.density` per cell against `volume.density` per voxel)
        and which the sensing cap's absolute value is anchored to.
        """
        a, r, f = params.agents, params.reaction, params.flow
        c, ho, pg = params.climate, params.homeostat, params.pigment

        # The sensing cap, made absolute. `sense_cap` is a multiple of the
        # equilibrium mean trail, which is exactly deposit-per-texel-per-tick
        # over decay; anchoring here is what lets one ratio hold across the
        # intensity macro and across both backends (see AgentParams). The
        # floor is a liveness bound: `recent` is an EMA of sensed -- and
        # therefore capped -- values, so a cap near `starve_threshold` reads
        # the whole population as starving and it respawns forever.
        sense_cap = 0.0
        if a.sense_cap > 0.0:
            equilibrium = agent_density * a.deposit / max(a.trail_decay, 1e-6)
            sense_cap = max(
                a.sense_cap * equilibrium, max(0.02, 4.0 * a.starve_threshold))

        # Homeostat slew per tick from its time constant in seconds.
        homeo_rate = 1.0 - math.exp(
            -1.0 / max(ho.tau_seconds * max(params.sim_hz, 1e-3), 1.0)
        )

        # The feature-size loop's per-tick rates, from their time constants, the
        # same way `homeo_rate` above is derived: the tempo macro moves
        # `sim_hz` by more than 2x, and a loop whose behaviour changed with it
        # would mean the tempo knob quietly retuned the controller.
        def _rate(tau_seconds: float) -> float:
            return 1.0 - math.exp(
                -1.0 / max(tau_seconds * max(params.sim_hz, 1e-3), 1.0)
            )

        # `du` goes to the shader as the base the scale macro set, with no
        # correction of its own: the global mean is the controller's now, and
        # it applies its own correction from the stats buffer, per texel,
        # alongside the climate's per-region deviation (DESIGN.md 4.7 step 5).
        # The only part of the loop that lives here is the setpoint walk, which
        # is accumulated state and so belongs with the hue phase.
        return {
            "speed": a.speed * tempo,
            "sensor_angle": a.sensor_angle,
            "sensor_distance": a.sensor_distance * feature,
            # In cells like the rest, so it scales with the layer the same way
            # the two lengths it relates do, and the ratio is per-layer
            # invariant.
            "sensor_distance_max": a.sensor_reach_max * a.trail_diffuse * feature,
            "turn_rate": a.turn_rate,
            "jitter": a.jitter,
            "deposit": a.deposit,
            "fusion_bias": a.fusion_bias,
            "fusion_max": a.fusion_max,
            "trail_decay": a.trail_decay,
            "trail_diffuse": a.trail_diffuse * feature,
            "income_rate": a.income_rate,
            "prune_gain": a.prune_gain,
            "deposit_cap": a.deposit_cap,
            "sense_cap": sense_cap,
            "trail_advect": a.trail_advect,
            "starve_threshold": a.starve_threshold,
            "max_age": a.max_age,
            "found_fraction": a.found_fraction,
            "found_period": a.found_period,
            "found_site_cells": a.found_site_cells,
            # In cells, like every other length here, so a half-resolution
            # layer founds a cohort of the same screen size rather than half of
            # one.
            "found_radius": a.found_radius * feature,

            "feed": r.feed, "kill": r.kill, "du": r.du, "dv": r.dv,
            "du_min": r.du_min, "du_max": r.du_max,
            "rdt": r.dt, "trail_feed_gain": r.trail_feed_gain,
            "kill_follows_feed": r.kill_follows_feed,
            "trail_seed_gain": r.trail_seed_gain,
            "trail_seed_falloff": r.trail_seed_falloff,
            "feed_min": r.feed_min, "feed_max": r.feed_max,
            "kill_band": r.kill_band,
            "kill_min": r.kill_min, "kill_max": r.kill_max,

            "psi_gain": f.psi_gain * tempo,
            "field_gain": f.field_gain * tempo,
            "psi_theta": f.psi_theta,
            "psi_sigma": f.psi_sigma,
            "psi_noise_scale": f.psi_noise_scale,
            "advect_dt": f.advect_dt,

            "clim_theta": c.theta, "clim_sigma": c.sigma,
            "clim_advect": c.advect_gain, "clim_diffuse": c.diffuse,
            "range_feed": c.range_feed, "range_kill": c.range_kill,
            "range_sensor_angle": c.range_sensor_angle,
            "range_sensor_distance": c.range_sensor_distance,
            "range_deposit": c.range_deposit, "range_decay": c.range_decay,
            "range_flow": c.range_flow, "range_hue": c.range_hue,
            "range_du": c.range_du, "range_prune": c.range_prune,
            "range_repel": c.range_repel,

            # Only the autonomous drift is baked into pigment; the palette macro
            # is applied at render time so turning the knob responds at once
            # instead of waiting for the field to be replaced.
            "hue_anchor": self.hue_phase,
            "hue_spread": params.render.hue_spread,
            "hue_from_orientation": pg.hue_from_orientation,
            "hue_inject_mix": pg.hue_inject_mix,
            "polychrome": params.render.polychrome,
            "polychrome_threshold": params.render.polychrome_threshold,
            "inject_rate": pg.inject_rate,
            "activity_rate": pg.activity_rate,
            "activity_gain": pg.activity_gain,
            "density_from_v": pg.density_from_v,
            "density_from_trail": pg.density_from_trail,
            "v_needs_trail": pg.v_needs_trail,
            "trail_knee": pg.trail_knee,

            "feature_scale": feature, "tempo_scale": tempo,

            "target_mass": ho.target_mass,
            "target_variance": ho.target_variance,
            "target_activity": ho.target_activity,
            "deadband": ho.deadband, "gain_p": ho.gain_p, "gain_i": ho.gain_i,
            "integral_limit": ho.integral_limit, "homeo_rate": homeo_rate,

            "ell_offset": r.ell_walk * self._ell_walk,
            "ell_rate": _rate(r.ell_tau_seconds),
            "ell_ref_rate": _rate(r.ell_ref_tau_seconds),
            "ell_corr_limit": r.ell_corr_limit,
        }

    # -- accumulated host-side state ----------------------------------------

    def _advance_hue_and_walk(self, params: Params, dt: float) -> None:
        self.hue_phase = (
            self.hue_phase
            + 2.0 * math.pi * params.render.hue_turns_per_hour * dt / 3600.0
        ) % (2.0 * math.pi)
        self._advance_ell_walk(dt, params.reaction.ell_walk_tau)

    def _advance_ell_walk(self, dt: float, tau: float) -> None:
        """One step of the Ornstein-Uhlenbeck walk on the feature-size setpoint.

        Kept unit-variance and dimensionless here, so the amplitude lives with
        the parameter it scales rather than being split across two places. The
        noise term is ``sqrt(1 - (1 - theta)^2)``, which is what makes the
        stationary variance exactly one whatever the tick rate is -- changing
        the tempo macro must not change how far the field wanders.

        This used to be a walk on the diffusion rate itself, and moving it to
        the setpoint is the difference between asking and getting. A walk on
        `du` asks for a diffusion rate and takes whatever texture the field
        chooses to produce; if the reaction is sitting in an attractor that
        pins its wavelength -- which is exactly the failure DESIGN.md §4.7
        exists to fix -- the walk moves and the picture does not. The setpoint
        it now drives is closed on a measurement of the texture, so a field
        that refuses to change feature size is a growing error rather than a
        thing nothing notices.

        It remains the *smaller* half of the mechanism. A global drift moves
        every feature on screen the same way at the same time, which §4.2 warns
        against, and it does nothing about the uniformity of size that is the
        actual accessibility problem; the per-region deviation in `climate_c`
        is what addresses that.
        """
        theta = 1.0 - math.exp(-dt / max(tau, 1e-3))
        sigma = math.sqrt(max(1.0 - (1.0 - theta) ** 2, 0.0))
        walk = self._ell_walk * (1.0 - theta) + float(self._walk_rng.normal()) * sigma
        # Bounded, so a tail excursion cannot ask for a feature size the
        # controller can only chase into its own clamp.
        self._ell_walk = max(-2.0, min(2.0, walk))

    def _update_parallax(self, params: Params, dt: float, count: int) -> None:
        """Where the viewpoint is, as a random walk rather than a sine.

        A sinusoidal drift would be periodic, and over a multi-hour session a
        periodic component is exactly what the design forbids -- the eye is very
        good at picking up a slow regular sway even when everything else is
        aperiodic.

        Written in the same form as :meth:`_advance_du_walk`, and for the same
        two reasons: the walk is kept unit-variance and dimensionless so the
        amplitude lives with the parameter that scales it rather than being
        split across two places, and its noise term is ``sqrt(1 - (1-theta)^2)``
        so the spread is exactly one whatever the frame rate happens to be.

        It did not used to be either of those. A fixed per-frame decay of 0.02
        against a ``sqrt(dt)`` noise term gave the walk a stationary spread of
        3e-4 across a travel of 1 -- which at ``render.parallax = 0.02`` is a
        viewpoint that moves a few hundredths of a pixel and, over two hours of
        simulation, never leaves a thousandth of its range. Motion parallax is
        the strongest depth cue either backend has, and it was in practice
        switched off in both of them.

        The offset the compositors read is ``tanh`` of the walk, not a clamp of
        it. Bounded either way, which is what makes ``render.parallax`` a
        maximum rather than a typical value; but a hard clamp on a *position*
        is a viewpoint that stops dead against a wall and stays there, where
        this eases to a halt and turns around.

        And the walk reaches the offset through one first-order lag, because an
        OU process is white in its increments and a position with white
        increments is a shaking image -- see :data:`PARALLAX_LAG`, which is the
        whole of that argument and the measurements behind it.

        One thing this deliberately does *not* do is tell the safety stage
        about itself. That stage reprojects its history through the screen-space
        velocity of the *material* (DESIGN.md §7), so a moving viewpoint leaves
        an uncompensated residual -- and at the amplitudes reachable here the
        residual is 0.02 px per frame against material that moves several,
        which is three orders of magnitude below anything the limiter responds
        to. Adding a camera term to the safety path would be complexity on the
        one path in the application that must stay simple enough to be obviously
        correct. It would start to matter somewhere around a pixel a frame,
        which needs `render.parallax` near 1 -- the whole frame width of drift.
        """
        if (
            self._parallax is None
            or len(self._parallax) != count
            or len(self._parallax_walk) != count
        ):
            # Padded and truncated rather than reset, so that a backend whose
            # record count changed keeps the viewpoints it can still use, and
            # a restore -- which brings back the offsets and nothing else --
            # continues from them. See `parallax_walk_for`.
            existing = self._parallax or []
            padded = (list(existing) + [[0.0, 0.0]] * count)[:count]
            self._parallax = [[float(x), float(y)] for x, y in padded]
            self._parallax_walk = [
                [parallax_walk_for(x), parallax_walk_for(y)]
                for x, y in self._parallax
            ]
            # Starting the lag at the walk is a viewpoint resuming at rest
            # rather than at whatever speed it happened to be moving, which is
            # the difference a restart is allowed to make.
            self._parallax_lag = [list(pair) for pair in self._parallax_walk]
        tau = max(params.render.parallax_tau, 1e-3)
        dt = max(dt, 0.0)
        theta = 1.0 - math.exp(-dt / tau)
        sigma = math.sqrt(max(1.0 - (1.0 - theta) ** 2, 0.0))
        lag = 1.0 - math.exp(-dt / (tau * PARALLAX_LAG))
        for walk, lagged, offset in zip(
            self._parallax_walk, self._parallax_lag, self._parallax
        ):
            for axis in (0, 1):
                walk[axis] = (
                    walk[axis] * (1.0 - theta)
                    + float(self._parallax_rng.normal()) * sigma
                )
                lagged[axis] += (walk[axis] - lagged[axis]) * lag
                offset[axis] = math.tanh(
                    PARALLAX_SHAPE * PARALLAX_NORM * lagged[axis])

    @staticmethod
    def _oklab_to_linear(
        lightness: float, a: float, b: float
    ) -> tuple[float, float, float]:
        """Host-side Oklab -> linear sRGB, matching common.wgsl."""
        l_ = lightness + 0.3963377774 * a + 0.2158037573 * b
        m_ = lightness - 0.1055613458 * a - 0.0638541728 * b
        s_ = lightness - 0.0894841775 * a - 1.2914855480 * b
        l, m, s = l_**3, m_**3, s_**3
        return (
            4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
            -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
            -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
        )

    # -- the output chain ---------------------------------------------------

    def _common_render_values(
        self, params: Params, frac: float, frame_dt: float
    ) -> dict:
        """Render-parameter fields that mean the same thing to both backends."""
        render = params.render
        safety = params.safety
        fog = self._oklab_to_linear(render.background_luma * 1.45, 0.0, -0.012)
        return {
            "out_w": self.width, "out_h": self.height,
            "frame": self.frame_count,
            "seed": self.seed,
            "frac": frac,
            "interp_dt": 1.0,
            "extinction": render.extinction,
            "fog_r": fog[0], "fog_g": fog[1], "fog_b": fog[2],
            "background_luma": render.background_luma,
            "filament_luma": render.filament_luma,
            "glow_gamma": render.glow_gamma,
            "l_max": render.l_max,
            "c_max": render.c_max,
            "chroma_activity_gain": render.chroma_activity_gain,
            "chroma_floor": render.chroma_floor,
            # The palette macro is applied here rather than baked into pigment,
            # so turning the knob responds immediately.
            "hue_global": render.hue_anchor,
            "max_luma_delta": safety.max_luma_delta,
            "max_chroma_delta": safety.max_chroma_delta,
            "iir_alpha": safety.iir_alpha,
            "exposure_target": safety.exposure_target,
            "exposure_attack": safety.exposure_attack,
            "exposure_release": safety.exposure_release,
            "exposure_max": safety.exposure_max,
            "dither_amount": safety.dither_amount,
            # How many sim ticks the previous frame is behind, so the history
            # can be reprojected by the right distance.
            "reproject_scale": max(frame_dt, 0.0) * params.sim_hz,
        }

    def render(
        self,
        params: Params,
        frac: float,
        target_view,
        target_format: str,
        frame_dt: float = 1.0 / 30.0,
    ) -> None:
        """One frame: the backend's compositor, then the shared output chain."""
        self._write_render_params(params, frac, frame_dt)

        encoder = self.device.create_command_encoder(label="render")
        cpass = encoder.begin_compute_pass()
        velocity_view = self._compose(cpass, params, frac)
        self._output_stage(cpass, velocity_view)
        cpass.end()
        self.final.flip()
        self._present(encoder, target_view, target_format)

        self.device.queue.submit([encoder.finish()])
        self.frame_count += 1

    def _output_stage(self, cpass, velocity_view) -> None:
        """Exposure governor, then the flash-safety stage. DESIGN.md §7.

        Both backends reach the screen through exactly this code. The safety
        stage is the one guarantee in the application that is enforced by
        construction rather than by taste, so a second copy of it -- one per
        depth backend, free to drift -- would be the most expensive kind of
        duplication there is.
        """
        render_bind = self._buffer_binding(self.render_buf)
        stats_bind = self._buffer_binding(self.stats_buf)

        # Exposure statistics, measured on the previous frame's output so
        # there is nothing to synchronise.
        cpass.set_pipeline(self.p_reduce_image)
        cpass.set_bind_group(0, self._bind(self.p_reduce_image, [
            render_bind, self.final.cur, self._buffer_binding(self.img_partials),
        ]))
        cpass.dispatch_workgroups(*self.img_tiles)

        cpass.set_pipeline(self.p_exposure)
        cpass.set_bind_group(0, self._bind(self.p_exposure, [
            render_bind, self._buffer_binding(self.img_partials), stats_bind,
        ]))
        cpass.dispatch_workgroups(1, 1, 1)

        # Reads the previous output as history and writes the new one, so the
        # ping-pong pair is also the history buffer.
        cpass.set_pipeline(self.p_safety)
        cpass.set_bind_group(0, self._bind(self.p_safety, [
            render_bind, self.hdr_view, self.final.cur,
            velocity_view, self.final.nxt, self.sampler, stats_bind,
        ]))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))

    def _present(self, encoder, target_view, target_format: str) -> None:
        """Present with blue-noise dithering."""
        pipeline = self._blit_pipeline(target_format)
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }])
        rpass.set_pipeline(pipeline)
        rpass.set_bind_group(0, self._bind(pipeline, [
            self._buffer_binding(self.render_buf), self.final.cur,
            self.noise_view, self.sampler,
        ]))
        rpass.draw(3, 1, 0, 0)
        rpass.end()

    # -- telemetry ----------------------------------------------------------

    def read_stats(self) -> dict[str, float]:
        """Read the stats buffer. For logging only -- the control loop never
        needs this, so it is safe to call rarely and never on the hot path."""
        raw = self.device.queue.read_buffer(self.stats_buf)
        record = np.frombuffer(
            memoryview(raw)[: gpu_params.STATS_DTYPE.itemsize],
            dtype=gpu_params.STATS_DTYPE,
        )[0]
        return {name: float(record[name]) for name in gpu_params.STATS_DTYPE.names}

    def read_final_rgba(self) -> np.ndarray:
        """Read back the current output as float32 RGBA. Used by the tests."""
        width, height = self.width, self.height
        raw = self.device.queue.read_texture(
            {"texture": self.final.textures[self.final.index],
             "mip_level": 0, "origin": (0, 0, 0)},
            {"offset": 0, "bytes_per_row": width * 8, "rows_per_image": height},
            (width, height, 1),
        )
        return (
            np.frombuffer(raw, dtype=np.float16)
            .reshape(height, width, 4)
            .astype(np.float32)
        )
