"""GPU resources, pipelines, and the tick/render sequence.

Resource notes worth knowing before reading further:

* **Everything is rgba16float.** Several fields need only one or two channels,
  but core WebGPU's storage-texture format list contains no r16float or
  rg16float, and the 32-bit single-channel formats are not filterable without an
  optional feature. rgba16float is both storage-capable and filterable in core,
  so it is used throughout. The wasted channels cost memory we have (DESIGN.md
  §8.1) and buy portability across Vulkan/Metal/DX12 that we want.

* **Bind groups are cached, not preallocated.** Ping-pong means a pass needs a
  different bind group depending on parity, and enumerating every combination by
  hand is error-prone. Instead they are created on demand and cached by resource
  identity, so the first couple of ticks populate the cache and steady state
  allocates nothing -- which is the property that actually matters for a
  multi-day run.

* **Simulation dimensions are rounded to multiples of 32** so that a texture row
  is a multiple of 256 bytes, which is what `write_texture` wants.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

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
MAX_LAYERS = 4
AGENT_STRIDE = 24  # vec2 pos, f32 heading, u32 rng, f32 recent, f32 age


def round_up(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


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
    """A pair of textures with an alternating current/next index."""

    def __init__(self, device: wgpu.GPUDevice, width: int, height: int, label: str):
        self.textures = [
            device.create_texture(
                size=(width, height, 1),
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


@dataclass
class LayerSpec:
    index: int
    width: int
    height: int
    agent_count: int
    feature_scale: float
    tempo_scale: float
    depth: float  # 0 = front, 1 = backmost


class Layer:
    """Per-layer GPU resources."""

    def __init__(self, device: wgpu.GPUDevice, spec: LayerSpec, params: Params):
        self.device = device
        self.spec = spec
        w, h = spec.width, spec.height

        self.trail = PingPong(device, w, h, f"trail{spec.index}")
        self.reaction = PingPong(device, w, h, f"reaction{spec.index}")
        self.pigment = PingPong(device, w, h, f"pigment{spec.index}")
        self.climate_a = PingPong(device, params.climate.width, params.climate.height,
                                  f"climate_a{spec.index}")
        self.climate_b = PingPong(device, params.climate.width, params.climate.height,
                                  f"climate_b{spec.index}")
        # Morphology: (scale, prune, fusion, spare). DESIGN.md §4.7. The other
        # two pairs are fully allocated, and a 64x36 rgba16f pair is ~9 KB.
        self.climate_c = PingPong(device, params.climate.width, params.climate.height,
                                  f"climate_c{spec.index}")

        psi_w = round_up(max(w // params.flow.psi_scale, 32), 32)
        psi_h = max(h // params.flow.psi_scale, 8)
        self.psi = PingPong(device, psi_w, psi_h, f"psi{spec.index}")
        self.psi_dims = (psi_w, psi_h)

        def plain(label: str, tw: int = w, th: int = h) -> wgpu.GPUTexture:
            return device.create_texture(
                size=(tw, th, 1), format=TEX_FORMAT, usage=FIELD_USAGE,
                label=f"{label}{spec.index}",
            )

        self.velocity = plain("velocity")
        self.velocity_view = self.velocity.create_view()
        self.scratch = plain("scratch")
        self.scratch_view = self.scratch.create_view()
        # Previous-tick reaction state, kept so the activity channel can measure
        # |dV/dt| without the substep loop clobbering it.
        self.reaction_prev = plain("reaction_prev")
        self.reaction_prev_view = self.reaction_prev.create_view()
        # Render-side: interpolated pigment, plus a scratch for the DOF blur.
        self.interp = plain("interp")
        self.interp_view = self.interp.create_view()
        self.interp_scratch = plain("interp_scratch")
        self.interp_scratch_view = self.interp_scratch.create_view()

        # --- Buffers -------------------------------------------------------
        self.params_buf = device.create_buffer(
            size=gpu_params.SIM_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label=f"sim_params{spec.index}",
        )
        # Separate parameter buffers per blur variant: a buffer write is a queue
        # operation, so a single buffer cannot carry different radii to two
        # dispatches inside one submission.
        self.blur_bufs = {
            name: device.create_buffer(
                size=gpu_params.SIM_DTYPE.itemsize,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                label=f"blur_{name}{spec.index}",
            )
            for name in ("trail_h", "trail_v", "dof_h", "dof_v")
        }
        self.sanitize_bufs = {
            name: device.create_buffer(
                size=gpu_params.SIM_DTYPE.itemsize,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                label=f"sanitize_{name}{spec.index}",
            )
            for name in ("trail", "reaction", "pigment")
        }

        self.agents_buf = device.create_buffer(
            size=max(spec.agent_count, 1) * AGENT_STRIDE,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label=f"agents{spec.index}",
        )
        self.deposit_buf = device.create_buffer(
            size=w * h * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label=f"deposit{spec.index}",
        )

        tiles_x = math.ceil(w / 16)
        tiles_y = math.ceil(h / 16)
        self.partials_buf = device.create_buffer(
            size=tiles_x * tiles_y * gpu_params.PARTIAL_SIZE,
            usage=wgpu.BufferUsage.STORAGE,
            label=f"partials{spec.index}",
        )
        self.tiles = (tiles_x, tiles_y)

        self._seed_state(params)

    # -- initial state ------------------------------------------------------

    def _seed_state(self, params: Params) -> None:
        device = self.device
        w, h = self.spec.width, self.spec.height
        rng = np.random.default_rng(0x5EED + self.spec.index)

        def upload(texture: wgpu.GPUTexture, array: np.ndarray) -> None:
            data = np.ascontiguousarray(array.astype(np.float16))
            device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                data,
                {"offset": 0, "bytes_per_row": data.shape[1] * 8,
                 "rows_per_image": data.shape[0]},
                (data.shape[1], data.shape[0], 1),
            )

        # Reaction starts as U=1, V=0 with scattered seeds of V, which is what
        # gives Gray-Scott something to nucleate from.
        reaction = np.zeros((h, w, 4), dtype=np.float32)
        reaction[..., 0] = 1.0
        seeds = max(8, (w * h) // 12000)
        for _ in range(seeds):
            cx, cy = rng.integers(0, w), rng.integers(0, h)
            radius = int(rng.integers(4, 12))
            ys, xs = np.ogrid[-radius:radius + 1, -radius:radius + 1]
            mask = xs * xs + ys * ys <= radius * radius
            yy = (np.arange(cy - radius, cy + radius + 1)) % h
            xx = (np.arange(cx - radius, cx + radius + 1)) % w
            patch = np.ix_(yy, xx)
            reaction[..., 1][patch] = np.where(mask, 0.45, reaction[..., 1][patch])
            reaction[..., 0][patch] = np.where(mask, 0.35, reaction[..., 0][patch])
        for tex in self.reaction.textures:
            upload(tex, reaction)
        upload(self.reaction_prev, reaction)

        zeros = np.zeros((h, w, 4), dtype=np.float32)
        for tex in self.trail.textures:
            upload(tex, zeros)
        upload(self.velocity, zeros)
        upload(self.scratch, zeros)

        # Pigment: hue must start as a valid unit vector or the first frames
        # would carry a degenerate (0,0) hue.
        pigment = np.zeros((h, w, 4), dtype=np.float32)
        pigment[..., 1] = 1.0
        for tex in self.pigment.textures:
            upload(tex, pigment)
        upload(self.interp, pigment)
        upload(self.interp_scratch, pigment)

        cw, ch = params.climate.width, params.climate.height
        for pair in (self.climate_a, self.climate_b, self.climate_c):
            for tex in pair.textures:
                upload(tex, rng.normal(0.0, 0.3, (ch, cw, 4)).clip(-1, 1))

        pw, ph = self.psi_dims
        psi = np.zeros((ph, pw, 4), dtype=np.float32)
        psi[..., 0] = rng.normal(0.0, 0.5, (ph, pw))
        for tex in self.psi.textures:
            upload(tex, psi)

        # Agents: uniform positions and headings.
        count = max(self.spec.agent_count, 1)
        agents = np.zeros(count, dtype=np.dtype([
            ("x", np.float32), ("y", np.float32), ("heading", np.float32),
            ("rng", np.uint32), ("recent", np.float32), ("age", np.float32),
        ]))
        agents["x"] = rng.random(count).astype(np.float32) * w
        agents["y"] = rng.random(count).astype(np.float32) * h
        agents["heading"] = rng.random(count).astype(np.float32) * (2 * math.pi)
        agents["rng"] = rng.integers(1, 2**32, count, dtype=np.uint64).astype(np.uint32)
        agents["recent"] = params.agents.starve_threshold * 4.0
        agents["age"] = rng.random(count).astype(np.float32) * params.agents.max_age
        device.queue.write_buffer(self.agents_buf, 0, agents.tobytes())

        device.queue.write_buffer(
            self.deposit_buf, 0, np.zeros(w * h, dtype=np.uint32).tobytes()
        )


class Engine:
    """Owns the device-side world and drives it."""

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        params: Params,
        seed: int | None = None,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        # The size the simulation layers were built for. The window may end up
        # a different size or shape later; the layers never change.
        self.sim_width = width
        self.sim_height = height
        self.tick_count = 0
        self.frame_count = 0
        self.seed = seed if seed is not None else int(time.time_ns() & 0xFFFFFFFF)
        self._bind_cache: dict[tuple, wgpu.GPUBindGroup] = {}
        self._layout_cache: dict[int, wgpu.GPUBindGroupLayout] = {}
        # Global mean of the reaction's diffusion rate, as a unit-variance OU
        # state. See `_advance_du_walk`.
        self._du_walk = 0.0
        self._walk_rng = np.random.default_rng(self.seed ^ 0xD1FF)

        self.sampler = device.create_sampler(
            address_mode_u="repeat", address_mode_v="repeat",
            mag_filter="linear", min_filter="linear", label="wrap_linear",
        )

        self.layers = [
            Layer(device, spec, params) for spec in self._layer_specs(params)
        ]

        # --- Render targets ------------------------------------------------
        self._make_output_targets(width, height)
        # Written only when a resize has to carry the on-screen frame across.
        self.resample_buf: wgpu.GPUBuffer | None = None

        # --- Shared buffers ------------------------------------------------
        self.render_buf = device.create_buffer(
            size=gpu_params.RENDER_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="render_params")
        self.layer_buf = device.create_buffer(
            size=gpu_params.LAYER_DTYPE.itemsize * MAX_LAYERS,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="layer_data")
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
        self._build_pipelines()

        log.info(
            "engine ready: %dx%d, %d layers (%s), %d agents total",
            width, height, len(self.layers),
            ", ".join(f"{l.spec.width}x{l.spec.height}" for l in self.layers),
            sum(l.spec.agent_count for l in self.layers),
        )

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

    def resize(self, width: int, height: int) -> None:
        """Follow a window resize *without* disturbing the simulation.

        Only the presentation chain depends on the window size, so only the
        presentation chain is rebuilt. The simulation layers keep the
        resolution they were created with and, with it, every bit of their
        state -- fields, agents, climate, the tick counter. The compositor
        samples layers in normalised coordinates, so it does not care that
        they no longer match the window; the aspect difference is corrected in
        `_write_render_params`.

        Rebuilding the layers instead would restart the world, and re-resolving
        a running simulation is itself the kind of visible discontinuity this
        application exists to avoid (DESIGN.md §8).
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
        ]

        self._make_output_targets(width, height)
        self._carry_history(on_screen)
        self._retire(retired)

        log.info(
            "output now %dx%d; simulation continues at %dx%d, tick %d",
            width, height, self.sim_width, self.sim_height, self.tick_count,
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

    def _layer_specs(self, params: Params) -> list[LayerSpec]:
        render = params.render
        count = max(1, min(render.layers, MAX_LAYERS))
        specs = []
        for i in range(count):
            shrink = render.scale_falloff**i
            w = round_up(max(int(self.width * render.base_scale * shrink), 64), 32)
            h = max(int(self.height * render.base_scale * shrink), 32)
            depth = i / max(count - 1, 1)
            specs.append(LayerSpec(
                index=i, width=w, height=h,
                agent_count=round_up(int(params.agents.density * w * h), 64),
                feature_scale=render.feature_falloff**i,
                tempo_scale=render.tempo_falloff**i,
                depth=depth,
            ))
        return specs

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

    def _build_pipelines(self) -> None:
        self.p_psi = self._compute("psi.wgsl")
        self.p_climate = self._compute("climate.wgsl")
        self.p_agents = self._compute("agents.wgsl")
        self.p_trail = self._compute("trail.wgsl")
        self.p_blur = self._compute("blur.wgsl")
        self.p_reaction = self._compute("reaction.wgsl")
        self.p_flow = self._compute("flow.wgsl")
        self.p_advect = self._compute("advect.wgsl")
        self.p_reduce = self._compute("reduce.wgsl", "reduce_tiles")
        self.p_homeostat = self._compute("homeostat.wgsl", "reduce_final")
        self.p_sanitize = self._compute("sanitize.wgsl")
        self.p_interp = self._compute("interp.wgsl")
        self.p_composite = self._compute("composite.wgsl")
        self.p_safety = self._compute("safety.wgsl")
        self.p_reduce_image = self._compute("reduce_image.wgsl")
        self.p_exposure = self._compute("exposure.wgsl")

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

    # -- parameter packing --------------------------------------------------

    def _sim_values(self, layer: Layer, params: Params, event_count: int) -> dict:
        a, r, f = params.agents, params.reaction, params.flow
        c, ho, pg = params.climate, params.homeostat, params.pigment
        spec = layer.spec

        # A cell on a half-resolution layer covers twice the screen distance, so
        # cell-space speeds must be scaled by the resolution ratio as well as by
        # the intended screen-relative tempo.
        tempo = spec.tempo_scale
        feature = spec.feature_scale

        # Homeostat slew per tick from its time constant in seconds.
        homeo_rate = 1.0 - math.exp(
            -1.0 / max(ho.tau_seconds * max(params.sim_hz, 1e-3), 1.0)
        )

        # The diffusion pair carries the global mean feature size; the climate
        # field carries the per-region deviation around it (DESIGN.md §4.7),
        # the same split feed and kill already use. Both are geometric, so the
        # shader's per-texel deviation composes with this exactly rather than
        # interacting with it. dv rides on du so the dv/du ratio is held: it is
        # scaling the pair together, and only together, that moves feature size
        # without moving mass -- which is what keeps this lever clear of the
        # homeostat and of the exposure governor.
        du_mean = min(
            max(r.du * math.exp(r.du_walk * self._du_walk), r.du_min), r.du_max)
        dv_mean = r.dv * (du_mean / max(r.du, 1e-6))

        return {
            "dims_x": spec.width, "dims_y": spec.height,
            "clim_w": c.width, "clim_h": c.height,
            "psi_w": layer.psi_dims[0], "psi_h": layer.psi_dims[1],
            "tick": self.tick_count,
            "seed": (self.seed ^ (spec.index * 0x9E3779B9)) & 0xFFFFFFFF,
            "agent_count": spec.agent_count,
            "layer_index": spec.index, "layer_count": len(self.layers),
            "event_count": event_count,

            "speed": a.speed * tempo,
            "sensor_angle": a.sensor_angle,
            "sensor_distance": a.sensor_distance * feature,
            "turn_rate": a.turn_rate,
            "jitter": a.jitter,
            "deposit": a.deposit,
            "fusion_bias": a.fusion_bias,
            "trail_decay": a.trail_decay,
            "trail_diffuse": a.trail_diffuse * feature,
            "income_rate": a.income_rate,
            "prune_gain": a.prune_gain,
            "starve_threshold": a.starve_threshold,
            "max_age": a.max_age,

            "feed": r.feed, "kill": r.kill, "du": du_mean, "dv": dv_mean,
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

            # Only the autonomous drift is baked into pigment; the palette macro
            # is applied at render time so turning the knob responds at once
            # instead of waiting for the field to be replaced.
            "hue_anchor": self.hue_phase,
            "hue_spread": params.render.hue_spread,
            "hue_from_orientation": pg.hue_from_orientation,
            "hue_inject_mix": pg.hue_inject_mix,
            "inject_rate": pg.inject_rate,
            "activity_rate": pg.activity_rate,
            "activity_gain": pg.activity_gain,
            "density_from_v": pg.density_from_v,
            "density_from_trail": pg.density_from_trail,

            "feature_scale": feature, "tempo_scale": tempo,

            "target_mass": ho.target_mass,
            "target_variance": ho.target_variance,
            "target_activity": ho.target_activity,
            "deadband": ho.deadband, "gain_p": ho.gain_p, "gain_i": ho.gain_i,
            "integral_limit": ho.integral_limit, "homeo_rate": homeo_rate,
        }

    def _write_layer_params(self, layer: Layer, params: Params, event_count: int) -> None:
        base = self._sim_values(layer, params, event_count)
        queue = self.device.queue
        queue.write_buffer(
            layer.params_buf, 0, gpu_params.pack(gpu_params.SIM_DTYPE, base).tobytes())

        # Blur variants need their own buffers: a buffer write is a queue
        # operation, so one buffer cannot carry two radii into one submission.
        for name, radius, dx, dy in (
            ("trail_h", base["trail_diffuse"], 1.0, 0.0),
            ("trail_v", base["trail_diffuse"], 0.0, 1.0),
            ("dof_h", params.render.dof_radius * layer.spec.depth, 1.0, 0.0),
            ("dof_v", params.render.dof_radius * layer.spec.depth, 0.0, 1.0),
        ):
            values = dict(base)
            values.update(blur_radius=radius, blur_dir_x=dx, blur_dir_y=dy)
            queue.write_buffer(
                layer.blur_bufs[name], 0,
                gpu_params.pack(gpu_params.SIM_DTYPE, values).tobytes())

        for name, lo, hi, fallback in (
            ("trail", 0.0, 8.0, 0.0),
            ("reaction", 0.0, 1.5, 0.0),
            ("pigment", -1.0, 4.0, 0.0),
        ):
            values = dict(base)
            values.update(sanitize_min=lo, sanitize_max=hi, sanitize_fallback=fallback)
            queue.write_buffer(
                layer.sanitize_bufs[name], 0,
                gpu_params.pack(gpu_params.SIM_DTYPE, values).tobytes())

    # -- simulation ---------------------------------------------------------

    hue_phase: float = 0.0

    def _advance_du_walk(self, dt: float, tau: float) -> None:
        """One step of the Ornstein-Uhlenbeck walk on the global feature size.

        Kept unit-variance and dimensionless here, so the amplitude lives with
        the parameter it scales rather than being split across two places. The
        noise term is ``sqrt(1 - (1 - theta)^2)``, which is what makes the
        stationary variance exactly one whatever the tick rate is -- changing
        the tempo macro must not change how far the field wanders.

        A walk rather than a constant because a fixed diffusion rate pins the
        Gray-Scott wavelength, and a pinned wavelength is what made the texture
        stationary (DESIGN.md §4.7). It is deliberately the *smaller* half of
        the mechanism: a global drift moves every feature on screen the same
        way at the same time, which §4.2 warns against, and it does nothing
        about the uniformity of size that is the actual accessibility problem.
        The per-region deviation in `climate_c` is what addresses that.
        """
        theta = 1.0 - math.exp(-dt / max(tau, 1e-3))
        sigma = math.sqrt(max(1.0 - (1.0 - theta) ** 2, 0.0))
        walk = self._du_walk * (1.0 - theta) + float(self._walk_rng.normal()) * sigma
        # Bounded, so a tail excursion cannot park the whole field at a clamp.
        self._du_walk = max(-2.0, min(2.0, walk))

    def tick(self, params: Params, event_rows: list[dict] | None = None) -> None:
        """Advance the simulation by one tick."""
        event_rows = event_rows or []
        event_count = min(len(event_rows), 8)

        dt = 1.0 / max(params.sim_hz, 1e-3)
        self.hue_phase = (
            self.hue_phase
            + 2.0 * math.pi * params.render.hue_turns_per_hour * dt / 3600.0
        ) % (2.0 * math.pi)
        self._advance_du_walk(dt, params.reaction.du_walk_tau)

        if event_count:
            self.device.queue.write_buffer(
                self.events_buf, 0,
                gpu_params.pack_array(
                    gpu_params.EVENT_DTYPE, event_rows[:event_count]).tobytes())

        for layer in self.layers:
            self._write_layer_params(layer, params, event_count)

        encoder = self.device.create_command_encoder(label="tick")
        cpass = encoder.begin_compute_pass()

        sampler = self.sampler
        events_bind = self._buffer_binding(self.events_buf)
        stats_bind = self._buffer_binding(self.stats_buf)

        for layer in self.layers:
            spec = layer.spec
            pbind = self._buffer_binding(layer.params_buf)
            agents_bind = self._buffer_binding(layer.agents_buf)
            deposit_bind = self._buffer_binding(layer.deposit_buf)
            gx, gy = self._groups(spec.width, spec.height)

            # 1. Vector potential.
            cpass.set_pipeline(self.p_psi)
            cpass.set_bind_group(0, self._bind(
                self.p_psi, [pbind, layer.psi.cur, layer.psi.nxt]))
            cpass.dispatch_workgroups(*self._groups(*layer.psi_dims))
            layer.psi.flip()

            # 2. Climate.
            cpass.set_pipeline(self.p_climate)
            cpass.set_bind_group(0, self._bind(self.p_climate, [
                pbind, layer.climate_a.cur, layer.climate_b.cur, layer.climate_c.cur,
                layer.climate_a.nxt, layer.climate_b.nxt, layer.climate_c.nxt,
                layer.psi.cur, sampler, events_bind,
            ]))
            cpass.dispatch_workgroups(*self._groups(params.climate.width,
                                                    params.climate.height))
            layer.climate_a.flip()
            layer.climate_b.flip()
            layer.climate_c.flip()

            # 3. Agents deposit into the fixed-point accumulator.
            cpass.set_pipeline(self.p_agents)
            cpass.set_bind_group(0, self._bind(self.p_agents, [
                pbind, agents_bind, layer.trail.cur, deposit_bind,
                layer.climate_a.cur, layer.climate_b.cur, sampler, stats_bind,
            ]))
            cpass.dispatch_workgroups(math.ceil(spec.agent_count / 64), 1, 1)

            # 4. Trail decay + deposit, then separable diffusion.
            cpass.set_pipeline(self.p_trail)
            cpass.set_bind_group(0, self._bind(self.p_trail, [
                pbind, layer.trail.cur, layer.trail.nxt, deposit_bind,
                layer.climate_b.cur, layer.climate_c.cur, sampler, stats_bind,
            ]))
            cpass.dispatch_workgroups(gx, gy)
            layer.trail.flip()

            cpass.set_pipeline(self.p_blur)
            cpass.set_bind_group(0, self._bind(self.p_blur, [
                self._buffer_binding(layer.blur_bufs["trail_h"]),
                layer.trail.cur, layer.scratch_view, sampler,
            ]))
            cpass.dispatch_workgroups(gx, gy)
            cpass.set_bind_group(0, self._bind(self.p_blur, [
                self._buffer_binding(layer.blur_bufs["trail_v"]),
                layer.scratch_view, layer.trail.nxt, sampler,
            ]))
            cpass.dispatch_workgroups(gx, gy)
            layer.trail.flip()

        # Reaction needs the pre-step state for the activity channel, and a
        # texture copy cannot be recorded inside a compute pass.
        cpass.end()
        for layer in self.layers:
            encoder.copy_texture_to_texture(
                {"texture": layer.reaction.textures[layer.reaction.index]},
                {"texture": layer.reaction_prev},
                (layer.spec.width, layer.spec.height, 1),
            )
        cpass = encoder.begin_compute_pass()

        for layer in self.layers:
            spec = layer.spec
            pbind = self._buffer_binding(layer.params_buf)
            gx, gy = self._groups(spec.width, spec.height)

            # 5. Reaction-diffusion substeps.
            cpass.set_pipeline(self.p_reaction)
            for _ in range(max(1, params.reaction.substeps)):
                cpass.set_bind_group(0, self._bind(self.p_reaction, [
                    pbind, layer.reaction.cur, layer.reaction.nxt,
                    layer.trail.cur, layer.climate_a.cur, layer.climate_c.cur,
                    sampler, stats_bind,
                ]))
                cpass.dispatch_workgroups(gx, gy)
                layer.reaction.flip()

            # 6. Velocity field.
            cpass.set_pipeline(self.p_flow)
            cpass.set_bind_group(0, self._bind(self.p_flow, [
                pbind, layer.psi.cur, layer.reaction.cur, layer.climate_b.cur,
                layer.velocity_view, sampler,
            ]))
            cpass.dispatch_workgroups(gx, gy)

            # 7. Advect pigment.
            cpass.set_pipeline(self.p_advect)
            cpass.set_bind_group(0, self._bind(self.p_advect, [
                pbind, layer.pigment.cur, layer.pigment.nxt,
                layer.reaction.cur, layer.reaction_prev_view, layer.trail.cur,
                layer.velocity_view, layer.climate_b.cur, sampler,
            ]))
            cpass.dispatch_workgroups(gx, gy)
            layer.pigment.flip()

            # 8. Field statistics for the homeostat.
            cpass.set_pipeline(self.p_reduce)
            cpass.set_bind_group(0, self._bind(self.p_reduce, [
                pbind, layer.reaction.cur, layer.reaction_prev_view,
                layer.trail.cur, self._buffer_binding(layer.partials_buf),
            ]))
            cpass.dispatch_workgroups(*layer.tiles)

        # 9. The controller runs once, on the front layer's statistics.
        front = self.layers[0]
        cpass.set_pipeline(self.p_homeostat)
        cpass.set_bind_group(0, self._bind(self.p_homeostat, [
            self._buffer_binding(front.params_buf),
            self._buffer_binding(front.partials_buf),
            stats_bind,
        ]))
        cpass.dispatch_workgroups(1, 1, 1)

        # 10. Periodic sanitisation. Cheap insurance against a permanent,
        #     unrecoverable failure mode (DESIGN.md §4.4).
        if self.tick_count % 60 == 0 and self.tick_count > 0:
            cpass.set_pipeline(self.p_sanitize)
            for layer in self.layers:
                gx, gy = self._groups(layer.spec.width, layer.spec.height)
                for name, field in (
                    ("trail", layer.trail), ("reaction", layer.reaction),
                    ("pigment", layer.pigment),
                ):
                    cpass.set_bind_group(0, self._bind(self.p_sanitize, [
                        self._buffer_binding(layer.sanitize_bufs[name]),
                        field.cur, field.nxt,
                    ]))
                    cpass.dispatch_workgroups(gx, gy)
                    field.flip()

        cpass.end()
        self.device.queue.submit([encoder.finish()])
        self.tick_count += 1

    # -- rendering ----------------------------------------------------------

    _parallax: list[list[float]] | None = None

    @staticmethod
    def _oklab_to_linear(lightness: float, a: float, b: float) -> tuple[float, float, float]:
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

    def _update_parallax(self, params: Params, dt: float) -> None:
        """Parallax offsets follow a random walk, not a sine.

        A sinusoidal drift would be periodic, and over a multi-hour session a
        periodic component is exactly what the design forbids -- the eye is very
        good at picking up a slow regular sway even when everything else is
        aperiodic.
        """
        if self._parallax is None:
            self._parallax = [[0.0, 0.0] for _ in self.layers]
        theta = 0.02
        sigma = params.render.parallax_drift * math.sqrt(max(dt, 1e-4))
        for state in self._parallax:
            for axis in (0, 1):
                state[axis] = (
                    state[axis] * (1.0 - theta)
                    + np.random.normal(0.0, 1.0) * sigma
                )
                state[axis] = max(-1.0, min(1.0, state[axis]))

    def _write_render_params(
        self, params: Params, frac: float, frame_dt: float
    ) -> None:
        render = params.render
        safety = params.safety
        fog = self._oklab_to_linear(render.background_luma * 1.45, 0.0, -0.012)

        values = {
            "out_w": self.width, "out_h": self.height,
            "layer_count": len(self.layers),
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
            "dither_amount": safety.dither_amount,
            # How many sim ticks the previous frame is behind, so the history
            # can be reprojected by the right distance.
            "reproject_scale": max(frame_dt, 0.0) * params.sim_hz,
        }
        self.device.queue.write_buffer(
            self.render_buf, 0,
            gpu_params.pack(gpu_params.RENDER_DTYPE, values).tobytes())

        aspect_x, aspect_y = aspect_correction(
            self.width, self.height, self.sim_width, self.sim_height)

        rows = []
        for i, layer in enumerate(self.layers):
            depth = layer.spec.depth
            offset = self._parallax[i] if self._parallax else [0.0, 0.0]
            rows.append({
                # >1 samples a wider area, so the layer reads as further away.
                "scale_x": (1.0 + 0.06 * depth) * aspect_x,
                "scale_y": (1.0 + 0.06 * depth) * aspect_y,
                "parallax_x": offset[0] * render.parallax * (1.0 - depth * 0.5),
                "parallax_y": offset[1] * render.parallax * (1.0 - depth * 0.5),
                "depth_dim": 1.0 + (render.depth_dim - 1.0) * depth,
                "depth_desat": 1.0 + (render.depth_desat - 1.0) * depth,
                "fog": render.fog_amount * depth,
                "opacity": 1.0,
            })
        while len(rows) < MAX_LAYERS:
            rows.append({k: 0.0 for k in gpu_params.LAYER_DTYPE.names})
        self.device.queue.write_buffer(
            self.layer_buf, 0,
            gpu_params.pack_array(gpu_params.LAYER_DTYPE, rows).tobytes())

    def render(
        self,
        params: Params,
        frac: float,
        target_view,
        target_format: str,
        frame_dt: float = 1.0 / 30.0,
    ) -> None:
        self._update_parallax(params, frame_dt)
        self._write_render_params(params, frac, frame_dt)

        sampler = self.sampler
        render_bind = self._buffer_binding(self.render_buf)
        layer_bind = self._buffer_binding(self.layer_buf)
        stats_bind = self._buffer_binding(self.stats_buf)

        encoder = self.device.create_command_encoder(label="render")
        cpass = encoder.begin_compute_pass()

        # 1. Motion-compensated interpolation between the last two sim states.
        for layer in self.layers:
            gx, gy = self._groups(layer.spec.width, layer.spec.height)
            cpass.set_pipeline(self.p_interp)
            cpass.set_bind_group(0, self._bind(self.p_interp, [
                self._buffer_binding(layer.params_buf), render_bind,
                layer.pigment.nxt,  # previous tick
                layer.pigment.cur,  # current tick
                layer.velocity_view, layer.interp_view, sampler,
            ]))
            cpass.dispatch_workgroups(gx, gy)

            # 2. Depth of field. The front layer has zero radius and is skipped.
            if params.render.dof_radius * layer.spec.depth > 0.05:
                cpass.set_pipeline(self.p_blur)
                cpass.set_bind_group(0, self._bind(self.p_blur, [
                    self._buffer_binding(layer.blur_bufs["dof_h"]),
                    layer.interp_view, layer.interp_scratch_view, sampler,
                ]))
                cpass.dispatch_workgroups(gx, gy)
                cpass.set_bind_group(0, self._bind(self.p_blur, [
                    self._buffer_binding(layer.blur_bufs["dof_v"]),
                    layer.interp_scratch_view, layer.interp_view, sampler,
                ]))
                cpass.dispatch_workgroups(gx, gy)

        # 3. Composite back to front.
        views = [layer.interp_view for layer in self.layers]
        while len(views) < MAX_LAYERS:
            views.append(views[0])  # unused; the shader loops to layer_count
        cpass.set_pipeline(self.p_composite)
        cpass.set_bind_group(0, self._bind(
            self.p_composite,
            [render_bind, layer_bind, *views, self.hdr_view, sampler],
        ))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))

        # 4. Exposure statistics, measured on the previous frame's output so
        #    there is nothing to synchronise.
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

        # 5. The safety stage. Reads the previous output as history and writes
        #    the new one, so the ping-pong pair is also the history buffer.
        cpass.set_pipeline(self.p_safety)
        cpass.set_bind_group(0, self._bind(self.p_safety, [
            render_bind, self.hdr_view, self.final.cur,
            self.layers[0].velocity_view, self.final.nxt, sampler, stats_bind,
        ]))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))
        cpass.end()
        self.final.flip()

        # 6. Present with blue-noise dithering.
        pipeline = self._blit_pipeline(target_format)
        rpass = encoder.begin_render_pass(color_attachments=[{
            "view": target_view,
            "load_op": wgpu.LoadOp.clear,
            "store_op": wgpu.StoreOp.store,
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        }])
        rpass.set_pipeline(pipeline)
        rpass.set_bind_group(0, self._bind(
            pipeline, [render_bind, self.final.cur, self.noise_view, sampler]))
        rpass.draw(3, 1, 0, 0)
        rpass.end()

        self.device.queue.submit([encoder.finish()])
        self.frame_count += 1

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
