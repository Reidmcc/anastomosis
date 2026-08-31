"""The layered 2.5D backend: its resources, pipelines, and tick/render sequence.

Everything below the compositor -- the exposure governor, the flash-safety
stage, the dither and present, the parameter mapping -- lives in
:mod:`anastomosis.backend`, which the volumetric slab
(:mod:`anastomosis.volume`) shares. What is here is the layered depth backend
itself: three independent 2D simulations at different scales and tempos,
composited back to front with Beer-Lambert transmittance, parallax, depth of
field and atmosphere (DESIGN.md §5).

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
from dataclasses import dataclass

import numpy as np
import wgpu

from . import gpu_params
from .backend import (
    FIELD_USAGE,
    TEX_FORMAT,
    Backend,
    PingPong,
    aspect_correction,
    plausible,
    round_up,
)
from .config import Params

log = logging.getLogger(__name__)

MAX_LAYERS = 4
AGENT_STRIDE = 24  # vec2 pos, f32 heading, u32 rng, f32 recent, f32 age

# A ceiling for an agent count that arrives from outside -- i.e. out of a
# checkpoint file, which decides how much memory a launch allocates before
# anything has validated it. Far above any density the config can ask for
# (0.44/cell); it only needs to catch nonsense.
MAX_AGENTS_PER_CELL = 8


def layer_cells(
    width: int, height: int, scale: float, count: int, falloff: float
) -> int:
    """Total simulation cells the stack would hold at this scale.

    The same rounding :meth:`Geometry.derive` does, because a ceiling checked
    against an estimate is a ceiling that is wrong by exactly the amount the
    rounding adds -- and the rounding is not small on the back layers, where
    the floors (64x32) do most of the deciding.
    """
    total = 0
    for i in range(count):
        shrink = falloff**i
        total += (
            round_up(max(int(width * scale * shrink), 64), 32)
            * max(int(height * scale * shrink), 32)
        )
    return total


def fit_cell_budget(
    width: int,
    height: int,
    scale: float,
    count: int,
    falloff: float,
    budget: int,
) -> float:
    """Shrink ``scale`` until the stack fits ``budget`` cells. DESIGN.md §8.3.

    Returns ``scale`` unchanged when there is no ceiling (``budget <= 0``) or
    when the stack already fits, which is every case on the target card of
    §8.1 and most cases anywhere else -- the ceiling is meant to be invisible
    until the window gets big.

    It only ever shrinks. A ceiling that *raised* the resolution of a small
    window would be spending an integrated GPU's bandwidth on detail nobody
    asked for, and `base_scale` would have stopped meaning what it says.

    The first guess is closed form -- cells go as the square of the scale, so
    the scale that fits is the square root of the ratio -- and then it walks
    down, because the rounding and the per-layer floors mean the closed form
    is an estimate rather than an answer. The walk is bounded and stops early
    once shrinking has stopped removing cells, which is what happens when
    every layer has hit its floor and the ceiling simply cannot be met.
    """
    if budget <= 0:
        return scale
    cells = layer_cells(width, height, scale, count, falloff)
    if cells <= budget:
        return scale

    area = max(width * height, 1)
    spread = sum((falloff**i) ** 2 for i in range(count)) or 1.0
    scale = min(scale, math.sqrt(budget / (area * spread)))

    for _ in range(64):
        cells = layer_cells(width, height, scale, count, falloff)
        if cells <= budget:
            break
        smaller = scale * 0.97
        if layer_cells(width, height, smaller, count, falloff) >= cells:
            # Every layer is on its floor; shrinking further buys nothing and
            # the ceiling is simply unreachable at this layer count.
            break
        scale = smaller
    return scale


@dataclass(frozen=True)
class LayerGeometry:
    """The allocation sizes of one layer.

    Kept apart from the rest of :class:`LayerSpec` because these are the numbers
    a saved field is *made of*: every texture and buffer in a checkpoint has one
    of these shapes, so they have to be settable from a checkpoint rather than
    always re-derived from the window (see :class:`Geometry`).
    """

    index: int
    width: int
    height: int
    agent_count: int
    psi_width: int
    psi_height: int
    climate_width: int
    climate_height: int


@dataclass(frozen=True)
class LayerSpec:
    """A layer's geometry, plus the scales its depth in the stack implies.

    The two halves have different lifetimes. The geometry is fixed for as long
    as the field exists, because changing it would mean reallocating the state.
    The scales are pure configuration, re-derived on every launch, so resuming a
    field never pins the look it is rendered with.
    """

    geometry: LayerGeometry
    feature_scale: float
    tempo_scale: float
    depth: float  # 0 = front, 1 = backmost

    @property
    def index(self) -> int:
        return self.geometry.index

    @property
    def width(self) -> int:
        return self.geometry.width

    @property
    def height(self) -> int:
        return self.geometry.height

    @property
    def agent_count(self) -> int:
        return self.geometry.agent_count

    @property
    def psi_dims(self) -> tuple[int, int]:
        return (self.geometry.psi_width, self.geometry.psi_height)

    @property
    def climate_dims(self) -> tuple[int, int]:
        return (self.geometry.climate_width, self.geometry.climate_height)


@dataclass(frozen=True)
class Geometry:
    """Every size the simulation's accumulated state is made of.

    Deliberately *not* the window size. The presentation follows the window
    (:meth:`Engine.resize`); the simulation keeps the resolution it was grown at.
    Making that a value an engine can be *given* rather than one it always
    derives is what lets a saved field be resumed into a window of any size: the
    launch reads what geometry the checkpoint needs and builds that, instead of
    asking whether the checkpoint happens to fit the window it found.

    Which means the config's structural values -- layer count, base scale,
    agent density, climate and psi sizes -- take effect when a *new* field is
    grown, not while an old one is being carried forward.
    """

    sim_width: int
    sim_height: int
    layers: tuple[LayerGeometry, ...]

    @classmethod
    def derive(cls, width: int, height: int, params: Params) -> "Geometry":
        """The geometry a fresh field of this size and configuration would have."""
        render = params.render
        count = max(1, min(render.layers, MAX_LAYERS))
        scale = fit_cell_budget(
            width, height, render.base_scale, count,
            render.scale_falloff, int(render.cell_budget),
        )
        layers = []
        for i in range(count):
            shrink = render.scale_falloff**i
            w = round_up(max(int(width * scale * shrink), 64), 32)
            h = max(int(height * scale * shrink), 32)
            layers.append(LayerGeometry(
                index=i,
                width=w,
                height=h,
                agent_count=round_up(int(params.agents.density * w * h), 64),
                # Rounded to 32 for the same reason the layers are: a texture row
                # has to be a multiple of 256 bytes.
                psi_width=round_up(max(w // params.flow.psi_scale, 32), 32),
                psi_height=max(h // params.flow.psi_scale, 8),
                climate_width=params.climate.width,
                climate_height=params.climate.height,
            ))
        return cls(sim_width=width, sim_height=height, layers=tuple(layers))

    def specs(self, params: Params) -> list[LayerSpec]:
        """These shapes, dressed with the scales the current config asks for."""
        render = params.render
        count = len(self.layers)
        return [
            LayerSpec(
                geometry=geometry,
                feature_scale=render.feature_falloff**i,
                tempo_scale=render.tempo_falloff**i,
                depth=i / max(count - 1, 1),
            )
            for i, geometry in enumerate(self.layers)
        ]

    def problems(self) -> list[str]:
        """Reasons an engine could not be built at this geometry.

        Geometry that came from a file is untrusted input, and it decides how
        much GPU memory a launch tries to allocate, so it is bounds-checked
        before anything is created from it.
        """
        if not 1 <= len(self.layers) <= MAX_LAYERS:
            return [f"{len(self.layers)} layers, expected 1 to {MAX_LAYERS}"]

        problems: list[str] = []
        if not (plausible(self.sim_width) and plausible(self.sim_height)):
            problems.append(
                f"simulation size {self.sim_width}x{self.sim_height}"
            )
        for i, layer in enumerate(self.layers):
            if layer.index != i:
                problems.append(f"layer {i} is indexed {layer.index}")
            for label, w, h in (
                ("size", layer.width, layer.height),
                ("psi size", layer.psi_width, layer.psi_height),
                ("climate size", layer.climate_width, layer.climate_height),
            ):
                if not (plausible(w) and plausible(h)):
                    problems.append(f"layer {i} {label} {w}x{h}")
            ceiling = MAX_AGENTS_PER_CELL * max(layer.width * layer.height, 0)
            if not 0 <= layer.agent_count <= ceiling:
                problems.append(f"layer {i} has {layer.agent_count} agents")
        return problems

    def differences(self, other: "Geometry") -> list[str]:
        """Human-readable list of where two geometries disagree."""
        differences: list[str] = []
        mine = (self.sim_width, self.sim_height)
        theirs = (other.sim_width, other.sim_height)
        if mine != theirs:
            differences.append(
                f"simulation {mine[0]}x{mine[1]} != {theirs[0]}x{theirs[1]}"
            )
        if len(self.layers) != len(other.layers):
            differences.append(
                f"{len(self.layers)} layers != {len(other.layers)}"
            )
            return differences
        for a, b in zip(self.layers, other.layers):
            for label, left, right in (
                ("size", (a.width, a.height), (b.width, b.height)),
                ("agent count", a.agent_count, b.agent_count),
                ("psi size", (a.psi_width, a.psi_height),
                 (b.psi_width, b.psi_height)),
                ("climate size", (a.climate_width, a.climate_height),
                 (b.climate_width, b.climate_height)),
            ):
                if left != right:
                    differences.append(f"layer {a.index} {label} {left} != {right}")
        return differences

    def describe(self) -> str:
        """One phrase, for the log line that explains which shape won."""
        layers = ", ".join(f"{l.width}x{l.height}" for l in self.layers)
        return f"{self.sim_width}x{self.sim_height} in {len(self.layers)} layers ({layers})"


class Layer:
    """Per-layer GPU resources."""

    def __init__(self, device: wgpu.GPUDevice, spec: LayerSpec, params: Params):
        self.device = device
        self.spec = spec
        w, h = spec.width, spec.height
        climate_w, climate_h = spec.climate_dims

        self.trail = PingPong(device, w, h, f"trail{spec.index}")
        self.reaction = PingPong(device, w, h, f"reaction{spec.index}")
        self.pigment = PingPong(device, w, h, f"pigment{spec.index}")
        self.climate_a = PingPong(device, climate_w, climate_h,
                                  f"climate_a{spec.index}")
        self.climate_b = PingPong(device, climate_w, climate_h,
                                  f"climate_b{spec.index}")
        # Morphology: (scale, prune, repel, spare). DESIGN.md §4.7. The other
        # two pairs are fully allocated, and a 64x36 rgba16f pair is ~9 KB.
        # Sized from the layer geometry, like its two siblings: the climate pass
        # dispatches over that, and a resumed engine is built at the geometry the
        # saved field needs rather than at whatever the live config now says.
        self.climate_c = PingPong(device, climate_w, climate_h,
                                  f"climate_c{spec.index}")

        self.psi = PingPong(device, *spec.psi_dims, f"psi{spec.index}")

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

        # COPY_SRC so the agent array can be read back for a checkpoint; the
        # deposit accumulator needs no such thing, since the trail pass drains it
        # every tick and it is therefore always zero between ticks.
        self.agents_buf = device.create_buffer(
            size=max(spec.agent_count, 1) * AGENT_STRIDE,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
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

        # From the layer's own geometry, not the live config: the two can differ
        # in a session resumed into the shape a saved field needs.
        cw, ch = self.spec.climate_dims
        for pair in (self.climate_a, self.climate_b, self.climate_c):
            for tex in pair.textures:
                upload(tex, rng.normal(0.0, 0.3, (ch, cw, 4)).clip(-1, 1))

        pw, ph = self.spec.psi_dims
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


class Engine(Backend):
    """The layered 2.5D backend: owns the device-side world and drives it."""

    name = "layered"

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        params: Params,
        seed: int | None = None,
        geometry: Geometry | None = None,
    ) -> None:
        # `width`/`height` are the *output* size and follow the window. The
        # simulation's own geometry is separate, and is passed in when the field
        # being loaded into this engine was grown at a different one -- a
        # checkpoint taken in another window, or under another config.
        self.geometry = (
            geometry if geometry is not None
            else Geometry.derive(width, height, params)
        )
        self.sim_width = self.geometry.sim_width
        self.sim_height = self.geometry.sim_height

        super().__init__(device, width, height, seed=seed)

        self.layers = [
            Layer(device, spec, params) for spec in self.geometry.specs(params)
        ]

        self.layer_buf = device.create_buffer(
            size=gpu_params.LAYER_DTYPE.itemsize * MAX_LAYERS,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="layer_data")

        self._build_pipelines()

        log.info(
            "engine ready: output %dx%d, simulation %s, %d agents total",
            width, height, self.geometry.describe(),
            sum(l.spec.agent_count for l in self.layers),
        )

    # -- setup --------------------------------------------------------------

    def _build_pipelines(self) -> None:
        """The layered backend's own passes. The output chain is on `Backend`."""
        self.p_psi = self._compute("psi.wgsl")
        self.p_climate = self._compute("climate.wgsl")
        self.p_agents = self._compute("agents.wgsl")
        self.p_trail = self._compute("trail.wgsl")
        self.p_reaction = self._compute("reaction.wgsl")
        self.p_flow = self._compute("flow.wgsl")
        self.p_advect = self._compute("advect.wgsl")
        self.p_reduce = self._compute("reduce.wgsl", "reduce_tiles")
        self.p_sanitize = self._compute("sanitize.wgsl")
        self.p_interp = self._compute("interp.wgsl")
        self.p_composite = self._compute("composite.wgsl")

    # -- parameter packing --------------------------------------------------

    def _sim_values(self, layer: Layer, params: Params, event_count: int) -> dict:
        """This layer's shape and identity, plus the shared physical parameters.

        A cell on a half-resolution layer covers twice the screen distance, so
        cell-space speeds and lengths are scaled by the resolution ratio as
        well as by the intended screen-relative tempo; that is what
        `feature_scale` and `tempo_scale` carry into `_physics_values`.

        The grid sizes come from the layer, never from the live params: those
        can be hot-reloaded mid-session, and the textures they describe were
        allocated once, when the field was created.
        """
        spec = layer.spec
        values = self._physics_values(
            params, spec.feature_scale, spec.tempo_scale, params.agents.density)
        values.update(
            dims_x=spec.width, dims_y=spec.height, dims_z=1,
            clim_w=spec.climate_dims[0], clim_h=spec.climate_dims[1], clim_d=1,
            psi_w=spec.psi_dims[0], psi_h=spec.psi_dims[1], psi_d=1,
            tick=self.tick_count,
            seed=(self.seed ^ (spec.index * 0x9E3779B9)) & 0xFFFFFFFF,
            agent_count=spec.agent_count,
            layer_index=spec.index, layer_count=len(self.layers),
            event_count=event_count,
        )
        return values

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

    def tick(self, params: Params, event_rows: list[dict] | None = None) -> None:
        """Advance the simulation by one tick."""
        event_rows = event_rows or []
        event_count = min(len(event_rows), 8)

        self._advance_hue_and_walk(params, 1.0 / max(params.sim_hz, 1e-3))

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
            cpass.dispatch_workgroups(*self._groups(*spec.psi_dims))
            layer.psi.flip()

            # 2. Climate.
            cpass.set_pipeline(self.p_climate)
            cpass.set_bind_group(0, self._bind(self.p_climate, [
                pbind, layer.climate_a.cur, layer.climate_b.cur, layer.climate_c.cur,
                layer.climate_a.nxt, layer.climate_b.nxt, layer.climate_c.nxt,
                layer.psi.cur, sampler, events_bind,
            ]))
            cpass.dispatch_workgroups(*self._groups(*spec.climate_dims))
            layer.climate_a.flip()
            layer.climate_b.flip()
            layer.climate_c.flip()

            # 3. Velocity field. Before the trail rather than after the
            # reaction, because the trail now advects through it (DESIGN.md
            # 4.7 step 6) and `velocity` must stay a derived field -- written
            # every tick before anything reads it -- or it would have to be
            # checkpointed. The structure-following component consequently
            # reads the previous tick's reaction, which at these tempos is the
            # same field to within one diffusion step.
            cpass.set_pipeline(self.p_flow)
            cpass.set_bind_group(0, self._bind(self.p_flow, [
                pbind, layer.psi.cur, layer.reaction.cur, layer.climate_b.cur,
                layer.velocity_view, sampler,
            ]))
            cpass.dispatch_workgroups(gx, gy)

            # 4. Agents deposit into the fixed-point accumulator.
            cpass.set_pipeline(self.p_agents)
            cpass.set_bind_group(0, self._bind(self.p_agents, [
                pbind, agents_bind, layer.trail.cur, deposit_bind,
                layer.climate_a.cur, layer.climate_b.cur, sampler, stats_bind,
                layer.climate_c.cur,
            ]))
            cpass.dispatch_workgroups(math.ceil(spec.agent_count / 64), 1, 1)

            # 5. Trail advection, decay and deposit, then separable diffusion.
            cpass.set_pipeline(self.p_trail)
            cpass.set_bind_group(0, self._bind(self.p_trail, [
                pbind, layer.trail.cur, layer.trail.nxt, deposit_bind,
                layer.climate_b.cur, layer.climate_c.cur, sampler, stats_bind,
                layer.velocity_view,
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

            # 6. Reaction-diffusion substeps.
            cpass.set_pipeline(self.p_reaction)
            for _ in range(max(1, params.reaction.substeps)):
                cpass.set_bind_group(0, self._bind(self.p_reaction, [
                    pbind, layer.reaction.cur, layer.reaction.nxt,
                    layer.trail.cur, layer.climate_a.cur, layer.climate_c.cur,
                    sampler, stats_bind,
                ]))
                cpass.dispatch_workgroups(gx, gy)
                layer.reaction.flip()

            # 7. Advect pigment, through the velocity written in step 3.
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
        self._submit_tick(encoder)
        self.tick_count += 1

    # -- rendering ----------------------------------------------------------

    def _write_render_params(
        self, params: Params, frac: float, frame_dt: float
    ) -> None:
        self._update_parallax(params, frame_dt, len(self.layers))

        render = params.render
        values = self._common_render_values(params, frac, frame_dt)
        values["layer_count"] = len(self.layers)
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

    def _compose(self, cpass, params: Params, frac: float):
        """Interpolate, blur for depth of field, and composite back to front.

        Returns the view the safety stage reprojects its history through: the
        front layer's velocity field, which is what most of the image is
        actually made of.
        """
        sampler = self.sampler
        render_bind = self._buffer_binding(self.render_buf)
        layer_bind = self._buffer_binding(self.layer_buf)

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

        return self.layers[0].velocity_view
