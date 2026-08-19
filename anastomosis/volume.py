"""The volumetric slab backend -- DESIGN.md §5.1.

A thin slab of voxels -- ``512 x 288 x 48`` at the default, about 7.1 million --
carrying the same three-system substrate the layered backend runs in the plane:
Physarum agents, a trail field, Gray-Scott reaction-diffusion, a
divergence-free flow, an advected pigment, and the climate field that varies
every one of their parameters by region. The image is a ray march through the
result rather than three sheets composited back to front.

What that buys, in §5.1's words, is "real Beer-Lambert attenuation through a
continuous medium, self-shadowing from a single soft light, and structures that
pass smoothly in front of and behind each other rather than living in three
discrete sheets". What it costs is lateral resolution -- 512 voxels across
where the layered front sheet has 2560 -- and that trade is the whole reason
the layered path was built first and is still the default.

Two things about the shape of this module are worth stating up front.

**The slab is a 3-torus, not a box.** Nothing here has a boundary case, because
there is no boundary: x, y and z all wrap, exactly as x and y do under the
layered backend, and for the same reason -- a reflecting or no-flux face
accumulates material against itself over a run measured in days. The single
consequence is paid in the renderer, where a smooth window over the two visible
faces turns material's arrival and departure into a fade instead of a step.

**Nothing below the compositor is duplicated.** The exposure governor, the
flash-safety stage, the dither, the present, and the mapping from macros to
shader parameters all live on :class:`~anastomosis.backend.Backend` and are the
same objects the layered backend uses. §5.1 says the output stages are
unchanged between backends and that this should be "a clean swap rather than a
fork"; the shared base class is what keeps that true as both sides change.

Memory is the one place the slab is materially more expensive. At the default
size it holds about 650 MB of ``rgba16float`` against roughly 90 MB for the
1440p layered stack -- comfortable on the target card (§8.1), and
``volume.width`` is there for anything smaller.

**Thickness is a knob, and it is the only geometry the panel moves.**
Forty-eight voxels is a slab a ray crosses one or two filaments of, which is
about as subtle as genuine volume gets, so ``volume.depth`` runs from 8 up to
the shorter of the two lateral axes -- 288 on a 16:9 display -- and
:meth:`VolumeGeometry.field_bytes` says what each setting costs, since the cost
is linear in it and reaches about 3.9 GB at the top. Everything the march is
calibrated with is expressed in *voxels* rather than as a fraction of the
thickness so that moving the knob changes how much material a ray passes
through and nothing else: the face window, the shadow reach, and the extinction
coefficient's filament width are all lengths, and the step count follows the
slice count up to ``volume.steps``.
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
    MAX_DIM_3D,
    TEX_FORMAT,
    Backend,
    PingPong,
    aspect_correction,
    plausible,
    round_up,
)
from .config import Params

log = logging.getLogger(__name__)

# vec3 pos, vec3 heading, u32 rng, f32 recent, f32 age -- nine 4-byte scalars.
# Written as scalars rather than as vec3s on purpose: std430 gives a vec3 an
# alignment of 16, which would pad this to 48 bytes and put two silent holes in
# a buffer the host also writes.
AGENT_STRIDE = 36

# A ceiling for an agent count that arrives from outside, i.e. out of a
# checkpoint file. Far above any density the config can ask for; it only needs
# to catch nonsense.
MAX_AGENTS_PER_VOXEL = 4

# The thinnest slab worth building, and a multiple of the 8 the depth axis
# rounds to. Below this the trail's diffusion kernel is wider than the axis it
# is separable along, and what is left is the layered backend with a great deal
# of extra work.
MIN_DEPTH = 8

# The steepest the camera is allowed to look through the slab, as a tangent:
# 0.8 is about 39 degrees off the slab's normal.
#
# Parallax is thickness times the tangent of the viewing angle, which is the
# geometric fact underneath every complaint that a thin slab looks flat -- 48
# voxels against 512 is a sheet of paper, and there is only so much depth to be
# had by walking around a sheet of paper. Swinging the viewpoint further does
# not buy more depth: it buys the same slab seen edge-on, with the march
# tracing a long oblique smear through it and the near and far faces sliding
# past each other faster than anything in between can explain. So the drift's
# amplitude is held to what the current thickness justifies, which is also what
# makes the thickness knob and the parallax knob compound rather than compete.
PARALLAX_MAX_TANGENT = 0.8

# What one voxel of slab costs on the card: eleven `rgba16float` volumes --
# three ping-pong pairs and five singles, see `Slab` -- plus the u32 deposit
# accumulator. Thickness is paid for exactly here, linearly, which is why the
# control panel quotes it.
BYTES_PER_VOXEL = 11 * 8 + 4


@dataclass(frozen=True)
class VolumeGeometry:
    """Every size the slab's accumulated state is made of.

    The counterpart of :class:`anastomosis.engine.Geometry`, and it exists for
    the same reason: these are the numbers a saved field is *made of*, so a
    launch that finds a checkpoint builds itself in the shape that field needs
    rather than asking whether the field happens to fit the window.
    """

    width: int
    height: int
    depth: int
    agent_count: int
    psi_width: int
    psi_height: int
    psi_depth: int
    climate_width: int
    climate_height: int
    climate_depth: int

    # Named to match `Geometry`, so the aspect correction and the resize log
    # do not have to know which backend they are looking at.
    @property
    def sim_width(self) -> int:
        return self.width

    @property
    def sim_height(self) -> int:
        return self.height

    @property
    def voxels(self) -> int:
        return self.width * self.height * self.depth

    @property
    def max_depth(self) -> int:
        """The thickest this slab is allowed to be, at this lateral shape.

        Both lateral axes are multiples of eight, so this is a thickness the
        depth axis can actually be built at as well as one it is allowed.
        """
        return min(self.width, self.height)

    @property
    def field_bytes(self) -> int:
        """Roughly what a field of this shape occupies on the card.

        Quoted by the control panel beside the thickness slider, because memory
        is what thickness costs and a user moving that slider is entitled to
        know the price before paying it rather than by watching the card run
        out. The full-slab volumes dominate; the agent buffer is counted
        because the agent count follows the voxel count, and the psi and
        climate grids are counted because they are cheap to count, not because
        they matter.
        """
        psi_voxels = self.psi_width * self.psi_height * self.psi_depth
        climate_voxels = (
            self.climate_width * self.climate_height * self.climate_depth)
        return (
            BYTES_PER_VOXEL * self.voxels
            + AGENT_STRIDE * self.agent_count
            + 8 * (2 * psi_voxels + 6 * climate_voxels)
        )

    @classmethod
    def derive(cls, width: int, height: int, params: Params) -> "VolumeGeometry":
        """The slab a fresh field of this configuration would have.

        The lateral width comes from the config; the height follows the
        *window's* aspect so the voxels stay cubic, which is what lets the
        renderer treat the slab as a box of world extent (1, h/w, d/w) and the
        simulation treat every axis alike. A 16:9 display therefore gets
        exactly the 512 x 288 x 48 of §5.1 at the default thickness.

        The thickness is capped at the shorter lateral axis. That is the bound
        the control panel offers, and it is applied here rather than there so
        that a hand-edited config cannot ask for something outside it either:
        past that point the shape is not a slab, and the extra voxels have in
        any case stopped arriving on screen -- a ray that already crosses that
        much material is opaque long before the far face.
        """
        vol = params.volume
        w = round_up(max(int(vol.width * vol.base_scale), 64), 32)
        aspect = height / max(width, 1)
        # Rounded to 8 so the 4-deep workgroups tile it without a large
        # remainder; the depth axis gets the same treatment.
        h = max(round_up(int(w * aspect), 8), 32)
        d = max(round_up(int(vol.depth * vol.base_scale), 8), MIN_DEPTH)
        d = min(d, min(w, h))

        psi_w = max(w // vol.psi_scale, 8)
        psi_h = max(h // vol.psi_scale, 8)
        psi_d = max(d // vol.psi_scale, 4)

        return cls(
            width=w,
            height=h,
            depth=d,
            agent_count=round_up(int(vol.density * w * h * d), 64),
            psi_width=psi_w,
            psi_height=psi_h,
            psi_depth=psi_d,
            climate_width=vol.climate_width,
            climate_height=vol.climate_height,
            climate_depth=vol.climate_depth,
        )

    @property
    def dims(self) -> tuple[int, int, int]:
        return (self.width, self.height, self.depth)

    @property
    def psi_dims(self) -> tuple[int, int, int]:
        return (self.psi_width, self.psi_height, self.psi_depth)

    @property
    def climate_dims(self) -> tuple[int, int, int]:
        return (self.climate_width, self.climate_height, self.climate_depth)

    def problems(self) -> list[str]:
        """Reasons an engine could not be built at this geometry.

        Geometry that came from a file is untrusted input and decides how much
        GPU memory a launch tries to allocate, so it is bounds-checked before
        anything is created from it. The ceiling is core WebGPU's guaranteed
        ``maxTextureDimension3D``, which is a quarter of the 2D one -- worth
        knowing, since a slab is easier to ask too much of than a sheet.
        """
        problems: list[str] = []
        for label, dims in (
            ("size", self.dims),
            ("psi size", self.psi_dims),
            ("climate size", self.climate_dims),
        ):
            if not all(plausible(v, MAX_DIM_3D) for v in dims):
                problems.append(
                    f"{label} {dims[0]}x{dims[1]}x{dims[2]}")
        ceiling = MAX_AGENTS_PER_VOXEL * max(self.voxels, 0)
        if not 0 <= self.agent_count <= ceiling:
            problems.append(f"{self.agent_count} agents")
        return problems

    def differences(self, other: "VolumeGeometry") -> list[str]:
        """Human-readable list of where two geometries disagree."""
        differences: list[str] = []
        for label, mine, theirs in (
            ("slab", self.dims, other.dims),
            ("agent count", self.agent_count, other.agent_count),
            ("psi size", self.psi_dims, other.psi_dims),
            ("climate size", self.climate_dims, other.climate_dims),
        ):
            if mine != theirs:
                differences.append(f"{label} {mine} != {theirs}")
        return differences

    def describe(self) -> str:
        """One phrase, for the log line that explains which shape won."""
        return (
            f"{self.width}x{self.height}x{self.depth} voxels "
            f"({self.voxels / 1e6:.1f} M)"
        )


def depth_limits(width: int, height: int, params: Params) -> tuple[int, int]:
    """The thinnest and thickest slab this window and configuration allow.

    The ceiling is a property of the lateral shape, which the thickness does
    not enter into, so this can be asked before deciding what to ask for -- it
    is what the control panel's slider spans.
    """
    geometry = VolumeGeometry.derive(width, height, params)
    return MIN_DEPTH, geometry.max_depth


class Slab:
    """The slab's GPU resources.

    Held in one object rather than spread over the engine for the same reason
    :class:`anastomosis.engine.Layer` is: the checkpoint reads and writes
    exactly this set, and a field it did not know about would be silently lost
    across a restart.
    """

    def __init__(
        self, device: wgpu.GPUDevice, geometry: VolumeGeometry, params: Params
    ) -> None:
        self.device = device
        self.geometry = geometry
        w, h, d = geometry.dims

        self.trail = PingPong(device, w, h, "vol_trail", depth=d)
        self.reaction = PingPong(device, w, h, "vol_reaction", depth=d)
        self.pigment = PingPong(device, w, h, "vol_pigment", depth=d)

        cw, ch, cd = geometry.climate_dims
        self.climate_a = PingPong(device, cw, ch, "vol_climate_a", depth=cd)
        self.climate_b = PingPong(device, cw, ch, "vol_climate_b", depth=cd)
        self.climate_c = PingPong(device, cw, ch, "vol_climate_c", depth=cd)

        pw, ph, pd = geometry.psi_dims
        self.psi = PingPong(device, pw, ph, "vol_psi", depth=pd)

        def plain(label: str) -> wgpu.GPUTexture:
            return device.create_texture(
                size=(w, h, d), dimension="3d", format=TEX_FORMAT,
                usage=FIELD_USAGE, label=label,
            )

        # The assembled vector potential, and its curl. Two textures rather
        # than one pass, because a divergence-free velocity in three dimensions
        # is the curl of a stored field -- see vol_potential.wgsl.
        self.potential = plain("vol_potential")
        self.potential_view = self.potential.create_view()
        self.velocity = plain("vol_velocity")
        self.velocity_view = self.velocity.create_view()
        self.scratch = plain("vol_scratch")
        self.scratch_view = self.scratch.create_view()
        # Previous-tick reaction state, kept so the activity channel can
        # measure |dV/dt| without the substep loop clobbering it.
        self.reaction_prev = plain("vol_reaction_prev")
        self.reaction_prev_view = self.reaction_prev.create_view()
        # Render-side: the interpolated pigment the march reads.
        self.interp = plain("vol_interp")
        self.interp_view = self.interp.create_view()

        # --- Buffers -------------------------------------------------------
        self.params_buf = device.create_buffer(
            size=gpu_params.SIM_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="vol_sim_params",
        )
        # One buffer per blur axis: a buffer write is a queue operation, so a
        # single buffer cannot carry three directions into one submission.
        self.blur_bufs = {
            axis: device.create_buffer(
                size=gpu_params.SIM_DTYPE.itemsize,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                label=f"vol_blur_{axis}",
            )
            for axis in ("x", "y", "z")
        }
        self.sanitize_bufs = {
            name: device.create_buffer(
                size=gpu_params.SIM_DTYPE.itemsize,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                label=f"vol_sanitize_{name}",
            )
            for name in ("trail", "reaction", "pigment")
        }

        # COPY_SRC so the agent array can be read back for a checkpoint; the
        # deposit accumulator needs no such thing, since the trail pass drains
        # it every tick and it is therefore always zero between ticks.
        self.agents_buf = device.create_buffer(
            size=max(geometry.agent_count, 1) * AGENT_STRIDE,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
            label="vol_agents",
        )
        self.deposit_buf = device.create_buffer(
            size=w * h * d * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="vol_deposit",
        )

        tiles = (math.ceil(w / 4), math.ceil(h / 4), math.ceil(d / 16))
        self.partials_buf = device.create_buffer(
            size=tiles[0] * tiles[1] * tiles[2] * gpu_params.PARTIAL_SIZE,
            usage=wgpu.BufferUsage.STORAGE,
            label="vol_partials",
        )
        self.tiles = tiles

        self._seed_state(params)

    # -- initial state ------------------------------------------------------

    def _seed_state(self, params: Params) -> None:
        device = self.device
        w, h, d = self.geometry.dims
        rng = np.random.default_rng(0x5EED ^ 0x3D)

        def upload(texture: wgpu.GPUTexture, array: np.ndarray) -> None:
            data = np.ascontiguousarray(array, dtype=np.float16)
            depth, height, width = data.shape[0], data.shape[1], data.shape[2]
            device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                data,
                {"offset": 0, "bytes_per_row": width * 8,
                 "rows_per_image": height},
                (width, height, depth),
            )

        # Reaction starts as U=1, V=0 with scattered balls of V, which is what
        # gives Gray-Scott something to nucleate from. Built in float16 rather
        # than float32 and converted: at the default slab one float32 copy of
        # this array is 113 MB, and there is no reason to hold two.
        reaction = np.zeros((d, h, w, 4), dtype=np.float16)
        reaction[..., 0] = 1.0
        seeds = max(8, (w * h * d) // 60_000)
        for _ in range(seeds):
            cx, cy, cz = (rng.integers(0, w), rng.integers(0, h),
                          rng.integers(0, d))
            radius = int(rng.integers(3, 8))
            zs, ys, xs = np.ogrid[
                -radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
            mask = xs * xs + ys * ys + zs * zs <= radius * radius
            patch = np.ix_(
                np.arange(cz - radius, cz + radius + 1) % d,
                np.arange(cy - radius, cy + radius + 1) % h,
                np.arange(cx - radius, cx + radius + 1) % w,
            )
            reaction[..., 1][patch] = np.where(
                mask, np.float16(0.45), reaction[..., 1][patch])
            reaction[..., 0][patch] = np.where(
                mask, np.float16(0.35), reaction[..., 0][patch])
        for tex in self.reaction.textures:
            upload(tex, reaction)
        upload(self.reaction_prev, reaction)
        del reaction

        zeros = np.zeros((d, h, w, 4), dtype=np.float16)
        for tex in self.trail.textures:
            upload(tex, zeros)
        upload(self.velocity, zeros)
        upload(self.potential, zeros)
        upload(self.scratch, zeros)

        # Pigment: hue must start as a valid unit vector or the first frames
        # would carry a degenerate (0, 0) hue. The zero array is reused rather
        # than a second one allocated -- everything that wanted it zeroed has
        # already been uploaded.
        zeros[..., 1] = 1.0
        for tex in self.pigment.textures:
            upload(tex, zeros)
        upload(self.interp, zeros)
        del zeros

        cw, ch, cd = self.geometry.climate_dims
        for pair in (self.climate_a, self.climate_b, self.climate_c):
            for tex in pair.textures:
                upload(tex, rng.normal(0.0, 0.3, (cd, ch, cw, 4)).clip(-1, 1))

        pw, ph, pd = self.geometry.psi_dims
        psi = np.zeros((pd, ph, pw, 4), dtype=np.float32)
        psi[..., :3] = rng.normal(0.0, 0.5, (pd, ph, pw, 3))
        for tex in self.psi.textures:
            upload(tex, psi)

        # Agents: uniform positions, isotropic headings flattened toward the
        # plane exactly as the shader will keep them.
        count = max(self.geometry.agent_count, 1)
        agents = np.zeros(count, dtype=np.dtype([
            ("px", np.float32), ("py", np.float32), ("pz", np.float32),
            ("dx", np.float32), ("dy", np.float32), ("dz", np.float32),
            ("rng", np.uint32), ("recent", np.float32), ("age", np.float32),
        ]))
        agents["px"] = rng.random(count).astype(np.float32) * w
        agents["py"] = rng.random(count).astype(np.float32) * h
        agents["pz"] = rng.random(count).astype(np.float32) * d
        heading = rng.normal(0.0, 1.0, (count, 3))
        heading[:, 2] *= max(min(params.volume.depth_agent, 1.0), 0.0)
        heading /= np.maximum(
            np.linalg.norm(heading, axis=1, keepdims=True), 1e-6)
        agents["dx"] = heading[:, 0].astype(np.float32)
        agents["dy"] = heading[:, 1].astype(np.float32)
        agents["dz"] = heading[:, 2].astype(np.float32)
        agents["rng"] = rng.integers(1, 2**32, count, dtype=np.uint64).astype(np.uint32)
        agents["recent"] = params.agents.starve_threshold * 4.0
        agents["age"] = rng.random(count).astype(np.float32) * params.agents.max_age
        device.queue.write_buffer(self.agents_buf, 0, agents.tobytes())

        device.queue.write_buffer(
            self.deposit_buf, 0, np.zeros(w * h * d, dtype=np.uint32).tobytes()
        )


class VolumeEngine(Backend):
    """The volumetric slab backend: owns the device-side world and drives it."""

    name = "volumetric"

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        params: Params,
        seed: int | None = None,
        geometry: VolumeGeometry | None = None,
    ) -> None:
        # `width`/`height` are the *output* size and follow the window; the
        # slab's own geometry is separate, and is passed in when the field
        # being loaded was grown at a different one.
        self.geometry = (
            geometry if geometry is not None
            else VolumeGeometry.derive(width, height, params)
        )
        self.sim_width = self.geometry.width
        self.sim_height = self.geometry.height

        super().__init__(device, width, height, seed=seed)

        self.slab = Slab(device, self.geometry, params)
        self._build_pipelines()

        log.info(
            "volumetric engine ready: output %dx%d, slab %s, %d agents",
            width, height, self.geometry.describe(), self.geometry.agent_count,
        )

    # -- output targets -----------------------------------------------------

    def _make_output_targets(self, width: int, height: int) -> None:
        """The shared window-sized resources, plus the march's motion output.

        The safety stage reprojects its history through a velocity field
        (DESIGN.md §7). The layered backend hands it the front layer's, which
        is already a 2D texture in the plane the limiter works in; a volume has
        no such thing, so the march accumulates one along each ray -- each
        sample's lateral velocity, weighted by how much that sample contributes
        to the pixel, in output pixels per tick. Being per-pixel makes it a
        window-sized resource, which is to say part of a resize.
        """
        super()._make_output_targets(width, height)
        self.motion = self.device.create_texture(
            size=(width, height, 1), format=TEX_FORMAT, usage=FIELD_USAGE,
            label="vol_motion")
        self.motion_view = self.motion.create_view()

    def _extra_output_targets(self) -> list:
        return [self.motion, self.motion_view]

    # -- setup --------------------------------------------------------------

    def _build_pipelines(self) -> None:
        """The slab's own passes. The output chain is on `Backend`."""
        self.p_psi = self._compute("vol_psi.wgsl")
        self.p_climate = self._compute("vol_climate.wgsl")
        self.p_agents = self._compute("vol_agents.wgsl")
        self.p_trail = self._compute("vol_trail.wgsl")
        self.p_vol_blur = self._compute("vol_blur.wgsl")
        self.p_reaction = self._compute("vol_reaction.wgsl")
        self.p_potential = self._compute("vol_potential.wgsl")
        self.p_flow = self._compute("vol_flow.wgsl")
        self.p_advect = self._compute("vol_advect.wgsl")
        self.p_reduce = self._compute("vol_reduce.wgsl", "reduce_tiles")
        self.p_sanitize = self._compute("vol_sanitize.wgsl")
        self.p_interp = self._compute("vol_interp.wgsl")
        self.p_raymarch = self._compute("raymarch.wgsl")

    # -- parameter packing --------------------------------------------------

    def _sim_values(self, params: Params, event_count: int) -> dict:
        """The slab's shape and identity, plus the shared physical parameters.

        Both scales are 1.0, and the second of those deserves a word. A voxel
        covers about five display pixels at the default slab on a 1440p
        display, so a voxel-space speed of one is five times the *absolute*
        screen speed the layered front sheet would have. It is not five times
        the perceived speed, because the features are five times larger too:
        motion at this timescale reads in feature-widths per second, and by
        that measure the two backends agree. Scaling the speed down to match in
        pixels would take the agents below the roughly one cell per tick that
        Physarum needs to lay a network down at all.
        """
        geometry = self.geometry
        values = self._physics_values(params, 1.0, 1.0)
        values.update(
            dims_x=geometry.width, dims_y=geometry.height, dims_z=geometry.depth,
            clim_w=geometry.climate_width, clim_h=geometry.climate_height,
            clim_d=geometry.climate_depth,
            psi_w=geometry.psi_width, psi_h=geometry.psi_height,
            psi_d=geometry.psi_depth,
            tick=self.tick_count,
            seed=self.seed,
            agent_count=geometry.agent_count,
            layer_index=0, layer_count=1,
            event_count=event_count,
            depth_flow=params.volume.depth_flow,
            depth_agent=params.volume.depth_agent,
        )
        return values

    def _write_sim_params(self, params: Params, event_count: int) -> None:
        slab = self.slab
        base = self._sim_values(params, event_count)
        queue = self.device.queue
        queue.write_buffer(
            slab.params_buf, 0,
            gpu_params.pack(gpu_params.SIM_DTYPE, base).tobytes())

        for axis, (dx, dy, dz) in (
            ("x", (1.0, 0.0, 0.0)),
            ("y", (0.0, 1.0, 0.0)),
            ("z", (0.0, 0.0, 1.0)),
        ):
            values = dict(base)
            values.update(
                blur_radius=base["trail_diffuse"],
                blur_dir_x=dx, blur_dir_y=dy, blur_dir_z=dz)
            queue.write_buffer(
                slab.blur_bufs[axis], 0,
                gpu_params.pack(gpu_params.SIM_DTYPE, values).tobytes())

        for name, lo, hi, fallback in (
            ("trail", 0.0, 8.0, 0.0),
            ("reaction", 0.0, 1.5, 0.0),
            ("pigment", -1.0, 4.0, 0.0),
        ):
            values = dict(base)
            values.update(sanitize_min=lo, sanitize_max=hi, sanitize_fallback=fallback)
            queue.write_buffer(
                slab.sanitize_bufs[name], 0,
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

        self._write_sim_params(params, event_count)

        slab = self.slab
        geometry = self.geometry
        sampler = self.sampler
        pbind = self._buffer_binding(slab.params_buf)
        events_bind = self._buffer_binding(self.events_buf)
        stats_bind = self._buffer_binding(self.stats_buf)
        groups = self._groups3(*geometry.dims)

        encoder = self.device.create_command_encoder(label="vol_tick")
        cpass = encoder.begin_compute_pass()

        # 1. Vector potential of the weather.
        cpass.set_pipeline(self.p_psi)
        cpass.set_bind_group(0, self._bind(
            self.p_psi, [pbind, slab.psi.cur, slab.psi.nxt]))
        cpass.dispatch_workgroups(*self._groups3(*geometry.psi_dims))
        slab.psi.flip()

        # 2. Climate.
        cpass.set_pipeline(self.p_climate)
        cpass.set_bind_group(0, self._bind(self.p_climate, [
            pbind, slab.climate_a.cur, slab.climate_b.cur, slab.climate_c.cur,
            slab.climate_a.nxt, slab.climate_b.nxt, slab.climate_c.nxt,
            slab.psi.cur, sampler, events_bind,
        ]))
        cpass.dispatch_workgroups(*self._groups3(*geometry.climate_dims))
        slab.climate_a.flip()
        slab.climate_b.flip()
        slab.climate_c.flip()

        # 3. Agents deposit into the fixed-point accumulator.
        cpass.set_pipeline(self.p_agents)
        cpass.set_bind_group(0, self._bind(self.p_agents, [
            pbind, self._buffer_binding(slab.agents_buf), slab.trail.cur,
            self._buffer_binding(slab.deposit_buf),
            slab.climate_a.cur, slab.climate_b.cur, sampler, stats_bind,
            slab.climate_c.cur,
        ]))
        cpass.dispatch_workgroups(math.ceil(geometry.agent_count / 64), 1, 1)

        # 4. Trail decay + deposit, then separable diffusion on three axes.
        cpass.set_pipeline(self.p_trail)
        cpass.set_bind_group(0, self._bind(self.p_trail, [
            pbind, slab.trail.cur, slab.trail.nxt,
            self._buffer_binding(slab.deposit_buf),
            slab.climate_b.cur, slab.climate_c.cur, sampler, stats_bind,
        ]))
        cpass.dispatch_workgroups(*groups)
        slab.trail.flip()

        # Separable diffusion: x through the ping-pong, y out to the scratch,
        # z back into the ping-pong. Three passes and one scratch texture,
        # which is what alternating between the pair's two slots buys -- a
        # naive x/y/z chain would need two.
        cpass.set_pipeline(self.p_vol_blur)
        cpass.set_bind_group(0, self._bind(self.p_vol_blur, [
            self._buffer_binding(slab.blur_bufs["x"]),
            slab.trail.cur, slab.trail.nxt, sampler,
        ]))
        cpass.dispatch_workgroups(*groups)
        slab.trail.flip()
        cpass.set_bind_group(0, self._bind(self.p_vol_blur, [
            self._buffer_binding(slab.blur_bufs["y"]),
            slab.trail.cur, slab.scratch_view, sampler,
        ]))
        cpass.dispatch_workgroups(*groups)
        cpass.set_bind_group(0, self._bind(self.p_vol_blur, [
            self._buffer_binding(slab.blur_bufs["z"]),
            slab.scratch_view, slab.trail.nxt, sampler,
        ]))
        cpass.dispatch_workgroups(*groups)
        slab.trail.flip()

        # Reaction needs the pre-step state for the activity channel, and a
        # texture copy cannot be recorded inside a compute pass.
        cpass.end()
        encoder.copy_texture_to_texture(
            {"texture": slab.reaction.textures[slab.reaction.index]},
            {"texture": slab.reaction_prev},
            geometry.dims,
        )
        cpass = encoder.begin_compute_pass()

        # 5. Reaction-diffusion substeps.
        cpass.set_pipeline(self.p_reaction)
        for _ in range(max(1, params.reaction.substeps)):
            cpass.set_bind_group(0, self._bind(self.p_reaction, [
                pbind, slab.reaction.cur, slab.reaction.nxt,
                slab.trail.cur, slab.climate_a.cur, slab.climate_c.cur,
                sampler, stats_bind,
            ]))
            cpass.dispatch_workgroups(*groups)
            slab.reaction.flip()

        # 6. Assemble the flow's vector potential, then take its curl.
        cpass.set_pipeline(self.p_potential)
        cpass.set_bind_group(0, self._bind(self.p_potential, [
            pbind, slab.psi.cur, slab.reaction.cur, slab.climate_b.cur,
            slab.potential_view, sampler,
        ]))
        cpass.dispatch_workgroups(*groups)

        cpass.set_pipeline(self.p_flow)
        cpass.set_bind_group(0, self._bind(self.p_flow, [
            pbind, slab.potential_view, slab.velocity_view,
        ]))
        cpass.dispatch_workgroups(*groups)

        # 7. Advect pigment.
        cpass.set_pipeline(self.p_advect)
        cpass.set_bind_group(0, self._bind(self.p_advect, [
            pbind, slab.pigment.cur, slab.pigment.nxt,
            slab.reaction.cur, slab.reaction_prev_view, slab.trail.cur,
            slab.velocity_view, slab.climate_b.cur, sampler,
        ]))
        cpass.dispatch_workgroups(*groups)
        slab.pigment.flip()

        # 8. Field statistics, and the controller on top of them.
        cpass.set_pipeline(self.p_reduce)
        cpass.set_bind_group(0, self._bind(self.p_reduce, [
            pbind, slab.reaction.cur, slab.reaction_prev_view,
            slab.trail.cur, self._buffer_binding(slab.partials_buf),
        ]))
        cpass.dispatch_workgroups(*slab.tiles)

        cpass.set_pipeline(self.p_homeostat)
        cpass.set_bind_group(0, self._bind(self.p_homeostat, [
            pbind, self._buffer_binding(slab.partials_buf), stats_bind,
        ]))
        cpass.dispatch_workgroups(1, 1, 1)

        # 9. Periodic sanitisation. Cheap insurance against a permanent,
        #    unrecoverable failure mode (DESIGN.md §4.4).
        if self.tick_count % 60 == 0 and self.tick_count > 0:
            cpass.set_pipeline(self.p_sanitize)
            for name, field in (
                ("trail", slab.trail), ("reaction", slab.reaction),
                ("pigment", slab.pigment),
            ):
                cpass.set_bind_group(0, self._bind(self.p_sanitize, [
                    self._buffer_binding(slab.sanitize_bufs[name]),
                    field.cur, field.nxt,
                ]))
                cpass.dispatch_workgroups(*groups)
                field.flip()

        cpass.end()
        self.device.queue.submit([encoder.finish()])
        self.tick_count += 1

    # -- rendering ----------------------------------------------------------

    def _write_render_params(
        self, params: Params, frac: float, frame_dt: float
    ) -> None:
        # One camera rather than one offset per layer: in a volume, parallax is
        # what a tilted ray does to the near and far faces, so there is exactly
        # one thing drifting.
        self._update_parallax(params, frame_dt, 1)
        drift = self._parallax[0] if self._parallax else [0.0, 0.0]

        render = params.render
        vol = params.volume
        geometry = self.geometry
        zoom_x, zoom_y = aspect_correction(
            self.width, self.height, geometry.width, geometry.height)

        # Voxels are cubic, so the slab's world thickness is just its aspect
        # against the lateral extent.
        slab_depth = geometry.depth / max(geometry.width, 1)
        # Scaled rather than clipped: clipping the drift where it ran past the
        # limit would put a corner in the viewpoint's path, and a corner in a
        # motion is exactly the punctuation nothing here is allowed to be.
        reach = min(render.parallax, PARALLAX_MAX_TANGENT * slab_depth)

        values = self._common_render_values(params, frac, frame_dt)
        values.update(
            layer_count=1,
            vol_w=geometry.width, vol_h=geometry.height, vol_d=geometry.depth,
            # One step per slice, up to the configured ceiling. The count has
            # to follow the thickness or a thick slab would be marched at the
            # same number of samples as a thin one, which is a ray stepping
            # straight over whole filaments.
            march_steps=min(vol.steps, geometry.depth),
            shadow_steps=vol.shadow_steps,
            zoom_x=zoom_x, zoom_y=zoom_y,
            cam_shear_x=drift[0] * reach,
            cam_shear_y=drift[1] * reach,
            converge=vol.converge,
            slab_depth=slab_depth,
            # A length in voxels on this side, a fraction of the depth on the
            # shader's -- which is the coordinate the march has. The shader
            # clamps it, so a window wider than the slab is a fully faded slab
            # rather than an error.
            depth_window=vol.depth_window_voxels / max(geometry.depth, 1),
            depth_dim=render.depth_dim,
            depth_desat=render.depth_desat,
            depth_fog=render.fog_amount,
            dof_radius=render.dof_radius,
            light_x=vol.light_x, light_y=vol.light_y, light_z=vol.light_z,
            light_ambient=vol.light_ambient,
            shadow_density=vol.shadow_density,
            # Also voxels here, world length there: the lateral extent is 1
            # across `vol_w` cubic voxels, so this is the conversion.
            shadow_reach=vol.shadow_voxels / max(geometry.width, 1),
        )
        self.device.queue.write_buffer(
            self.render_buf, 0,
            gpu_params.pack(gpu_params.RENDER_DTYPE, values).tobytes())

    def _compose(self, cpass, params: Params, frac: float):
        """Interpolate the slab to this instant, then march it.

        Returns the screen-space velocity the safety stage reprojects through,
        which the march accumulates along each ray.
        """
        slab = self.slab
        render_bind = self._buffer_binding(self.render_buf)

        cpass.set_pipeline(self.p_interp)
        cpass.set_bind_group(0, self._bind(self.p_interp, [
            self._buffer_binding(slab.params_buf), render_bind,
            slab.pigment.nxt,  # previous tick
            slab.pigment.cur,  # current tick
            slab.velocity_view, slab.interp_view, self.sampler,
        ]))
        cpass.dispatch_workgroups(*self._groups3(*self.geometry.dims))

        cpass.set_pipeline(self.p_raymarch)
        cpass.set_bind_group(0, self._bind(self.p_raymarch, [
            render_bind, slab.interp_view, slab.velocity_view,
            self.hdr_view, self.motion_view, self.noise_view, self.sampler,
        ]))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))

        return self.motion_view
