"""The Small Strange Things backend: a port with identity preservation.

DESIGN.md §18. The fourth backend, and the first whose referent is a prior
artwork rather than a metaphor: ``docs/founding/small_strange_thing.html``,
~December 2025, in which little Things wander, befriend each other, sparkle
for no reason, and spawn children near their parents. Every run of it died
with its window; this backend is the same souls on a body that persists.

The fidelity criterion for every choice in this module (§18.1): someone who
loved the original must recognise them instantly. Where a constant below
looks naive -- the 50-unit friend radius, the 0.7 body alpha, the hue+60
sparkle -- it is the founding file's, conserved on purpose; check the
reference implementation before "improving" one.

Structure (§18.3): the world is a population buffer (double-buffered, one
record per slot, a slot an identity for life because nothing dies) and one
canvas field (ping-pong; simultaneously the trail layer and the image, the
founding file's one-canvas law). A tick is three dispatches: update, deposit
(bodies + bonds into an atomic accumulator), and the canvas fade-and-drain.
The compositor samples the canvas onto the window and the shared output
chain does the rest -- the velocity texture is zeros, because nothing here
moves fast enough for reprojection to owe it anything.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import wgpu

from . import engine as engine_module
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
from .rhizotron import BufferPair

log = logging.getLogger(__name__)

# One Thing record: position, the five birth traits, age, flags, three
# friend slots, the friend count, and one spare word. Mirrored word for
# word by things_common.wgsl.
THING_STRIDE = 56
THING_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32),
    ("hue", np.float32), ("size", np.float32), ("speed", np.float32),
    ("curiosity", np.float32), ("shyness", np.float32),
    ("age", np.uint32), ("flags", np.uint32),
    ("friend0", np.uint32), ("friend1", np.uint32), ("friend2", np.uint32),
    ("friend_count", np.uint32), ("spare", np.uint32),
])

THING_ALIVE = 1
NO_FRIEND = 0xFFFFFFFF

# The founding file ran at requestAnimationFrame's ~60 fps; every per-frame
# quantity it wrote is converted through this reference rate so the Things
# keep their tempo at any tick rate (§18.1 soul 10).
FOUNDING_FPS = 60.0

# The width, in texels, at which one world unit is one texel -- the
# resolution the porch's review ratified the register at. Every length in
# ThingsParams is a world unit; the engine multiplies by field_width /
# WORLD_WIDTH when packing, so the Things are the same beings at every
# resolution (the round-2 law: the founding file was pixel-native because
# it only ever lived at one window; the port lives at every size). A
# constant, not a knob: changing it would rescale the meaning of every
# saved world.
WORLD_WIDTH = 960.0

# How many clicks one tick can consume; later clicks wait their turn in the
# host-side queue rather than being dropped.
MAX_CLICKS_PER_TICK = 4

# Slots kept beyond the population cap for click-born Things. In the
# founding file the cap only ever gated *reproduction* -- the click handler
# pushed unconditionally past 200 -- so a full village that ignored the
# finger would break soul 9, not honour soul 4. The engine's buffer must
# be finite where the founding array was not; a reserve of a couple of
# enthusiastic click-bursts keeps the verb answered without letting the
# census become a different sociology.
CLICK_RESERVE = 24


@dataclass(frozen=True)
class ThingsGeometry:
    """Every size the Things' accumulated state is made of.

    Same contract as the other geometry classes: settable from a checkpoint
    rather than always re-derived, so a saved world resumes at the shape it
    was lived in whatever window it is shown in. ``capacity`` is here
    because the population buffer is sized by it and slots are identities
    (§18.1 soul 5) -- a saved village only means anything at the capacity
    it grew in.
    """

    width: int
    height: int
    capacity: int = 200

    @classmethod
    def derive(cls, width: int, height: int, params: Params) -> "ThingsGeometry":
        # One full-window layer, like the rhizotron: the same cell ceiling
        # for the same shared-memory reasons (§8.3).
        scale = engine_module.fit_cell_budget(
            width, height, params.render.base_scale, 1, 1.0,
            int(params.render.cell_budget),
        )
        sim_w = round_up(max(int(width * scale), 64), 32)
        sim_h = max(int(height * scale), 32)
        # Buffer = the lottery's cap plus the click reserve: the click
        # outranks the cap (§18.1 souls 4 and 9).
        return cls(
            width=sim_w, height=sim_h,
            capacity=int(params.things.capacity) + CLICK_RESERVE,
        )

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not (plausible(self.width) and plausible(self.height)):
            problems.append(f"canvas size {self.width}x{self.height}")
        # The cap bounds mirror `validate`'s plus the reserve: a village,
        # not a metropolis (§18.1 soul 4), and the product decides a
        # buffer allocation.
        if not (isinstance(self.capacity, int)
                and 1 <= self.capacity <= 2048 + CLICK_RESERVE):
            problems.append(f"population capacity {self.capacity}")
        return problems

    def differences(self, other: "ThingsGeometry") -> list[str]:
        differences: list[str] = []
        for label, mine, theirs in (
            ("size", (self.width, self.height), (other.width, other.height)),
            ("capacity", self.capacity, other.capacity),
        ):
            if mine != theirs:
                differences.append(f"{label} {mine} != {theirs}")
        return differences

    def describe(self) -> str:
        return (
            f"{self.width}x{self.height} canvas, "
            f"{self.capacity} Thing slots"
        )


class ThingsEngine(Backend):
    """Owns the Things and their canvas, and drives them."""

    name = "things"

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        params: Params,
        seed: int | None = None,
        geometry: ThingsGeometry | None = None,
    ) -> None:
        self.geometry = (
            geometry if geometry is not None
            else ThingsGeometry.derive(width, height, params)
        )

        super().__init__(device, width, height, seed=seed)

        g = self.geometry
        # The canvas field: image and trail, one object (§18.3).
        self.canvas = PingPong(device, g.width, g.height, "things_canvas")
        # The population, double-buffered exactly as the rhizotron's tips
        # are: no pass has a scheduling-dependent read.
        self.things = BufferPair(
            device, max(g.capacity, 1) * THING_STRIDE, "things")
        # The deposit accumulator: four words per texel (rgb + one spare
        # lane for addressing simplicity), drained by atomicExchange every
        # tick and so empty between ticks.
        self.deposit_buf = device.create_buffer(
            size=g.width * g.height * 16,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="things_deposit")
        # The painter's order (§18 round 5): one word per texel naming the
        # topmost body standing there this tick (index + 1; zero is
        # nobody). atomicMax gives later slots the brush, which is exactly
        # the founding file's draw order; the compositor paints the
        # owner's own colour source-over everything beneath, so a Thing's
        # core belongs to that Thing whatever the crowd is doing.
        # Rebuilt every tick, so it is derived state and never
        # checkpointed.
        self.owner_buf = device.create_buffer(
            size=g.width * g.height * 4,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="things_owner")

        self.things_buf = device.create_buffer(
            size=gpu_params.THINGS_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="things_params")

        # The zero velocity texture the safety stage reprojects through:
        # written once, never again. Brownian wander is genuine change at a
        # fraction of a texel per tick; there is no bulk motion to
        # compensate.
        self.zero_vel = device.create_texture(
            size=(32, 1, 1), format=TEX_FORMAT, usage=FIELD_USAGE,
            label="things_velocity")
        self.zero_vel_view = self.zero_vel.create_view()
        device.queue.write_texture(
            {"texture": self.zero_vel, "mip_level": 0, "origin": (0, 0, 0)},
            np.zeros((1, 32, 4), dtype=np.float16),
            {"offset": 0, "bytes_per_row": 32 * 8, "rows_per_image": 1},
            (32, 1, 1),
        )

        # The pulse phase (the founding `time * 0.05`, on an accumulator so
        # a tempo change bends the breath rather than snapping it) and the
        # pending clicks. Phase rides the checkpoint; clicks are transient.
        self._pulse_phase = 0.0
        self._pending_clicks: list[tuple[float, float]] = []

        self._build_pipelines()
        self._seed_state(params)

        log.info(
            "small strange things ready: output %dx%d, %s",
            width, height, self.geometry.describe(),
        )

    # -- setup ----------------------------------------------------------------

    def _build_pipelines(self) -> None:
        self.p_update = self._compute("things_update.wgsl")
        self.p_clear_owner = self._compute("things_clear.wgsl")
        self.p_bodies = self._compute("things_deposit.wgsl", "bodies")
        self.p_bonds = self._compute("things_deposit.wgsl", "bonds")
        self.p_canvas = self._compute("things_canvas.wgsl")
        self.p_things_composite = self._compute("things_composite.wgsl")

    def _seed_state(self, params: Params) -> None:
        """A fresh world: a dark canvas and a few founding Things.

        The founding file started with five, scattered anywhere; traits are
        rolled host-side from the seed in the constructor's own order, so a
        seeded world is deterministic and the first five are as random as
        every child after them.
        """
        g = self.geometry
        # All four channels start dark: rgb is the light, alpha the breath
        # layer (§18.3), and a fresh world has been nowhere yet.
        blank = np.zeros((g.height, g.width, 4), dtype=np.float16)
        for texture in self.canvas.textures:
            self.device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                np.ascontiguousarray(blank),
                {"offset": 0, "bytes_per_row": g.width * 8,
                 "rows_per_image": g.height},
                (g.width, g.height, 1),
            )
        self.device.queue.write_buffer(
            self.deposit_buf, 0,
            np.zeros(g.width * g.height * 4, dtype=np.uint32).tobytes())

        rng = np.random.default_rng(self.seed ^ 0x5741)
        pop = np.zeros(max(g.capacity, 1), dtype=THING_DTYPE)
        pop["friend0"] = NO_FRIEND
        pop["friend1"] = NO_FRIEND
        pop["friend2"] = NO_FRIEND
        count = min(int(params.things.seed_count), g.capacity)
        for i in range(count):
            pop["x"][i] = float(rng.random()) * g.width
            pop["y"][i] = float(rng.random()) * g.height
            pop["size"][i] = float(rng.random()) * 3.0 + 1.0
            pop["speed"][i] = float(rng.random()) * 0.5 + 0.1
            pop["hue"][i] = float(rng.random()) * 360.0
            pop["curiosity"][i] = float(rng.random())
            pop["shyness"][i] = float(rng.random())
            pop["flags"][i] = THING_ALIVE
        for buffer in self.things.buffers:
            self.device.queue.write_buffer(buffer, 0, pop.tobytes())

    # -- participation ---------------------------------------------------------

    def queue_click(self, u: float, v: float) -> None:
        """Someone pointed somewhere (§18.1 soul 9).

        Takes window-normalised coordinates; the mapping to the canvas is
        the exact inverse of the compositor's sampling, so the Things
        appear where the finger was, whatever the aspect difference. The
        queue is drained a few clicks per tick, in arrival order.
        """
        if not (math.isfinite(u) and math.isfinite(v)):
            return
        g = self.geometry
        sx, sy = aspect_correction(self.width, self.height, g.width, g.height)
        fx = ((u - 0.5) * sx + 0.5) % 1.0
        fy = ((v - 0.5) * sy + 0.5) % 1.0
        self._pending_clicks.append((fx * g.width, fy * g.height))
        # Bounded: a held-down button is enthusiasm, not a workload.
        del self._pending_clicks[64:]

    # -- parameter packing -----------------------------------------------------

    def _things_values(self, params: Params, clicks: list[tuple[float, float]]) -> dict:
        g = self.geometry
        things = params.things
        dt = 1.0 / max(params.sim_hz, 1e-3)

        def per_tick_prob(rate_per_second: float) -> float:
            return 1.0 - math.exp(-max(rate_per_second, 0.0) * dt)

        # The same-beings-at-every-resolution law: every length below is a
        # world unit (one texel at WORLD_WIDTH), scaled here to this
        # world's texels. Sociology, gait and presence are all fractions
        # of the world, not of the pixel grid.
        scale = g.width / WORLD_WIDTH

        sx, sy = aspect_correction(self.width, self.height, g.width, g.height)
        values = {
            "dims_x": g.width, "dims_y": g.height,
            "capacity": g.capacity,
            # The lottery stops at the cap; clicks spend the reserve.
            "soft_cap": min(
                int(things.capacity), max(g.capacity - CLICK_RESERVE, 1)),
            "tick": self.tick_count,
            "seed": self.seed & 0xFFFFFFFF,
            "click_count": len(clicks),
            "per_click": int(things.per_click),
            "click_scatter": things.click_scatter * scale,
            # sqrt(60 * dt): the founding per-frame step, variance-matched
            # per second at any tick rate -- in world units, so the walk
            # covers the same fraction of the world at any resolution.
            "step_scale": math.sqrt(FOUNDING_FPS * dt) * scale,
            "world_scale": scale,
            "friend_prob": per_tick_prob(things.friend_rate),
            "friend_radius": things.friend_radius * scale,
            "spawn_prob": per_tick_prob(things.spawn_rate),
            "mature_ticks": things.mature_seconds * max(params.sim_hz, 1e-3),
            "spawn_radius": things.spawn_radius * scale,
            "fadein_ticks": max(
                things.fadein_seconds * max(params.sim_hz, 1e-3), 1.0),
            "fade": per_tick_prob(things.fade_rate),
            # Sustained emitters arrive per tick so steady state is
            # emit/fade_rate whatever the tick rate; the sparkle is an
            # event and its amplitude is absolute (§18.3).
            "body_emit": things.body_emit * dt,
            "bond_emit": things.bond_emit * dt,
            "bond_width": things.bond_width * scale,
            "bond_near_width": things.bond_near_width * scale,
            "bond_near_gain": things.bond_near_gain,
            # The breath layer: the ghost max-tracks the *canvas* value
            # (which is emit/fade, rate-invariant) rather than the raw
            # per-tick deposit, so the breath does not depend on the tick
            # rate; only its slow fade converts per second to per tick.
            "ghost_gain": things.ghost_gain,
            "ghost_fade": per_tick_prob(things.ghost_fade_rate),
            "ghost_luma": things.ghost_luma,
            "sparkle_amp": things.sparkle_amp,
            "sparkle_prob": per_tick_prob(things.sparkle_rate),
            "sparkle_offset": things.sparkle_offset * scale,
            "glow_mult": things.glow_mult,
            "glow_gain": things.glow_gain,
            "pulse_phase": self._pulse_phase,
            # Radians per world unit: divided by the scale so the breath's
            # spatial pattern rides the world, not the pixel grid.
            "pulse_x": things.pulse_x / max(scale, 1e-6),
            "pulse_amp": things.pulse_amp * scale,
            "x_scale": sx, "y_scale": sy,
            "out_gain": things.out_gain,
            # The owned core's display level: the trail's own steady
            # state, so painter's-order bodies keep the ratified
            # brightness (§18 round 5).
            "body_level": things.body_emit / max(things.fade_rate, 1e-3),
        }
        for index in range(MAX_CLICKS_PER_TICK):
            x, y = clicks[index] if index < len(clicks) else (0.0, 0.0)
            values[f"click{index}_x"] = x
            values[f"click{index}_y"] = y
        return values

    def _write_things_params(
        self, params: Params, clicks: list[tuple[float, float]]
    ) -> None:
        self.device.queue.write_buffer(
            self.things_buf, 0,
            gpu_params.pack(
                gpu_params.THINGS_DTYPE,
                self._things_values(params, clicks)).tobytes())

    # -- simulation ------------------------------------------------------------

    def tick(self, params: Params, event_rows: list[dict] | None = None) -> None:
        del event_rows  # no weather in their world; the scheduler's sky is not theirs
        dt = 1.0 / max(params.sim_hz, 1e-3)

        # The shared walks advance so the noise stream's saved position
        # stays meaningful across backends; nothing here reads the hue
        # phase -- a trait hue is fixed for life.
        self._advance_hue_and_walk(params, dt)
        # The founding `time * 0.05` at 60 fps = 3 rad/s, accumulated so a
        # tempo change bends the breath rather than snapping it.
        self._pulse_phase = (
            self._pulse_phase + params.things.pulse_rate * dt
        ) % (2.0 * math.pi * 1024.0)

        clicks = self._pending_clicks[:MAX_CLICKS_PER_TICK]
        del self._pending_clicks[:MAX_CLICKS_PER_TICK]
        self._write_things_params(params, clicks)

        g = self.geometry
        encoder = self.device.create_command_encoder(label="things_tick")
        cpass = encoder.begin_compute_pass()

        # 0. Yesterday's painter's order is wiped; this tick's bodies
        #    claim their own ground below.
        cpass.set_pipeline(self.p_clear_owner)
        cpass.set_bind_group(0, self._bind(self.p_clear_owner, [
            self._buffer_binding(self.things_buf),
            self._buffer_binding(self.owner_buf),
        ]))
        cpass.dispatch_workgroups(math.ceil(g.width * g.height / 64), 1, 1)

        # 1. The lives: age, wander, befriend, be born.
        cpass.set_pipeline(self.p_update)
        cpass.set_bind_group(0, self._bind(self.p_update, [
            self._buffer_binding(self.things_buf),
            self._buffer_binding(self.things.cur),
            self._buffer_binding(self.things.nxt),
        ]))
        cpass.dispatch_workgroups(math.ceil(g.capacity / 64), 1, 1)
        self.things.flip()

        # 2. The drawing: bodies and sparkles, then bonds, into the
        #    accumulator. Reads the state the update just wrote.
        cpass.set_pipeline(self.p_bodies)
        cpass.set_bind_group(0, self._bind(self.p_bodies, [
            self._buffer_binding(self.things_buf),
            self._buffer_binding(self.things.cur),
            self._buffer_binding(self.deposit_buf),
            self._buffer_binding(self.owner_buf),
        ]))
        cpass.dispatch_workgroups(math.ceil(g.capacity / 64), 1, 1)

        cpass.set_pipeline(self.p_bonds)
        cpass.set_bind_group(0, self._bind(self.p_bonds, [
            self._buffer_binding(self.things_buf),
            self._buffer_binding(self.things.cur),
            self._buffer_binding(self.deposit_buf),
        ]))
        cpass.dispatch_workgroups(g.capacity * 3, 1, 1)

        # 3. The canvas: fade, then drink the tick.
        cpass.set_pipeline(self.p_canvas)
        cpass.set_bind_group(0, self._bind(self.p_canvas, [
            self._buffer_binding(self.things_buf),
            self.canvas.cur,
            self.canvas.nxt,
            self._buffer_binding(self.deposit_buf),
        ]))
        cpass.dispatch_workgroups(*self._groups(g.width, g.height))
        cpass.end()
        self.canvas.flip()

        self._submit_tick(encoder)
        self.tick_count += 1

    # -- rendering -------------------------------------------------------------

    def _write_render_params(
        self, params: Params, frac: float, frame_dt: float
    ) -> None:
        values = self._common_render_values(params, frac, frame_dt)
        self.device.queue.write_buffer(
            self.render_buf, 0,
            gpu_params.pack(gpu_params.RENDER_DTYPE, values).tobytes())
        # Refresh the aspect correction for this frame's window without
        # re-consuming clicks: the tick owns those.
        self._write_things_params(params, [])

    def _compose(self, cpass, params: Params, frac: float):
        del frac  # sub-texel wander per tick; nothing to interpolate
        cpass.set_pipeline(self.p_things_composite)
        cpass.set_bind_group(0, self._bind(self.p_things_composite, [
            self._buffer_binding(self.things_buf),
            self._buffer_binding(self.render_buf),
            self.canvas.cur,
            self.hdr_view,
            self.sampler,
            self._buffer_binding(self.owner_buf),
            self._buffer_binding(self.things.cur),
        ]))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))
        return self.zero_vel_view

    # -- checkpointing ---------------------------------------------------------

    def pulse_state(self) -> dict:
        """The breath's accumulated phase, for the checkpoint."""
        return {"phase": float(self._pulse_phase)}

    def restore_pulse(self, saved: dict) -> None:
        """Put the breath back. Untrusted input, like all of it."""
        try:
            phase = float(saved.get("phase", 0.0))
        except (TypeError, ValueError):
            return
        if math.isfinite(phase):
            self._pulse_phase = phase % (2.0 * math.pi * 1024.0)
