"""The rhizotron backend: a pane of living soil, descending. DESIGN.md §15.

The third backend, and the first on a second metaphor. Where `engine.py` and
`volume.py` are two ways of drawing the fungal field, this is a different
world behind the same output chain: a stratified soil column seen face-on,
moisture percolating down through it after rain, and the whole window sinking
-- at hour-hand speed -- through soil that is generated below and retired
above, forever.

Three structural facts carry the design, all from §15.4:

* **The soil is a function, not a field.** Strata, stones and grain are pure
  functions of (seed, world position), evaluated wherever a pass wants them
  (`rhiz_common.wgsl`), so there is no soil texture to scroll, generate into,
  or checkpoint. What *is* stateful -- moisture -- is one ping-pong pair.

* **The vertical coordinate is an unbounded u64 row counter.** The world row
  of the texture's top edge only ever grows, it is used only as hash input,
  and it cannot repeat or lose precision on any timescale -- §3's discipline,
  applied to depth instead of time.

* **The descent is integer in the simulation and fractional in the
  presentation.** Fields scroll by whole rows, folded into each pass's source
  read, so a scroll is an exact translation costing nothing. The compositor
  glides over the sub-row remainder, interpolated between ticks, so what the
  eye sees is a continuous sink at a fraction of a pixel per second. The
  safety stage is told about it: the scroll is written into the velocity
  texture its reprojection reads, so honest bulk motion is compensated
  exactly rather than charged as change (§15.7).

The descent rate is, for now, a drifting constant -- §15's step 1 has no
roots whose growth front could set the pace. The front controller replaces
`_advance_descent`'s rate, not its plumbing.
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
    plausible,
    round_up,
)
from .config import Params

log = logging.getLogger(__name__)

# Generated soil above and below the view, in rows. The margin above is where
# the display offset reaches between ticks (the presentation trails the
# simulation by up to one tick's scroll); the margin below is soil the
# percolation can already reach before the view arrives at it.
MARGIN_ROWS = 4

# The most the world may move in one tick, in rows -- what the margins are
# sized for. The descent controller clamps to this; at defaults it asks for
# under a hundredth of it.
MAX_SCROLL_PER_TICK = 2.0

# Where the depth counter starts. Far from zero so that every relative
# coordinate the shaders derive from it -- tilted stratum rows reach a few
# hundred rows either side -- stays comfortably inside the u64 without any
# pass having to reason about underflow. The descent only ever increases it.
START_ORIGIN = 1 << 30

# One tip record: pos.xy, heading, rng, flags, age, since_branch, generation.
TIP_STRIDE = 32
TIP_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32), ("heading", np.float32),
    ("rng", np.uint32), ("flags", np.uint32), ("age", np.float32),
    ("since_branch", np.float32), ("generation", np.float32),
])

# The tips pass's fixed-point scale on the front row (rhiz_tips.wgsl).
FRONT_SCALE = 64.0

# How often the front controller reads its two words back, in ticks. Rare on
# purpose: a readback synchronises the queue, and the controller is meant to
# drift after the plant rather than track it. Three seconds at the default
# tick rate.
FRONT_READ_TICKS = 60

# Tip flag bits, mirroring rhiz_tips.wgsl.
TIP_ALIVE = 8
TIP_SPENT = 16


class BufferPair:
    """A pair of storage buffers with an alternating current/next index.

    The tips are double-buffered for the same reason every field is
    ping-ponged: each invocation reads any slot's previous state and writes
    only its own next state, so the pass has no scheduling-dependent reads --
    the property the bit-identical resume test holds every mechanism to.
    """

    def __init__(self, device: wgpu.GPUDevice, size: int, label: str):
        self.buffers = [
            device.create_buffer(
                size=size,
                usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                       | wgpu.BufferUsage.COPY_SRC),
                label=f"{label}[{i}]",
            )
            for i in range(2)
        ]
        self.index = 0

    @property
    def cur(self) -> wgpu.GPUBuffer:
        return self.buffers[self.index]

    @property
    def nxt(self) -> wgpu.GPUBuffer:
        return self.buffers[1 - self.index]

    def flip(self) -> None:
        self.index = 1 - self.index


@dataclass(frozen=True)
class RhizotronGeometry:
    """Every size the rhizotron's accumulated state is made of.

    The same contract as the other two geometry classes: settable from a
    checkpoint rather than always re-derived, so a saved column resumes at
    the shape it was grown at whatever window it is shown in. The tip pool's
    three dimensions are here because the tips buffer is sized from them and
    slots are *addressed* by them (rhiz_tips.wgsl) -- a saved plant only
    means anything in the tree shape it grew in.
    """

    width: int
    height: int  # texture rows: the view plus both margins
    view_rows: int
    max_axes: int = 6
    laterals_per_axis: int = 48
    fines_per_lateral: int = 6

    @property
    def tips_total(self) -> int:
        a, l, f = self.max_axes, self.laterals_per_axis, self.fines_per_lateral
        return a + a * l + a * l * f

    @classmethod
    def derive(cls, width: int, height: int, params: Params) -> "RhizotronGeometry":
        scale = params.render.base_scale
        rhiz = params.rhizotron
        sim_w = round_up(max(int(width * scale), 64), 32)
        view = max(int(height * scale), 32)
        return cls(
            width=sim_w, height=view + 2 * MARGIN_ROWS, view_rows=view,
            max_axes=int(rhiz.max_axes),
            laterals_per_axis=int(rhiz.laterals_per_axis),
            fines_per_lateral=int(rhiz.fines_per_lateral),
        )

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not (plausible(self.width) and plausible(self.height)):
            problems.append(f"column size {self.width}x{self.height}")
        if not plausible(self.view_rows):
            problems.append(f"view of {self.view_rows} rows")
        elif self.height != self.view_rows + 2 * MARGIN_ROWS:
            problems.append(
                f"{self.height} rows for a {self.view_rows}-row view "
                f"(expected {self.view_rows + 2 * MARGIN_ROWS})"
            )
        # The pool bounds mirror `validate`'s; a file claiming a pool outside
        # them is nonsense, and the product decides a buffer allocation.
        if not (
            1 <= self.max_axes <= 16
            and 1 <= self.laterals_per_axis <= 128
            and 0 <= self.fines_per_lateral <= 16
        ):
            problems.append(
                f"tip pool {self.max_axes}x{self.laterals_per_axis}"
                f"x{self.fines_per_lateral}"
            )
        return problems

    def differences(self, other: "RhizotronGeometry") -> list[str]:
        differences: list[str] = []
        for label, mine, theirs in (
            ("size", (self.width, self.height), (other.width, other.height)),
            ("view rows", self.view_rows, other.view_rows),
            (
                "tip pool",
                (self.max_axes, self.laterals_per_axis, self.fines_per_lateral),
                (other.max_axes, other.laterals_per_axis,
                 other.fines_per_lateral),
            ),
        ):
            if mine != theirs:
                differences.append(f"{label} {mine} != {theirs}")
        return differences

    def describe(self) -> str:
        return (
            f"{self.width}x{self.view_rows} soil column "
            f"(+{2 * MARGIN_ROWS} margin rows, {self.tips_total} tip slots)"
        )


class RhizotronEngine(Backend):
    """Owns the soil column and drives it."""

    name = "rhizotron"

    def __init__(
        self,
        device: wgpu.GPUDevice,
        width: int,
        height: int,
        params: Params,
        seed: int | None = None,
        geometry: RhizotronGeometry | None = None,
    ) -> None:
        self.geometry = (
            geometry if geometry is not None
            else RhizotronGeometry.derive(width, height, params)
        )

        super().__init__(device, width, height, seed=seed)

        g = self.geometry
        self.moisture = PingPong(device, g.width, g.height, "moisture")
        # The root map: density, age (seconds), fineness. Cumulative rather
        # than decaying -- the descent is what retires it (§15.4).
        self.structure = PingPong(device, g.width, g.height, "structure")
        # The tips, double-buffered for the deterministic birth mechanism
        # (rhiz_tips.wgsl), and their fixed-point deposit accumulator: two
        # words per texel, density and order-weight, drained every tick.
        self.tips = BufferPair(
            device, max(g.tips_total, 1) * TIP_STRIDE, "tips")
        self.deposit_buf = device.create_buffer(
            size=g.width * g.height * 8,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="root_deposit")
        # The front controller's two words: deepest living row, living count.
        # Derived -- zeroed before every tips dispatch, read back rarely.
        self.front_buf = device.create_buffer(
            size=8,
            usage=(wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST
                   | wgpu.BufferUsage.COPY_SRC),
            label="root_front")
        self._front_zero = np.zeros(2, dtype=np.uint32).tobytes()
        # The controller's accumulated state: the rate multiplier the front
        # error steers, and the last front reading (view fraction; negative
        # means nothing living). Multiplier is checkpointed with the descent;
        # the reading is derived and refreshed within FRONT_READ_TICKS.
        self._descent_mult = 1.0
        self._front_frac = -1.0
        self._alive_tips = 0

        # The uniform scroll velocity the safety stage reprojects through:
        # one row of texels, rewritten each frame from the descent's actual
        # displacement. 32 wide so a single-row write meets the 256-byte row
        # alignment `write_texture` wants.
        self.scroll_vel = device.create_texture(
            size=(32, 1, 1), format=TEX_FORMAT, usage=FIELD_USAGE,
            label="scroll_velocity")
        self.scroll_vel_view = self.scroll_vel.create_view()

        self.rhiz_buf = device.create_buffer(
            size=gpu_params.RHIZ_DTYPE.itemsize,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            label="rhiz_params")

        # --- The descent (§15.4) -------------------------------------------
        # `_descent_pos` is the world row of the view's top edge, in float64 --
        # exact to a whole row out to 2^52, and the GPU never sees it: what
        # crosses the boundary is the integer origin (as a u64) plus small
        # relative offsets. `_descent_prev` is the previous tick's position,
        # which is what the presentation interpolates from; `_display_prev`
        # is the previous *frame's* displayed position, which is what the
        # reprojection velocity is measured from.
        self._descent_pos = float(START_ORIGIN)
        self._descent_prev = float(START_ORIGIN)
        self._display_prev = float(START_ORIGIN)
        self._descent_walk = 0.0
        self._scroll_rows = 0

        self._build_pipelines()
        self._seed_state(params)

        log.info(
            "rhizotron ready: output %dx%d, %s",
            width, height, self.geometry.describe(),
        )

    # -- setup ----------------------------------------------------------------

    def _build_pipelines(self) -> None:
        self.p_moisture = self._compute("rhiz_moisture.wgsl")
        self.p_tips = self._compute("rhiz_tips.wgsl")
        self.p_structure = self._compute("rhiz_structure.wgsl")
        self.p_rhiz_composite = self._compute("rhiz_composite.wgsl")

    def _seed_state(self, params: Params) -> None:
        """A fresh column: resting moisture, bare structure, and one plant.

        The moisture is deliberately structureless -- the local baselines, the
        pooling above stones and the wetness bands all arrive through the same
        relaxation and transport a running field uses, so the first minutes
        are a grow-in rather than a painted-on state the dynamics then
        contradict. The plant is a few axis tips just below the window's top,
        headed down with a little scatter; everything else about it -- every
        lateral, every fine -- is grown, not seeded.
        """
        g = self.geometry
        w = float(params.rhizotron.moisture_baseline)
        state = np.zeros((g.height, g.width, 4), dtype=np.float16)
        state[..., 0] = w
        state[..., 1] = w

        def upload(texture, data):
            self.device.queue.write_texture(
                {"texture": texture, "mip_level": 0, "origin": (0, 0, 0)},
                np.ascontiguousarray(data),
                {"offset": 0, "bytes_per_row": g.width * 8,
                 "rows_per_image": g.height},
                (g.width, g.height, 1),
            )

        for texture in self.moisture.textures:
            upload(texture, state)
        bare = np.zeros((g.height, g.width, 4), dtype=np.float16)
        for texture in self.structure.textures:
            upload(texture, bare)
        self.device.queue.write_buffer(
            self.deposit_buf, 0,
            np.zeros(g.width * g.height * 2, dtype=np.uint32).tobytes())

        rng = np.random.default_rng(self.seed ^ 0x9007)
        tips = np.zeros(max(g.tips_total, 1), dtype=TIP_DTYPE)
        count = min(int(params.rhizotron.axes_per_plant), g.max_axes)
        spread = float(params.rhizotron.axis_spread)
        for a in range(count):
            # Axes fan out from one crown, spaced a little so the plant is a
            # plant and not a cord.
            tips["x"][a] = g.width * (
                0.5 + 0.06 * (a - (count - 1) / 2.0) + 0.01 * rng.normal())
            tips["y"][a] = MARGIN_ROWS + g.view_rows * 0.12
            tips["heading"][a] = math.pi / 2.0 + spread * float(rng.normal())
            tips["rng"][a] = int(rng.integers(1, 2**32))
            tips["flags"][a] = TIP_ALIVE | TIP_SPENT
            tips["generation"][a] = 1.0
        # Axis slots beyond the first seeding get their own noise streams and
        # stay dormant: seeds in the bank, which the germination path wakes
        # over time as the weather allows (§15.4's succession).
        for a in range(count, g.max_axes):
            tips["rng"][a] = int(rng.integers(1, 2**32))
        for buffer in self.tips.buffers:
            self.device.queue.write_buffer(buffer, 0, tips.tobytes())

    # -- the descent ----------------------------------------------------------

    def _advance_descent(self, params: Params, dt: float) -> int:
        """One tick of the sink. Returns the whole rows the world moves.

        The rate is the configured window-heights-per-hour, modulated by a
        slow log-scale OU walk -- stateful and aperiodic like every slow
        variation here (§3). It draws from the same saved noise stream the
        feature-size walk uses, so a restored session continues the walk
        rather than restarting it.
        """
        rhiz = params.rhizotron
        tau = max(rhiz.descent_wander_tau, 1e-3)
        theta = 1.0 - math.exp(-dt / tau)
        sigma = math.sqrt(max(1.0 - (1.0 - theta) ** 2, 0.0))
        walk = (
            self._descent_walk * (1.0 - theta)
            + float(self._walk_rng.normal()) * sigma
        )
        self._descent_walk = max(-2.0, min(2.0, walk))

        rows_per_second = (
            rhiz.descent_rate * self.geometry.view_rows / 3600.0
            * math.exp(rhiz.descent_wander * self._descent_walk)
            # The front controller's slow correction (§15.4): the window
            # drifts after the growing front rather than sinking blindly.
            * self._descent_mult
        )
        advance = min(max(rows_per_second * dt, 0.0), MAX_SCROLL_PER_TICK)

        self._descent_prev = self._descent_pos
        self._descent_pos += advance
        scroll = int(math.floor(self._descent_pos) - math.floor(self._descent_prev))
        return scroll

    def _origin_words(self) -> tuple[int, int]:
        """The world row of texture row 0, split for the GPU."""
        origin = int(math.floor(self._descent_pos)) - MARGIN_ROWS
        return origin & 0xFFFFFFFF, (origin >> 32) & 0xFFFFFFFF

    def _update_front_controller(self, params: Params) -> None:
        """Read the front words back and steer the descent's multiplier.

        Rarely, on the tick counter -- an 8-byte readback synchronises the
        queue, and the controller is meant to *drift* after the plant. The
        error acts through a deadband and a bounded multiplier, relaxed over
        `front_tau` seconds, so the window never chases: growth surges, and
        tens of seconds later the descent has leaned into it. With nothing
        living the descent idles at its floor, carrying the window gently
        down toward the seeds that will wake (§15.4).
        """
        raw = np.frombuffer(
            self.device.queue.read_buffer(self.front_buf), dtype=np.uint32)
        self._alive_tips = int(raw[1])
        rhiz = params.rhizotron
        g = self.geometry
        if self._alive_tips > 0:
            front_rows = float(raw[0]) / FRONT_SCALE
            self._front_frac = (front_rows - MARGIN_ROWS) / max(g.view_rows, 1)
            error = self._front_frac - rhiz.front_target
            if abs(error) < rhiz.front_deadband:
                error = 0.0
            else:
                error -= math.copysign(rhiz.front_deadband, error)
            desired = min(
                max(1.0 + rhiz.front_gain * error, rhiz.descent_mult_min),
                rhiz.descent_mult_max)
        else:
            self._front_frac = -1.0
            desired = rhiz.descent_mult_min
        interval = FRONT_READ_TICKS / max(params.sim_hz, 1e-3)
        alpha = 1.0 - math.exp(-interval / max(rhiz.front_tau, 1e-3))
        self._descent_mult += (desired - self._descent_mult) * alpha

    # -- parameter packing ----------------------------------------------------

    def _rhiz_values(
        self, params: Params, scroll_rows: int, base_row: float
    ) -> dict:
        g = self.geometry
        rhiz = params.rhizotron
        dt = 1.0 / max(params.sim_hz, 1e-3)
        origin_lo, origin_hi = self._origin_words()

        def per_tick(per_second: float) -> float:
            return per_second * dt

        def rate(per_second: float) -> float:
            # A relaxation coefficient rather than a plain scale, so a slow
            # tick rate cannot push it past stability.
            return 1.0 - math.exp(-max(per_second, 0.0) * dt)

        # Vertical lattices as powers of two of rows (see rhiz_common.wgsl);
        # the horizontal ones as whole cells around the cylinder.
        strata_rows = max(rhiz.strata_thickness * g.view_rows, 4.0)
        strata_shift = min(max(int(round(math.log2(strata_rows))), 2), 20)
        stone_shift = min(max(int(round(math.log2(rhiz.stone_cells))), 1), 12)

        return {
            "dims_x": g.width, "dims_y": g.height,
            "view_rows": g.view_rows, "margin_top": MARGIN_ROWS,
            "origin_lo": origin_lo, "origin_hi": origin_hi,
            "scroll_rows": max(scroll_rows, 0),
            "tick": self.tick_count,
            "seed": self.seed & 0xFFFFFFFF,
            "event_count": 0,  # overwritten by tick
            "strata_shift": strata_shift,
            "stone_shift": stone_shift,
            "grain_shift": 1,
            "stone_cells_x": max(int(round(g.width / max(rhiz.stone_cells, 2.0))), 4),
            "grain_cells_x": max(g.width // 2, 8),
            "base_row": base_row,
            "x_span": (
                (self.width / max(self.height, 1))
                / (g.width / max(g.view_rows, 1))
            ),
            "perc_rate": per_tick(rhiz.percolation_rate),
            "perc_pow": rhiz.percolation_power,
            "lat_spread": min(per_tick(rhiz.lateral_spread), 0.2),
            "drain_rate": rate(rhiz.drainage_rate),
            "rain_base": per_tick(rhiz.rain_base),
            "rain_event_gain": per_tick(rhiz.rain_event_gain),
            "cond_floor": rhiz.conductivity_floor,
            "moisture_baseline": rhiz.moisture_baseline,
            "wet_ema_rate": rate(1.0 / max(rhiz.wet_ema_tau, 1e-3)),
            "strata_tilt": rhiz.strata_tilt,
            "stone_amount": rhiz.stone_amount,
            "grain_amount": rhiz.grain_amount,
            "soil_l_range": rhiz.soil_l_range,
            "wet_darken": rhiz.wet_darken,
            "wet_chroma": rhiz.wet_chroma,
            # The plant. Per-second quantities become per-tick here, in the
            # form each one wants: distances scale with dt, relaxations go
            # through 1 - exp(-rate*dt) so a slow tick cannot overshoot, and
            # the steering noise scales with sqrt(dt) so the wander's
            # variance per second is what the config says whatever the tempo.
            "max_axes": g.max_axes,
            "laterals_per_axis": g.laterals_per_axis,
            "fines_per_lateral": g.fines_per_lateral,
            "tips_total": g.tips_total,
            "elong_axis": rhiz.elong_axis * dt,
            "elong_lateral": rhiz.elong_lateral * dt,
            "elong_fine": rhiz.elong_fine * dt,
            "gsa_lateral": rhiz.gsa_lateral,
            "gsa_fine": rhiz.gsa_fine,
            "gsa_gain_axis": rate(rhiz.gsa_gain_axis),
            "gsa_gain_lateral": rate(rhiz.gsa_gain_lateral),
            "gsa_gain_fine": rate(rhiz.gsa_gain_fine),
            "thigmo_gain": rhiz.thigmo_gain,
            "hydro_gain": rhiz.hydro_gain,
            "avoid_gain": rhiz.avoid_gain,
            "tip_turn": min(rhiz.tip_turn * dt, 1.2),
            "tip_jitter": rhiz.tip_jitter * math.sqrt(dt),
            "sense_dist": rhiz.sense_dist,
            "sense_angle": rhiz.sense_angle,
            "spacing_axis": rhiz.spacing_axis,
            "spacing_lateral": rhiz.spacing_lateral,
            "branch_prob": rate(rhiz.branch_prob),
            "branch_angle": rhiz.branch_angle,
            "branch_jitter": rhiz.branch_jitter,
            "tip_deposit": rhiz.tip_deposit,
            "splat_axis": rhiz.splat_axis,
            "splat_lateral": rhiz.splat_lateral,
            "splat_fine": rhiz.splat_fine,
            "fine_life": rhiz.fine_life / dt,
            "lateral_life": rhiz.lateral_life / dt,
            "axis_life": rhiz.axis_life / dt,
            "dt_seconds": dt,
            "root_knee": rhiz.root_knee,
            "root_edge": rhiz.root_edge,
            "root_age_scale": rhiz.root_age_scale,
            "root_brown": rhiz.root_brown,
            "senesce_rate": per_tick(rhiz.senescence_rate),
            "senesce_delay": rhiz.senescence_delay,
            "regerm_delay": rhiz.regermination_delay / dt,
            "germ_prob": rate(rhiz.germination_rate),
            "germ_floor": rate(rhiz.germination_floor),
            "germ_moisture": rhiz.germination_moisture,
        }

    def _write_rhiz_params(
        self, params: Params, scroll_rows: int, base_row: float,
        event_count: int,
    ) -> None:
        values = self._rhiz_values(params, scroll_rows, base_row)
        values["event_count"] = event_count
        self.device.queue.write_buffer(
            self.rhiz_buf, 0,
            gpu_params.pack(gpu_params.RHIZ_DTYPE, values).tobytes())

    # -- simulation -----------------------------------------------------------

    def tick(self, params: Params, event_rows: list[dict] | None = None) -> None:
        event_rows = event_rows or []
        event_count = min(len(event_rows), 8)
        dt = 1.0 / max(params.sim_hz, 1e-3)

        # The hue phase doubles as the soil-family drift (§15.2), and the
        # shared walk advances so the noise stream's saved position stays
        # meaningful across backends.
        self._advance_hue_and_walk(params, dt)
        scroll = self._advance_descent(params, dt)
        self._scroll_rows = scroll

        if event_count:
            self.device.queue.write_buffer(
                self.events_buf, 0,
                gpu_params.pack_array(
                    gpu_params.EVENT_DTYPE, event_rows[:event_count]).tobytes())

        # The tick's own base_row is provisional -- the render path rewrites
        # it every frame -- but the buffer must always hold a coherent one.
        base_row = MARGIN_ROWS + (
            self._descent_pos - math.floor(self._descent_pos))
        self._write_rhiz_params(params, scroll, base_row, event_count)

        g = self.geometry
        encoder = self.device.create_command_encoder(label="rhiz_tick")
        cpass = encoder.begin_compute_pass()

        # 1. Tips first, in the pre-shift frame the current textures are in:
        #    sense, steer, move, deposit, branch. They write their positions
        #    already shifted for the frame everything after them produces.
        #    The front words are zeroed going in; the living rebuild them.
        self.device.queue.write_buffer(self.front_buf, 0, self._front_zero)
        cpass.set_pipeline(self.p_tips)
        cpass.set_bind_group(0, self._bind(self.p_tips, [
            self._buffer_binding(self.rhiz_buf),
            self._buffer_binding(self.tips.cur),
            self._buffer_binding(self.tips.nxt),
            self.moisture.cur,
            self.structure.cur,
            self._buffer_binding(self.deposit_buf),
            self.sampler,
            self._buffer_binding(self.front_buf),
        ]))
        cpass.dispatch_workgroups(math.ceil(g.tips_total / 64), 1, 1)

        # 2. Moisture: percolation, rain, the shift.
        cpass.set_pipeline(self.p_moisture)
        cpass.set_bind_group(0, self._bind(self.p_moisture, [
            self._buffer_binding(self.rhiz_buf),
            self.moisture.cur,
            self.moisture.nxt,
            self._buffer_binding(self.events_buf),
        ]))
        cpass.dispatch_workgroups(*self._groups(g.width, g.height))

        # 3. Structure: the shift, and the deposit accumulator drained into it.
        cpass.set_pipeline(self.p_structure)
        cpass.set_bind_group(0, self._bind(self.p_structure, [
            self._buffer_binding(self.rhiz_buf),
            self.structure.cur,
            self.structure.nxt,
            self._buffer_binding(self.deposit_buf),
        ]))
        cpass.dispatch_workgroups(*self._groups(g.width, g.height))

        cpass.end()
        self.tips.flip()
        self.moisture.flip()
        self.structure.flip()

        self.device.queue.submit([encoder.finish()])
        self.tick_count += 1

        # The front controller reads back on the tick counter -- deterministic
        # in the state, so a resumed session reproduces its corrections.
        if self.tick_count % FRONT_READ_TICKS == 0:
            self._update_front_controller(params)

    # -- rendering ------------------------------------------------------------

    def _write_render_params(
        self, params: Params, frac: float, frame_dt: float
    ) -> None:
        values = self._common_render_values(params, frac, frame_dt)
        # The palette anchor plus the autonomous drift: under this backend the
        # pair walks the soil-family ring rather than the hue circle.
        values["hue_global"] = params.render.hue_anchor + self.hue_phase
        self.device.queue.write_buffer(
            self.render_buf, 0,
            gpu_params.pack(gpu_params.RENDER_DTYPE, values).tobytes())

        # Where the view sits this frame: the descent interpolated between the
        # last two ticks, so the sink glides instead of stepping per tick.
        display = self._descent_prev + (
            self._descent_pos - self._descent_prev) * min(max(frac, 0.0), 1.0)
        base_row = MARGIN_ROWS + (display - math.floor(self._descent_pos))
        self._write_rhiz_params(params, 0, base_row, 0)

        # Tell the reprojection about the scroll (§15.7): the displayed
        # image's world displacement since the previous frame, expressed in
        # the units safety.wgsl turns back into a screen-space offset. With
        # this in place the descent costs the luminance budget nothing, at
        # any speed the controller can reach.
        reproject = max(frame_dt, 0.0) * params.sim_hz
        delta_rows = display - self._display_prev
        self._display_prev = display
        vy = 0.0
        if reproject > 1e-6:
            vy = -(delta_rows / max(self.geometry.view_rows, 1)) / reproject
        texel = np.zeros((1, 32, 4), dtype=np.float16)
        texel[..., 1] = vy
        self.device.queue.write_texture(
            {"texture": self.scroll_vel, "mip_level": 0, "origin": (0, 0, 0)},
            np.ascontiguousarray(texel),
            {"offset": 0, "bytes_per_row": 32 * 8, "rows_per_image": 1},
            (32, 1, 1),
        )

    def _compose(self, cpass, params: Params, frac: float):
        cpass.set_pipeline(self.p_rhiz_composite)
        cpass.set_bind_group(0, self._bind(self.p_rhiz_composite, [
            self._buffer_binding(self.rhiz_buf),
            self._buffer_binding(self.render_buf),
            self.moisture.cur,
            self.structure.cur,
            self.hdr_view,
            self.sampler,
        ]))
        cpass.dispatch_workgroups(*self._groups(self.width, self.height))
        return self.scroll_vel_view

    # -- checkpointing --------------------------------------------------------

    def descent_state(self) -> dict:
        """The descent's accumulated state, for the checkpoint.

        The position is split into its integer origin and sub-row remainder
        so the u64-scale part travels as a JSON integer (exact at any depth)
        and the fractional part as a float64 (exact by round-trip).
        """
        origin = int(math.floor(self._descent_pos))
        return {
            "origin": origin,
            "remainder": float(self._descent_pos - origin),
            "prev_delta": float(self._descent_pos - self._descent_prev),
            "walk": float(self._descent_walk),
            "rate_mult": float(self._descent_mult),
        }

    def restore_descent(self, saved: dict) -> None:
        """Put the descent back where it was. Untrusted input, like all of it."""
        try:
            origin = int(saved.get("origin", START_ORIGIN))
            remainder = float(saved.get("remainder", 0.0))
            prev_delta = float(saved.get("prev_delta", 0.0))
            walk = float(saved.get("walk", 0.0))
        except (TypeError, ValueError):
            return
        if not (
            math.isfinite(remainder) and math.isfinite(prev_delta)
            and math.isfinite(walk) and origin >= 0
        ):
            return
        self._descent_pos = origin + min(max(remainder, 0.0), 1.0)
        self._descent_prev = self._descent_pos - min(
            max(prev_delta, 0.0), MAX_SCROLL_PER_TICK)
        # The presentation resumes at rest; the first frame's velocity is
        # zero, which the limiter treats as a still image -- correct for a
        # relaunch.
        self._display_prev = self._descent_pos
        self._descent_walk = max(-2.0, min(2.0, walk))
        try:
            mult = float(saved.get("rate_mult", 1.0))
        except (TypeError, ValueError):
            mult = 1.0
        # Bounded by the widest limits `validate` admits, so a foreign value
        # cannot pin the descent anywhere the controller could not have.
        self._descent_mult = (
            min(max(mult, 0.05), 10.0) if math.isfinite(mult) else 1.0)
