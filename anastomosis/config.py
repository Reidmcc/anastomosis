"""Parameter model, macro mapping, TOML persistence, and safety validation.

Two tiers, per DESIGN.md §9:

* **Macros** — eight knobs in 0..1, the normal interface.
* **Primitives** — the ~50 values the shaders actually read.

Macros drive primitives through a curve table -- one per *mode*, in
:data:`MODE_CURVES` (DESIGN.md §14): the same eight knobs mean the same things
in both modes, and only the endpoints they reach differ. The config file may
also pin individual primitives, which override the macro result.

Every value that could affect flash safety is clamped to a hard ceiling here
(:data:`SAFETY_CEILINGS`) before it can reach the GPU. That clamp is the last
line of defence for the invariant in DESIGN.md §7, so it lives in the
parameter layer rather than in the renderer, where a future refactor might
route around it.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import math
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TAU = math.tau


# --------------------------------------------------------------------------
# Primitive parameter groups
# --------------------------------------------------------------------------


@dataclass
class SafetyParams:
    """Output-stage limits. See DESIGN.md §7.

    ``max_luma_delta`` is the load-bearing one: at 30 FPS a value of 0.01 means
    a 10% luminance excursion takes >=333 ms, capping the system at 1.5
    flashes/second against the WCAG threshold of 3.
    """

    max_luma_delta: float = 0.010
    max_chroma_delta: float = 0.030
    iir_alpha: float = 0.20
    dither_amount: float = 1.0
    exposure_target: float = 0.16
    # Asymmetric: darkening is allowed to act faster than brightening, since
    # the unsafe direction is always "gets brighter".
    exposure_attack: float = 0.0035
    exposure_release: float = 0.0090


@dataclass
class AgentParams:
    """Physarum layer. Deposits are soft splats; see DESIGN.md §2."""

    density: float = 0.27  # agents per simulation cell
    speed: float = 0.90  # cells per tick
    sensor_angle: float = 0.42  # radians, off-axis sensors
    # How far ahead an agent senses, in cells. Bounded relative to the width of
    # what it is sensing: `sensor_distance` above roughly 4x `trail_diffuse`
    # puts the layer in a condensing regime where the whole population ends up
    # on one axis-aligned strand that wraps the torus (DESIGN.md §4.9). This is
    # the *base*; the shader clamps the climate's deviation against
    # `sensor_reach_max` so no region can cross the threshold on its own.
    sensor_distance: float = 3.6  # cells
    # The bound, as a multiple of the trail's diffusion sigma. Measured on a
    # 256x160 field: the layer condenses at a ratio of 5.8, is still drifting
    # toward it at 4.0, and holds a distributed network at 3.7 and below. On a
    # smaller field the same ratios bite harder -- 3.2 was still half-condensed
    # at 192x128 where 2.5 was not -- so the base sits at 2.6, with the ceiling
    # far enough above it for the climate to deviate on both sides.
    sensor_reach_max: float = 3.2
    turn_rate: float = 0.32  # radians per tick
    jitter: float = 0.10  # radians, stochastic steering
    deposit: float = 0.018  # per tick, kept far below trail_decay
    fusion_bias: float = 0.55  # commitment to sensed junctions, 0..1
    # The ceiling on that commitment, and the anti-fusion mechanism entire
    # (DESIGN.md 4.7 step 4). The steering term is `turn * (1 - commitment)`,
    # so the axis is not two-sided but three-valued: 0 leaves the agent
    # glancing along whatever it sensed, 1 cancels the turn so it drives
    # straight through the junction and fuses, and past 1 the term changes
    # sign and the agent veers *away*. Held at 0.92 the layer had attraction
    # and no repulsion, so every fusion added a cycle and nothing could remove
    # one. That old ceiling was never actually reached -- the bias tops out at
    # 0.72 across the intensity macro -- so widening it changes nothing on its
    # own; what it does is let the climate `repel` channel and rift events push
    # a region past the crossing. The look at a neutral climate is unchanged.
    fusion_max: float = 1.85
    # Founding respawn. A respawned agent used to land alone on uniform random
    # ground, where a single agent's deposit is far below what can hold against
    # decay -- so it could never found anything, and all growth accreted onto
    # the network that already existed. DESIGN.md 4.7 identifies that as what
    # made flux pruning (below) concentrate the network rather than turn it
    # over: the resorbed mass had nowhere to go but back into the existing
    # structure. This is the mechanism that half of the argument asked for.
    #
    # This fraction of respawns instead land together at a shared site, which
    # is reselected every `found_period` ticks, so the arrivals of a whole
    # epoch pile onto one patch of bare ground and can actually establish
    # something. The site is chosen as the barest of four candidates, which
    # costs four trail samples on the respawn path only.
    found_fraction: float = 0.55
    found_period: int = 240  # ticks a founding site lasts, ~12 s at 20 Hz
    # Simulation cells per concurrent founding site. A *density* rather than a
    # count, so the arrival rate per site is the same on a 128-cell test layer
    # as on a 1440p one -- agent count scales with area, and a fixed number of
    # sites would concentrate a hundred times more traffic on the large layer,
    # which would land as a visible flare rather than as a founding.
    found_site_cells: int = 16_384
    # Cells; the scatter of a cohort around its site. Bounded by the sensing
    # reach for the same reason that reach is bounded by the trail width
    # (DESIGN.md §4.9): a cohort scattered wider than its members can see does
    # not find itself, so the site never establishes and the founding mechanism
    # quietly stops working. Measured, at a reach of 3.6 cells a radius of 6.0
    # left a rifted disc at 1-3% of the control's trail for 5400 ticks -- bare
    # ground that never came back -- where 2.55 heals it to 80%.
    found_radius: float = 2.55
    trail_decay: float = 0.055
    trail_diffuse: float = 1.15  # gaussian sigma in cells
    # Flux pruning -- the autolysis half of anastomosis (DESIGN.md 4.7).
    # Without it the trail field can only gain edges: agents fuse and reinforce,
    # decay is uniform and traffic-blind, so nothing can ever remove a strand.
    #
    # `income_rate` is the EMA on arriving traffic, and must be meaningfully
    # *faster* than the trail's own decay or the average simply reproduces the
    # trail and the deficit carries no information at all: at 0.05 against a
    # decay of 0.055 the measured deficit is 0.02 everywhere and the term is
    # inert. At 0.15 it spreads across the full range.
    income_rate: float = 0.15
    # Off by default, and that is a measured decision rather than caution.
    # Enabled, the mechanism does what it says -- it removes strands traffic has
    # abandoned, and the network carries the same mass in a third less area --
    # but the field it leaves behind is the *persistent* part of itself, so the
    # trail's autocorrelation at a 1050-tick lag rises from 0.11 to 0.27. It
    # buys concentration at the cost of exactly the monotony 4.7 exists to fix,
    # and one run in four at gain 1.5 fell into a sparse state it never left.
    # The mechanism is built, plumbed and tested; see DESIGN.md 4.7 for what
    # would have to change for it to earn being switched on.
    prune_gain: float = 0.0
    # Deposit capacity -- the counter to winner-take-all trail following, and
    # the mechanism behind the persistent white dots (DESIGN.md 4.7, "what the
    # dots turned out to be"). Trail following has no capacity limit (4.9):
    # agents pile onto the strongest signal and their deposits make it
    # stronger, so the network ends up holding almost half its mass in its top
    # 2% of texels -- round, stationary hubs five times the level of the
    # filaments between them. A deposit landing on trail at this level is
    # halved (`1 / (1 + trail / cap)`), so hubs stop out-competing while bare
    # ground and ordinary filaments -- an order of magnitude below the cap --
    # barely notice; founding cohorts land on bare ground and are untouched.
    #
    # What the capacity withholds is measured (the trail texture's `.a` channel
    # carries a withheld EMA alongside the income EMA in `.g`) and handed back
    # through the agent deposit, exactly as flux pruning's removal is: the
    # capacity is a redistribution from hubs to wherever traffic is, not a
    # sink, so the homeostat has nothing to cancel. Zero disables.
    #
    # The value is measured, and lower is not stronger. At 1.2, across three
    # seeds, the network's top-2% mass share falls 0.49 -> 0.33 and the
    # bright-blob count roughly halves (72 -> 34 on average), with trail mass
    # matching the uncapped control to 1% and the return settling around 1
    # against its bound of 3. At 2.0 the effect fades (0.42); at 0.6 the
    # deposit-weighted withheld ratio exceeds the bound, the return pins, and
    # the capacity becomes exactly the sink it must not be: trail mass falls
    # 11% and the hubs survive *better* than at 1.2. If this is ever lowered,
    # cap_return in the telemetry must stay clear of its clamp.
    deposit_cap: float = 1.2
    # Sensing saturation -- the missing half of the capacity, and the reason
    # the layer produced knots instead of a network at all (DESIGN.md 4.7,
    # "the network that was never there"). The capacity bounds what a hub can
    # *store*; nothing bounded what it could *attract*: sensed trail was
    # unbounded, so a hub at ten times filament level out-competed every
    # strand in sensor range forever, and grown from scratch the layer reached
    # a stable field of round milling knots with no filaments anywhere --
    # which was the persistent-dot complaint, on every build tried back to
    # PR #16. What the sensors read is therefore clamped, so a healthy
    # filament is exactly as attractive as any hub -- and inside a saturated
    # plateau the three sensors tie, which reads as "keep going", so agents
    # drive straight out of a knot instead of orbiting it.
    #
    # Measured against an uncapped control (same seed, 320x180 and 128x128,
    # 4000 ticks): the trail becomes an anastomosing network -- strands,
    # junctions, closed loops, stable for the whole run -- with total mass
    # identical to the control and its p99/mean concentration down from ~16
    # to ~5. The obvious parameter-space alternative (higher turn rate and
    # jitter) was tried and is much worse: sparse lone strands plus knots.
    #
    # The value is a *multiple of the equilibrium mean trail*, which is
    # exactly `density * deposit / trail_decay` (each backend packs the
    # absolute cap in `_physics_values`, from its own agent density). An
    # absolute cap cannot be right at more than one point of the intensity
    # macro, and the failure at the quiet end is distinctive: with density
    # and deposit at their low ends an absolute cap sits far above the level
    # any filament can sustain, the plateau survives only at knot cores, and
    # the layer draws *rings* -- agents orbiting the rim of their own
    # saturated deposit. Anchored to the equilibrium mean, the quiet end
    # grows the same wispy network as the default (p99/mean ~5.0 against an
    # uncapped 13-19, with the ring state at ~10) and the dense end holds
    # (~4.7-5.0 against an uncapped 14-16). At 3.3 the
    # absolute cap is ~0.30 at defaults: about twice the level a trafficked
    # filament equilibrates to, comfortably under the knot level of 1+.
    #
    # Zero disables. The packed absolute value is also floored well clear of
    # `starve_threshold`: `recent` is an EMA of sensed -- and therefore
    # capped -- values, so a cap near the threshold would read the whole
    # population as starving and it would respawn forever.
    sense_cap: float = 3.3
    # How much of the velocity field the trail rides -- DESIGN.md 4.7 step 6,
    # at last. Pigment is advected at 1.0; the trail at this fraction, so the
    # network is carried and sheared by the same flow that carries the colour,
    # instead of sitting bolted to the grid while the picture slides over it.
    # A hub under shear stops being a stationary circle, which is the half of
    # the dot complaint no morphology lever could reach. Zero disables, and
    # the trail pass is then texel-exact.
    #
    # What is verified is the mechanism and the invariants, in step 4's
    # tradition: a blob rides a known velocity at exactly this fraction of it
    # (test_agents), and across seeds the sweep shows mass, mean V, activity
    # and corr_decay unmoved with it on. The aggregate -- whether the network's
    # 400-tick autocorrelation actually falls -- did not resolve above
    # run-to-run variance at test resolution, exactly as step 4's churn did
    # not. Two costs are measured and accepted: the capacity's de-hubbing
    # weakens somewhat with the trail sliding under the depositors (top-2%
    # mass share 0.33 -> 0.39 across three seeds), and cap_return rides
    # higher (~1.6 against ~1.0), still well inside its bound.
    trail_advect: float = 0.5
    starve_threshold: float = 0.004
    max_age: float = 2400.0  # ticks before forced respawn


@dataclass
class ReactionParams:
    """Gray-Scott, coupled to the trail field."""

    # Chosen by sweeping the Gray-Scott map (see tests/reference.py and
    # test_regime.py). This point sits on the persistently-live ridge: it holds
    # mean V ~0.12 with variance ~0.009 and, critically, does not settle. The
    # more familiar F=0.038/K=0.062 looks similar for a few minutes and then
    # goes static, which is useless here.
    feed: float = 0.0180
    kill: float = 0.0510
    du: float = 0.2097
    dv: float = 0.1050
    dt: float = 0.85
    substeps: int = 2
    # Diffusion is the morphology lever (DESIGN.md 4.7). At fixed diffusion a
    # Gray-Scott regime has exactly one characteristic wavelength, so a
    # constant `du` pins the feature size and the field settles into a
    # monodisperse spot texture -- a lattice of same-sized round holes, which
    # is both monotonous and a trypophobia trigger. `du` is the lever to move
    # because it is nearly orthogonal to everything the homeostat measures:
    # over du 0.17-0.40 the feature count changes 8.6x while mean V stays
    # within 0.102-0.114. kill and the dv/du ratio both move mass, which the
    # exposure governor would turn into a slow global luminance swing.
    #
    # `dv` is never varied independently: the ratio dv/du is held fixed and
    # only the pair is scaled, because the ratio moves mass hard.
    #
    # --- The feature-size loop (DESIGN.md 4.7 step 5) ---------------------
    #
    # The global mean of `du` is not a constant and is no longer an open-loop
    # walk either. It is the output of a controller that measures the field's
    # characteristic length scale `ell = mean V / mean |grad V|` and drives it
    # to a setpoint which is itself a slow bounded walk. The difference
    # matters: an open-loop walk on `du` asks for a diffusion rate and hopes
    # the texture follows, and if the field is sitting in an attractor that
    # pins its wavelength, nothing objects. A loop closed on `ell` notices --
    # that is the whole point of measuring the one quantity that is not
    # invariant under rearrangement -- and pushes `du` until the texture
    # actually moves.
    #
    # `ell_walk` is the log-`ell` deviation the setpoint asks for at one
    # standard deviation of the walk, so 0.09 is about +-9% in feature size
    # typically and +-19% at the walk's +-2 bound. Feature *count* goes as
    # roughly ell^-2, so that is a swing of about 2x across the walk's range --
    # comparable to the 2.7x the offline drift experiment in DESIGN.md 4.7
    # produced, and several times what the open-loop du walk it replaces
    # delivered (+-7% in `du` is +-3% in ell at one standard deviation).
    ell_walk: float = 0.09
    ell_walk_tau: float = 420.0  # seconds; well past any visible timescale
    # The controller's own time constant. It has to sit between two others: far
    # slower than the field's response to a change in `du` (a few hundred
    # ticks, from the offline drift runs) so the loop is not chasing the
    # reaction's own dynamics, and rather faster than `ell_walk_tau` so it
    # tracks the setpoint instead of lagging it. The plant gain is about 0.5
    # -- ell goes as sqrt(du) across the whole measured sweep -- so the closed
    # loop settles at about twice this.
    ell_tau_seconds: float = 90.0
    # The reference the setpoint is a deviation *from*: a very slow average of
    # the field's own length scale, with the modulation subtracted before it is
    # averaged. Both halves are load-bearing.
    #
    # A reference rather than an absolute setpoint, because there is no
    # absolute number to use. The `scale` macro moves `du` by design, and a
    # controller holding ell at a fixed value would cancel it; the agent layer
    # and the feed/kill machinery move ell too, so the value the full engine
    # settles at is not the value the isolated reaction does. Referenced to
    # itself, the loop has no opinion about where feature size should sit and
    # only insists that it move.
    #
    # And the modulation is subtracted before averaging because otherwise the
    # reference chases its own output: ell tracks the setpoint, the reference
    # averages ell, the setpoint is built from the reference, and the walk gets
    # integrated into a drift with no fixed point. Subtracting it leaves the
    # reference tracking the field's *natural* length scale, which is what it
    # is supposed to be following.
    ell_ref_tau_seconds: float = 1800.0
    # Bound on the accumulated log multiplier, and so on the controller's whole
    # authority: +-0.36 is x/1.43 on `du`, i.e. 0.146 to 0.301 at the shipped
    # base. That is very nearly the 0.16-0.34 span the offline drift experiment
    # in DESIGN.md 4.7 ran at and measured mass and activity staying inside
    # both homeostat deadbands through. It doubles as the anti-windup: the
    # integrator is clamped, not merely its effect, so a field that refuses to
    # move its length scale parks the controller at a bound rather than winding
    # up somewhere it then has to unwind from.
    ell_corr_limit: float = 0.36
    # Hard bounds on the local `du` after the controller's global correction
    # and the climate deviation are both applied. A survival bound, not a
    # target: the usable band in DESIGN.md 4.7 is [0.17, 0.40], and these are
    # deliberately wider so the drift has somewhere to go.
    #
    # The floor is measured. At du = 0.06 the field collapses -- mean V 0.015,
    # activity indistinguishable from zero; at 0.08 it survives but barely
    # moves; at 0.12 it is a live fine-grained texture (mean V 0.129, activity
    # 4.3e-4). 0.12 is a factor of two above the collapse and the lowest point
    # with a real measurement behind it. The ceiling has more headroom than it
    # needs (du = 0.50 still runs clean) and exists to keep a hand-edited
    # override away from the explicit scheme's stability limit, which for this
    # averaging-form Laplacian is near dt*du = 1.0.
    du_min: float = 0.120
    du_max: float = 0.420
    # Trail raises local feed through a saturating curve, so a heavily
    # trafficked texel cannot drive feed off the map. This is the coupling that
    # makes filaments nucleate reaction structure.
    trail_feed_gain: float = 0.012
    # Kill tracks feed along the live band rather than staying fixed. Without
    # this, a downward climate excursion in feed drives whole regions to zero
    # and they never recover -- verified in test_regime.py.
    kill_follows_feed: float = 0.55
    # Agents seed V directly, not only via feed. Without this, V = 0 is an
    # absorbing state -- dV/dt is zero when V is zero, so the reaction can
    # never restart anywhere it has been fully extinguished, and one bad
    # excursion or a sanitised NaN would end the run permanently with a black
    # screen. Verified by test_recovers_from_a_corrupted_field.
    #
    # The falloff makes this act only where there is room: established
    # structure is untouched, empty ground is slowly reseeded by passing
    # filaments.
    trail_seed_gain: float = 0.0020
    trail_seed_falloff: float = 8.0
    # Hard bounds applied in the shader after climate deviation, trail
    # coupling and homeostat correction are all summed. These are a liveness
    # floor, not a stylistic range: below kill_min the reaction dies at low
    # feed and cannot restart, and a dead region would then be advected around
    # by the climate flow rather than recovering. Verified by
    # test_climate_range_stays_on_the_live_band.
    # The live region of the Gray-Scott map is a diagonal strip, not a
    # rectangle, so kill is bounded *relative to the band centre* (which
    # follows feed) as well as absolutely. A fixed box either admits dead
    # corners at both ends or needlessly restricts the middle.
    feed_min: float = 0.0100
    feed_max: float = 0.0300
    kill_band: float = 0.0050  # permitted excursion off the band centre
    kill_min: float = 0.0470  # absolute floor, protects the low-feed end
    kill_max: float = 0.0630  # absolute ceiling, protects the high-feed end


@dataclass
class FlowParams:
    """Divergence-free velocity field, ``v = curl(psi)``. DESIGN.md §2.

    ``psi`` is a *stateful* field evolved by a spatial OU process, never a
    function of wall-clock time -- see DESIGN.md §3 for why that distinction
    matters over multi-day runs.
    """

    psi_gain: float = 1.30  # weight of the imposed-weather component
    field_gain: float = 0.85  # weight of the structure-following component
    psi_theta: float = 0.0022  # OU mean reversion per tick
    psi_sigma: float = 0.085  # OU noise amplitude per tick
    # Structural: psi texture size divisor. Fixed at startup, since changing
    # it would mean reallocating textures mid-session.
    psi_scale: int = 4
    # The `scale` macro drives this instead -- same perceptual effect on
    # feature size, no reallocation.
    psi_noise_scale: float = 3.0
    advect_dt: float = 1.0


@dataclass
class PigmentParams:
    """The advected field that is actually shaded. DESIGN.md §2."""

    inject_rate: float = 0.055  # how fast pigment adopts local structure
    # What the image is *made of*, and the balance is the whole question.
    #
    # At 2.9 against 0.85 the reaction carried the picture and the network was
    # nearly invisible in it. That is worth stating in numbers, because the
    # ratio understates it: mean V is about 0.118 and a spot reaches 0.3-0.4,
    # so a single spot alone reached 0.9-1.2 against a ceiling of 1 and every
    # feature on screen rendered as a flat-topped disc with a hard rim. The
    # field of similar-sized round dots that made the texture aversive
    # (DESIGN.md 4.7) was not only in the simulation; the shading stage was
    # picking it out and clipping it.
    #
    # Rebalanced, the peak of the reaction lands at 0.86 of the ceiling and
    # reads as a soft bump instead of a clipped disc, while the network carries
    # the structure. Measured on a mature field, the fraction of texels the
    # ceiling clips falls from 6.7% to 4.3%, and the number of separate bright
    # blobs the rendered image is made of from 147 to 25. Nothing about the
    # simulation changes; this is the last stage before colour, and it is the
    # cheapest place to stop amplifying the one geometry the application must
    # not dwell on.
    #
    # The trail weight is high enough to carry the image and no higher. Past
    # about 1.3 the trail term starts reaching the ceiling on its own, and
    # where it does, the reaction's contribution is discarded -- which would
    # move the clipping from the spots to the filaments rather than removing
    # it. At 1.25 that is 2% of texels and 2.8% of the brightest reaction.
    density_from_v: float = 1.90
    density_from_trail: float = 1.25
    # How much of the reaction's contribution is gated on there being network
    # under it. 0 renders V wherever it is, which is what produced a
    # free-standing lattice of dots across bare ground; 1 renders it only where
    # filaments are.
    #
    # This is the difference between the two things V does. Trail raises the
    # local feed rate, so the reaction nucleates *on* the network and gives
    # filaments the internal texture DESIGN.md 2 wants from the coupling. Away
    # from the network the same reaction is just Gray-Scott in its spot regime,
    # and that is the monodisperse lattice -- the same size everywhere,
    # answering to nothing, and the part of the image with no structural story
    # behind it. Gating keeps the first and drops the second.
    #
    # Deliberately well short of 1, and the reason is measured rather than
    # aesthetic. Gating harder keeps winning on blob count -- 25 separate bright
    # regions at 0.25, 11 at 0.40 -- but it wins by hiding the reaction, and the
    # reaction is what carries the *variation* in feature size that the rest of
    # DESIGN.md 4.7 works to produce. Past this point the spread of local
    # feature size in the rendered image starts falling back through the
    # unrebalanced control (c.v. 0.222 ungated, 0.203 here, 0.183 at 0.40,
    # against 0.191 before any of this). Uniformity is the trigger, so trading
    # it away for a lower blob count is trading the wrong way.
    #
    # This is the single default most likely to want moving once someone has
    # watched the result on real hardware; see DESIGN.md 13.
    v_needs_trail: float = 0.25
    # Soft knee on the trail's rendered contribution. The hubs sit at trail
    # levels around five times the filaments', so under a linear term they are
    # the one thing on screen that reaches the density ceiling and clips to a
    # flat white disc -- measured, a third of the bright-blob texels were
    # clipped. `knee * tanh(trail / knee)` is within 12% of linear at the
    # filament level (trail ~0.07-0.3) and bounded at `knee` above it, which
    # takes the clipped fraction to 3% while moving filament brightness by
    # less than 0.003. It dims hubs rather than removes them -- the removal is
    # `deposit_cap`'s job -- but a soft translucent maximum is a different
    # thing to look at than a hard-rimmed white circle. Zero means linear.
    trail_knee: float = 0.45
    activity_rate: float = 0.020  # lowpass on |dV/dt|; deliberately very slow
    activity_gain: float = 26.0
    # Material keeps the hue it was born with and carries it along the flow.
    # Low values make structures of different ages chromatically distinct, so
    # the field marbles instead of shifting as one; high values make the whole
    # field track the drifting anchor together.
    hue_inject_mix: float = 0.010
    hue_from_orientation: float = 0.55


@dataclass
class ClimateParams:
    """Slowly drifting field of local parameter values. DESIGN.md §4.1."""

    width: int = 64
    height: int = 36
    theta: float = 0.0016  # OU mean reversion per tick
    sigma: float = 0.055  # OU noise per tick
    advect_gain: float = 0.22  # how fast regimes migrate
    diffuse: float = 0.30
    # Deviation amplitudes, applied as base + range * climate_texel.
    range_feed: float = 0.0080
    range_kill: float = 0.0035
    range_sensor_angle: float = 0.22
    # Kept inside the band the sensing reach is safe in (DESIGN.md §4.9) rather
    # than the +-3 cells it used to span, which put the tails of an ordinary
    # climate excursion well over the condensation threshold. The shader clamps
    # the top as well, so this only has to be the typical spread.
    range_sensor_distance: float = 0.8
    range_deposit: float = 0.008
    range_decay: float = 0.018
    range_flow: float = 0.55
    range_hue: float = 1.15  # radians
    # Feature size, per region -- the third climate pair, DESIGN.md 4.7.
    # Geometric rather than absolute, unlike the ranges above: it is the log of
    # the multiplier on `du`. `du` is driven by the `scale` macro, so an
    # absolute deviation would mean a different spread at each end of that
    # knob, and the survivable band is asymmetric around the base, so an
    # additive deviation spends its downside against the floor while its upside
    # is still unused.
    #
    # Note the magnitude against the others. Every `range_*` is the deviation
    # at a climate value of 1, but the climate field does not reach 1: the
    # per-tick diffusion is applied to a spatially white OU drive, so almost
    # all of the injected power is at high spatial frequency and is removed
    # immediately. The realised deviation settles at s.d. ~0.11 with extremes
    # near +-0.44 (measured off a running engine at ticks 1200-3600, and the
    # same for every channel). So this 3.2 is a realised x1.42 / /1.42 at one
    # standard deviation, and the field spans the whole du band across its
    # extremes -- which is the band the sweeps in DESIGN.md 4.7 explored. Some
    # 5% of texels sit at the du_min clamp; that is intended, and it is why the
    # floor is a measured survival bound rather than a guess.
    range_du: float = 3.2
    # Pruning strength, per region -- geometric, like range_du and for the same
    # reasons. Zones where the network visibly comes apart, migrating past zones
    # where it holds together. Inert while agents.prune_gain is zero, which it
    # is by default.
    range_prune: float = 1.5
    # Junction behaviour, per region -- the `repel` channel of the third pair.
    # Additive, unlike range_du and range_prune, because what matters on this
    # axis is a *threshold* rather than a ratio: at a commitment of 1 the
    # steering term vanishes and the agent crosses the junction, and past 1 it
    # changes sign and the agent veers away (agents.wgsl). An additive
    # deviation puts that crossing at a fixed distance in climate units, so the
    # fraction of the field that is repelling can be reasoned about directly.
    #
    # Against the realised climate amplitude (s.d. ~0.11, see range_du), 2.6
    # puts the crossing at 1.6 s.d. above the mean, so roughly 6% of the field
    # is coming apart at any moment while the rest fuses -- and those zones
    # migrate with the climate. A rift event saturates the channel at its clamp
    # for the length of its envelope -- an event adds its amplitude every tick
    # against a mean reversion of 0.0016, so any channel it names ends up
    # pinned -- which puts the whole event disc at `agents.fusion_max`.
    range_repel: float = 2.6


@dataclass
class HomeostatParams:
    """Loose, slow regulation that keeps the system alive. DESIGN.md §4.2.

    Deliberately wide deadband and long time constant: a tight controller makes
    the output feel regulated and monotonous, and becomes itself a source of
    coordinated global change (i.e. punctuation).
    """

    target_mass: float = 0.118  # mean V
    target_variance: float = 0.0090  # var V, proxy for "has structure"
    target_activity: float = 0.0012  # mean |dV/dt|, measured not guessed
    deadband: float = 0.30  # fractional, +-30%
    gain_p: float = 0.010
    gain_i: float = 0.0009
    integral_limit: float = 0.35
    tau_seconds: float = 120.0


@dataclass
class EventParams:
    """Poisson-arrival localised perturbations. DESIGN.md §4.3.

    Events are applied to *climate*, never to pigment or luminance directly, so
    their effect reaches the image only after several stages of diffusion.
    """

    enabled: bool = True
    rate_per_hour: float = 7.5
    attack_seconds: float = 45.0
    hold_seconds: float = 60.0
    release_seconds: float = 90.0
    strength: float = 0.85
    max_radius_frac: float = 0.24  # <= 0.25, the WCAG flash-area threshold
    max_concurrent: int = 4


@dataclass
class VolumeParams:
    """The volumetric slab backend -- DESIGN.md §5.1.

    Only read when ``Config.backend`` is ``"volumetric"``. The slab is a thin
    3-torus: ``512 x 288 x 48`` cubic voxels, which is a lateral extent of 1 by
    0.56 and a thickness of 0.094, so "thin slab" is a fact about the geometry
    rather than a way of talking about it.

    Nothing here duplicates a parameter the layered backend already has.
    Agents, reaction, flow, climate, pigment, homeostat and the whole of
    ``render``'s colour and depth work are shared, because a macro has to mean
    the same thing under either backend; what is here is the handful of things
    a volume has and a stack of sheets does not.

    Memory is the one place the slab is genuinely more expensive than the
    layers: at the default size a field is about 650 MB of rgba16float, against
    roughly 90 MB for the 1440p layered stack. That is comfortable on the
    target card (DESIGN.md §8.1), and it is also why ``depth`` -- the knob the
    control panel exposes -- is the parameter that decides what this backend
    costs. Everything below that is calibrated against a *filament*, in voxels,
    rather than against the slab's thickness, so that moving it changes how
    much material a ray passes through and nothing else.
    """

    # --- Slab geometry (structural: changing it needs a new field) --------
    # Lateral width in voxels. The height follows the window's aspect, so the
    # voxels stay cubic and a 16:9 display gets exactly the 512x288 of §5.1.
    #
    # Normally set by `Config.volume_detail` rather than here: this is the
    # primitive, and the three named sizes in `VOLUME_DETAIL` are what the
    # command line and the control panel actually offer. Setting it directly
    # through `[overrides]` still wins, which is the escape hatch for a size
    # that is not one of the three.
    width: int = 512
    # Thickness in voxels, and the one piece of the slab's geometry the control
    # panel can move. It is a knob rather than a constant because depth is the
    # entire point of this backend and forty-eight voxels of it is a subtle
    # effect: a ray crosses one or two filaments, so there is not much
    # occlusion for the eye to read. The ceiling is the shorter of the two
    # lateral axes -- past that this is not a slab, and `VolumeGeometry.derive`
    # enforces it. What thickness costs is memory, linearly: about 13 MB per
    # slice at the default 512x288, so the 288 ceiling is about 3.9 GB against
    # the default's 650 MB. What it buys stops arriving somewhere short of that
    # ceiling, when the near material has become opaque enough to hide the far
    # face -- see DESIGN.md §5.1.
    depth: int = 48
    base_scale: float = 1.0  # fraction of `width` actually allocated
    # Agents per voxel, against 0.27 per cell in the plane. Not a like-for-like
    # comparison and not meant to be: a filament network occupies a much smaller
    # fraction of a volume than of a plane, so the agents that do find the
    # structure concentrate harder on it, while the ones that do not are spread
    # through three dimensions of empty space. 0.09 gives about 640,000 agents
    # at the default slab, which is the same order as the 1.5 M the layered
    # stack runs at 1440p and comfortably inside the budget either way. It is
    # also the parameter most likely to want moving once somebody has watched
    # this on a real GPU -- see DESIGN.md §13.
    density: float = 0.09
    # The climate is a volume too, so a regime occupies a region of the slab
    # rather than a column through it. Coarse for the same reason its 2D
    # counterpart is: it is sampled trilinearly and must never develop an edge.
    climate_width: int = 32
    climate_height: int = 18
    climate_depth: int = 6
    psi_scale: int = 8  # vector-potential grid divisor

    # --- Motion anisotropy ------------------------------------------------
    # The slab is four or five feature-widths deep, so isotropic flow would
    # carry material through the whole thickness in a few seconds and the
    # depth axis would read as churn rather than as depth. Both of these damp
    # motion along the slab normal; see the notes on the matching fields in
    # `gpu_params` for why the flow one weights the *potential* and not the
    # velocity.
    depth_flow: float = 0.45
    depth_agent: float = 0.45

    # --- Raymarch ---------------------------------------------------------
    # A *ceiling* on the march, which otherwise takes one step per slice --
    # ample for parallax and occlusion given that the far field is fogged and
    # blurred anyway (§5.1). At the 48-deep default the two are the same
    # number and this changes nothing; past 160 slices the step grows longer
    # than a voxel, which the march's static jitter turns into fixed noise
    # rather than into banding, and which the depth-of-field blur is already
    # doing much worse to at that end of the slab. Render cost is linear in it,
    # so this is the value to lower first if a thick slab will not hold the
    # frame budget.
    steps: int = 160
    # How far off orthographic the rays are. Small on purpose: a
    # near-orthographic camera keeps ray coherence excellent, and the depth cue
    # this application wants comes from occlusion and attenuation rather than
    # from perspective.
    converge: float = 0.055
    # How far each face fades over, in voxels. Material leaving the near face
    # reappears at the far one -- the domain is toroidal in all three axes,
    # exactly as it is in two under the layered backend -- and this window is
    # what makes that arrival and departure a fade rather than a step.
    #
    # In voxels rather than as a fraction of the thickness, because what has to
    # be smooth is material's *arrival*, and that happens at a rate set by how
    # fast the material moves and how large it is -- neither of which knows how
    # deep the slab is. Held as a fraction, a thick slab would dim its whole
    # crisp near face to hide a seam that a few voxels already hide. 8.6 voxels
    # is what the 0.18 of the depth this used to be came to at the 48-deep slab
    # it was chosen on.
    depth_window_voxels: float = 8.6

    # --- The single soft light -------------------------------------------
    # Direction the light comes *from*, in slab coordinates. Slightly above,
    # slightly to one side, and mostly from behind the viewer, which is the
    # arrangement that reads as volume without casting shadows long enough to
    # look theatrical.
    light_x: float = -0.45
    light_y: float = 0.70
    light_z: float = -0.55
    # How much of the lighting survives full shadow. High, because the point of
    # self-shadowing here is a depth cue and not drama, and because a deep
    # shadow moving across the field is a large local luminance change -- which
    # the safety stage would bound anyway, but at the cost of lagging the image.
    light_ambient: float = 0.55
    shadow_steps: int = 6
    shadow_density: float = 1.8
    # How far the shadow ray probes, in voxels -- filament-relative rather than
    # slab-relative, for the same reason the face window is. Self-shadowing is
    # interior shading of a structure about six voxels across, so the useful
    # reach is a few filament widths whatever the slab's thickness; measured in
    # slab depths it would grow with the thickness and six steps would be
    # sampling a reach they cannot resolve, which is blotches rather than
    # shading. 21.6 voxels is what the 0.45 slab depths this used to be came to
    # at the 48-deep slab it was calibrated on.
    shadow_voxels: float = 21.6


@dataclass
class RenderParams:
    """Compositing, depth, and the Oklab colour mapping. DESIGN.md §5-6.

    Everything from ``parallax`` down is read by both depth backends, which is
    what makes the `depth` macro mean one thing: under the layered backend the
    values describe the backmost layer, under the volumetric one they describe
    the far face of the slab. ``layers`` and the three falloffs above that are
    layered-only, and the slab's own geometry lives in :class:`VolumeParams`.
    """

    layers: int = 3
    base_scale: float = 1.0  # front layer, fraction of display resolution
    scale_falloff: float = 0.5  # each layer back is this much smaller
    # Back layers get larger on-screen features for free by being simulated at
    # lower resolution, so this is a fine-tuning knob rather than the main
    # mechanism; >1 exaggerates the difference.
    feature_falloff: float = 1.0
    # Screen-relative speed of each layer back. Combined with scale_falloff in
    # the engine, since a cell on a half-resolution layer covers twice the
    # screen distance.
    tempo_falloff: float = 0.70

    # How far the viewpoint drifts, as a fraction of the field's lateral
    # extent. A *maximum* rather than a typical value: the walk behind it is
    # bounded, and sits at about half of this most of the time.
    #
    # Under the layered backend this offsets each sheet by a different amount,
    # which is what separates them; under the volumetric one it is the whole of
    # the camera's motion. Either way it is the only depth cue here that comes
    # from the scene *moving*: everything else on this list -- the focus
    # falloff, the fog, the dimming, the desaturation -- is a function of
    # normalised depth, so it says the same thing about the far face however
    # far away that face actually is. Which makes this the one that carries
    # depth the others cannot describe, and the one worth raising first if a
    # display makes the image look flat.
    parallax: float = 0.020
    # Seconds; how long the drift takes to change its mind. Slow enough to read
    # as weather rather than as sway, and floored in `validate` for the same
    # reason -- a short one would turn the strongest depth cue here into the
    # kind of coordinated global movement DESIGN.md 4.2 exists to prevent.
    parallax_tau: float = 75.0
    dof_radius: float = 3.2  # cells, at the backmost layer
    fog_amount: float = 0.42  # atmospheric attenuation at the backmost layer
    depth_dim: float = 0.55  # luminance retained at the backmost layer
    depth_desat: float = 0.45  # chroma retained at the backmost layer
    extinction: float = 2.6  # Beer-Lambert coefficient

    # Luminance. Defaults sit in the "dark ground, moderate luminous filament"
    # register.
    background_luma: float = 0.030
    filament_luma: float = 0.360  # user-adjustable filament brightness
    glow_gamma: float = 0.78  # <1 lifts faint structure without raising peaks
    l_max: float = 0.620  # hard ceiling on Oklab L

    # Chroma and hue.
    c_max: float = 0.145
    chroma_activity_gain: float = 5.5
    chroma_floor: float = 0.012
    hue_turns_per_hour: float = 1.33  # ~45 min per full rotation
    hue_anchor: float = 0.0  # radians; set by the palette macro
    hue_spread: float = 0.85  # spatial hue variation, radians
    # The polychrome palette (DESIGN.md §14.4): the climate hue channel picks
    # among three hue families +-120 degrees apart, through a smooth
    # three-plateau warp. `polychrome` scales the family separation -- 0 is
    # the regulation mapping exactly, 1 the full triad -- and is driven by the
    # activation table's intensity curve; regulation pins it at 0.
    # `polychrome_threshold` is where the transitions sit, in the channel's
    # *realised* units (the field settles at s.d. ~0.11 -- §4.1): 0.06 puts
    # roughly two fifths of the field in the middle family and three tenths
    # in each of the others. The transition width is tied to it (2.5 / t in
    # the shader), so this one value moves the wells and their ramps together.
    polychrome: float = 0.0
    polychrome_threshold: float = 0.06


@dataclass
class Params:
    """Complete primitive parameter set."""

    sim_hz: float = 20.0
    max_fps: int = 30
    agents: AgentParams = field(default_factory=AgentParams)
    reaction: ReactionParams = field(default_factory=ReactionParams)
    flow: FlowParams = field(default_factory=FlowParams)
    pigment: PigmentParams = field(default_factory=PigmentParams)
    climate: ClimateParams = field(default_factory=ClimateParams)
    volume: VolumeParams = field(default_factory=VolumeParams)
    homeostat: HomeostatParams = field(default_factory=HomeostatParams)
    events: EventParams = field(default_factory=EventParams)
    render: RenderParams = field(default_factory=RenderParams)
    safety: SafetyParams = field(default_factory=SafetyParams)


@dataclass
class Macros:
    """The knobs the control panel exposes. All in 0..1."""

    intensity: float = 0.50
    scale: float = 0.50
    tempo: float = 0.45
    palette: float = 0.30
    brightness: float = 0.35
    filament_glow: float = 0.45
    depth: float = 0.60
    # How far the viewpoint swings, and how briskly. Its own knob rather than
    # a part of `depth` for the same reason `event_rate` is not part of
    # `intensity`: the two are different questions. Everything `depth` moves is
    # a shading trick applied to a *normalised* depth -- how much the far face
    # is fogged, dimmed, desaturated and blurred -- and says the same thing
    # whatever is actually back there. This one is the only cue that comes from
    # the scene moving, and it is the one that answers "is there really
    # something behind that". Split out so it can be turned up on a display
    # where the shading tricks are not enough on their own.
    #
    # Defaulted high while the mechanism is being judged on real hardware --
    # see `load`, and lower it once it has been.
    parallax: float = 0.60
    # How often the scheduler's own events arrive. Its own knob rather than a
    # part of `intensity`, because the two are not the same question: a dense,
    # busy field that is left alone for an hour at a time is a coherent thing
    # to want, and so is a sparse one that keeps being interrupted. Coupling
    # them meant nobody could ask for either. See `MACRO_CURVES`.
    event_rate: float = 0.50


# --------------------------------------------------------------------------
# Macro -> primitive curves
# --------------------------------------------------------------------------

# path, low value, high value, gamma. The macro is raised to ``gamma`` before
# lerping, so gamma > 1 gives finer control at the low end.
#
# This is the *regulation* table -- the tuning the application shipped with,
# and the one every value in it was measured against. The activation mode
# (DESIGN.md §14) has its own, :data:`ACTIVATION_CURVES` below, reached
# through :data:`MODE_CURVES`; this name stays because everything historical
# about these values ("the shipped curve", the legacy event-rate inversion)
# means this table specifically.
MACRO_CURVES: dict[str, list[tuple[str, float, float, float]]] = {
    "intensity": [
        ("agents.density", 0.10, 0.44, 1.0),
        ("agents.deposit", 0.009, 0.028, 1.0),
        ("agents.fusion_bias", 0.35, 0.72, 1.0),
        ("reaction.trail_feed_gain", 0.012, 0.034, 1.0),
        ("render.chroma_activity_gain", 3.5, 8.0, 1.0),
        ("pigment.inject_rate", 0.032, 0.085, 1.0),
        # Held flat here and driven for real by the activation table: the two
        # tables must drive the same paths under each macro (a slider that
        # goes dead, or gains a hidden effect, crossing modes is what the
        # structure test forbids), and regulation's look was tuned with these
        # at their defaults, so here the curve pins them there.
        ("render.c_max", 0.145, 0.145, 1.0),
        ("render.chroma_floor", 0.012, 0.012, 1.0),
        ("render.polychrome", 0.0, 0.0, 1.0),
    ],
    "scale": [
        # Larger scale == coarser features: slower agents, longer sensors,
        # faster diffusion. Sensing and diffusion move *together*, holding
        # `sensor_distance / trail_diffuse` at ~2.6 across the whole range:
        # above ~4 the agent layer condenses onto a single axis-aligned strand
        # (DESIGN.md §4.9), and the old 4.0-12.0 sweep was over that threshold
        # along its entire length.
        ("agents.sensor_distance", 2.2, 4.9, 1.0),
        ("agents.trail_diffuse", 0.85, 1.90, 1.0),
        # A cohort is a length like the other two, and has to stay inside the
        # sensing reach that moves alongside it.
        ("agents.found_radius", 1.6, 3.5, 1.0),
        ("reaction.du", 0.16, 0.26, 1.0),
        ("reaction.dv", 0.080, 0.130, 1.0),
        ("flow.psi_noise_scale", 2.0, 5.0, 1.0),
    ],
    "tempo": [
        ("sim_hz", 12.0, 26.0, 1.0),
        ("agents.speed", 0.55, 1.35, 1.0),
        ("flow.psi_gain", 0.70, 2.10, 1.0),
        ("flow.field_gain", 0.45, 1.30, 1.0),
        ("flow.psi_theta", 0.0012, 0.0038, 1.0),
        ("climate.advect_gain", 0.12, 0.38, 1.0),
        ("render.hue_turns_per_hour", 0.55, 2.60, 1.2),
        # Held flat here, driven by the activation table (§14.6): regulation
        # events take the §4.3 minute-or-two to come up whatever the tempo.
        # Same paired-constant pattern as intensity's c_max above.
        ("events.attack_seconds", 45.0, 45.0, 1.0),
        ("events.release_seconds", 90.0, 90.0, 1.0),
        # The structure rides the flow at the shipped rate whatever the tempo
        # (§4.7 step 6, measured on main): this is a *constant at the default*,
        # not a disable. Pinning it to zero here would switch off the trail
        # advection that dissolves the hubs, which is a fix for a real visual
        # defect rather than an activation flourish.
        ("agents.trail_advect", 0.5, 0.5, 1.0),
    ],
    "palette": [
        # Palette selects a hue anchor; the spatial spread widens slightly at
        # the extremes so the ends of the range are not flat monochrome.
        ("render.hue_spread", 0.55, 1.25, 1.0),
    ],
    "brightness": [
        ("render.background_luma", 0.012, 0.075, 1.0),
        ("render.l_max", 0.44, 0.78, 1.0),
        ("safety.exposure_target", 0.10, 0.26, 1.0),
    ],
    "filament_glow": [
        # The user-facing "how luminous are the filaments" control.
        ("render.filament_luma", 0.16, 0.62, 1.0),
        ("render.glow_gamma", 0.92, 0.62, 1.0),
        ("render.extinction", 1.9, 3.6, 1.0),
    ],
    "depth": [
        ("render.dof_radius", 1.2, 5.4, 1.0),
        ("render.fog_amount", 0.18, 0.62, 1.0),
        ("render.depth_dim", 0.78, 0.38, 1.0),
        ("render.depth_desat", 0.72, 0.30, 1.0),
    ],
    # How far the viewpoint swings and how fast, together, because they are one
    # question: more parallax means both more travel and more of it per second,
    # and a knob that moved only the travel would take four times as long to
    # show twice as much.
    #
    # The travel reaches a quarter of the screen's width between the near and
    # far material, which is a great deal -- far past anything that would be
    # called restrained, and deliberately so. The old range topped out at 0.038
    # and was chosen against a mechanism that did not work (see
    # `Backend._update_parallax`), so it is not evidence of anything. The
    # volumetric backend holds itself to whatever fraction of this its slab is
    # thick enough to justify; see `volume.py`.
    "parallax": [
        ("render.parallax", 0.0, 0.25, 1.0),
        # Seconds. Shorter is faster, so this runs the other way. The fast end
        # is 60 rather than something brisker because the travel is doing the
        # work: at the top of this knob the viewpoint crosses a quarter of the
        # screen, and it does not also need to hurry.
        ("render.parallax_tau", 150.0, 60.0, 1.0),
    ],
    # Mean arrival rate of the scheduler's own events, and nothing else: this
    # knob moves *when* perturbations come, never what they do when they get
    # here. Amplitude, radius, envelope and the concurrency cap are all
    # untouched, so turning the rate up cannot make an event punctuate; it can
    # only make the field spend more of its time inside one (DESIGN.md 4.3).
    #
    # The span is wide on purpose -- 0.5/hour is roughly one every two hours,
    # 20/hour roughly one every three minutes -- because this is the knob whose
    # useful setting varies most between "background presence" and "something
    # to watch". gamma 1.5 puts the interesting end where the hand is: the
    # centre of the travel resolves to 7.4/hour -- the ~8-minute mean interval
    # 4.3 describes, and within a couple of per cent of the 7.5/hour
    # `EventParams.rate_per_hour` has always defaulted to -- so an untouched
    # slider is the setting the field was tuned at, and the top half of the
    # travel covers everything faster than that.
    "event_rate": [
        ("events.rate_per_hour", 0.5, 20.0, 1.5),
        # A constant is not a coupling: the knob still moves nothing but the
        # rate (§4.3's promise, kept), and the *mode* sets the cap -- 4 here,
        # 6 under activation, where overlap is the point of the fast end.
        ("events.max_concurrent", 4.0, 4.0, 1.0),
    ],
}

# Macro values that are not a simple lerp.
def _palette_hue_anchor(v: float) -> float:
    """Palette macro 0..1 maps to a full hue circle."""
    return v * TAU


# --------------------------------------------------------------------------
# Modes -- DESIGN.md §14
# --------------------------------------------------------------------------

# Two tunings of one instrument. Regulation is everything the application has
# always been; activation is the same eight macros over endpoints retuned for
# sensory seeking -- more motion, more colour, more happening. The mode decides
# which curve table `resolve` reads and nothing else: the safety ceilings are
# one table serving both, every value a mode moves is one the ramp smooths, and
# no geometry follows it -- which is why, unlike `backend`, switching mode is a
# live transition on the running field rather than a reset.
MODES = ("regulation", "activation")
DEFAULT_MODE = "regulation"

# The activation table. Same eight macros, same meanings, same paths in the
# same order (the structure test holds the two tables to that); what differs
# is where the top of each travel reaches. Low ends stay at regulation's, so
# the modes overlap rather than abut and the bottom of activation is
# recognisably the same instrument.
#
# The endpoints rest on the step 2 measurements (DESIGN.md §14.8): the WCAG
# area criterion does not bind the tempo axis anywhere up to 6x the
# regulation tops -- worst per-frame per-pixel |dL| grows only 0.018 -> 0.043
# across that whole sweep -- so the tempo tops here (1.6-1.9x) are perceptual
# choices carrying an order of magnitude of measured headroom, to be judged
# on real hardware. And the homeostat holds every measure in band at these
# endpoints with *smaller* corrections than the regulation busy corner needs,
# so there are no per-mode homeostat targets; the bands of §4.2 serve both.
ACTIVATION_CURVES: dict[str, list[tuple[str, float, float, float]]] = {
    "intensity": [
        ("agents.density", 0.10, 0.44, 1.0),
        ("agents.deposit", 0.009, 0.028, 1.0),
        ("agents.fusion_bias", 0.35, 0.72, 1.0),
        ("reaction.trail_feed_gain", 0.012, 0.034, 1.0),
        # Chroma carries this mode (§14.1): change is far less provocative in
        # chroma than in luminance, and the budget was mostly unspent.
        ("render.chroma_activity_gain", 3.5, 11.0, 1.0),
        ("pigment.inject_rate", 0.032, 0.085, 1.0),
        # Toward the 0.22 ceiling, not to it -- gamut-mapping pressure rises
        # with chroma and the margin is deliberate.
        ("render.c_max", 0.145, 0.205, 1.0),
        # Quiet regions stay coloured instead of falling to grey.
        ("render.chroma_floor", 0.012, 0.035, 1.0),
        # The polychrome warp (§14.4): at the top, three full hue families
        # +-120 degrees apart, chosen per region by the climate. On intensity
        # rather than palette because it is a "how much colour" question --
        # palette says *where* the families sit, this says how far apart they
        # are -- and the low end at zero keeps the bottom of the travel the
        # same instrument as regulation, like every other curve here.
        ("render.polychrome", 0.0, 1.0, 1.0),
    ],
    "scale": [
        # The whole knob biased ~15% finer at the top: busier texture. The
        # low ends hold at regulation's, since going *below* them has
        # measured risk -- du under ~0.17 walks activity toward the homeostat
        # floor (§4.7), and the shipped 0.16 is already at the edge. Sensing
        # and diffusion move together as ever, ratio ~2.6 across the whole
        # travel (§4.9); found_radius keeps its ~0.72x of the reach.
        ("agents.sensor_distance", 2.2, 4.2, 1.0),
        ("agents.trail_diffuse", 0.85, 1.63, 1.0),
        ("agents.found_radius", 1.6, 3.0, 1.0),
        ("reaction.du", 0.16, 0.22, 1.0),
        ("reaction.dv", 0.080, 0.110, 1.0),  # dv/du held at 0.50, as §4.7 requires
        ("flow.psi_noise_scale", 2.0, 4.3, 1.0),
    ],
    "tempo": [
        # 30 Hz at the top: one sim tick per displayed frame at the 30 FPS
        # cap. Budget: ~21 GB/s at 30 Hz, ~3% of the target card (§8.1).
        ("sim_hz", 12.0, 30.0, 1.0),
        ("agents.speed", 0.55, 2.2, 1.0),
        ("flow.psi_gain", 0.70, 4.0, 1.0),
        ("flow.field_gain", 0.45, 2.4, 1.0),
        # Faster reversion: the weather changes its mind sooner.
        ("flow.psi_theta", 0.0012, 0.0055, 1.0),
        ("climate.advect_gain", 0.12, 0.55, 1.0),
        # A full turn in 7-8 minutes at the top -- visible drift, not a spin;
        # the 8 s ramp tau on this path smooths any adjustment to it.
        ("render.hue_turns_per_hour", 0.55, 8.0, 1.2),
        # Shorter envelopes at the fast end (§14.6): an event still arrives
        # as a raised cosine, still through the climate's diffusion and the
        # colour pipeline's lowpasses, still under the 25% area cap -- it
        # just takes ~15 s to come up instead of a minute. On tempo rather
        # than event_rate, because §4.3 promises that knob moves timing and
        # nothing else, and how briskly a perturbation builds is exactly a
        # tempo question. The heal claim is re-verified at this pace: the
        # rift recovery test has always run a 10 s attack, faster than this
        # end's worst case (15 x 0.75 jitter).
        ("events.attack_seconds", 45.0, 15.0, 1.0),
        ("events.release_seconds", 90.0, 40.0, 1.0),
        # §4.7 step 6 shipped on main at 0.5 while this branch was in flight,
        # so activation's contribution is no longer the mechanism but *more of
        # it*: the structure is carried at up to 0.8 of the pigment's speed at
        # the top of tempo, against the 0.5 both modes share at the bottom.
        # Below 1.0 deliberately -- at parity the network would ride the flow
        # exactly as the pigment does, and the shear that stretches and pinches
        # filaments is precisely the *difference* between the two rates.
        ("agents.trail_advect", 0.5, 0.80, 1.0),
    ],
    "palette": [
        # Most of the hue circle in play at once. This is spread around the
        # anchor; simultaneous *contrasting* families are step 4's warp.
        ("render.hue_spread", 0.55, 2.8, 1.0),
    ],
    # Nothing about activation wants a different luminance architecture, so
    # brightness, glow, depth and the viewpoint keep regulation's curves.
    "brightness": [
        ("render.background_luma", 0.012, 0.075, 1.0),
        ("render.l_max", 0.44, 0.78, 1.0),
        ("safety.exposure_target", 0.10, 0.26, 1.0),
    ],
    "filament_glow": [
        ("render.filament_luma", 0.16, 0.62, 1.0),
        ("render.glow_gamma", 0.92, 0.62, 1.0),
        ("render.extinction", 1.9, 3.6, 1.0),
    ],
    "depth": [
        ("render.dof_radius", 1.2, 5.4, 1.0),
        ("render.fog_amount", 0.18, 0.62, 1.0),
        ("render.depth_dim", 0.78, 0.38, 1.0),
        ("render.depth_desat", 0.72, 0.30, 1.0),
    ],
    "parallax": [
        ("render.parallax", 0.0, 0.25, 1.0),
        ("render.parallax_tau", 150.0, 60.0, 1.0),
    ],
    "event_rate": [
        # One every ~90 s at the top. Still arrival-time only: what an event
        # *does* is untouched by this knob; the envelope rides tempo above.
        ("events.rate_per_hour", 0.5, 40.0, 1.5),
        # Six at once instead of four: at one event every ~90 s with ~2-minute
        # envelopes, overlap is the normal condition rather than the corner,
        # and a cap that refused it would turn the fast end into a queue of
        # refusals. Constant, so the knob itself still moves only the rate.
        ("events.max_concurrent", 6.0, 6.0, 1.0),
    ],
}

MODE_CURVES: dict[str, dict[str, list[tuple[str, float, float, float]]]] = {
    "regulation": MACRO_CURVES,
    "activation": ACTIVATION_CURVES,
}


def normalise_mode(name: object) -> str:
    """The mode a name asks for, or the default with a warning.

    Untrusted like every other value off disk, and handled the way
    :func:`normalise_backend` handles its argument: an unrecognised mode is a
    typo, and regulation -- the application's whole character until §14 -- is
    the safe thing to open with.
    """
    text = str(name or "").strip().lower()
    if text in MODES:
        return text
    if text:
        log.warning(
            "unknown mode %r; using %r (known: %s)",
            name, DEFAULT_MODE, ", ".join(MODES),
        )
    return DEFAULT_MODE


# --------------------------------------------------------------------------
# Hard safety ceilings -- see DESIGN.md §7
# --------------------------------------------------------------------------

SAFETY_CEILINGS: dict[str, tuple[float, float]] = {
    # path: (minimum, maximum)
    # 0.012 gives 1.8 flashes/s at 30 FPS in the worst case (a sustained
    # maximum-rate oscillation), against the WCAG limit of 3. The 0.03 this
    # was originally set to allows 4.5/s and is NOT safe -- see
    # test_ceiling_implies_wcag_margin. This entry alone is not the whole
    # bound: the flash arithmetic is per-frame times frame rate, so `validate`
    # additionally holds the *product* to MAX_LUMA_PER_SECOND -- at the 60 FPS
    # this table permits, 0.012 per frame would be 3.6 flashes/s, over the
    # WCAG limit rather than under it.
    "safety.max_luma_delta": (0.0005, 0.012),
    "safety.max_chroma_delta": (0.0005, 0.100),
    "safety.iir_alpha": (0.02, 1.000),
    "safety.exposure_attack": (0.0, 0.050),
    "safety.exposure_release": (0.0, 0.050),
    "safety.exposure_target": (0.02, 0.400),
    "render.l_max": (0.05, 0.900),
    "render.c_max": (0.0, 0.220),
    "render.filament_luma": (0.0, 0.900),
    "render.background_luma": (0.0, 0.300),
    "events.max_radius_frac": (0.0, 0.250),
    "sim_hz": (4.0, 60.0),
    "max_fps": (5, 60),
}

# The lightness slew budget per *second* -- the quantity the WCAG arithmetic
# actually runs on. The per-frame ceiling above is this at the design's 30 FPS
# (0.012 x 30), and the worst case it permits is budget / 0.2 = 1.8 flashes/s,
# whatever the frame rate: a sustained maximum-rate oscillation spends
# 2 x 10% of lightness per flash pair however the frames are sliced. Found
# while writing DESIGN.md §14.5(2): the per-frame ceiling alone, at the 60 FPS
# the table permits, allows 3.6/s -- above the WCAG limit of 3, not below it.
MAX_LUMA_PER_SECOND = 0.36


# --------------------------------------------------------------------------
# Dataclass path helpers
# --------------------------------------------------------------------------


def clamp_du(du_raw: float, reaction: "ReactionParams") -> float:
    """Constrain a local diffusion rate to the survivable band.

    Mirrors ``reaction.wgsl`` exactly, for the same reason
    :func:`clamp_reaction` does: the morphology tests must exercise the values
    the shader actually reaches, not the unclamped ones. If one changes, the
    other must.
    """
    return min(max(du_raw, reaction.du_min), reaction.du_max)


def clamp_sensor_distance(distance_raw: float, agents: "AgentParams") -> float:
    """Constrain the sensing reach to the width of what it senses.

    Mirrors ``agents.wgsl`` exactly, for the same reason :func:`clamp_du` does.
    An agent that senses much further than a strand is wide stops following the
    strand it is on and starts driving at whatever it can see from a distance,
    which on a torus ends with the whole population on one straight
    axis-aligned strand (DESIGN.md §4.9). If one changes, the other must.
    """
    return min(max(distance_raw, 1.0), agents.sensor_reach_max * agents.trail_diffuse)


def clamp_reaction(
    feed_raw: float, kill_raw: float, reaction: "ReactionParams"
) -> tuple[float, float]:
    """Constrain (feed, kill) to the live band.

    This mirrors the clamping in ``reaction.wgsl`` exactly and exists so the
    regime tests exercise the same logic the shader does rather than an
    approximation of it. If one changes, the other must.
    """
    feed = min(max(feed_raw, reaction.feed_min), reaction.feed_max)
    centre = reaction.kill + reaction.kill_follows_feed * (feed - reaction.feed)
    lo = max(centre - reaction.kill_band, reaction.kill_min)
    hi = min(centre + reaction.kill_band, reaction.kill_max)
    if lo > hi:  # bands crossed; the absolute bounds win
        lo = hi = min(max(centre, reaction.kill_min), reaction.kill_max)
    return feed, min(max(kill_raw, lo), hi)


def get_path(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    current = getattr(obj, parts[-1])
    # Preserve integer-typed primitives (substeps, psi_scale, layers, ...).
    if isinstance(current, bool):
        setattr(obj, parts[-1], bool(value))
    elif isinstance(current, int):
        setattr(obj, parts[-1], int(round(value)))
    else:
        setattr(obj, parts[-1], float(value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def curve_value(macro: str, path: str, value: float, mode: str = DEFAULT_MODE) -> float:
    """The value ``path`` takes when the macro driving it sits at ``value``.

    One curve, evaluated on its own rather than by building a whole
    :class:`Params`. The control panel needs this to say what a slider position
    *means* -- "about one every eight minutes" rather than "0.50" -- and the
    live parameters cannot answer that, because they are ramping and would show
    the value catching up rather than the one just asked for.

    ``mode`` picks the curve table, because the same slider position means
    different values under different modes (DESIGN.md §14) and a readout that
    quoted the regulation curve while the activation one was driving would be
    lying in exactly the way this function exists to prevent.

    The macro curve only. An explicit override of the same path beats the macro
    in :meth:`Config.resolve` and is not visible here, which is the honest
    answer for a slider readout: the slider is not what is deciding then.
    """
    value = min(1.0, max(0.0, float(value)))
    for entry_path, lo, hi, gamma in MODE_CURVES[normalise_mode(mode)].get(macro, ()):
        if entry_path == path:
            return _lerp(lo, hi, value**gamma if gamma != 1.0 else value)
    raise KeyError(f"macro {macro!r} does not drive {path!r}")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


BACKENDS = ("layered", "volumetric")
DEFAULT_BACKEND = "layered"

# The slab's lateral resolution, as three named sizes rather than a free
# integer. ``volume.width`` is still the primitive the geometry is derived
# from and a hand-written override of it still wins; what this adds is a small
# set of sizes that have been costed against each other, so that choosing one
# does not mean working out for yourself what a given width is worth.
#
# Why the width is the knob worth having. Every tier leaves ``volume.depth``
# wherever the thickness knob has put it, and lets the height follow the
# window, so the voxels stay cubic and a Gray-Scott feature stays the same
# handful of voxels across -- which is
# exactly why a wider slab is sharper rather than merely bigger: the same
# feature is drawn into more voxels and therefore lands on fewer display
# pixels. At the standard size a filament is about five pixels wide at 1440p,
# which is the blur the layered backend does not have (DESIGN.md §5.1); at the
# finest it is about two and a half.
#
# What it costs, and what it does not. Voxel count goes with the square of the
# ratio, and three things follow it: the simulation passes, the per-frame
# interpolation over the slab, and memory. The ray march follows none of them
# -- its cost is output pixels times ``volume.steps``, and the step count
# follows the depth, which no tier changes. So the render side of a 1440p
# frame costs the same at every size here, and only the simulation grows.
#
# Agent count follows the voxel count too, since ``volume.density`` is per
# voxel: about 640,000 at the standard size, 1.4 M at fine, 2.6 M at finest.
# The last of those is above the 1.5 M the layered backend runs at 1440p, so
# somebody running `finest` on a smaller card may want `volume.density` a
# little lower; it is left alone here because the deposit is atomic adds into
# a buffer rather than bandwidth, and it is not what was blurry.
VOLUME_DETAIL: dict[str, int] = {
    "standard": 512,
    "fine": 768,
    "finest": 1024,
}
DEFAULT_VOLUME_DETAIL = "standard"


@dataclass
class Config:
    """Macros plus explicit primitive overrides.

    ``resolve()`` produces the :class:`Params` the engine consumes.

    ``backend`` chooses how depth is put on the screen (DESIGN.md §5): the
    layered 2.5D stack, or the volumetric slab of §5.1. It sits here rather
    than in :class:`Params` because it is not a parameter -- nothing ramps it,
    and it decides which class the engine *is*. Like every other structural
    value it takes effect when a field is grown, so switching it needs a reset
    or a relaunch; the two backends keep separate saved fields, so switching
    away and back resumes what was there.

    ``volume_detail`` chooses how wide the volumetric slab is, from
    :data:`VOLUME_DETAIL`. It sits beside ``backend`` for the same reasons --
    it is a name rather than a quantity, nothing ramps it, and it decides the
    shape a field is grown in -- and it is read only when the volumetric
    backend is running. Unlike a backend switch there is only one volumetric
    field, so changing the size does not set the old one aside for later: the
    saved field keeps the size it was grown at until it is reset.

    ``mode`` chooses which curve table the macros resolve through
    (:data:`MODE_CURVES`, DESIGN.md §14). It sits beside ``backend`` because it
    is the other "which application is this" choice, but it has the opposite
    character: it is *not* structural. It moves no geometry, allocates nothing,
    and every value it changes is one :class:`ParamRamp` smooths -- so
    switching mode is a live, ramped transition on the running field, exactly
    like a preset switch, and a field checkpointed in one mode resumes cleanly
    into the other.
    """

    macros: Macros = field(default_factory=Macros)
    overrides: dict[str, float] = field(default_factory=dict)
    preset_name: str = "default"
    backend: str = DEFAULT_BACKEND
    volume_detail: str = DEFAULT_VOLUME_DETAIL
    mode: str = DEFAULT_MODE

    def resolve(self) -> Params:
        params = Params()

        for macro_name, curves in MODE_CURVES[normalise_mode(self.mode)].items():
            value = getattr(self.macros, macro_name)
            value = min(1.0, max(0.0, float(value)))
            for path, lo, hi, gamma in curves:
                t = value**gamma if gamma != 1.0 else value
                set_path(params, path, _lerp(lo, hi, t))

        # Non-lerp macro effects.
        params.render.hue_anchor = _palette_hue_anchor(
            min(1.0, max(0.0, self.macros.palette))
        )

        # The named slab size, which is a choice rather than a curve. Applied
        # after the macros and before the overrides, so that it beats nothing
        # (no macro drives it) and loses to an explicit `volume.width`, which
        # is the same precedence every other named setting has.
        params.volume.width = VOLUME_DETAIL[
            normalise_volume_detail(self.volume_detail)
        ]

        # Explicit overrides win over macros.
        for path, value in self.overrides.items():
            try:
                set_path(params, path, value)
            except AttributeError:
                log.warning("unknown parameter override %r, ignoring", path)

        validate(params)
        return params


def validate(params: Params) -> Params:
    """Clamp every safety-relevant value to its hard ceiling.

    Clamps rather than raises: this runs on hot-reload of a file the user is
    editing by hand during a multi-hour session, and killing the session over a
    typo would be a worse outcome than silently correcting it.
    """
    for path, (lo, hi) in SAFETY_CEILINGS.items():
        value = get_path(params, path)
        clamped = min(hi, max(lo, value))
        if clamped != value:
            log.warning(
                "parameter %s = %g is outside the permitted range [%g, %g]; "
                "clamped to %g",
                path,
                value,
                lo,
                hi,
                clamped,
            )
            set_path(params, path, clamped)

    # The luma slew ceiling is per frame, and the flash arithmetic multiplies
    # it by the frame rate, so the two ceilings above are only jointly safe:
    # the product is held to MAX_LUMA_PER_SECOND here, after both have been
    # clamped. At the design's 30 FPS this binds at exactly the table's 0.012
    # and changes nothing; at a raised frame-rate cap the per-frame allowance
    # shrinks so the worst case stays 1.8 flashes/s. Conservative on purpose:
    # `max_fps` is the fastest the canvas may present, and a frame rate below
    # it only slows the worst case further.
    fps = max(float(params.max_fps), 1.0)
    per_frame = MAX_LUMA_PER_SECOND / fps
    if params.safety.max_luma_delta > per_frame:
        log.warning(
            "safety.max_luma_delta = %g at max_fps = %g exceeds the "
            "%g/second flash budget; clamped to %g",
            params.safety.max_luma_delta, params.max_fps,
            MAX_LUMA_PER_SECOND, per_frame,
        )
        params.safety.max_luma_delta = per_frame

    # The diffusion band is a stability bound, not a stylistic one: this is an
    # explicit scheme, and the averaging-form Laplacian in reaction.wgsl goes
    # unstable somewhere near dt*du = 1. A hand-edited override of du_max is
    # the one way to reach that, so it is bounded here for the same reason the
    # flash ceilings are -- by the time it shows up in the image the run is
    # already lost.
    reaction = params.reaction
    stability_ceiling = 0.9 / max(reaction.dt, 1e-3)
    reaction.du_max = min(max(reaction.du_max, 1e-3), stability_ceiling)
    reaction.du_min = min(max(reaction.du_min, 1e-3), reaction.du_max)

    # Structural values that are not floats.
    params.render.layers = max(1, min(5, int(params.render.layers)))
    params.reaction.substeps = max(1, min(8, int(params.reaction.substeps)))
    params.flow.psi_scale = max(1, min(16, int(params.flow.psi_scale)))
    params.events.max_concurrent = max(0, min(8, int(params.events.max_concurrent)))
    # A founding period of zero would divide by zero picking the epoch, and a
    # period of one would reselect the site every tick, which is the uniform
    # respawn this exists to replace.
    params.agents.found_period = max(2, min(20_000, int(params.agents.found_period)))
    params.agents.found_site_cells = max(
        64, min(1 << 24, int(params.agents.found_site_cells)))
    # The sensing cap is a ratio to the equilibrium mean trail; the absolute
    # value -- and the liveness floor that keeps it clear of the starve
    # threshold -- is computed where it is packed, in `_physics_values`. Here
    # only the ratio's own sanity: zero (or below) stays zero, which disables
    # the cap outright, and a huge ratio is indistinguishable from disabled
    # but would still pack a live clamp, so it is bounded.
    agents = params.agents
    agents.sense_cap = 0.0 if agents.sense_cap <= 0.0 else min(
        agents.sense_cap, 100.0)
    # The viewpoint's drift has to stay a drift. The flash bound does not
    # depend on this -- the limiter is per-pixel and holds whatever the camera
    # does -- but a time constant of a second or two would make the whole image
    # sway, which is the coordinated global motion of DESIGN.md 4.2 rather than
    # the parallax this is for.
    params.render.parallax_tau = min(max(params.render.parallax_tau, 8.0), 3600.0)
    # And it has to stay on the screen. Beyond a whole screen width between the
    # near and far material there is nothing left that reads as one scene.
    params.render.parallax = min(max(params.render.parallax, 0.0), 1.0)

    # The slab's own structural values. The ceilings are core WebGPU's
    # guaranteed maxTextureDimension3D (2048), and the floors are what the
    # passes need to be meaningful at all: a slab one voxel deep is the layered
    # backend with extra steps, and a march of a couple of steps cannot resolve
    # occlusion. The thickness gets the full 2048 here because the bound that
    # actually applies to it is relative -- no deeper than the shorter lateral
    # axis -- and `VolumeGeometry.derive` is where that can be known.
    volume = params.volume
    volume.width = max(32, min(2048, int(volume.width)))
    volume.depth = max(8, min(2048, int(volume.depth)))
    volume.climate_width = max(4, min(256, int(volume.climate_width)))
    volume.climate_height = max(4, min(256, int(volume.climate_height)))
    volume.climate_depth = max(2, min(64, int(volume.climate_depth)))
    volume.psi_scale = max(1, min(32, int(volume.psi_scale)))
    volume.steps = max(8, min(256, int(volume.steps)))
    volume.shadow_steps = max(0, min(32, int(volume.shadow_steps)))
    # Both are lengths in voxels. The floor on the window is what keeps the
    # toroidal seam a fade: at zero it is a step, which is exactly the
    # punctuation §7 exists to prevent.
    volume.depth_window_voxels = min(
        max(float(volume.depth_window_voxels), 1.0), 4096.0)
    volume.shadow_voxels = min(max(float(volume.shadow_voxels), 0.0), 4096.0)
    return params


def normalise_backend(name: object) -> str:
    """The backend a name asks for, or the default with a warning.

    Untrusted like every other value off disk or off a command line, and not
    worth failing a launch over: an unrecognised backend is a typo, and the
    layered one is the safe thing to open with.
    """
    text = str(name or "").strip().lower()
    if text in BACKENDS:
        return text
    if text:
        log.warning(
            "unknown backend %r; using %r (known: %s)",
            name, DEFAULT_BACKEND, ", ".join(BACKENDS),
        )
    return DEFAULT_BACKEND


def normalise_volume_detail(name: object) -> str:
    """The slab size a name asks for, or the default with a warning.

    Untrusted for the same reasons ``normalise_backend``'s argument is, and
    handled the same way: a size that is not one of the three is a typo, and
    the standard one is both the smallest and the one every other default was
    chosen against, so it is the safe thing to open with.

    A bare width is accepted as well as a name, since "1024" is the obvious
    thing to write in a config file whose neighbouring key is a number of
    voxels, and refusing it would be pedantry.
    """
    text = str(name or "").strip().lower()
    if text in VOLUME_DETAIL:
        return text
    for tier, width in VOLUME_DETAIL.items():
        if text == str(width):
            return tier
    if text:
        log.warning(
            "unknown volume detail %r; using %r (known: %s)",
            name, DEFAULT_VOLUME_DETAIL,
            ", ".join(f"{t} ({w})" for t, w in VOLUME_DETAIL.items()),
        )
    return DEFAULT_VOLUME_DETAIL


# --------------------------------------------------------------------------
# Parameter ramping -- no parameter change is ever a step. DESIGN.md §9.
# --------------------------------------------------------------------------

# Per-path time constants in seconds; anything unlisted uses the default.
RAMP_TAU_DEFAULT = 1.5
RAMP_TAU: dict[str, float] = {
    "render.background_luma": 4.0,
    "render.filament_luma": 4.0,
    "render.l_max": 4.0,
    "render.c_max": 4.0,
    "safety.exposure_target": 6.0,
    "render.hue_turns_per_hour": 8.0,
    "render.fog_amount": 3.0,
    "render.depth_dim": 3.0,
    "render.depth_desat": 3.0,
    # Structural / integer values snap immediately; ramping them is meaningless.
    "sim_hz": 0.5,
    "render.hue_anchor": 12.0,
}

# Paths whose value is an angle in radians and must ramp along the shortest
# arc. Lerping these linearly would send the palette knob the long way round
# the colour circle -- a slow but very visible global hue sweep.
CIRCULAR_PATHS = frozenset({"render.hue_anchor"})


class ParamRamp:
    """Exponentially approaches a target parameter set.

    Adjusting a control must not itself cause visual punctuation, so every
    float reaching the GPU is smoothed. Integers and bools snap, since they are
    structural (layer count, substeps) rather than perceptual.
    """

    def __init__(self, params: Params) -> None:
        self.current = copy.deepcopy(params)
        self.target = copy.deepcopy(params)

    def set_target(self, params: Params) -> None:
        self.target = copy.deepcopy(params)

    def update(self, dt: float) -> Params:
        _ramp_dataclass(self.current, self.target, dt, prefix="")
        return self.current

    def snap(self, params: Params) -> None:
        self.current = copy.deepcopy(params)
        self.target = copy.deepcopy(params)


def _ramp_dataclass(cur: Any, tgt: Any, dt: float, prefix: str) -> None:
    for f in fields(cur):
        name = f.name
        path = f"{prefix}{name}"
        cur_value = getattr(cur, name)
        tgt_value = getattr(tgt, name)

        if is_dataclass(cur_value):
            _ramp_dataclass(cur_value, tgt_value, dt, prefix=f"{path}.")
        elif isinstance(cur_value, bool) or isinstance(cur_value, int):
            setattr(cur, name, tgt_value)
        elif isinstance(cur_value, float):
            tau = RAMP_TAU.get(path, RAMP_TAU_DEFAULT)
            alpha = 1.0 - math.exp(-dt / max(tau, 1e-4))
            if path in CIRCULAR_PATHS:
                delta = (tgt_value - cur_value + math.pi) % TAU - math.pi
                setattr(cur, name, (cur_value + delta * alpha) % TAU)
            else:
                setattr(cur, name, cur_value + (tgt_value - cur_value) * alpha)


# --------------------------------------------------------------------------
# TOML persistence
# --------------------------------------------------------------------------

_HEADER = """\
# Anastomosis configuration.
#
# Edit and save: changes are hot-reloaded, and every parameter is ramped
# smoothly rather than stepped, so it is safe to adjust while running.
#
# `backend` chooses how depth is drawn: "layered" (three 2.5D sheets) or
# "volumetric" (a raymarched slab). It is structural, so it takes effect on a
# new field -- reset the simulation, or relaunch. Each backend keeps its own
# saved field.
#
# `volume_detail` is how wide that slab is, and only matters under the
# volumetric backend: "standard" (512 voxels across), "fine" (768) or
# "finest" (1024). Wider is sharper and costs more simulation -- roughly the
# square of the ratio, in both memory and GPU time, while the raymarch itself
# costs the same at all three. Structural like `backend`, so a saved field
# keeps the size it grew at until the simulation is reset.
#
# `mode` is which tuning the eight knobs move through: "regulation" (calm,
# the original) or "activation" (more motion and colour, for sensory seeking
# rather than settling). Not structural: switching it is a smooth transition
# on the field you already have. The flash-safety bound is identical in both.
#
# [macros] are the normal interface -- eight knobs, all 0..1.
# [overrides] pins individual primitive parameters by dotted path, e.g.
#   "render.filament_luma" = 0.42
# Overrides take precedence over macros.
#
# Safety-relevant values are clamped to hard ceilings on load (see
# DESIGN.md §7); an out-of-range value is corrected with a warning rather
# than rejected.
"""


# The arrival rate used to be one of the primitives `intensity` drove, over the
# curve ``lerp(2.5, 14.0, intensity ** 1.3)``. A config file written before the
# rate became a knob of its own says nothing about it and yet implies one, and
# someone who turned the intensity down in order to be left alone should not
# find the field interrupting them half again as often because they upgraded.
# So the old curve is inverted here, once, for a file that predates the split.
_LEGACY_INTENSITY_RATE = (2.5, 14.0, 1.3)


def _event_rate_from_intensity(intensity: float) -> float:
    """The `event_rate` a pre-split file meant by its `intensity`."""
    lo_was, hi_was, gamma_was = _LEGACY_INTENSITY_RATE
    was = _lerp(lo_was, hi_was, min(1.0, max(0.0, float(intensity))) ** gamma_was)
    for path, lo, hi, gamma in MACRO_CURVES["event_rate"]:
        if path != "events.rate_per_hour" or hi == lo:
            continue
        t = min(1.0, max(0.0, (was - lo) / (hi - lo)))
        return t ** (1.0 / gamma)
    return Macros().event_rate


def load(path: str | Path) -> Config:
    import tomllib

    path = Path(path)
    if not path.exists():
        log.info("no config at %s, using defaults", path)
        return Config()

    with path.open("rb") as fh:
        data = tomllib.load(fh)

    macros = Macros()
    table = data.get("macros") or {}
    for key, value in table.items():
        if hasattr(macros, key):
            setattr(macros, key, min(1.0, max(0.0, float(value))))
        else:
            log.warning("unknown macro %r in %s, ignoring", key, path)

    if "parallax" not in table and "depth" in table:
        # A file written before the viewpoint drift became a knob of its own.
        # Unlike the event-rate split, there is nothing here to carry across:
        # `depth` did drive `render.parallax`, but over a range chosen against
        # a walk that never moved (see `Backend._update_parallax`), so whatever
        # the old file says about parallax is a description of a setting that
        # did nothing. Inheriting it would be preserving a bug's configuration.
        # The new default stands instead, and says so, because a knob appearing
        # at a value the file did not ask for should not be silent.
        log.info(
            "%s predates the parallax macro; the viewpoint drift starts at "
            "%.2f, which is deliberately strong -- the mechanism it drives was "
            "inert until now and wants judging on a real display",
            path, macros.parallax,
        )

    if "event_rate" not in table and "intensity" in table:
        macros.event_rate = _event_rate_from_intensity(macros.intensity)
        log.info(
            "%s predates the event-rate macro; carrying its intensity of %.2f "
            "across as an event rate of %.2f",
            path, macros.intensity, macros.event_rate,
        )

    overrides = {str(k): v for k, v in (data.get("overrides") or {}).items()}
    return Config(
        macros=macros,
        overrides=overrides,
        preset_name=str(data.get("preset_name", "default")),
        backend=normalise_backend(data.get("backend", DEFAULT_BACKEND)),
        volume_detail=normalise_volume_detail(
            data.get("volume_detail", DEFAULT_VOLUME_DETAIL)),
        # A file predating the mode says nothing about it, and needs no
        # migration note the way the split-out macros did: every such file was
        # written for the regulation mode, which is what the default is.
        mode=normalise_mode(data.get("mode", DEFAULT_MODE)),
    )


def save(config: Config, path: str | Path) -> None:
    import tomlkit

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = tomlkit.document()
    for line in _HEADER.splitlines():
        doc.add(tomlkit.comment(line.lstrip("#").rstrip()))
    doc.add(tomlkit.nl())

    doc.add("preset_name", config.preset_name)
    doc.add("backend", normalise_backend(config.backend))
    doc.add("volume_detail", normalise_volume_detail(config.volume_detail))
    doc.add("mode", normalise_mode(config.mode))

    macros = tomlkit.table()
    for f in fields(config.macros):
        macros.add(f.name, round(float(getattr(config.macros, f.name)), 4))
    doc.add("macros", macros)

    overrides = tomlkit.table()
    for key, value in sorted(config.overrides.items()):
        overrides.add(key, value)
    doc.add("overrides", overrides)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(tomlkit.dumps(doc), encoding="utf-8")
    tmp.replace(path)  # atomic, so a hot-reload never sees a half-written file
    log.info("wrote config to %s", path)


def default_config_path() -> Path:
    import os

    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "anastomosis" / "config.toml"
