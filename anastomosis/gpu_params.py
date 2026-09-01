"""GPU parameter block layout.

The WGSL struct definitions here are *generated* from the field lists, and the
same lists drive the numpy packing. A mismatch between shader-side and
host-side layout is the classic source of silent, baffling GPU bugs, so the two
are derived from one declaration rather than maintained in parallel.

Every field is a 4-byte scalar (``f32`` or ``u32``). No vectors, so std430
packing is exactly "one word per field, in order", with no padding rules to get
wrong.
"""

from __future__ import annotations

import numpy as np

Field = tuple[str, str]

# --------------------------------------------------------------------------
# Simulation parameters -- bound to every sim pass.
# --------------------------------------------------------------------------

SIM_FIELDS: list[Field] = [
    # Dimensions and identity
    ("dims_x", "u32"),
    ("dims_y", "u32"),
    # The third dimension of each grid. One under the layered backend, which
    # simply never reads it; the slab's real depth under the volumetric one.
    # Shared rather than forked because these are the numbers a *macro* has to
    # mean the same thing through, and one struct is one place to keep the
    # host-side packing and the shader-side layout in step.
    ("dims_z", "u32"),
    ("clim_w", "u32"),
    ("clim_h", "u32"),
    ("clim_d", "u32"),
    ("psi_w", "u32"),
    ("psi_h", "u32"),
    ("psi_d", "u32"),
    ("tick", "u32"),
    ("seed", "u32"),
    ("agent_count", "u32"),
    ("layer_index", "u32"),
    ("layer_count", "u32"),
    ("event_count", "u32"),
    # Agents
    ("speed", "f32"),
    ("sensor_angle", "f32"),
    ("sensor_distance", "f32"),
    ("sensor_distance_max", "f32"),
    ("turn_rate", "f32"),
    ("jitter", "f32"),
    ("deposit", "f32"),
    ("fusion_bias", "f32"),
    ("fusion_max", "f32"),
    ("trail_decay", "f32"),
    ("trail_diffuse", "f32"),
    ("income_rate", "f32"),
    ("prune_gain", "f32"),
    # Deposit capacity: trail level at which a deposit is halved. The counter
    # to winner-take-all trail following -- DESIGN.md 4.9 notes the layer has
    # "no capacity limit", and the measured consequence is a network holding
    # half its mass in its top 2% of texels: the white hubs. Zero disables.
    ("deposit_cap", "f32"),
    # Sensing saturation: ceiling on the trail value the agents' sensors can
    # read. The capacity above bounds what a hub stores; this bounds what it
    # can *attract* -- unbounded sensing is what collapsed the layer into
    # stationary knots instead of a network (DESIGN.md 4.7, "the network that
    # was never there"). Zero disables.
    ("sense_cap", "f32"),
    # Fraction of the velocity field the trail is advected by (pigment gets
    # 1.0). Zero disables, and the pass is then texel-exact.
    ("trail_advect", "f32"),
    ("starve_threshold", "f32"),
    ("max_age", "f32"),
    ("found_fraction", "f32"),
    ("found_period", "u32"),
    ("found_site_cells", "u32"),
    ("found_radius", "f32"),
    # Reaction
    ("feed", "f32"),
    ("kill", "f32"),
    ("du", "f32"),
    ("dv", "f32"),
    ("du_min", "f32"),
    ("du_max", "f32"),
    ("rdt", "f32"),
    ("trail_feed_gain", "f32"),
    ("kill_follows_feed", "f32"),
    ("trail_seed_gain", "f32"),
    ("trail_seed_falloff", "f32"),
    ("feed_min", "f32"),
    ("feed_max", "f32"),
    ("kill_band", "f32"),
    ("kill_min", "f32"),
    ("kill_max", "f32"),
    # Flow
    ("psi_gain", "f32"),
    ("field_gain", "f32"),
    ("psi_theta", "f32"),
    ("psi_sigma", "f32"),
    ("psi_noise_scale", "f32"),
    ("advect_dt", "f32"),
    # Climate
    ("clim_theta", "f32"),
    ("clim_sigma", "f32"),
    ("clim_advect", "f32"),
    ("clim_diffuse", "f32"),
    ("range_feed", "f32"),
    ("range_kill", "f32"),
    ("range_sensor_angle", "f32"),
    ("range_sensor_distance", "f32"),
    ("range_deposit", "f32"),
    ("range_decay", "f32"),
    ("range_flow", "f32"),
    ("range_hue", "f32"),
    ("range_du", "f32"),
    ("range_prune", "f32"),
    ("range_repel", "f32"),
    # Pigment / colour injection
    ("hue_anchor", "f32"),
    ("hue_spread", "f32"),
    ("hue_from_orientation", "f32"),
    ("hue_inject_mix", "f32"),
    # The polychrome palette's multi-well warp (DESIGN.md §14.4): gain and
    # transition threshold. Gain zero -- the regulation mode -- makes the
    # warp identically zero and the hue mapping what it always was.
    ("polychrome", "f32"),
    ("polychrome_threshold", "f32"),
    ("inject_rate", "f32"),
    ("activity_rate", "f32"),
    ("activity_gain", "f32"),
    ("density_from_v", "f32"),
    ("density_from_trail", "f32"),
    ("v_needs_trail", "f32"),
    # Soft knee on the trail's rendered contribution: the shaded term is
    # `density_from_trail * knee * tanh(trail / knee)`, near-linear below the
    # knee and bounded above it, so a hub cannot clip to a flat disc however
    # much mass it holds. Zero or negative means linear.
    ("trail_knee", "f32"),
    # Generic blur pass control (reused by trail diffuse and DOF)
    ("blur_radius", "f32"),
    ("blur_dir_x", "f32"),
    ("blur_dir_y", "f32"),
    ("blur_dir_z", "f32"),
    # Per-layer feel
    ("feature_scale", "f32"),
    ("tempo_scale", "f32"),
    # Generic sanitise pass bounds
    ("sanitize_min", "f32"),
    ("sanitize_max", "f32"),
    ("sanitize_fallback", "f32"),
    # Homeostat
    ("target_mass", "f32"),
    ("target_variance", "f32"),
    ("target_activity", "f32"),
    ("deadband", "f32"),
    ("gain_p", "f32"),
    ("gain_i", "f32"),
    ("integral_limit", "f32"),
    ("homeo_rate", "f32"),
    # The feature-size loop (DESIGN.md 4.7 step 5). `ell_offset` is the log
    # deviation the setpoint walk is asking for *this* tick, which is the only
    # part of the mechanism that lives on the host: the walk is accumulated
    # state, so it belongs with the hue phase and the parallax drift rather
    # than in a shader. The three rates are per-tick forms of the time
    # constants in `ReactionParams`, computed against `sim_hz` the same way
    # `homeo_rate` is, so the tempo macro cannot change how the loop behaves.
    ("ell_offset", "f32"),
    ("ell_rate", "f32"),
    ("ell_ref_rate", "f32"),
    ("ell_corr_limit", "f32"),
    # Volumetric slab only (DESIGN.md §5.1).
    #
    # `depth_flow` weights the *lateral* components of the flow's vector
    # potential, which is how the slab gets motion that is mostly in plane
    # without giving up the exactly divergence-free velocity field: velocity is
    # the curl of whatever potential is stored, so weighting the potential
    # keeps the identity while scaling `v_z` by roughly this factor. Scaling
    # `v_z` directly would not -- `div(g*v) = grad(g).v` -- and pigment would
    # accumulate in the places the scaling compresses.
    ("depth_flow", "f32"),
    # The agents' step along the slab normal, as a fraction of their lateral
    # step. Agents are not a conserved density, so this one is a plain scale.
    ("depth_agent", "f32"),
]

# --------------------------------------------------------------------------
# Rhizotron parameters -- bound to the root backend's passes (DESIGN.md §15).
#
# Its own block rather than more fields on SimParams, because the rhizotron
# shares no simulation machinery with the fungal backends: nothing here means
# anything to agents.wgsl, and none of the Gray-Scott plumbing means anything
# to soil. What the backends *do* share -- the output chain -- speaks
# RenderParams, which the rhizotron packs exactly as the other two do.
# --------------------------------------------------------------------------

RHIZ_FIELDS: list[Field] = [
    # Texture shape, and where the window sits inside it. The texture is the
    # view plus a margin above and below (see rhizotron.py): the margin above
    # is where the display offset may reach between ticks, the margin below is
    # where rows are generated before the view arrives at them.
    ("dims_x", "u32"),
    ("dims_y", "u32"),
    ("view_rows", "u32"),
    ("margin_top", "u32"),
    # The world row of texture row 0, as a u64 split across two words. This is
    # the depth counter of DESIGN.md §15.4: an integer, never a float, used
    # only as hash input -- §3's pattern exactly -- so the soil below can never
    # repeat and never loses precision however long the descent runs.
    ("origin_lo", "u32"),
    ("origin_hi", "u32"),
    # How many whole rows the world moves up through the texture this tick.
    # Zero on most ticks; the pass reads its source at (x, y + scroll) so the
    # shift costs nothing and resamples nothing.
    ("scroll_rows", "u32"),
    ("tick", "u32"),
    ("seed", "u32"),
    ("event_count", "u32"),
    # Vertical noise lattice sizes, as log2 of rows. Powers of two because the
    # lattice cell of an unbounded u64 row is found by shift and mask, which is
    # exact forever; a division would need 64-bit arithmetic WGSL does not have.
    ("strata_shift", "u32"),
    ("stone_shift", "u32"),
    ("grain_shift", "u32"),
    # Horizontal lattice sizes, in cells around the wrapping x axis -- integers
    # so the noise tiles the cylinder exactly (see common.wgsl on why tiling
    # is a requirement, not a nicety).
    ("stone_cells_x", "u32"),
    ("grain_cells_x", "u32"),
    # Where the top of the *view* sits in the texture this frame, in fractional
    # rows -- margin_top plus the sub-row remainder of the descent, interpolated
    # between the last two ticks so the scroll presents as a continuous glide
    # rather than a per-tick step.
    ("base_row", "f32"),
    # Lateral aspect correction: how much of the (wrapping) x axis the window
    # samples. The vertical axis never wraps, so it takes no correction; see
    # rhiz_composite.wgsl.
    ("x_span", "f32"),
    # Moisture physics, all in per-tick units -- converted from the per-second
    # values in RhizotronParams against sim_hz where they are packed, so the
    # tempo macro cannot retune the physics by changing the tick rate.
    ("perc_rate", "f32"),
    ("perc_pow", "f32"),
    ("lat_spread", "f32"),
    ("drain_rate", "f32"),
    ("rain_base", "f32"),
    ("rain_event_gain", "f32"),
    ("cond_floor", "f32"),
    ("moisture_baseline", "f32"),
    ("wet_ema_rate", "f32"),
    # Soil generation and its look.
    ("strata_tilt", "f32"),
    ("stone_amount", "f32"),
    ("grain_amount", "f32"),
    ("hardpan_amount", "f32"),
    ("biopore_amount", "f32"),
    # The nutrient economy (step 5, second half). Rates arrive per tick.
    ("nutrient_baseline", "f32"),
    ("nutrient_cache", "f32"),
    ("nutrient_uptake", "f32"),
    ("nutrient_recycle", "f32"),
    ("nutrient_spread", "f32"),
    ("chemo_gain", "f32"),
    ("forage_gain", "f32"),
    ("soil_l_floor", "f32"),
    ("soil_l_range", "f32"),
    ("wet_darken", "f32"),
    ("wet_chroma", "f32"),
    # The surface (§17.4): the texture row of the soil line, fractional.
    # Negative or past the bottom means no surface is in the pane (a sunk
    # §15 column, or surface_frac zero).
    ("surface_row", "f32"),
    # --- The plant (§15.11 step 3) ----------------------------------------
    # Tip pool shape. Slots are allocated by position in the tree, not by an
    # atomic bump: axis a is slot a, lateral (a, l) is A + a*L + l, fine
    # (a, l, f) is A + A*L + (a*L + l)*F + f. Each parent owns its children's
    # slots outright and hands them out from its own counter, so branching is
    # deterministic under any GPU scheduling -- the property the bit-identical
    # resume test needs -- and needs no atomics at all.
    ("max_axes", "u32"),
    ("laterals_per_axis", "u32"),
    ("fines_per_lateral", "u32"),
    ("tips_total", "u32"),
    # Tropism steering, per order where an order needs its own value. The
    # gravitropic setpoint angle is measured off straight down; the axis's is
    # zero by definition.
    ("elong_axis", "f32"),      # cells per tick, this order's elongation
    ("elong_lateral", "f32"),
    ("elong_fine", "f32"),
    # Growth decelerates with age toward a floor: factor
    # floor + (1-floor) * exp(-age * elong_slow), with elong_slow the
    # per-tick decay. What lets the window keep pace with the front.
    ("elong_floor", "f32"),
    ("elong_slow", "f32"),
    ("gsa_lateral", "f32"),     # radians off vertical
    ("gsa_fine", "f32"),
    ("gsa_gain_axis", "f32"),   # angular relaxation toward the setpoint, /tick
    ("gsa_gain_lateral", "f32"),
    ("gsa_gain_fine", "f32"),
    ("thigmo_gain", "f32"),
    ("hydro_gain", "f32"),
    ("avoid_gain", "f32"),
    ("tip_turn", "f32"),        # radians per tick, the flank-steering step
    ("tip_jitter", "f32"),
    ("sense_dist", "f32"),      # cells ahead of the tip
    ("sense_angle", "f32"),     # radians off-axis for the flank probes
    # Branching.
    ("spacing_axis", "f32"),    # cells between laterals along an axis
    ("spacing_lateral", "f32"), # cells between fines along a lateral
    ("branch_prob", "f32"),     # per tick, once the spacing is met
    ("branch_angle", "f32"),    # radians the child leaves its parent at
    ("branch_jitter", "f32"),
    # Deposits into the structure field.
    ("tip_deposit", "f32"),     # per cell travelled
    ("splat_axis", "f32"),      # gaussian sigma, cells
    ("splat_lateral", "f32"),
    ("splat_fine", "f32"),
    # Lifetimes, in ticks (converted from seconds where they are packed).
    ("fine_life", "f32"),
    ("lateral_life", "f32"),
    ("axis_life", "f32"),
    ("dt_seconds", "f32"),      # one tick, for ageing the structure in seconds
    # Root shading.
    ("root_knee", "f32"),       # density at half coverage
    ("root_edge", "f32"),       # transfer softness; the §15.7(2) sweep's knob
    ("root_age_scale", "f32"),  # seconds to brown
    ("root_brown", "f32"),      # how far a browned root sinks toward the soil
    ("root_hair", "f32"),       # the pale skirt around young material
    ("mycorrhiza", "f32"),      # the cool shimmer in young fine fuzz
    # --- The record layer (§17.6) -----------------------------------------
    # Lignification per tick (relaxation form), the avoidance weight of wood,
    # and the wood shading's transfer and maturation.
    ("lignify_rate", "f32"),
    # Seasons (§17.6): the interment's per-tick rate (zero outside one),
    # the ghost's share of interred mass, and the generational dim -- 1.0
    # on every ordinary tick, (1 - ghost_fade) for exactly the one tick
    # of the fossil moment, when the standing strata step a generation
    # deeper before the new burial lays its own.
    ("intern_rate", "f32"),
    ("ghost_gain", "f32"),
    ("ghost_dim", "f32"),
    ("wood_avoid", "f32"),
    ("wood_edge", "f32"),
    ("wood_age_scale", "f32"),
    # --- The long-duration core (§15.11 step 4) ---------------------------
    # Senescence: per-tick decay of fine structure, gated by age.
    ("senesce_rate", "f32"),
    ("senesce_delay", "f32"),   # seconds before fine material starts to go
    # Succession: how long a spent axis rests before it may re-germinate, in
    # ticks, and the per-tick germination chances -- one earned by moisture,
    # one unconditional floor so a drought cannot end the world (§15.4's
    # absorbing-state argument).
    ("regerm_delay", "f32"),
    ("germ_prob", "f32"),
    ("germ_floor", "f32"),
    ("germ_moisture", "f32"),   # the wetness that makes a seed eager
]

# --------------------------------------------------------------------------
# Small Strange Things parameters -- bound to the port's passes (DESIGN.md
# §18).
#
# Its own block for the same reason the rhizotron has one: the Things share
# no simulation machinery with the other backends. Everything dynamic
# arrives per-tick, converted from the per-second values in ThingsParams
# against sim_hz where it is packed, so the tick rate cannot retune their
# world (§18.1 soul 10).
# --------------------------------------------------------------------------

THINGS_FIELDS: list[Field] = [
    ("dims_x", "u32"),
    ("dims_y", "u32"),
    ("capacity", "u32"),
    # The lottery's ceiling (§18.1 soul 4). The buffer is larger by the
    # click reserve, because in the founding file the cap only ever gated
    # reproduction -- the click handler pushed unconditionally. The click
    # outranks the cap (soul 9); a full village still answers the finger.
    ("soft_cap", "u32"),
    ("tick", "u32"),
    ("seed", "u32"),
    # Click-to-add (§18.1 soul 9): up to four clicks consumed per tick, in
    # field texels, each spawning `per_click` Things. Empty slots claim them
    # by rank; see things_update.wgsl.
    ("click_count", "u32"),
    ("per_click", "u32"),
    ("click0_x", "f32"),
    ("click0_y", "f32"),
    ("click1_x", "f32"),
    ("click1_y", "f32"),
    ("click2_x", "f32"),
    ("click2_y", "f32"),
    ("click3_x", "f32"),
    ("click3_y", "f32"),
    ("click_scatter", "f32"),
    # Wander: the per-tick scale on the trait speed. The founding file
    # stepped `(rand - 0.5) * speed` per 60 fps frame; this factor is
    # sqrt(60 * dt) so the realised diffusion per second matches at any
    # tick rate -- times the world scale below, so the diffusion covers
    # the same *fraction* of the world at any resolution.
    ("step_scale", "f32"),
    # The same-beings-at-every-resolution law (§18, round 2): every
    # length in this block arrives pre-multiplied by field_width /
    # WORLD_WIDTH, and this factor carries the scale for the lengths the
    # shaders derive themselves (the trait size, the sparkle's stamp).
    # The founding file was pixel-native because it only ever lived at
    # one window; the port lives at every size.
    ("world_scale", "f32"),
    # Friendship (soul 2). Probability per tick; radius in texels. The
    # three-friend cap is a law, not a knob -- it is a constant in the
    # shader.
    ("friend_prob", "f32"),
    ("friend_radius", "f32"),
    # Lineage (soul 4). Probability per empty slot per tick, the maturity
    # gate in ticks, and the birth scatter.
    ("spawn_prob", "f32"),
    ("mature_ticks", "f32"),
    ("spawn_radius", "f32"),
    # The birth fade-in, in ticks (the founding file's age/50 frames).
    ("fadein_ticks", "f32"),
    # The canvas field (soul 7): per-tick decay, already 1 - exp(-rate*dt).
    ("fade", "f32"),
    # Deposit amplitudes. Bodies and bonds are sustained emitters and
    # arrive per-tick (emit-per-second * dt) so steady-state canvas level
    # is emit/fade_rate whatever the tick rate; a sparkle is an *event* and
    # its amplitude is absolute.
    ("body_emit", "f32"),
    ("bond_emit", "f32"),
    ("bond_width", "f32"),
    # Bond span differentiation (§18.2, the round-2 ruling): intra-village
    # bonds are background hum at near-founding hairline; the long emigrant
    # lines keep their earned width and presence. The ramp runs from
    # friend_radius (formation scale) to a few multiples of it.
    ("bond_near_width", "f32"),
    ("bond_near_gain", "f32"),
    ("sparkle_amp", "f32"),
    ("sparkle_prob", "f32"),
    ("sparkle_offset", "f32"),
    # The glow skirt (§18.2): radius as a multiple of the body's, and its
    # amplitude relative to the core.
    ("glow_mult", "f32"),
    ("glow_gain", "f32"),
    # The breath layer (§18.1 soul 7, incarnated properly): the founding
    # file's wander-shadows were an 8-bit canvas quantisation floor -- a
    # rounding error that became a soul. Here it is explicit: the canvas
    # alpha channel max-accumulates a ghost of everywhere light has been,
    # decaying on a slow clock (ghost_fade is per tick), rendered at
    # ghost_luma as a faint cool-grey breath under the villages.
    ("ghost_gain", "f32"),
    ("ghost_fade", "f32"),
    ("ghost_luma", "f32"),
    # The pulse (the founding sin(time*0.05 + x*0.01) * 0.5): accumulated
    # phase (host state, checkpointed), spatial frequency, amplitude.
    ("pulse_phase", "f32"),
    ("pulse_x", "f32"),
    ("pulse_amp", "f32"),
    # Composite: aspect correction (both axes wrap; see
    # backend.aspect_correction) and the canvas-to-HDR gain.
    ("x_scale", "f32"),
    ("y_scale", "f32"),
    ("out_gain", "f32"),
    # The painter's order, restored (§18 round 5): the display-linear
    # level a body's own colour is drawn at when the compositor paints
    # the owned core source-over the canvas -- emit/fade, the same
    # steady-state level the trail holds, so the look the review ratified
    # keeps its brightness while neighbours lose their vote in it.
    ("body_level", "f32"),
]

# --------------------------------------------------------------------------
# Render parameters -- bound to composite / safety / blit.
# --------------------------------------------------------------------------

RENDER_FIELDS: list[Field] = [
    ("out_w", "u32"),
    ("out_h", "u32"),
    ("layer_count", "u32"),
    ("frame", "u32"),
    ("seed", "u32"),
    # Volumetric slab only: the grid the ray marches through, and how many
    # steps it takes doing it.
    ("vol_w", "u32"),
    ("vol_h", "u32"),
    ("vol_d", "u32"),
    ("march_steps", "u32"),
    ("shadow_steps", "u32"),
    ("pad0", "u32"),
    ("pad1", "u32"),
    # Temporal interpolation between the last two sim states.
    ("frac", "f32"),
    ("interp_dt", "f32"),
    # Compositing
    ("extinction", "f32"),
    ("fog_r", "f32"),
    ("fog_g", "f32"),
    ("fog_b", "f32"),
    # Tone mapping
    ("background_luma", "f32"),
    ("filament_luma", "f32"),
    ("glow_gamma", "f32"),
    ("l_max", "f32"),
    ("c_max", "f32"),
    ("chroma_activity_gain", "f32"),
    ("chroma_floor", "f32"),
    ("hue_global", "f32"),
    # Safety stage
    ("max_luma_delta", "f32"),
    ("max_chroma_delta", "f32"),
    ("iir_alpha", "f32"),
    ("exposure_target", "f32"),
    ("exposure_attack", "f32"),
    ("exposure_release", "f32"),
    ("exposure_max", "f32"),
    ("dither_amount", "f32"),
    ("reproject_scale", "f32"),
    # --- Volumetric slab only (DESIGN.md §5.1) ---------------------------
    # Lateral zoom, which is the aspect correction the layered backend puts in
    # its per-layer records.
    ("zoom_x", "f32"),
    ("zoom_y", "f32"),
    # Camera. `shear` is the differential lateral offset between the near and
    # far faces -- this is what parallax *is* in a volume, rather than a
    # per-layer offset -- and `converge` spreads the rays slightly off
    # orthographic so the slab is seen at an angle away from the centre.
    ("cam_shear_x", "f32"),
    ("cam_shear_y", "f32"),
    ("converge", "f32"),
    # World thickness of the slab, as a fraction of its lateral extent. Voxels
    # are cubic, so this is just depth/width.
    ("slab_depth", "f32"),
    # The soft window that fades the two faces, as a fraction of the thickness.
    # The slab is a 3-torus like the rest of the domain (there are no walls
    # anywhere in this simulation), so material leaving the near face reappears
    # at the far one; without a window that arrival and departure would be a
    # step at the depth extremes. Configured in voxels -- see
    # `VolumeParams.depth_window_voxels` -- and divided by the thickness on the
    # way here, since a fraction is the coordinate the march has.
    ("depth_window", "f32"),
    # Atmospheric attenuation with depth, matching the layered backend's
    # per-layer values at the backmost layer.
    ("depth_dim", "f32"),
    ("depth_desat", "f32"),
    ("depth_fog", "f32"),
    # Depth of field: lateral blur radius in voxels at the far face.
    ("dof_radius", "f32"),
    # The single soft light: direction, and how much of the lighting is
    # ambient rather than shadowed.
    ("light_x", "f32"),
    ("light_y", "f32"),
    ("light_z", "f32"),
    ("light_ambient", "f32"),
    ("shadow_density", "f32"),
    # How far the shadow ray probes, as a world length -- the lateral extent of
    # the slab is 1. Configured in voxels for the same reason the face window
    # is: it is calibrated against a filament, not against the thickness.
    ("shadow_reach", "f32"),
]

# Per-layer compositing data, one record per layer in a storage array.
LAYER_FIELDS: list[Field] = [
    ("scale_x", "f32"),
    ("scale_y", "f32"),
    ("parallax_x", "f32"),
    ("parallax_y", "f32"),
    ("depth_dim", "f32"),
    ("depth_desat", "f32"),
    ("fog", "f32"),
    ("opacity", "f32"),
]

# Events, one record per active event.
EVENT_FIELDS: list[Field] = [
    ("pos_x", "f32"),
    ("pos_y", "f32"),
    # Where the event sits through the slab. Ignored by the layered backend,
    # whose climate has no third axis; the scheduler draws it either way, so an
    # event that outlives a backend switch does not have to be re-placed.
    ("pos_z", "f32"),
    ("radius", "f32"),
    ("strength", "f32"),
    ("chan_feed", "f32"),
    ("chan_kill", "f32"),
    ("chan_flow", "f32"),
    ("chan_hue", "f32"),
    # Added for rift events (DESIGN.md 4.7 step 4). Without these an event can
    # thin material -- `dieback` lowers feed and raises kill -- but it cannot
    # sever anything, because severance lives in the trail decay, the pruning
    # term and the agents' junction behaviour, and none of those was reachable
    # from an event.
    ("chan_decay", "f32"),
    ("chan_prune", "f32"),
    ("chan_repel", "f32"),
]


# Per-tile output of the reduction pass. Unlike every block above, this one is
# never packed on the host -- only its size is, to allocate the buffer -- so it
# is written as a pair of vec4s rather than derived from a scalar field list.
# The second vec4 carries the trail flux balance the pruning term needs
# centred, and the gradient sum the feature-size loop needs; see reduce.wgsl
# and DESIGN.md §4.7.
PARTIAL_WGSL = """\
struct Partial {
    field: vec4<f32>,  // sum V, sum V^2, sum |dV/dt|, count
    flux: vec4<f32>,   // sum trail, sum trail*deficit, sum |grad V|, spare
    dep: vec4<f32>,    // sum income EMA, sum withheld EMA, spare, spare
};"""

# Three vec4s. std430 gives a vec4 16-byte alignment and this struct holds
# nothing else, so the stride is exactly the sum of the members. The third was
# added for the deposit-capacity accounting: the return needs both what the
# capacity let through and what it withheld, and the second vec4 had one lane
# left where two were needed.
PARTIAL_SIZE = 48


def wgsl_struct(name: str, fields: list[Field]) -> str:
    lines = [f"struct {name} {{"]
    lines += [f"    {fname}: {ftype}," for fname, ftype in fields]
    lines.append("};")
    return "\n".join(lines)


def _dtype(fields: list[Field]) -> np.dtype:
    return np.dtype(
        [(name, np.float32 if ftype == "f32" else np.uint32) for name, ftype in fields]
    )


SIM_DTYPE = _dtype(SIM_FIELDS)
RHIZ_DTYPE = _dtype(RHIZ_FIELDS)
THINGS_DTYPE = _dtype(THINGS_FIELDS)
RENDER_DTYPE = _dtype(RENDER_FIELDS)
LAYER_DTYPE = _dtype(LAYER_FIELDS)
EVENT_DTYPE = _dtype(EVENT_FIELDS)


def pack(dtype: np.dtype, values: dict[str, float]) -> np.ndarray:
    """Build a single record, defaulting any unset field to zero.

    Unknown keys raise: a typo'd parameter name that silently did nothing would
    be very hard to notice in a system whose output is a slowly drifting image.
    """
    unknown = set(values) - set(dtype.names)
    if unknown:
        raise KeyError(f"unknown parameter field(s): {sorted(unknown)}")
    record = np.zeros(1, dtype=dtype)
    for key, value in values.items():
        record[key] = value
    return record


def pack_array(dtype: np.dtype, rows: list[dict[str, float]]) -> np.ndarray:
    if not rows:
        return np.zeros(1, dtype=dtype)
    out = np.zeros(len(rows), dtype=dtype)
    for i, row in enumerate(rows):
        unknown = set(row) - set(dtype.names)
        if unknown:
            raise KeyError(f"unknown parameter field(s): {sorted(unknown)}")
        for key, value in row.items():
            out[i][key] = value
    return out


# Statistics / homeostat state, shared between the reduce pass and the climate
# pass. Lives entirely on the GPU -- the control loop never round-trips to the
# CPU (DESIGN.md §4.2); the host reads it only for telemetry.
STATS_FIELDS: list[Field] = [
    ("sum_v", "f32"),
    ("sum_v2", "f32"),
    ("sum_activity", "f32"),
    ("count", "f32"),
    ("mean_v", "f32"),
    ("var_v", "f32"),
    ("mean_activity", "f32"),
    ("alive_frac", "f32"),
    # Homeostat outputs: additive corrections applied to the climate bases.
    ("corr_feed", "f32"),
    ("corr_kill", "f32"),
    ("corr_deposit", "f32"),
    ("corr_decay", "f32"),
    ("int_mass", "f32"),
    ("int_var", "f32"),
    ("int_activity", "f32"),
    # Fraction of the trail field's throughput that flux pruning removes, so
    # the agent deposit can hand back exactly that much. Measured rather than
    # assumed: it moves by 3x across the intensity macro, and an unreturned
    # prune term is a net mass sink that the homeostat cancels through
    # corr_decay.
    ("prune_return", "f32"),
    # What the deposit capacity withheld, as a fraction of what it let through,
    # measured from the trail texture's income and withheld EMAs and handed
    # back through the agent deposit exactly as `prune_return` is. Without the
    # return the capacity is a net deposit sink; with it, capacity is a pure
    # redistribution from hubs to wherever traffic is.
    ("cap_return", "f32"),
    # The feature-size loop, DESIGN.md 4.7 step 5. `ell` is the characteristic
    # length scale `mean V / mean |grad V|` in cells -- the one measure of the
    # field that is *not* invariant under rearrangement, which is why the
    # controller was blind to a frozen texture without it. `ell_ref` is the
    # slow reference it is regulated against, in logarithms, and `corr_du` is
    # the accumulated log multiplier on the diffusion rate that the loop uses
    # to get there. `ell_samples` counts the ticks that reference has averaged,
    # which is what lets it converge fast when it is new and slowly once it is
    # established -- a fresh field must not spend its first half hour dragging
    # `corr_du` toward a reference taken before it had a texture to measure.
    ("mean_grad_v", "f32"),
    ("ell", "f32"),
    ("ell_ref", "f32"),
    ("corr_du", "f32"),
    ("ell_samples", "f32"),
    # Image statistics, used by the exposure governor.
    ("img_sum_l", "f32"),
    ("img_max_l", "f32"),
    ("img_count", "f32"),
    ("exposure", "f32"),
]

STATS_DTYPE = _dtype(STATS_FIELDS)
