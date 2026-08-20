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
    ("trail_advect", "f32"),
    ("income_rate", "f32"),
    ("prune_gain", "f32"),
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
# centred; see reduce.wgsl and DESIGN.md §4.7.
PARTIAL_WGSL = """\
struct Partial {
    field: vec4<f32>,  // sum V, sum V^2, sum |dV/dt|, count
    flux: vec4<f32>,   // sum trail, sum trail*deficit, spare, spare
};"""

# Two vec4s. std430 gives a vec4 16-byte alignment and this struct holds
# nothing else, so the stride is exactly the sum of the members.
PARTIAL_SIZE = 32


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
    # Image statistics, used by the exposure governor.
    ("img_sum_l", "f32"),
    ("img_max_l", "f32"),
    ("img_count", "f32"),
    ("exposure", "f32"),
]

STATS_DTYPE = _dtype(STATS_FIELDS)
