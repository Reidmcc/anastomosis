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
    ("clim_w", "u32"),
    ("clim_h", "u32"),
    ("psi_w", "u32"),
    ("psi_h", "u32"),
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
    ("turn_rate", "f32"),
    ("jitter", "f32"),
    ("deposit", "f32"),
    ("fusion_bias", "f32"),
    ("trail_decay", "f32"),
    ("trail_diffuse", "f32"),
    ("starve_threshold", "f32"),
    ("max_age", "f32"),
    # Reaction
    ("feed", "f32"),
    ("kill", "f32"),
    ("du", "f32"),
    ("dv", "f32"),
    ("rdt", "f32"),
    ("trail_feed_gain", "f32"),
    ("kill_follows_feed", "f32"),
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
    # Pigment / colour injection
    ("hue_anchor", "f32"),
    ("hue_spread", "f32"),
    ("hue_from_orientation", "f32"),
    ("hue_inject_mix", "f32"),
    ("inject_rate", "f32"),
    ("activity_rate", "f32"),
    ("activity_gain", "f32"),
    ("density_from_v", "f32"),
    ("density_from_trail", "f32"),
    # Generic blur pass control (reused by trail diffuse and DOF)
    ("blur_radius", "f32"),
    ("blur_dir_x", "f32"),
    ("blur_dir_y", "f32"),
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
    ("pad0", "u32"),
    ("pad1", "u32"),
    ("pad2", "u32"),
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
    ("radius", "f32"),
    ("strength", "f32"),
    ("chan_feed", "f32"),
    ("chan_kill", "f32"),
    ("chan_flow", "f32"),
    ("chan_hue", "f32"),
]


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
    ("stats_pad", "f32"),
    # Image statistics, used by the exposure governor.
    ("img_sum_l", "f32"),
    ("img_max_l", "f32"),
    ("img_count", "f32"),
    ("exposure", "f32"),
]

STATS_DTYPE = _dtype(STATS_FIELDS)
