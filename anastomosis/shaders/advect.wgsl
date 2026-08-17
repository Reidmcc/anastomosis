// Pigment advection -- the field that is actually shaded.
//
// This pass is what makes the motion read as fluid rather than as the crawling
// quality that raw reaction-diffusion and raw Physarum both have: structure is
// *carried* by the velocity field, not recomputed in place.
//
// Semi-Lagrangian with bilinear reconstruction. Unconditionally stable at any
// timestep, which matters for a process that must never blow up across a
// multi-day run, and mildly diffusive -- which here is a feature, since
// diffusion is smoothness and smoothness is the absence of punctuation.
//
// Channels: (density, hue_cos, hue_sin, activity)
//
// Hue is carried as a unit vector rather than an angle so that advection and
// blending interpolate along the shortest arc; lerping raw angles would tear
// wherever the field crosses the +/-pi branch cut.

//!include common.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var pigment_in: texture_2d<f32>;
@group(0) @binding(2) var pigment_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var reaction_cur: texture_2d<f32>;
@group(0) @binding(4) var reaction_prev: texture_2d<f32>;
@group(0) @binding(5) var trail_tex: texture_2d<f32>;
@group(0) @binding(6) var velocity_tex: texture_2d<f32>;
@group(0) @binding(7) var clim_b: texture_2d<f32>;
@group(0) @binding(8) var samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = vec2<i32>(gid.xy);
    let idims = vec2<i32>(dims);
    let fdims = vec2<f32>(dims);
    let uv = (vec2<f32>(gid.xy) + 0.5) / fdims;

    // --- Advect -----------------------------------------------------------
    let velocity = textureLoad(velocity_tex, p, 0).rg;
    let src = wrap_uv(uv - velocity * params.advect_dt / fdims);
    let carried = finite_or4(textureSampleLevel(pigment_in, samp, src, 0.0), 0.0);

    // --- Local structure --------------------------------------------------
    let v_now = textureLoad(reaction_cur, p, 0).g;
    let v_prev = textureLoad(reaction_prev, p, 0).g;
    let trail = textureLoad(trail_tex, p, 0).r;

    let structure = clamp(
        params.density_from_v * v_now + params.density_from_trail * trail,
        0.0,
        1.0,
    );
    let density = mix(carried.x, structure, params.inject_rate);

    // --- Hue --------------------------------------------------------------
    // Local field orientation gives fine hue variation that follows the shape
    // of the structure; the climate channel gives large-scale regional hue that
    // migrates. Together they make colour a function of simulation state rather
    // than of a clock.
    let gx = textureLoad(reaction_cur, wrap_texel(p + vec2<i32>(1, 0), idims), 0).g
           - textureLoad(reaction_cur, wrap_texel(p - vec2<i32>(1, 0), idims), 0).g;
    let gy = textureLoad(reaction_cur, wrap_texel(p + vec2<i32>(0, 1), idims), 0).g
           - textureLoad(reaction_cur, wrap_texel(p - vec2<i32>(0, 1), idims), 0).g;
    let orientation = atan2(gy, gx);

    let cb = textureSampleLevel(clim_b, samp, uv, 0.0);
    let target_hue = params.hue_anchor
        + params.range_hue * cb.w * params.hue_spread
        + params.hue_from_orientation * orientation;

    // Material keeps the hue it was born with and carries it along the flow;
    // only a small fraction re-adopts the current anchor each tick, so
    // structures of different ages stay chromatically distinct and the field
    // marbles instead of shifting as one.
    let carried_hue = normalize_hue_vec(carried.yz);
    let hue_vec = normalize_hue_vec(
        mix(carried_hue, hue_to_vec(target_hue), params.hue_inject_mix)
    );

    // --- Activity ---------------------------------------------------------
    // Heavily lowpassed, and drives chroma rather than luminance downstream --
    // a fast-responding brightness term would be exactly the punctuation the
    // brief rules out.
    let instantaneous = abs(v_now - v_prev) * params.activity_gain;
    let activity = clamp(
        mix(carried.w, instantaneous, params.activity_rate),
        0.0,
        4.0,
    );

    textureStore(
        pigment_out,
        p,
        vec4<f32>(clamp(density, 0.0, 4.0), hue_vec.x, hue_vec.y, activity),
    );
}
