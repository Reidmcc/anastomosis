// Pigment advection through the slab -- the field the ray march actually
// shades.
//
// Semi-Lagrangian with trilinear reconstruction, unconditionally stable at any
// timestep and mildly diffusive, which here is a feature: diffusion is
// smoothness and smoothness is the absence of punctuation. This pass is what
// makes the motion read as fluid rather than as the crawling quality that raw
// reaction-diffusion and raw Physarum both have -- structure is *carried* by
// the velocity field, not recomputed in place.
//
// Channels: (density, hue_cos, hue_sin, activity), as in 2D. Hue is a unit
// vector rather than an angle so that advection and blending interpolate along
// the shortest arc; lerping raw angles would tear wherever the field crosses
// the +/-pi branch cut.
//
// One thing genuinely differs. Hue is seeded from the local field orientation,
// which in two dimensions is `atan2` of the gradient of V -- a single angle. A
// gradient in three dimensions has two angles and no canonical one, so the hue
// is taken from the gradient's *azimuth about the view axis*, which is the
// component of orientation a viewer looking down that axis can actually see.
// Structure lying in the plane therefore colours the way it does under the
// layered backend, and structure leaning through the slab is coloured by how
// it projects, which is what the eye reads anyway.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var pigment_in: texture_3d<f32>;
@group(0) @binding(2) var pigment_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var reaction_cur: texture_3d<f32>;
@group(0) @binding(4) var reaction_prev: texture_3d<f32>;
@group(0) @binding(5) var trail_tex: texture_3d<f32>;
@group(0) @binding(6) var velocity_tex: texture_3d<f32>;
@group(0) @binding(7) var clim_b: texture_3d<f32>;
@group(0) @binding(8) var samp: sampler;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let p = vec3<i32>(gid);
    let idims = vec3<i32>(dims);
    let fdims = vec3<f32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / fdims;

    // --- Advect -----------------------------------------------------------
    let velocity = textureLoad(velocity_tex, p, 0).xyz;
    let src = wrap_uvw(uvw - velocity * params.advect_dt / fdims);
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
    let gx = textureLoad(reaction_cur, wrap_texel3(p + vec3<i32>(1, 0, 0), idims), 0).g
           - textureLoad(reaction_cur, wrap_texel3(p - vec3<i32>(1, 0, 0), idims), 0).g;
    let gy = textureLoad(reaction_cur, wrap_texel3(p + vec3<i32>(0, 1, 0), idims), 0).g
           - textureLoad(reaction_cur, wrap_texel3(p - vec3<i32>(0, 1, 0), idims), 0).g;
    let orientation = atan2(gy, gx);

    let cb = textureSampleLevel(clim_b, samp, uvw, 0.0);
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
