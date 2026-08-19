// The flow's vector potential, assembled at full resolution.
//
// In two dimensions a divergence-free velocity is the curl of a scalar, and
// `flow.wgsl` can build both of its components -- imposed weather and
// structure-following -- and add them, because the sum of two curls is a curl.
// In three dimensions the structure term is not a curl of anything obvious:
// "flow along the iso-surfaces of V" wants `grad(V) x e` for some direction
// `e`, and `grad(V) x e` is only divergence-free when `e` is constant.
//
// So the potential is built here and differentiated in vol_flow.wgsl, rather
// than being computed and added as velocities. That makes the whole question
// go away: `div(curl(A))` cancels term for term under central differences,
// whatever A is and however A was assembled, because the difference operators
// commute. Pigment advected through the result is neither compressed into
// bright accumulations nor drained into holes -- which is the property the
// entire flow stage exists for.
//
// Two things ride on that licence, and neither would be exactly
// divergence-free applied to a velocity:
//
//   * **The climate's flow gain.** Multiplying a velocity by a spatially
//     varying gain adds `grad(g) . v` to the divergence, and material piles up
//     wherever the gain falls off. Applied to the potential it is free.
//
//   * **The depth anisotropy.** `depth_flow` weights the lateral components of
//     A, which scales `v_z = dx(A_y) - dy(A_x)` by that factor while leaving
//     the lateral velocity dominated by the terms in `A_z`. Scaling `v_z`
//     itself would not be divergence-free at all.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var psi_tex: texture_3d<f32>;
@group(0) @binding(2) var reaction_tex: texture_3d<f32>;
@group(0) @binding(3) var clim_b: texture_3d<f32>;
@group(0) @binding(4) var potential_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var samp: sampler;

// Reads V through a small blur so that the flow follows the overall shape of a
// filament rather than its per-voxel roughness. Centre-weighted seven-point;
// the potential is differentiated twice over before it reaches the image, so
// this is the cheapest place to spend smoothness.
fn smooth_v(p: vec3<i32>, idims: vec3<i32>) -> f32 {
    var acc = textureLoad(reaction_tex, wrap_texel3(p, idims), 0).g * 2.0;
    acc = acc + textureLoad(reaction_tex, wrap_texel3(p + vec3<i32>(1, 0, 0), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel3(p + vec3<i32>(-1, 0, 0), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel3(p + vec3<i32>(0, 1, 0), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel3(p + vec3<i32>(0, -1, 0), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel3(p + vec3<i32>(0, 0, 1), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel3(p + vec3<i32>(0, 0, -1), idims), 0).g;
    return acc / 8.0;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let p = vec3<i32>(gid);
    let idims = vec3<i32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);

    // Imposed weather, from the coarse stored potential.
    let weather = textureSampleLevel(psi_tex, samp, uvw, 0.0).xyz * params.psi_gain;

    // Structure-following. `curl(V * e)` is `grad(V) x e + V curl(e)`, so
    // material slides *around* structure rather than through it -- the second
    // term is not a defect but the reason the axis is taken from the weather
    // rather than fixed: a constant `e` would give the slab a permanent
    // preferred plane, and every filament in it would slide the same way.
    let axis_raw = weather;
    let axis_len = length(axis_raw);
    let axis = select(
        vec3<f32>(0.0, 0.0, 1.0), axis_raw / max(axis_len, 1e-6), axis_len > 1e-5);
    let structure = axis * smooth_v(p, idims) * params.field_gain;

    // Local flow strength varies by region and migrates with the climate.
    // Applied here rather than to the velocity; see the header.
    let cb = textureSampleLevel(clim_b, samp, uvw, 0.0);
    let gain = max(0.0, 1.0 + params.range_flow * cb.z);

    var potential = (weather + structure) * gain;
    // Damp motion along the slab normal by weighting the lateral components.
    let lateral = clamp(params.depth_flow, 0.0, 1.0);
    potential = vec3<f32>(potential.x * lateral, potential.y * lateral, potential.z);
    potential = clamp(
        finite_or4(vec4<f32>(potential, 0.0), 0.0).xyz,
        vec3<f32>(-64.0), vec3<f32>(64.0));

    textureStore(potential_out, p, vec4<f32>(potential, 1.0));
}
