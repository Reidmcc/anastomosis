// Velocity field construction.
//
// Both components are curls, so the result is divergence-free and pigment
// advected through it is neither compressed into bright accumulations nor
// drained into holes:
//
//   * curl(psi)    -- imposed weather, a slow stateful random field.
//   * curl(blur(V)) -- flow along the iso-contours of the reaction field, so
//                      material slides *around* structure rather than through
//                      it. This is the feedback term: the field pushes itself
//                      around, which is a significant source of the system's
//                      unpredictability.

//!include common.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var psi_tex: texture_2d<f32>;
@group(0) @binding(2) var reaction_tex: texture_2d<f32>;
@group(0) @binding(3) var clim_b: texture_2d<f32>;
@group(0) @binding(4) var velocity_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var samp: sampler;

fn curl_psi(uv: vec2<f32>) -> vec2<f32> {
    let step = vec2<f32>(1.0 / f32(params.psi_w), 1.0 / f32(params.psi_h));
    let px = textureSampleLevel(psi_tex, samp, wrap_uv(uv + vec2<f32>(step.x, 0.0)), 0.0).r;
    let mx = textureSampleLevel(psi_tex, samp, wrap_uv(uv - vec2<f32>(step.x, 0.0)), 0.0).r;
    let py = textureSampleLevel(psi_tex, samp, wrap_uv(uv + vec2<f32>(0.0, step.y)), 0.0).r;
    let my = textureSampleLevel(psi_tex, samp, wrap_uv(uv - vec2<f32>(0.0, step.y)), 0.0).r;
    return vec2<f32>(py - my, -(px - mx)) * 0.5;
}

// Reads V through a small blur so that the flow follows the overall shape of a
// filament rather than its per-texel roughness.
fn smooth_v(p: vec2<i32>, idims: vec2<i32>) -> f32 {
    var acc = textureLoad(reaction_tex, wrap_texel(p, idims), 0).g * 4.0;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(1, 0), idims), 0).g * 2.0;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(-1, 0), idims), 0).g * 2.0;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(0, 1), idims), 0).g * 2.0;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(0, -1), idims), 0).g * 2.0;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(1, 1), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(1, -1), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(-1, 1), idims), 0).g;
    acc = acc + textureLoad(reaction_tex, wrap_texel(p + vec2<i32>(-1, -1), idims), 0).g;
    return acc / 16.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = vec2<i32>(gid.xy);
    let idims = vec2<i32>(dims);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    let weather = curl_psi(uv) * params.psi_gain;

    let dvdx = smooth_v(p + vec2<i32>(1, 0), idims) - smooth_v(p - vec2<i32>(1, 0), idims);
    let dvdy = smooth_v(p + vec2<i32>(0, 1), idims) - smooth_v(p - vec2<i32>(0, 1), idims);
    let structure = vec2<f32>(dvdy, -dvdx) * 0.5 * params.field_gain;

    // Local flow strength varies by region and migrates with the climate.
    let cb = textureSampleLevel(clim_b, samp, uv, 0.0);
    let gain = max(0.0, 1.0 + params.range_flow * cb.z);

    // Clamped to a few cells per tick: semi-Lagrangian advection is stable at
    // any step, but very long steps would sample uncorrelated regions and read
    // as tearing rather than flow.
    var velocity = (weather + structure) * gain;
    velocity = clamp(finite_or2(velocity, 0.0), vec2<f32>(-8.0), vec2<f32>(8.0));

    textureStore(velocity_out, p, vec4<f32>(velocity, 0.0, 1.0));
}
