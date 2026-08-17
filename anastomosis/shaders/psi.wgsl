// Vector potential update. The velocity field is v = curl(psi), so evolving
// psi rather than v keeps the flow divergence-free by construction: pigment is
// carried around without ever accumulating or draining.
//
// psi is a *stored* field driven by an Ornstein-Uhlenbeck process. The noise
// increments are white in time -- it is psi's own integration that supplies
// temporal smoothness. This is what lets the flow evolve forever without any
// reference to a clock (DESIGN.md §3).

//!include common.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var psi_in: texture_2d<f32>;
@group(0) @binding(2) var psi_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.psi_w, params.psi_h);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = vec2<i32>(gid.xy);
    let idims = vec2<i32>(dims);
    let current = textureLoad(psi_in, p, 0).r;

    // Mild diffusion keeps psi smooth under repeated perturbation, which
    // matters because we differentiate it: any roughness in psi becomes
    // visible turbulence in the velocity field.
    var neighbours = 0.0;
    neighbours = neighbours + textureLoad(psi_in, wrap_texel(p + vec2<i32>(1, 0), idims), 0).r;
    neighbours = neighbours + textureLoad(psi_in, wrap_texel(p + vec2<i32>(-1, 0), idims), 0).r;
    neighbours = neighbours + textureLoad(psi_in, wrap_texel(p + vec2<i32>(0, 1), idims), 0).r;
    neighbours = neighbours + textureLoad(psi_in, wrap_texel(p + vec2<i32>(0, -1), idims), 0).r;
    let smoothed = mix(current, neighbours * 0.25, 0.25);

    // Spatially smooth increment: sampled from coarse value noise so that psi
    // gains large-scale structure rather than per-texel hash.
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let increment = value_noise_octaves(uv * 3.0, params.tick ^ params.seed ^ 0x5bf03635u);

    var value = smoothed * (1.0 - params.psi_theta) + increment * params.psi_sigma;
    value = clamp(finite_or(value, 0.0), -8.0, 8.0);

    textureStore(psi_out, p, vec4<f32>(value, 0.0, 0.0, 1.0));
}
