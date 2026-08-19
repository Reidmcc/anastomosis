// The slab's weather: a stored vector potential, evolved by an OU process.
//
// The 2D backend keeps a scalar stream function, because in the plane
// `v = curl(psi)` needs only one component. A volume needs all three: the
// divergence-free velocity field is the curl of a *vector* potential, so this
// pass maintains three coupled OU fields rather than one.
//
// Everything else is unchanged from psi.wgsl, including the part that matters
// most: the noise increments are white in time and the field's own integration
// supplies the temporal smoothness, so the weather evolves forever without any
// reference to a clock (DESIGN.md §3). The increment tiles on the domain for
// the same reason too -- an untiled forcing on a toroidal integrated field
// builds a permanent ridge along the wrap seam, and curl turns a ridge into a
// jet.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var psi_in: texture_3d<f32>;
@group(0) @binding(2) var psi_out: texture_storage_3d<rgba16float, write>;

fn component(
    uvw: vec3<f32>,
    lo_periods: vec3<i32>,
    hi_periods: vec3<i32>,
    blend: f32,
    norm: f32,
    seed: u32,
) -> f32 {
    return norm * (
        value_noise_octaves_tiled3(uvw, lo_periods, seed) * (1.0 - blend)
        + value_noise_octaves_tiled3(uvw, hi_periods, seed) * blend);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.psi_w, params.psi_h, params.psi_d);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let p = vec3<i32>(gid);
    let idims = vec3<i32>(dims);
    let current = textureLoad(psi_in, p, 0).xyz;

    // Mild diffusion keeps the potential smooth under repeated perturbation,
    // which matters because we differentiate it: any roughness here becomes
    // visible turbulence in the velocity field.
    var neighbours = vec3<f32>(0.0);
    neighbours += textureLoad(psi_in, wrap_texel3(p + vec3<i32>(1, 0, 0), idims), 0).xyz;
    neighbours += textureLoad(psi_in, wrap_texel3(p + vec3<i32>(-1, 0, 0), idims), 0).xyz;
    neighbours += textureLoad(psi_in, wrap_texel3(p + vec3<i32>(0, 1, 0), idims), 0).xyz;
    neighbours += textureLoad(psi_in, wrap_texel3(p + vec3<i32>(0, -1, 0), idims), 0).xyz;
    neighbours += textureLoad(psi_in, wrap_texel3(p + vec3<i32>(0, 0, 1), idims), 0).xyz;
    neighbours += textureLoad(psi_in, wrap_texel3(p + vec3<i32>(0, 0, -1), idims), 0).xyz;
    let smoothed = mix(current, neighbours / 6.0, 0.25);

    // Spatially smooth increment, crossfaded between the two bracketing
    // integer periods so that `psi_noise_scale` -- which the `scale` macro
    // drives continuously -- never has to land on a period that tiles. The
    // weights are normalised in quadrature rather than summing to one, because
    // the two fields are incoherent: linear weights would thin the increment
    // by 30% halfway between periods, quietly slowing the weather at half the
    // settings of a knob that is not supposed to touch it.
    let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
    let noise_scale = max(params.psi_noise_scale, 1.0);
    let period = i32(floor(noise_scale));
    let blend = noise_scale - floor(noise_scale);
    let norm = inverseSqrt(
        max((1.0 - blend) * (1.0 - blend) + blend * blend, 1e-6));
    let lo_periods = slab_periods(period, dims);
    let hi_periods = slab_periods(period + 1, dims);

    // A stream per component, or the three would be identical fields and the
    // potential would lie along (1, 1, 1) everywhere -- which is a potential
    // whose curl is a single shear, not weather.
    let base_seed = params.tick ^ params.seed ^ 0x5bf03635u;
    let increment = vec3<f32>(
        component(uvw, lo_periods, hi_periods, blend, norm, base_seed),
        component(uvw, lo_periods, hi_periods, blend, norm,
                  base_seed ^ 0x9e3779b9u),
        component(uvw, lo_periods, hi_periods, blend, norm,
                  base_seed ^ 0x165667b1u),
    );

    var value = smoothed * (1.0 - params.psi_theta) + increment * params.psi_sigma;
    value = clamp(finite_or4(vec4<f32>(value, 0.0), 0.0).xyz,
                  vec3<f32>(-8.0), vec3<f32>(8.0));

    textureStore(psi_out, p, vec4<f32>(value, 1.0));
}
