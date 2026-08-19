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
    //
    // The noise has to *close on the domain*, and this is the field where that
    // matters most, because psi is both toroidal and integrated: a forcing that
    // jumps across the wrap seam is re-applied there every tick, and what
    // accumulates is a permanent ridge of steep gradient along u=0 and v=0.
    // Curl turns a ridge in psi into a jet in the velocity field -- measured at
    // 9x the interior shear with the untiled noise -- and pigment, trail and
    // climate all stretch along it. That was the line-like structure visible
    // near the window edges: not an edge in the simulation, but the seam of the
    // noise lattice, drawn inward by the compositor's aspect and depth scaling.
    //
    // Tiling needs a whole number of lattice cells across the domain, and
    // `psi_noise_scale` is continuous -- the `scale` macro drives it -- so the
    // two bracketing periods are crossfaded. The weights are normalised in
    // quadrature rather than summing to one, because the two fields are
    // incoherent: linear weights would thin the increment by 30% halfway
    // between periods, which would quietly slow the weather down at half the
    // settings of a knob that is not supposed to touch it.
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let noise_scale = max(params.psi_noise_scale, 1.0);
    let period = i32(floor(noise_scale));
    let blend = noise_scale - floor(noise_scale);
    let norm = inverseSqrt(
        max((1.0 - blend) * (1.0 - blend) + blend * blend, 1e-6));
    let noise_seed = params.tick ^ params.seed ^ 0x5bf03635u;
    let increment = norm * (
        value_noise_octaves_tiled(uv, period, noise_seed) * (1.0 - blend)
        + value_noise_octaves_tiled(uv, period + 1, noise_seed) * blend);

    var value = smoothed * (1.0 - params.psi_theta) + increment * params.psi_sigma;
    value = clamp(finite_or(value, 0.0), -8.0, 8.0);

    textureStore(psi_out, p, vec4<f32>(value, 0.0, 0.0, 1.0));
}
