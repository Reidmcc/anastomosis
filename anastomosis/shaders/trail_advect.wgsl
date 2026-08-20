// Semi-Lagrangian advection of the trail field -- DESIGN.md §4.7 step 6.
//
// The last and riskiest of the §4.7 mechanisms, and the largest payoff for
// dynamism: with a nonzero gain the *structure* is carried by the flow, at a
// fraction of the pigment's speed, so filaments experience the shear that
// previously only their colour did -- stretched, bent, and pinched by the
// currents instead of sitting still while pigment streams through them.
//
// It runs as its own pass, after the velocity field is written and before the
// pigment reads it, for two reasons that matter more than tidiness. First,
// the trail-decay pass runs before the velocity exists each tick, so folding
// advection into it would read the *previous* tick's velocity -- state the
// checkpoint deliberately does not carry (§4.6), which would break resume
// determinism in the one mode that uses this. Read in the same tick it is
// written, the velocity stays a derived quantity and the snapshot is
// untouched. Second, the dispatch is skipped entirely at gain zero, so the
// regulation mode -- whose curve table pins the gain at zero -- pays nothing
// and remains bit-identical to a build without this file.
//
// All four channels ride together: a strand's income EMA and prune term
// belong to the strand, not to the texel it used to be at. Semi-Lagrangian
// with bilinear reconstruction, unconditionally stable like the pigment's
// (§2), and mildly diffusive -- which for a field the agents re-sharpen every
// tick is a feature.

//!include common.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var trail_in: texture_2d<f32>;
@group(0) @binding(2) var trail_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var velocity_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let fdims = vec2<f32>(dims);
    let uv = (vec2<f32>(gid.xy) + 0.5) / fdims;

    let velocity = textureLoad(velocity_tex, vec2<i32>(gid.xy), 0).rg;
    let src = wrap_uv(
        uv - velocity * params.trail_advect * params.advect_dt / fdims);
    let carried = finite_or4(textureSampleLevel(trail_in, samp, src, 0.0), 0.0);

    // The same clamps the trail pass applies; advection must not be a way to
    // smuggle a value past them.
    textureStore(trail_out, vec2<i32>(gid.xy), vec4<f32>(
        clamp(carried.x, 0.0, 8.0),
        clamp(carried.y, 0.0, 8.0),
        clamp(carried.z, 0.0, 8.0),
        1.0,
    ));
}
