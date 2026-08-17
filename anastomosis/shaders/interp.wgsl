// Motion-compensated temporal interpolation between the last two sim states.
//
// The simulation runs at ~20 Hz and the display at 30, so most frames fall
// between ticks. A naive lerp of two states would be mushy and would
// reintroduce exactly the crawling quality the advection stage exists to
// remove. Instead each state is advected to the intermediate time through the
// velocity field before blending, which makes 20 Hz look *smoother* than
// simulating at 30 -- and lets the budget governor lower the tick rate without
// anything becoming visible.

//!include common.wgsl

//!struct SimParams
//!struct RenderParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var<storage, read> render: RenderParams;
@group(0) @binding(2) var pigment_prev: texture_2d<f32>;
@group(0) @binding(3) var pigment_cur: texture_2d<f32>;
@group(0) @binding(4) var velocity_tex: texture_2d<f32>;
@group(0) @binding(5) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let fdims = vec2<f32>(dims);
    let uv = (vec2<f32>(gid.xy) + 0.5) / fdims;
    let velocity = textureLoad(velocity_tex, vec2<i32>(gid.xy), 0).rg / fdims;
    let frac = clamp(render.frac, 0.0, 1.0);
    let dt = render.interp_dt;

    // Advance the older state forward by frac, and rewind the newer state by
    // (1 - frac); both then describe the same instant.
    let prev_uv = wrap_uv(uv - velocity * frac * dt);
    let cur_uv = wrap_uv(uv + velocity * (1.0 - frac) * dt);

    let a = textureSampleLevel(pigment_prev, samp, prev_uv, 0.0);
    let b = textureSampleLevel(pigment_cur, samp, cur_uv, 0.0);
    var blended = mix(a, b, frac);

    // Hue is a unit vector; blending two of them shortens the result, so
    // renormalise or the colour would desaturate wherever the two states
    // disagree.
    let hue = normalize_hue_vec(blended.yz);

    textureStore(
        out_tex,
        vec2<i32>(gid.xy),
        finite_or4(vec4<f32>(blended.x, hue.x, hue.y, blended.w), 0.0),
    );
}
