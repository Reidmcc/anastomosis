// Motion-compensated temporal interpolation between the last two slab states.
//
// The simulation runs at ~20 Hz and the display at 30, so most frames fall
// between ticks. A naive lerp of two states would be mushy and would
// reintroduce exactly the crawling quality the advection stage exists to
// remove. Instead each state is advected to the intermediate time through the
// velocity field before blending, which makes 20 Hz look *smoother* than
// simulating at 30 -- and lets the budget governor lower the tick rate without
// anything becoming visible.
//
// Done as a pass over the volume rather than inside the ray march, which would
// otherwise have to take two trilinear taps at every step of every ray. One
// pass over the voxels costs less than that at any march length worth having,
// and it leaves the march reading a single texture.

//!include common3d.wgsl

//!struct SimParams
//!struct RenderParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var<storage, read> render: RenderParams;
@group(0) @binding(2) var pigment_prev: texture_3d<f32>;
@group(0) @binding(3) var pigment_cur: texture_3d<f32>;
@group(0) @binding(4) var velocity_tex: texture_3d<f32>;
@group(0) @binding(5) var out_tex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var samp: sampler;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let fdims = vec3<f32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / fdims;
    let velocity = textureLoad(velocity_tex, vec3<i32>(gid), 0).xyz / fdims;
    let frac = clamp(render.frac, 0.0, 1.0);
    let dt = render.interp_dt;

    // Advance the older state forward by frac, and rewind the newer state by
    // (1 - frac); both then describe the same instant.
    let prev_uvw = wrap_uvw(uvw - velocity * frac * dt);
    let cur_uvw = wrap_uvw(uvw + velocity * (1.0 - frac) * dt);

    let a = textureSampleLevel(pigment_prev, samp, prev_uvw, 0.0);
    let b = textureSampleLevel(pigment_cur, samp, cur_uvw, 0.0);
    var blended = mix(a, b, frac);

    // Hue is a unit vector; blending two of them shortens the result, so
    // renormalise or the colour would desaturate wherever the two states
    // disagree.
    let hue = normalize_hue_vec(blended.yz);

    textureStore(
        out_tex,
        vec3<i32>(gid),
        finite_or4(vec4<f32>(blended.x, hue.x, hue.y, blended.w), 0.0),
    );
}
