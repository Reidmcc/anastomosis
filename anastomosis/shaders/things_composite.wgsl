// The Things' image: the canvas, lifted onto the void -- §18.3.
//
// The canvas field already *is* the picture (image and trail are one
// object, the founding law), so the compositor's whole job is sampling it
// onto the window: aspect correction that wraps both axes (the world is a
// torus, §18.1 soul 8), the house void underneath (the founding #0a0a0a,
// spoken as background_luma with the fungal fog's faint coolness, packed
// host-side into fog_r/g/b), and the perceptual bounds in Oklab. Those
// bounds are shaping; §7's stage downstream is the guarantee.

//!include things_common.wgsl

//!struct RenderParams

@group(0) @binding(0) var<storage, read> params: ThingsParams;
@group(0) @binding(1) var<storage, read> render: RenderParams;
@group(0) @binding(2) var canvas_tex: texture_2d<f32>;
@group(0) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var samp: sampler;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= render.out_w || gid.y >= render.out_h) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5)
        / vec2<f32>(f32(render.out_w), f32(render.out_h));

    // Both axes wrap: the axis that gained on the simulation's aspect
    // samples more of the torus, seamlessly (backend.aspect_correction).
    let suv = wrap_uv((uv - 0.5) * vec2<f32>(params.x_scale, params.y_scale)
                      + 0.5);

    let lit = max(
        finite_or4(textureSampleLevel(canvas_tex, samp, suv, 0.0), 0.0).rgb,
        vec3<f32>(0.0));
    let fog = vec3<f32>(render.fog_r, render.fog_g, render.fog_b);
    let rgb = fog + lit * params.out_gain;

    // Perceptual bounds, applied to the target as everywhere else.
    var lab = linear_srgb_to_oklab(rgb);
    lab.x = clamp(lab.x, 0.0, render.l_max);
    let chroma = length(lab.yz);
    if (chroma > render.c_max && chroma > 1e-6) {
        lab = vec3<f32>(lab.x, lab.yz * (render.c_max / chroma));
    }
    let out = oklab_to_linear_srgb(lab);
    textureStore(
        hdr_out, vec2<i32>(gid.xy), finite_or4(vec4<f32>(out, 1.0), 0.0));
}
