// Final presentation: linear -> sRGB with blue-noise dithering.
//
// Dithering is not polish here. An 8-bit display showing a very slowly drifting
// smooth gradient bands visibly, and worse, the band boundaries *crawl* as the
// gradient moves -- a moving hard edge is a form of visual punctuation, which
// is the one thing this application must not produce. A blue-noise mask breaks
// the bands into noise the eye integrates away.
//
// The mask is deliberately *static*. Animating dither per frame is the usual
// practice and gives better numerical results, but it is per-pixel temporal
// noise -- exactly the shimmer this application exists to avoid. A fixed mask
// contributes precisely zero frame-to-frame change.

//!include common.wgsl

//!struct RenderParams

@group(0) @binding(0) var<storage, read> render: RenderParams;
@group(0) @binding(1) var final_tex: texture_2d<f32>;
@group(0) @binding(2) var noise_tex: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VertexOutput {
    // Fullscreen triangle: one primitive, no vertex buffer, no seam down the
    // diagonal that a two-triangle quad can show under some samplers.
    var out: VertexOutput;
    let x = f32((index << 1u) & 2u);
    let y = f32(index & 2u);
    out.uv = vec2<f32>(x, y);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let linear = textureSampleLevel(final_tex, samp, in.uv, 0.0).rgb;
    var srgb = linear_to_srgb(linear);

    let noise_dims = textureDimensions(noise_tex, 0);
    let coords = vec2<i32>(in.position.xy) % vec2<i32>(noise_dims);
    let threshold = textureLoad(noise_tex, coords, 0).r;

    // Triangular PDF: two independent-ish draws, which gives noise that is
    // independent of the signal level rather than modulating with it.
    let t2 = textureLoad(
        noise_tex,
        (coords + vec2<i32>(37, 17)) % vec2<i32>(noise_dims),
        0,
    ).r;
    let dither = (threshold + t2 - 1.0) * render.dither_amount / 255.0;

    srgb = clamp(srgb + vec3<f32>(dither), vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(srgb, 1.0);
}
