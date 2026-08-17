// Generic separable Gaussian blur, reused for trail diffusion and for
// depth-of-field on the back layers.
//
// Wide, high-quality kernels rather than a minimal 5-tap stencil: smoother
// fields carry less high-frequency content, and high-frequency content is what
// shimmers. On the target hardware the extra taps are free (DESIGN.md §8.1).

//!include common.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var samp: sampler;

const MAX_TAPS: i32 = 12;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let fdims = vec2<f32>(dims);
    let uv = (vec2<f32>(gid.xy) + 0.5) / fdims;
    let dir = vec2<f32>(params.blur_dir_x, params.blur_dir_y) / fdims;
    let sigma = max(params.blur_radius, 1e-3);

    // The sampler wraps, matching the toroidal simulation domain.
    var acc = textureSampleLevel(src, samp, uv, 0.0);
    var weight_sum = 1.0;

    let radius = min(i32(ceil(sigma * 2.5)), MAX_TAPS);
    for (var i = 1; i <= radius; i = i + 1) {
        let x = f32(i);
        let w = exp(-(x * x) / (2.0 * sigma * sigma));
        acc = acc
            + textureSampleLevel(src, samp, uv + dir * x, 0.0) * w
            + textureSampleLevel(src, samp, uv - dir * x, 0.0) * w;
        weight_sum = weight_sum + 2.0 * w;
    }

    textureStore(dst, vec2<i32>(gid.xy), finite_or4(acc / weight_sum, 0.0));
}
