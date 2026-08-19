// Generic separable Gaussian blur over the slab, reused for trail diffusion.
//
// Separable in three passes rather than two, which is what makes an isotropic
// 3D diffusion affordable at all: a 12-tap radius costs 36 taps separably
// against the 15,625 a gather of the same width would.
//
// The sampler wraps on all three axes, matching the toroidal domain, so there
// is no boundary case here either.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var samp: sampler;

const MAX_TAPS: i32 = 12;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let fdims = vec3<f32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / fdims;
    let dir = vec3<f32>(params.blur_dir_x, params.blur_dir_y, params.blur_dir_z) / fdims;
    let sigma = max(params.blur_radius, 1e-3);

    var acc = textureSampleLevel(src, samp, uvw, 0.0);
    var weight_sum = 1.0;

    let radius = min(i32(ceil(sigma * 2.5)), MAX_TAPS);
    for (var i = 1; i <= radius; i = i + 1) {
        let x = f32(i);
        let w = exp(-(x * x) / (2.0 * sigma * sigma));
        acc = acc
            + textureSampleLevel(src, samp, uvw + dir * x, 0.0) * w
            + textureSampleLevel(src, samp, uvw - dir * x, 0.0) * w;
        weight_sum = weight_sum + 2.0 * w;
    }

    textureStore(dst, vec3<i32>(gid), finite_or4(acc / weight_sum, 0.0));
}
