// Image statistics, stage 1: per-tile sum and max of Oklab lightness.
//
// Feeds the exposure governor. Runs on the previous frame's output, so it costs
// no synchronisation -- one frame of latency is irrelevant to a controller
// whose time constant is measured in seconds.

//!include common.wgsl

//!struct RenderParams

@group(0) @binding(0) var<storage, read> render: RenderParams;
@group(0) @binding(1) var final_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    var sample = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    if (gid.x < render.out_w && gid.y < render.out_h) {
        let rgb = finite_or4(textureLoad(final_tex, vec2<i32>(gid.xy), 0), 0.0).rgb;
        let lightness = linear_srgb_to_oklab(rgb).x;
        sample = vec4<f32>(lightness, lightness, 1.0, 0.0);
    }

    tile[lid] = sample;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = tile[lid + stride];
            tile[lid] = vec4<f32>(
                tile[lid].x + other.x,
                max(tile[lid].y, other.y),
                tile[lid].z + other.z,
                0.0,
            );
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wid.y * nwg.x + wid.x] = tile[0];
    }
}
