// Field statistics, stage 1: per-tile reduction.
//
// Stage 2 (the homeostat proper) lives in homeostat.wgsl -- WGSL forbids two
// resource variables sharing a (group, binding) pair within one module, so the
// two stages cannot share a file.

//!include common.wgsl

//!struct SimParams
//!struct Stats

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var reaction_cur: texture_2d<f32>;
@group(0) @binding(2) var reaction_prev: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<vec4<f32>>;

var<workgroup> tile: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn reduce_tiles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    var sample = vec4<f32>(0.0);
    if (gid.x < dims.x && gid.y < dims.y) {
        let p = vec2<i32>(gid.xy);
        let v = finite_or(textureLoad(reaction_cur, p, 0).g, 0.0);
        let v_prev = finite_or(textureLoad(reaction_prev, p, 0).g, 0.0);
        sample = vec4<f32>(v, v * v, abs(v - v_prev), 1.0);
    }

    tile[lid] = sample;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            tile[lid] = tile[lid] + tile[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        partials[wid.y * nwg.x + wid.x] = tile[0];
    }
}
