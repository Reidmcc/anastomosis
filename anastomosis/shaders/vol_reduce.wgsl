// Field statistics over the slab, stage 1: per-tile reduction.
//
// Stage 2 is homeostat.wgsl, shared unchanged with the layered backend: it
// sums an array of `Partial` and runs the controller, and neither operation
// cares what shape the field they came from was. That is deliberate -- the
// homeostat is the thing keeping a multi-day run alive (DESIGN.md §4.2), and
// one implementation of it is worth more than one tuned per backend.
//
// The tile here is a 4 x 4 x 16 box rather than a 16 x 16 square, which is
// still 256 invocations and still one `Partial` out. The long axis is z
// because the slab is shallow: a cubic 6 x 6 x 6 tile would be more coherent
// but leaves a ragged remainder against a depth of 48, and the reduction is
// not where the time goes.

//!include common3d.wgsl

//!struct SimParams
//!struct Stats

//!struct Partial

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var reaction_cur: texture_3d<f32>;
@group(0) @binding(2) var reaction_prev: texture_3d<f32>;
@group(0) @binding(3) var trail_tex: texture_3d<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<Partial>;

var<workgroup> tile: array<vec4<f32>, 256>;
var<workgroup> tile_flux: array<vec4<f32>, 256>;

@compute @workgroup_size(4, 4, 16)
fn reduce_tiles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    let idims = vec3<i32>(dims);
    var sample = vec4<f32>(0.0);
    var flux = vec4<f32>(0.0);
    if (gid.x < dims.x && gid.y < dims.y && gid.z < dims.z) {
        let p = vec3<i32>(gid);
        let v = finite_or(textureLoad(reaction_cur, p, 0).g, 0.0);
        let v_prev = finite_or(textureLoad(reaction_prev, p, 0).g, 0.0);
        sample = vec4<f32>(v, v * v, abs(v - v_prev), 1.0);

        // Weighted by trail, because the quantity that has to balance is mass:
        // the prune term multiplies decay, and decay acts on what is there.
        let t = textureLoad(trail_tex, p, 0);
        let trail = max(finite_or(t.r, 0.0), 0.0);
        let prune_relative = clamp(finite_or(t.b, 0.0), 0.0, 8.0);

        // |grad V| for the feature-size loop, as in reduce.wgsl but over three
        // axes. The length scale it feeds is in voxels, and the controller
        // regulates it against a reference it takes from the field itself, so
        // nothing here has to know that a voxel is five display pixels wide
        // where a layered cell is one.
        let vxp = finite_or(
            textureLoad(reaction_cur, wrap_texel3(p + vec3<i32>(1, 0, 0), idims), 0).g, 0.0);
        let vxm = finite_or(
            textureLoad(reaction_cur, wrap_texel3(p - vec3<i32>(1, 0, 0), idims), 0).g, 0.0);
        let vyp = finite_or(
            textureLoad(reaction_cur, wrap_texel3(p + vec3<i32>(0, 1, 0), idims), 0).g, 0.0);
        let vym = finite_or(
            textureLoad(reaction_cur, wrap_texel3(p - vec3<i32>(0, 1, 0), idims), 0).g, 0.0);
        let vzp = finite_or(
            textureLoad(reaction_cur, wrap_texel3(p + vec3<i32>(0, 0, 1), idims), 0).g, 0.0);
        let vzm = finite_or(
            textureLoad(reaction_cur, wrap_texel3(p - vec3<i32>(0, 0, 1), idims), 0).g, 0.0);
        let grad = length(vec3<f32>(
            (vxp - vxm) * 0.5, (vyp - vym) * 0.5, (vzp - vzm) * 0.5));

        flux = vec4<f32>(trail, trail * prune_relative, grad, 0.0);
    }

    tile[lid] = sample;
    tile_flux[lid] = flux;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            tile[lid] = tile[lid] + tile[lid + stride];
            tile_flux[lid] = tile_flux[lid] + tile_flux[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let index = (wid.z * nwg.y + wid.y) * nwg.x + wid.x;
        partials[index].field = tile[0];
        partials[index].flux = tile_flux[0];
    }
}
