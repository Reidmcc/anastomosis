// Field statistics, stage 1: per-tile reduction.
//
// Stage 2 (the homeostat proper) lives in homeostat.wgsl -- WGSL forbids two
// resource variables sharing a (group, binding) pair within one module, so the
// two stages cannot share a file.
//
// Two vec4s per tile, not one. The second carries two things the first has no
// room for.
//
// How much of the trail field's throughput the pruning term in trail.wgsl is
// removing, so the agent layer can hand exactly that much back (DESIGN.md
// §4.7). Without the return, pruning is a net mass sink and the homeostat
// simply cancels it through corr_decay. It has to be measured rather than
// assumed: the figure is 0.21 at the default intensity and 0.07 at the top of
// that macro's range, where the network concentrates two thirds of its mass
// into a few percent of texels.
//
// And the sum of |grad V|, which with the sum of V already in the first vec4
// gives the characteristic length scale `ell = mean V / mean |grad V|`. That
// is the measure the homeostat was missing (DESIGN.md §4.7 step 5): mass,
// variance and activity are all invariant under rearrangement, so a field can
// be perfectly on-band and morphologically frozen and the controller has no
// term that objects. Feature size is the one thing a frozen texture cannot
// hide from. It costs four more texel loads here -- neighbours every adjacent
// invocation has loaded anyway -- in the cheapest pass over the field.

//!include common.wgsl

//!struct SimParams
//!struct Stats

//!struct Partial

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var reaction_cur: texture_2d<f32>;
@group(0) @binding(2) var reaction_prev: texture_2d<f32>;
@group(0) @binding(3) var trail_tex: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<Partial>;

var<workgroup> tile: array<vec4<f32>, 256>;
var<workgroup> tile_flux: array<vec4<f32>, 256>;
var<workgroup> tile_dep: array<vec4<f32>, 256>;

@compute @workgroup_size(16, 16, 1)
fn reduce_tiles(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    let idims = vec2<i32>(dims);
    var sample = vec4<f32>(0.0);
    var flux = vec4<f32>(0.0);
    var dep = vec4<f32>(0.0);
    if (gid.x < dims.x && gid.y < dims.y) {
        let p = vec2<i32>(gid.xy);
        let v = finite_or(textureLoad(reaction_cur, p, 0).g, 0.0);
        let v_prev = finite_or(textureLoad(reaction_prev, p, 0).g, 0.0);
        sample = vec4<f32>(v, v * v, abs(v - v_prev), 1.0);

        // Weighted by trail, because the quantity that has to balance is mass:
        // the prune term multiplies decay, and decay acts on what is there. The
        // ratio of these two sums is the fraction of the trail field's
        // throughput that pruning removes, which is what the agent deposit has
        // to put back.
        let t = textureLoad(trail_tex, p, 0);
        let trail = max(finite_or(t.r, 0.0), 0.0);
        let prune_relative = clamp(finite_or(t.b, 0.0), 0.0, 8.0);

        // Central differences, matching tests/morphology.py exactly: the
        // offline sweeps that chose the du band are quoted in this ell, so a
        // different discretisation here would be measuring a different
        // quantity from the one every number in DESIGN.md §4.7 refers to.
        let vxp = finite_or(
            textureLoad(reaction_cur, wrap_texel(p + vec2<i32>(1, 0), idims), 0).g, 0.0);
        let vxm = finite_or(
            textureLoad(reaction_cur, wrap_texel(p - vec2<i32>(1, 0), idims), 0).g, 0.0);
        let vyp = finite_or(
            textureLoad(reaction_cur, wrap_texel(p + vec2<i32>(0, 1), idims), 0).g, 0.0);
        let vym = finite_or(
            textureLoad(reaction_cur, wrap_texel(p - vec2<i32>(0, 1), idims), 0).g, 0.0);
        let grad = length(vec2<f32>((vxp - vxm) * 0.5, (vyp - vym) * 0.5));

        flux = vec4<f32>(trail, trail * prune_relative, grad, 0.0);

        // The deposit-capacity accounting: what traffic brought (the income
        // EMA) against what the capacity withheld (its sibling EMA in `.a`).
        // Their difference is what actually landed, and the ratio of withheld
        // to landed is the boost the agent deposit needs to make the capacity
        // a redistribution rather than a sink.
        dep = vec4<f32>(
            clamp(finite_or(t.g, 0.0), 0.0, 8.0),
            clamp(finite_or(t.a, 0.0), 0.0, 8.0),
            0.0, 0.0);
    }

    tile[lid] = sample;
    tile_flux[lid] = flux;
    tile_dep[lid] = dep;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            tile[lid] = tile[lid] + tile[lid + stride];
            tile_flux[lid] = tile_flux[lid] + tile_flux[lid + stride];
            tile_dep[lid] = tile_dep[lid] + tile_dep[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let index = wid.y * nwg.x + wid.x;
        partials[index].field = tile[0];
        partials[index].flux = tile_flux[0];
        partials[index].dep = tile_dep[0];
    }
}
