// Gray-Scott reaction-diffusion in the slab, coupled to the trail field.
//
// Every term is the one in reaction.wgsl -- the saturating trail coupling that
// makes filaments nucleate reaction structure, kill following feed along the
// live band, the geometric per-region deviation on `du` that makes feature
// size polydisperse (DESIGN.md §4.7), and the trail seeding that keeps V = 0
// from being an absorbing state the run can never leave. Read that file for
// why each is shaped the way it is; what follows is the difference.
//
// The Laplacian is the difference. In 2D the nine-point stencil is used over
// the five-point one because the five-point leaves a visible square bias in
// the growth of large structures. The same is true in 3D and worse -- a
// six-point stencil biases growth toward the three axes, and the resulting
// structures look extruded. This is the 27-point isotropic stencil, with
// weights split 0.6 / 0.3 / 0.1 across faces, edges and corners, which is the
// direct generalisation of the 0.8 / 0.2 split in 2D.
//
// That costs 27 loads per voxel per substep, which is the single most
// expensive thing the volumetric backend does: at the default slab and two
// substeps it is roughly 15 GB/s of the ~30 GB/s the whole simulation uses.
// The stencil is highly cache-coherent -- every load is reused by 26
// neighbouring invocations -- so the measured cost is far below the naive
// figure, and DESIGN.md §8.1's headroom check still holds. Dropping to a
// seven-point stencil is the lever if it ever does not.

//!include common3d.wgsl

//!struct SimParams
//!struct Stats

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var reaction_in: texture_3d<f32>;
@group(0) @binding(2) var reaction_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var trail_tex: texture_3d<f32>;
@group(0) @binding(4) var clim_a: texture_3d<f32>;
@group(0) @binding(5) var clim_c: texture_3d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> stats: Stats;

// Averaging form, as in 2D: the diffusion rates are applied by the caller per
// voxel, not here. Faces, edges and corners are weighted 0.1, 0.025 and 0.0125
// so the 6 + 12 + 8 neighbours contribute 0.6 + 0.3 + 0.1 = 1.
fn laplacian(p: vec3<i32>, idims: vec3<i32>) -> vec2<f32> {
    let c = textureLoad(reaction_in, p, 0).rg;
    var acc = vec2<f32>(0.0);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let manhattan = abs(dx) + abs(dy) + abs(dz);
                if (manhattan == 0) {
                    continue;
                }
                // 1 -> face, 2 -> edge, 3 -> corner.
                var w = 0.0125;
                if (manhattan == 1) {
                    w = 0.1;
                } else if (manhattan == 2) {
                    w = 0.025;
                }
                let s = textureLoad(
                    reaction_in, wrap_texel3(p + vec3<i32>(dx, dy, dz), idims), 0).rg;
                acc = acc + s * w;
            }
        }
    }
    return acc - c;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let p = vec3<i32>(gid);
    let idims = vec3<i32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);

    let pair = textureLoad(reaction_in, p, 0).rg;
    var u = finite_or(pair.x, 1.0);
    var v = finite_or(pair.y, 0.0);

    let trail = textureLoad(trail_tex, p, 0).r;
    let ca = textureSampleLevel(clim_a, samp, uvw, 0.0);
    let cc = textureSampleLevel(clim_c, samp, uvw, 0.0);

    // Saturating trail coupling: a heavily trafficked voxel can accumulate a
    // trail value many times the field mean, and a linear term would drive
    // feed clean off the usable map there.
    let trail_boost = params.trail_feed_gain * (trail / (1.0 + trail));
    let feed_deviation = params.range_feed * ca.x + trail_boost;
    let feed = clamp(
        params.feed + feed_deviation + stats.corr_feed,
        params.feed_min, params.feed_max);

    // Kill follows feed along the live band. Holding kill fixed while the
    // climate varies feed walks whole regions off the edge of the map, where
    // the reaction dies and cannot restart. Must stay in step with
    // config.clamp_reaction.
    let band_centre = params.kill
        + params.kill_follows_feed * (feed - params.feed);
    let band_lo = max(band_centre - params.kill_band, params.kill_min);
    let band_hi = min(band_centre + params.kill_band, params.kill_max);
    let kill_raw = params.kill
        + params.range_kill * ca.y
        + params.kill_follows_feed * feed_deviation
        + stats.corr_kill;
    let kill = clamp(kill_raw, min(band_lo, band_hi), max(band_lo, band_hi));

    // Local feature size, geometric so it composes exactly with the two terms
    // it rides on. Must stay in step with config.clamp_du.
    //
    // `stats.corr_du` is the feature-size controller's global correction and
    // `params.du` the base the scale macro sets; the climate texel is the
    // per-region deviation on top. All three are geometric, so they compose
    // exactly rather than interacting -- and `dv` riding on the result holds
    // the dv/du ratio through every one of them, which is what keeps the whole
    // mechanism clear of mass and so of the exposure governor.
    let du_global = params.du * exp(stats.corr_du);
    let du_local = clamp(
        du_global * exp(params.range_du * cc.x),
        params.du_min, params.du_max);
    let dv_local = params.dv * (du_local / max(params.du, 1e-6));

    let lap = laplacian(p, idims);
    let reaction_rate = u * v * v;
    let dt = params.rdt;

    u = u + dt * (du_local * lap.x - reaction_rate + feed * (1.0 - u));
    v = v + dt * (dv_local * lap.y + reaction_rate - (feed + kill) * v);

    // Hyphal seeding. V = 0 is an absorbing state for Gray-Scott, so without a
    // direct injection path the reaction can never restart on ground where it
    // has been extinguished -- and over a multi-day run something will
    // eventually extinguish some.
    let seed_room = clamp(1.0 - v * params.trail_seed_falloff, 0.0, 1.0);
    v = v + params.trail_seed_gain * (trail / (1.0 + trail)) * seed_room;

    u = clamp(finite_or(u, 1.0), 0.0, 1.5);
    v = clamp(finite_or(v, 0.0), 0.0, 1.0);

    textureStore(reaction_out, p, vec4<f32>(u, v, 0.0, 1.0));
}
