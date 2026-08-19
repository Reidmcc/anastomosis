// Gray-Scott reaction-diffusion, coupled to the trail field.
//
// The coupling is the point: agent trails raise the local feed rate, so the
// filaments the agents lay down become the places where reaction structure
// nucleates. That gives filaments internal texture that neither system
// produces alone -- Physarum trails are otherwise smooth ridges, and free
// Gray-Scott has no reason to organise into a network.
//
// Feed and kill also carry a per-region offset from the climate field, so
// different parts of the screen sit in different Gray-Scott regimes
// simultaneously, and those regions migrate.
//
// So does the diffusion rate, and that one is load-bearing for a different
// reason (DESIGN.md §4.7). At fixed diffusion a Gray-Scott regime has one
// characteristic wavelength, so every feature on screen ends up the same size
// -- a monodisperse spot field, which is monotonous over long viewing and is
// also a trypophobia trigger. Varying `du` per region makes the feature size
// polydisperse and migratory: coarse zones drift through fine ones, structures
// merge as their neighbourhood coarsens and split as it refines.

//!include common.wgsl

//!struct SimParams
//!struct Stats

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var reaction_in: texture_2d<f32>;
@group(0) @binding(2) var reaction_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var trail_tex: texture_2d<f32>;
@group(0) @binding(4) var clim_a: texture_2d<f32>;
@group(0) @binding(5) var clim_c: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> stats: Stats;

// Nine-point Laplacian: more isotropic than the five-point stencil, which
// leaves a visible square bias in the growth of large structures.
//
// Note that the diffusion *rates* are applied by the caller, per texel, not
// here: this is the plain averaging-form Laplacian of both species.
fn laplacian(p: vec2<i32>, idims: vec2<i32>) -> vec2<f32> {
    let c = textureLoad(reaction_in, p, 0).rg;
    var ortho = vec2<f32>(0.0);
    ortho = ortho + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(1, 0), idims), 0).rg;
    ortho = ortho + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(-1, 0), idims), 0).rg;
    ortho = ortho + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(0, 1), idims), 0).rg;
    ortho = ortho + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(0, -1), idims), 0).rg;
    var diag = vec2<f32>(0.0);
    diag = diag + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(1, 1), idims), 0).rg;
    diag = diag + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(1, -1), idims), 0).rg;
    diag = diag + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(-1, 1), idims), 0).rg;
    diag = diag + textureLoad(reaction_in, wrap_texel(p + vec2<i32>(-1, -1), idims), 0).rg;
    return ortho * 0.2 + diag * 0.05 - c;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = vec2<i32>(gid.xy);
    let idims = vec2<i32>(dims);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    let uv_pair = textureLoad(reaction_in, p, 0).rg;
    var u = finite_or(uv_pair.x, 1.0);
    var v = finite_or(uv_pair.y, 0.0);

    let trail = textureLoad(trail_tex, p, 0).r;
    let ca = textureSampleLevel(clim_a, samp, uv, 0.0);
    let cc = textureSampleLevel(clim_c, samp, uv, 0.0);

    // Saturating trail coupling: a heavily trafficked texel can accumulate a
    // trail value many times the field mean, and a linear term would drive feed
    // clean off the usable map there.
    let trail_boost = params.trail_feed_gain * (trail / (1.0 + trail));
    let feed_deviation = params.range_feed * ca.x + trail_boost;
    let feed = clamp(
        params.feed + feed_deviation + stats.corr_feed,
        params.feed_min, params.feed_max);

    // Kill follows feed along the live band. Holding kill fixed while the
    // climate varies feed walks regions off the edge of the map, where the
    // reaction dies and cannot restart -- the single most dangerous failure
    // mode for a run measured in days.
    // Bounded relative to the band centre as well as absolutely. The live
    // region is a diagonal strip: a fixed box in (feed, kill) admits dead
    // corners at both ends. Must stay in step with config.clamp_reaction.
    let band_centre = params.kill
        + params.kill_follows_feed * (feed - params.feed);
    let band_lo = max(band_centre - params.kill_band, params.kill_min);
    let band_hi = min(band_centre + params.kill_band, params.kill_max);
    let kill_raw = params.kill
        + params.range_kill * ca.y
        + params.kill_follows_feed * feed_deviation
        + stats.corr_kill;
    let kill = clamp(kill_raw, min(band_lo, band_hi), max(band_lo, band_hi));

    // Local feature size. The deviation is geometric, not additive, for two
    // reasons. `du` is driven by the `scale` macro, so a fixed offset would
    // mean a different spread at each end of that knob. And the band is not
    // symmetric around the base -- there is a factor of two of headroom above
    // and only 0.57 below -- so an additive deviation spends its downside
    // against the floor long before it has used its upside. Geometric, the
    // same deviation costs 1% of texels at the clamp where additive costs 5%,
    // for the same spread of feature size.
    //
    // The bounds are a survival floor and a stability ceiling, not a stylistic
    // range: under du_min the reaction thins toward collapse (mean V 0.015 at
    // du = 0.06) and its activity drops below the homeostat's deadband, so the
    // controller starts fighting the drift with kill -- which moves mass, and
    // so moves the exposure governor. Over du_max the explicit diffusion runs
    // out of headroom. Must stay in step with config.clamp_du.
    //
    // dv is scaled by the same factor rather than varied on its own: the
    // dv/du *ratio* moves mass hard (0.36 -> 0.64 takes mean V from 0.144 to
    // 0.084), while the pair scaled together barely moves it at all. That is
    // the whole reason this lever was chosen over kill.
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
    // has been extinguished -- and over a multi-day run, something will
    // eventually extinguish some. Agents wander everywhere, so trail-driven
    // seeding guarantees recovery without needing a separate mechanism.
    let seed_room = clamp(1.0 - v * params.trail_seed_falloff, 0.0, 1.0);
    v = v + params.trail_seed_gain * (trail / (1.0 + trail)) * seed_room;

    u = clamp(finite_or(u, 1.0), 0.0, 1.5);
    v = clamp(finite_or(v, 0.0), 0.0, 1.0);

    textureStore(reaction_out, p, vec4<f32>(u, v, 0.0, 1.0));
}
