// Trail decay plus deposit application, and flux-based pruning.
//
// `atomicExchange` both reads and clears the deposit accumulator, so a single
// pass consumes this tick's deposits and leaves the buffer ready for the next
// one -- no separate clear dispatch.
//
// Channel layout: `.r` is the trail, `.g` is an exponential moving average of
// the traffic arriving at this texel, `.b` is the extra decay the pruning term
// applies, relative to the base rate. The reduce pass sums `.b` weighted by
// trail to find how much of the field's throughput pruning is taking, which is
// what the agent layer has to hand back. The texture is rgba16float and only `.r`
// was used, so the income channel costs no new texture and no extra bandwidth,
// and the separable blur that follows this pass smooths all four channels --
// which means the income channel is spatially diffused for free. That is
// wanted rather than tolerated: per-texel agent arrivals are Poisson-spiky,
// and a decay rate steered by the raw count would be noise.
//
// The pruning is the missing half of anastomosis (DESIGN.md §4.7). Agents fuse
// and reinforce, decay is uniform and traffic-blind, so every fusion adds a
// cycle to the network and nothing ever destroys one: the mesh can only
// accrete. Real hyphal networks resorb unused hyphae and Physarum prunes
// low-flux tubes, and without that half the topology is one-way. Raising decay
// where income falls short of expenditure gives the field the ability to lose
// an edge -- and once a strand thins below sensing range agents stop finding
// it, which starves it further, so the feedback severs the edge outright and
// merges the two cells it separated.

//!include common.wgsl

//!struct SimParams
//!struct Stats

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var trail_in: texture_2d<f32>;
@group(0) @binding(2) var trail_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(4) var clim_b: texture_2d<f32>;
@group(0) @binding(5) var clim_c: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> stats: Stats;

const DEPOSIT_SCALE: f32 = 1048576.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let index = gid.y * dims.x + gid.x;
    let deposited = f32(atomicExchange(&deposit_buf[index], 0u)) / DEPOSIT_SCALE;

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let cb = textureSampleLevel(clim_b, samp, uv, 0.0);
    let cc = textureSampleLevel(clim_c, samp, uv, 0.0);
    let decay = clamp(
        params.trail_decay + params.range_decay * cb.y + stats.corr_decay,
        0.001,
        0.5,
    );

    let prior = textureLoad(trail_in, vec2<i32>(gid.xy), 0).rg;
    let previous = finite_or(prior.r, 0.0);
    let income = mix(finite_or(prior.g, 0.0), deposited, params.income_rate);

    // What decay will take this tick, against what traffic is bringing in. A
    // strand carrying enough agents to replace what it loses runs no deficit; a
    // strand that has been bypassed runs a full one.
    let expenditure = decay * previous;
    let deficit = clamp(1.0 - income / (expenditure + 1e-6), 0.0, 1.0);

    // Pruning strength varies by region, so the network visibly comes apart in
    // some places while holding together in others, and those zones migrate
    // with the climate.
    let prune_gain = params.prune_gain * exp(params.range_prune * cc.y);

    // One-sided: decay is raised on starved strands and never lowered anywhere.
    // The obvious alternative -- centring the term on the field's mean deficit,
    // so that what is taken from starved strands is handed straight back to
    // well-fed ones -- is mass-neutral by construction but freezes the field
    // solid. Well-fed strands have a deficit near zero, so centring gives them
    // several times the memory of everything else, and the network locks into
    // whatever shape it happened to hold: measured, the trail's autocorrelation
    // at a 1050-tick lag went from 0.11 to 0.72. That is a worse failure than
    // the one this whole mechanism exists to fix.
    //
    // The mass still has to be returned, or the term is a sink and the
    // homeostat cancels it through corr_decay. It is returned through the agent
    // deposit instead (`stats.prune_return`, applied in agents.wgsl), which
    // puts it wherever traffic currently is rather than wherever mass already
    // is. That is also the better model: fungi translocate what they resorb to
    // the growing tips, not to the trunk.
    let prune_relative = prune_gain * deficit;
    let decay_eff = clamp(decay * (1.0 + prune_relative), 0.001, 0.5);

    var value = previous * (1.0 - decay_eff) + deposited;
    value = clamp(finite_or(value, 0.0), 0.0, 8.0);

    // `.b` carries the relative extra removal, which the reduce pass weights by
    // trail to get the fraction of the field's throughput that pruning takes.
    // That is the quantity the deposit compensation needs, and measuring it
    // rather than deriving it keeps the accounting right when `prune_gain`
    // varies by region.
    textureStore(
        trail_out, vec2<i32>(gid.xy),
        vec4<f32>(value, clamp(income, 0.0, 8.0), clamp(prune_relative, 0.0, 8.0), 1.0));
}
