// Trail decay plus deposit application, and flux-based pruning, in the slab.
//
// The 3D counterpart of trail.wgsl and identical in every mechanism; see there
// for why pruning is one-sided, why the mass it removes is returned through
// the agent deposit rather than through decay, and what each channel carries:
//
//   .r  trail
//   .g  exponential moving average of arriving traffic
//   .b  the extra decay the pruning term applies, relative to the base rate
//
// `atomicExchange` both reads and clears the deposit accumulator, so a single
// pass consumes this tick's deposits and leaves the buffer ready for the next
// one -- no separate clear dispatch over seven million voxels.

//!include common3d.wgsl

//!struct SimParams
//!struct Stats

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var trail_in: texture_3d<f32>;
@group(0) @binding(2) var trail_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(4) var clim_b: texture_3d<f32>;
@group(0) @binding(5) var clim_c: texture_3d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> stats: Stats;

const DEPOSIT_SCALE: f32 = 1048576.0;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let index = (gid.z * dims.y + gid.y) * dims.x + gid.x;
    let deposited = f32(atomicExchange(&deposit_buf[index], 0u)) / DEPOSIT_SCALE;

    let p = vec3<i32>(gid);
    let uvw = (vec3<f32>(gid) + 0.5) / vec3<f32>(dims);
    let cb = textureSampleLevel(clim_b, samp, uvw, 0.0);
    let cc = textureSampleLevel(clim_c, samp, uvw, 0.0);
    let decay = clamp(
        params.trail_decay + params.range_decay * cb.y + stats.corr_decay,
        0.001,
        0.5,
    );

    let prior = textureLoad(trail_in, p, 0).rg;
    let previous = finite_or(prior.r, 0.0);
    let income = mix(finite_or(prior.g, 0.0), deposited, params.income_rate);

    // What decay will take this tick, against what traffic is bringing in. A
    // strand carrying enough agents to replace what it loses runs no deficit;
    // a strand that has been bypassed runs a full one.
    let expenditure = decay * previous;
    let deficit = clamp(1.0 - income / (expenditure + 1e-6), 0.0, 1.0);

    let prune_gain = params.prune_gain * exp(params.range_prune * cc.y);
    let prune_relative = prune_gain * deficit;
    let decay_eff = clamp(decay * (1.0 + prune_relative), 0.001, 0.5);

    var value = previous * (1.0 - decay_eff) + deposited;
    value = clamp(finite_or(value, 0.0), 0.0, 8.0);

    textureStore(
        trail_out, p,
        vec4<f32>(value, clamp(income, 0.0, 8.0), clamp(prune_relative, 0.0, 8.0), 1.0));
}
