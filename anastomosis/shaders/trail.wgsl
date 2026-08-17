// Trail decay plus deposit application.
//
// `atomicExchange` both reads and clears the deposit accumulator, so a single
// pass consumes this tick's deposits and leaves the buffer ready for the next
// one -- no separate clear dispatch.

//!include common.wgsl

//!struct SimParams
//!struct Stats

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var trail_in: texture_2d<f32>;
@group(0) @binding(2) var trail_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(4) var clim_b: texture_2d<f32>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var<storage, read> stats: Stats;

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
    let decay = clamp(
        params.trail_decay + params.range_decay * cb.y + stats.corr_decay,
        0.001,
        0.5,
    );

    let previous = textureLoad(trail_in, vec2<i32>(gid.xy), 0).r;
    var value = previous * (1.0 - decay) + deposited;
    value = clamp(finite_or(value, 0.0), 0.0, 8.0);

    textureStore(trail_out, vec2<i32>(gid.xy), vec4<f32>(value, 0.0, 0.0, 1.0));
}
