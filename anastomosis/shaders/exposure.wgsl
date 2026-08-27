// Image statistics, stage 2: the exposure governor.
//
// Holds mean image lightness near a target so the overall level is stable over
// a multi-hour session and cannot drift bright as structure accumulates.
//
// Deliberately asymmetric: brightening is slower than darkening, because the
// unsafe direction is always "gets brighter". The governor only ever *asks* for
// a level; the per-pixel slew limiter in safety.wgsl decides how fast that
// request is honoured, so no exposure correction can arrive as a step.

//!include common.wgsl

//!struct RenderParams
//!struct Stats

@group(0) @binding(0) var<storage, read> render: RenderParams;
@group(0) @binding(1) var<storage, read> partials: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> stats: Stats;

var<workgroup> totals: array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_index) lid: u32) {
    let n = arrayLength(&partials);

    var acc = vec4<f32>(0.0);
    var i = lid;
    loop {
        if (i >= n) { break; }
        let value = partials[i];
        acc = vec4<f32>(acc.x + value.x, max(acc.y, value.y), acc.z + value.z, 0.0);
        i = i + 256u;
    }
    totals[lid] = acc;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = totals[lid + stride];
            totals[lid] = vec4<f32>(
                totals[lid].x + other.x,
                max(totals[lid].y, other.y),
                totals[lid].z + other.z,
                0.0,
            );
        }
        workgroupBarrier();
    }

    if (lid != 0u) {
        return;
    }

    let sums = totals[0];
    let count = max(sums.z, 1.0);
    let mean_l = sums.x / count;

    stats.img_sum_l = sums.x;
    stats.img_max_l = sums.y;
    stats.img_count = count;

    // The amplification ceiling: modes whose honest mean falls over time
    // (the perennial's record inking in) pin this at 1 so the governor only
    // ever attenuates; a zeroed or garbage field value falls back to the
    // full range rather than freezing the image dark.
    var ceiling = finite_or(render.exposure_max, 20.0);
    if (ceiling < 0.5) {
        ceiling = 20.0;
    }
    let current = clamp(finite_or(stats.exposure, 1.0), 0.02, ceiling);

    // Bounded request: even a completely black or blown-out frame can only ask
    // for a factor of two per frame, before rate limiting.
    let desired = clamp(
        current * render.exposure_target / max(mean_l, 1e-4),
        current * 0.5,
        current * 2.0,
    );
    let rate = select(render.exposure_release, render.exposure_attack, desired > current);
    stats.exposure = clamp(
        mix(current, desired, clamp(rate, 0.0, 1.0)), 0.02, ceiling);
}
