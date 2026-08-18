// Field statistics, stage 2: sum the partials and run the homeostat.
//
// The entire control loop runs on the GPU: this pass writes corrections into
// the stats buffer and the consuming passes read them next tick. Nothing
// round-trips to the CPU, so the controller costs no pipeline stall and the
// host reads the buffer only for telemetry.
//
// Two deliberate choices, both about *not* controlling too well:
//
//   * A wide deadband. A tight controller makes the output feel regulated and
//     monotonous -- it actively fights the regional variety the climate field
//     exists to create.
//   * A long time constant. The controller must be far slower than anything
//     visible, or it becomes itself a source of coordinated global change,
//     which is precisely the punctuation the brief forbids.

//!include common.wgsl

//!struct SimParams
//!struct Stats

//!struct Partial

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var<storage, read> partials_in: array<Partial>;
@group(0) @binding(2) var<storage, read_write> stats: Stats;

var<workgroup> totals: array<vec4<f32>, 256>;
var<workgroup> flux_totals: array<vec4<f32>, 256>;

// Slew on the measured prune return. It is already an average over the whole
// field and so barely moves, but it scales every agent's deposit and a tick of
// jitter there would be a tick of jitter in the whole trail field. Fast enough
// to follow a regime change well inside the homeostat's own time constant --
// this is an accounting measurement, not a control output, and lagging it would
// reintroduce exactly the mass bias it exists to remove.
const PRUNE_RETURN_RATE: f32 = 0.05;

// Relative error with a deadband: zero inside the band, and continuous at the
// edges so the controller never switches on abruptly.
fn banded_error(setpoint: f32, actual: f32, deadband: f32) -> f32 {
    let safe_setpoint = max(setpoint, 1e-6);
    let relative = (setpoint - actual) / safe_setpoint;
    if (abs(relative) <= deadband) {
        return 0.0;
    }
    return relative - sign(relative) * deadband;
}

@compute @workgroup_size(256, 1, 1)
fn reduce_final(@builtin(local_invocation_index) lid: u32) {
    let n = arrayLength(&partials_in);

    var acc = vec4<f32>(0.0);
    var acc_flux = vec4<f32>(0.0);
    var i = lid;
    loop {
        if (i >= n) { break; }
        acc = acc + partials_in[i].field;
        acc_flux = acc_flux + partials_in[i].flux;
        i = i + 256u;
    }
    totals[lid] = acc;
    flux_totals[lid] = acc_flux;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            totals[lid] = totals[lid] + totals[lid + stride];
            flux_totals[lid] = flux_totals[lid] + flux_totals[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid != 0u) {
        return;
    }

    let sums = totals[0];
    let count = max(sums.w, 1.0);
    let mean_v = sums.x / count;
    let var_v = max(sums.y / count - mean_v * mean_v, 0.0);
    let mean_activity = sums.z / count;

    stats.sum_v = sums.x;
    stats.sum_v2 = sums.y;
    stats.sum_activity = sums.z;
    stats.count = count;
    stats.mean_v = mean_v;
    stats.var_v = var_v;
    stats.mean_activity = mean_activity;

    // What flux pruning is taking out of the trail field, as a fraction of its
    // throughput. agents.wgsl multiplies the deposit by 1 + this, so the mass
    // goes back where traffic currently is. Without the return the term is a
    // straight sink and corr_decay below would undo it within a couple of time
    // constants -- a globally weaker network and no severance, the opposite of
    // the intent. Bounded because it multiplies the deposit: a transient in the
    // measurement must not be able to flood the field.
    let flux = flux_totals[0];
    let measured_return = clamp(flux.y / max(flux.x, 1e-6), 0.0, 2.0);
    stats.prune_return = mix(stats.prune_return, measured_return, PRUNE_RETURN_RATE);

    let err_mass = banded_error(params.target_mass, mean_v, params.deadband);
    let err_var = banded_error(params.target_variance, var_v, params.deadband);
    let err_activity = banded_error(params.target_activity, mean_activity, params.deadband);

    // Integral terms, clamped so a long excursion cannot wind up into a
    // correction large enough to swamp the climate field's regional variety.
    let limit = params.integral_limit;
    stats.int_mass = clamp(stats.int_mass + err_mass * params.gain_i, -limit, limit);
    stats.int_var = clamp(stats.int_var + err_var * params.gain_i, -limit, limit);
    stats.int_activity = clamp(
        stats.int_activity + err_activity * params.gain_i, -limit, limit);

    let ctrl_mass = err_mass * params.gain_p + stats.int_mass;
    let ctrl_var = err_var * params.gain_p + stats.int_var;
    let ctrl_activity = err_activity * params.gain_p + stats.int_activity;

    // Kill is the primary lever. Mean V and activity both respond
    // monotonically to -kill, so a single control serves both objectives.
    // Feed cannot: its effect on activity is non-monotonic and collapses
    // abruptly at the top of its range, so it is used only for mass and
    // structure, and gently. The agent layer trims activity from the other
    // side. All of this was measured, not assumed -- see test_regime.py.
    let want_kill = -(ctrl_mass * 0.008 + ctrl_activity * 0.006) + ctrl_var * 0.002;
    let want_feed = ctrl_mass * 0.004 - ctrl_var * 0.003;
    let want_deposit = ctrl_activity * 0.006;
    let want_decay = -ctrl_activity * 0.004;

    // Slew toward the target rather than snapping: the controller's own output
    // must not be a step either.
    let rate = clamp(params.homeo_rate, 0.0, 1.0);
    stats.corr_feed = clamp(
        mix(stats.corr_feed, want_feed, rate), -0.006, 0.006);
    stats.corr_kill = clamp(
        mix(stats.corr_kill, want_kill, rate), -0.006, 0.006);
    stats.corr_deposit = clamp(
        mix(stats.corr_deposit, want_deposit, rate), -0.012, 0.012);
    stats.corr_decay = clamp(
        mix(stats.corr_decay, want_decay, rate), -0.010, 0.010);
}
