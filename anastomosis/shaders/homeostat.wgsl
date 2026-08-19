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
//
// The feature-size loop at the bottom is deliberately none of that, and the
// difference is worth being explicit about because it looks like a
// contradiction. Mass, variance and activity are quantities the field is
// *allowed* to wander in; defending them tightly would flatten the regional
// variety the climate field exists to create, so they get a wide deadband and
// a slow hand. Feature size is not being defended -- its setpoint is itself
// walking, and the loop's job is to make sure the texture follows it rather
// than sitting in whatever attractor the reaction found. So it has no deadband
// and a much shorter time constant. A deadband there would be a licence to
// freeze, which is the failure this whole mechanism exists to catch
// (DESIGN.md 4.7 step 5).

//!include common.wgsl

//!struct SimParams
//!struct Stats

//!struct Partial

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var<storage, read> partials_in: array<Partial>;
@group(0) @binding(2) var<storage, read_write> stats: Stats;

var<workgroup> totals: array<vec4<f32>, 256>;
var<workgroup> flux_totals: array<vec4<f32>, 256>;
var<workgroup> dep_totals: array<vec4<f32>, 256>;

// Slew on the measured prune return. It is already an average over the whole
// field and so barely moves, but it scales every agent's deposit and a tick of
// jitter there would be a tick of jitter in the whole trail field. Fast enough
// to follow a regime change well inside the homeostat's own time constant --
// this is an accounting measurement, not a control output, and lagging it would
// reintroduce exactly the mass bias it exists to remove.
const PRUNE_RETURN_RATE: f32 = 0.05;

// The field is too thin for the length scale to mean anything below this: `ell`
// is a ratio of two means that both go to zero together, and on a field that
// has been extinguished it is 0/0. The mass loop is what matters then, and it
// is unaffected; this one holds its output until there is a field to measure.
const ELL_LIVE_FLOOR: f32 = 1e-4;

// Mature ticks the feature-size reference is averaged over before it becomes
// the slow exponential its time constant asks for. It has to start fast: its
// first value is otherwise its value for the next half hour, and the
// controller spends that half hour driving `du` to a bound chasing a baseline
// taken before the field had a texture. It has to *stop* being fast just as
// firmly -- a reference that keeps converging quickly tracks the modulation it
// is supposed to be the baseline for, and the loop cancels itself. 600 ticks
// is half a minute at the default rate, long enough to average the field's own
// jitter and far short of the walk that rides on top of it.
//
// The counter it is compared against runs for the life of the session and is
// an f32, so past 2^24 ticks -- about ten days at the default rate -- it stops
// incrementing exactly. That is harmless and deliberate rather than merely
// tolerated: by then its only remaining job is to keep this comparison false,
// which a saturated counter does perfectly well.
const ELL_SEED_SAMPLES: f32 = 600.0;

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
    var acc_dep = vec4<f32>(0.0);
    var i = lid;
    loop {
        if (i >= n) { break; }
        acc = acc + partials_in[i].field;
        acc_flux = acc_flux + partials_in[i].flux;
        acc_dep = acc_dep + partials_in[i].dep;
        i = i + 256u;
    }
    totals[lid] = acc;
    flux_totals[lid] = acc_flux;
    dep_totals[lid] = acc_dep;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            totals[lid] = totals[lid] + totals[lid + stride];
            flux_totals[lid] = flux_totals[lid] + flux_totals[lid + stride];
            dep_totals[lid] = dep_totals[lid] + dep_totals[lid + stride];
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

    // What the deposit capacity withheld, against what it let through: income
    // EMA minus withheld EMA is the deposit that actually landed, and boosting
    // the deposit by withheld/landed is what makes the total deposited mass
    // come out the same as if the capacity were not there -- taken from the
    // hubs, returned wherever traffic is. Slewed at the same rate as the
    // pruning return and for the same reason: it scales every agent's deposit,
    // so a tick of jitter here is a tick of jitter in the whole trail field.
    // The bound needs more headroom than the pruning return's. Deposits land
    // on the network by construction -- agents ride the strands they follow --
    // so the deposit-weighted trail level is several times the field mean, and
    // the equilibrium ratio is well above one. A bound the measurement rests
    // against would under-return, and an under-returned capacity is a net
    // deposit sink the homeostat then cancels through corr_decay: a globally
    // weaker network and hubs intact, the opposite of the intent.
    let dep = dep_totals[0];
    let measured_cap = clamp(dep.y / max(dep.x - dep.y, 1e-6), 0.0, 3.0);
    stats.cap_return = mix(stats.cap_return, measured_cap, PRUNE_RETURN_RATE);

    let err_mass = banded_error(params.target_mass, mean_v, params.deadband);

    // --- Feature size -----------------------------------------------------
    //
    // The one measure here that is not invariant under rearrangement, which is
    // why it is the one that can tell a live field from a frozen picture of
    // one. Everything in this block is in logarithms: `du` acts on the length
    // scale as roughly its square root -- measured on a running engine, a
    // x1.43 correction moved ell from 2.39 to 2.97 -- so a multiplicative
    // correction is the natural actuator, and a walk symmetric in log is
    // symmetric in what it does to the texture.
    let mean_grad = flux.z / count;
    let ell = mean_v / max(mean_grad, 1e-9);
    stats.mean_grad_v = mean_grad;
    stats.ell = ell;

    // Steered only while the field is one whose feature size means anything,
    // and `err_mass` is already the test for that: zero exactly when mass sits
    // inside the deadband above. A field still growing into its band has a
    // length scale that is going to change for reasons that have nothing to do
    // with this loop, and a field a dieback has just emptied has one that says
    // more about what is left than about what the texture is doing. Outside
    // the band the correction simply holds, which is the right thing for an
    // integrator whose measurement has stopped being informative.
    //
    // It doubles as this loop's NaN quarantine (DESIGN.md 4.6), and that is
    // load-bearing rather than incidental: comparisons against NaN are false,
    // so a poisoned reduction skips the block entirely and `corr_du` keeps its
    // last good value. It matters more here than for the corrections above
    // because this one is applied through an `exp`, so a NaN that reached the
    // accumulator would not merely be a wrong diffusion rate.
    if (err_mass == 0.0 && mean_v > ELL_LIVE_FLOOR && mean_grad > 0.0) {
        let ln_ell = log(max(ell, 1e-6));
        // The demanded modulation, subtracted here and added back below. This
        // is what the reference is *not* allowed to absorb; see
        // ReactionParams.ell_ref_tau_seconds.
        let natural = ln_ell - params.ell_offset;

        // A running mean for the first ELL_SEED_SAMPLES ticks, an exponential
        // one after that: exact adoption of the first measurement, 1/n
        // convergence while the reference is being established, and the
        // configured time constant from then on. Both halves matter; see the
        // constant.
        let n = stats.ell_samples + 1.0;
        stats.ell_samples = n;
        var ref_rate = params.ell_ref_rate;
        if (n <= ELL_SEED_SAMPLES) {
            ref_rate = 1.0 / n;
        }

        // Pure integral, and pure integral on purpose: it is the accumulated
        // correction that *is* the actuator, so the controller has no
        // steady-state error against a slowly moving setpoint and its output
        // is slew-limited by construction. Clamping the accumulator rather
        // than its effect is the anti-windup.
        let ell_err = (stats.ell_ref + params.ell_offset) - ln_ell;
        stats.corr_du = clamp(
            stats.corr_du + params.ell_rate * ell_err,
            -params.ell_corr_limit, params.ell_corr_limit);

        // Updated after the error is taken, so one tick's reference is the one
        // the same tick's setpoint was built from.
        stats.ell_ref = stats.ell_ref + ref_rate * (natural - stats.ell_ref);
    }

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
