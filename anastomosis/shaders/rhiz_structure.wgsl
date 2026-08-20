// The root map: what the tips have built -- DESIGN.md §15.3.
//
// A field, not a polyline list: tips deposit soft splats and the structure
// is their accumulation, exactly the trail-field discipline (§2) with the
// decay turned off -- roots are *cumulative*, and what bounds them is the
// descent carrying old structure out of the window (§15.4), not forgetting.
// Step 4's senescence adds the forgetting, for the fine material only.
//
// Channels: r = density; g = age in *seconds* since last reinforcement
// (seconds, not ticks: ages span hours, and f16 stores 4000 s to a couple of
// seconds where a tick count would overflow in minutes); b = fineness, an
// EMA of the branching order of what deposits here, which is what lets the
// shading tell a taproot from fuzz. The age resets *smoothly* under deposit
// -- a blend by deposit magnitude, never a step -- so the pallor-by-age
// mapping downstream can never inherit an edge in time.
//
// The scroll is folded into the source read as everywhere else; a row that
// did not exist last tick starts empty, because fresh soil has no roots in
// it yet -- that is the entire narrative axis of the mode.

//!include rhiz_common.wgsl

@group(0) @binding(0) var<storage, read> rp: RhizParams;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> deposit_buf: array<atomic<u32>>;

const DEPOSIT_SCALE: f32 = 1048576.0;

// The deposit level that fully re-youngs a texel's age in one tick. Deposits
// per tick are ~2e-2 at the default rates, so a growing tip holds its texels
// near age zero while a texel it has left ages away smoothly.
const AGE_RESET_REF: f32 = 0.02;

const DENSITY_CAP: f32 = 4.0;
const AGE_CAP: f32 = 4000.0;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= rp.dims_x || gid.y >= rp.dims_y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);

    // Source state, shifted; a row from below the old texture starts bare.
    var density = 0.0;
    var age = 0.0;
    var fineness = 0.0;
    let sy = y + i32(rp.scroll_rows);
    if (sy < i32(rp.dims_y)) {
        let v = textureLoad(src_tex, vec2<i32>(x, sy), 0);
        density = clamp(finite_or(v.x, 0.0), 0.0, DENSITY_CAP);
        age = clamp(finite_or(v.y, 0.0), 0.0, AGE_CAP);
        fineness = clamp(finite_or(v.z, 0.0), 0.0, 1.0);
    }

    // Drain the accumulator. The tips deposited in the pre-shift frame, so
    // the texel that owns those quanta now is the shifted one -- the same
    // (x, y + scroll) the state came from.
    var deposited = 0.0;
    var deposited_order = 0.0;
    if (sy < i32(rp.dims_y)) {
        let index = 2u * (u32(sy) * rp.dims_x + u32(x));
        deposited = f32(atomicExchange(&deposit_buf[index], 0u)) / DEPOSIT_SCALE;
        deposited_order =
            f32(atomicExchange(&deposit_buf[index + 1u], 0u)) / DEPOSIT_SCALE;
    }
    // Rows that scrolled off the top this tick have no destination texel, so
    // their quanta would otherwise sit in the accumulator and land on the
    // wrong texel next tick. The top-margin threads drain and discard them:
    // each source row is drained exactly once either way.
    if (y < i32(rp.scroll_rows)) {
        let orphan = 2u * (u32(y) * rp.dims_x + u32(x));
        atomicExchange(&deposit_buf[orphan], 0u);
        atomicExchange(&deposit_buf[orphan + 1u], 0u);
    }

    density = min(density + deposited, DENSITY_CAP);

    // Age: a second older, then re-younged in proportion to what just landed
    // -- a smooth blend, so time never steps.
    let youth = clamp(deposited / AGE_RESET_REF, 0.0, 1.0);
    age = min(age + rp.dt_seconds, AGE_CAP) * (1.0 - youth);

    // Senescence (§15.11 step 4): fine material fades once it is old, at a
    // rate scaling with its fineness -- fuzz in minutes, mixed lateral paths
    // in tens of minutes, the woody axes effectively never. Every factor is
    // smooth: the age gate is a saturating ramp past the grace period, so
    // nothing anywhere switches. This is the turnover that keeps the visible
    // plant a process instead of an accumulating painting, and it is why
    // §15.7(2)'s sweep was re-run once it landed: sustained churn is now the
    // steady state, not just the growth burst.
    let ripeness = 1.0 - exp(
        -max(age - rp.senesce_delay, 0.0) / max(rp.senesce_delay, 1.0));
    density = density
        * (1.0 - rp.senesce_rate * fineness * sqrt(fineness) * ripeness);

    // Fineness follows what deposits: order 0 pulls toward 0, fines toward 1.
    if (deposited > 1e-6) {
        let order_here = clamp(deposited_order / deposited, 0.0, 1.0);
        fineness = mix(fineness, order_here, youth);
    }

    textureStore(
        dst_tex, vec2<i32>(x, y),
        vec4<f32>(density, age, fineness, 0.0));
}
