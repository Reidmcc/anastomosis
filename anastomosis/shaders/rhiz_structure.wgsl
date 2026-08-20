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

// The floor on the mass the deposit share is measured against, so a deposit
// onto near-bare ground still counts as (almost) the whole of it: a fresh
// path is fully young and fully its own order, while established mass
// dilutes later arrivals in proportion.
const AGE_RESET_REF: f32 = 0.02;

const DENSITY_CAP: f32 = 4.0;
const AGE_CAP: f32 = 4000.0;

// The source state of the texel landing at (x, y) after this tick's scroll:
// (density, age, fineness, nutrient). A row from below the old texture
// starts bare of roots but arrives with the generator's richness -- and so
// does any texel whose nutrient reads non-positive, which is both the fresh
// seeding (the host writes zeros rather than mirroring the hash in numpy)
// and a column saved before the economy existed. Living depletion never
// reaches zero exactly; the floor in `nutrient_seed` sees to that.
fn state_at(x: i32, y: i32) -> vec4<f32> {
    // Above the texture (a diffusion probe from the top row) reads the top
    // row itself -- no-flux; below it is fresh soil from the generator.
    let sy = max(y + i32(rp.scroll_rows), 0);
    let sx = ((x % i32(rp.dims_x)) + i32(rp.dims_x)) % i32(rp.dims_x);
    let ux = (f32(sx) + 0.5) / f32(rp.dims_x);
    if (sy >= i32(rp.dims_y)) {
        return vec4<f32>(
            0.0, 0.0, 0.0, nutrient_seed(rp, ux, f32(y) + 0.5));
    }
    let v = textureLoad(src_tex, vec2<i32>(sx, sy), 0);
    var n = finite_or(v.w, 0.0);
    if (n <= 0.0) {
        n = nutrient_seed(rp, ux, f32(y) + 0.5);
    }
    return vec4<f32>(
        clamp(finite_or(v.x, 0.0), 0.0, DENSITY_CAP),
        clamp(finite_or(v.y, 0.0), 0.0, AGE_CAP),
        clamp(finite_or(v.z, 0.0), 0.0, 1.0),
        clamp(n, 0.0, 2.0),
    );
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= rp.dims_x || gid.y >= rp.dims_y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);

    let state = state_at(x, y);
    var density = state.x;
    var age = state.y;
    var fineness = state.z;
    var nutrient = state.w;
    let sy = y + i32(rp.scroll_rows);

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

    // Age and fineness update by the deposit's *share of the texel's mass*,
    // not by its absolute size. The first build weighted them by the deposit
    // against a fixed reference, and the consequence is only visible in a
    // grown plant: one fine root crossing a woody axis re-younged and
    // re-labelled the axis's texels as fine, and senescence then ate holes
    // in the trunk -- the central roots breaking up while their plant lived. A
    // hairline crossing a trunk is a fraction of a percent of its mass, and
    // now moves its identity by exactly that much.
    let share = clamp(deposited / max(density, AGE_RESET_REF), 0.0, 1.0);
    age = min(age + rp.dt_seconds, AGE_CAP) * (1.0 - share);

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
    let kept = density
        * (1.0 - rp.senesce_rate * fineness * sqrt(fineness) * ripeness);

    // Fineness follows what deposits, by the same mass share: order 0 pulls
    // toward 0, fines toward 1, and a trunk stays a trunk under crossings.
    if (deposited > 1e-6) {
        let order_here = clamp(deposited_order / deposited, 0.0, 1.0);
        fineness = mix(fineness, order_here, share);
    }

    // --- The nutrient economy (§15.11 step 5) -------------------------------
    // Growth eats: what a tip laid down here cost the texel a multiple of
    // itself in richness. Death feeds: the mass senescence just removed
    // returns as richness, which is the recycling memory -- ground where a
    // plant died is ground the next plant forages into. And a whisper of
    // diffusion lets a cache feed its surroundings rather than only its own
    // texels; the neighbours are read shifted through `state_at`, so the
    // exchange is frame-consistent under any scroll.
    nutrient = nutrient - rp.nutrient_uptake * deposited
        + rp.nutrient_recycle * (density - kept);
    let n_left = state_at(x - 1, y).w;
    let n_right = state_at(x + 1, y).w;
    let n_up = state_at(x, y - 1).w;
    let n_down = state_at(x, y + 1).w;
    nutrient = nutrient + rp.nutrient_spread
        * (0.25 * (n_left + n_right + n_up + n_down) - nutrient);
    nutrient = clamp(finite_or(nutrient, 0.002), 0.002, 2.0);
    density = kept;

    textureStore(
        dst_tex, vec2<i32>(x, y),
        vec4<f32>(density, age, fineness, nutrient));
}
