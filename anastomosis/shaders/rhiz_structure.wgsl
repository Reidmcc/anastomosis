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
@group(0) @binding(4) var record_src: texture_2d<f32>;
@group(0) @binding(5) var record_dst: texture_storage_2d<rgba16float, write>;
// The controller words (rhizotron.py): this pass owes the season controller
// the third and fourth -- living mass and wood mass, fixed point.
@group(0) @binding(6) var<storage, read_write> front_buf: array<atomic<u32>>;

const DEPOSIT_SCALE: f32 = 1048576.0;
const MASS_SCALE: f32 = 256.0;

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

// The record layer's source state, shifted exactly as the living layer is:
// (lignin, biographical age, graft glow, ghost). Rows the texture did not
// hold last tick arrive bare -- fresh soil has no history in it.
fn record_at(x: i32, y: i32) -> vec4<f32> {
    let sy = y + i32(rp.scroll_rows);
    let sx = ((x % i32(rp.dims_x)) + i32(rp.dims_x)) % i32(rp.dims_x);
    if (sy >= i32(rp.dims_y)) {
        return vec4<f32>(0.0);
    }
    let v = textureLoad(record_src, vec2<i32>(sx, max(sy, 0)), 0);
    return vec4<f32>(
        clamp(finite_or(v.x, 0.0), 0.0, DENSITY_CAP),
        clamp(finite_or(v.y, 0.0), 0.0, AGE_CAP),
        clamp(finite_or(v.z, 0.0), 0.0, 4.0),
        clamp(finite_or(v.w, 0.0), 0.0, DENSITY_CAP),
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
    // Ages advance in 64-tick batches, not per tick: an f16 texel truncates
    // an increment below one ulp, so per-tick seconds stall at increment x
    // 1024 -- about a minute -- and every age-driven gradient in the mode
    // silently saturated there (found when the record layer's biographical
    // clock froze at 64.0 exactly; the living channel had been capped since
    // it was built). Batching multiplies the ceiling past AGE_CAP, the
    // step is a couple of seconds on mappings whose scales are minutes --
    // far below anything the eye or the slew limiter can see -- and the
    // tick counter is checkpointed, so a resume replays the same batches.
    let share = clamp(deposited / max(density, AGE_RESET_REF), 0.0, 1.0);
    let age_batch = select(
        0.0, rp.dt_seconds * 64.0, (rp.tick & 63u) == 0u);
    age = min(age + age_batch, AGE_CAP) * (1.0 - share);

    // --- The commitment transfer (§17.6) -----------------------------------
    // Coarse living material converts into wood at a steady slow rate --
    // independent of re-touch, which is what the recency age above can never
    // be: an axis shaft lignifies behind its tip whatever grows across it.
    // Fineness squared discounts the rate, so fuzz never commits (senescence
    // is its exit, below) and the width hierarchy becomes a *time* hierarchy:
    // the boldest strokes are the first into the record. Wood is append-only
    // while the season lives -- nothing below ever decrements it.
    let rec = record_at(x, y);
    var lignin = rec.x;
    var bio_age = rec.y;
    // The discount stays squared -- fuzz and fuzz-adjacent mass must NOT
    // commit, or the record inks in every path ever taken and the pane
    // clutters (tried at the three-halves power: a black mesh). The race
    // against senescence is won by rate instead: at the shipped
    // lignify_rate a lateral shaft (fineness ~0.5) commits in ~5 minutes
    // against its 8-minute life, so branch systems stand as wood instead
    // of vanishing -- the first live viewing's second finding.
    let coarse = (1.0 - fineness) * (1.0 - fineness);
    let transfer = min(rp.lignify_rate * coarse * density, density);
    density = density - transfer;
    lignin = min(lignin + transfer, DENSITY_CAP);
    var ghost = rec.w;

    // The interment (§17.6): the one licensed writer of wood's exit, zero
    // outside a burial. Lignin leaves for the ghost at the drive's eased
    // rate; the *existing* ghost fades under the same cover, so the ground
    // reads a couple of seasons deep and no deeper. The biographical clock
    // clears with the wood it was counting for, slowly, wherever none is
    // left to age.
    // Young wood is spared most of the burial: the interment is of the
    // completed season's record, and a straggler plant still writing
    // through it keeps the skeleton it is laying down rather than having
    // it erased under its living tips.
    let interred = lignin * rp.intern_rate
        * mix(0.15, 1.0, smoothstep(120.0, 480.0, bio_age));
    lignin = lignin - interred;
    ghost = min(
        ghost * (1.0 - rp.ghost_fade) + interred * rp.ghost_gain,
        DENSITY_CAP);
    let presence = smoothstep(0.004, 0.02, lignin);
    bio_age = bio_age * (1.0 - 0.03 * (1.0 - presence));
    // Biographical age: seconds since wood first held here. Advances only
    // where there is wood to age, smoothly gated so a whisper of lignin does
    // not start a clock the eye will later read; never reset by anything.
    bio_age = min(
        bio_age + select(0.0, rp.dt_seconds * 64.0, (rp.tick & 63u) == 0u)
            * smoothstep(0.01, 0.05, lignin),
        AGE_CAP);

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
    // The remnant cleanup: faint old mass -- what patchy senescence leaves
    // of a fuzz path -- fades several times faster than established
    // material, and partly regardless of its fineness label, so the field
    // never holds a confetti of sub-salience dashes (seen the moment the
    // record layer gave the living material a contrasting ground). Smooth
    // in density, gated by the same ripeness, and irrelevant to any path
    // still being reinforced.
    let remnant = exp(-density * 18.0);
    let kept = density
        * (1.0 - rp.senesce_rate * ripeness
            * (fineness * sqrt(fineness) * (1.0 + 4.0 * remnant)
                + 0.8 * remnant));

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
    textureStore(
        record_dst, vec2<i32>(x, y),
        vec4<f32>(lignin, bio_age, rec.z, ghost));

    // The season controller's masses, accumulated only where there is
    // something to count -- integer atomics, so the sum is deterministic
    // under any dispatch order.
    if (density > 1e-4) {
        atomicAdd(&front_buf[2], u32(density * MASS_SCALE));
    }
    if (lignin > 1e-4) {
        atomicAdd(&front_buf[3], u32(lignin * MASS_SCALE));
    }
}
