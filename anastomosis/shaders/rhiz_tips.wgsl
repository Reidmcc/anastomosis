// Root tips: the apical meristems -- DESIGN.md §15.3.
//
// A few thousand individuals, not a population: each tip has an order (axis,
// lateral, fine), an age, a side, and a place in a *tree of slots*. Steering
// is a weighted sum of tropisms -- the actual botanical control vocabulary:
//
//   * gravitropism, toward the order's gravitropic setpoint angle (the axis
//     plunges, laterals hold an oblique angle, fines barely answer);
//   * thigmotropism, slowing into and deflecting around stones;
//   * hydrotropism, toward moisture;
//   * self-avoidance, away from sensed structure -- the fungal fusion bias
//     with its sign flipped, which is what makes a tree instead of a mesh
//     (§15.1).
//
// Two structural decisions carry the pass:
//
// **The pool is a tree of slots, and birth is self-construction.** Axis a is
// slot a; lateral (a, l) is A + a*L + l; fine (a, l, f) follows. A parent
// never writes its child's slot: it only increments its own child counter,
// and on the *next* tick the child's own thread notices the counter has
// passed its index and constructs itself from the parent's previous state.
// With the tips double-buffered (read src, write own slot in dst, nothing
// else) there is no write to any slot but the thread's own, no atomics, and
// no dependence on GPU scheduling -- which is what the bit-identical resume
// test demands of every mechanism here.
//
// **Tips live in the texture frame and shift with it.** The world scrolls up
// through the textures by whole rows (§15.4); a tip senses and moves in the
// pre-shift frame this tick's source textures are in, deposits there, and
// writes its position already shifted for the frame the next tick will read.
// A tip carried off the top of the column is done; the slot stays spent.
//
// Deposits are soft gaussian splats into a fixed-point accumulator, exactly
// the fungal discipline (§2): magnitude proportional to distance actually
// travelled -- a stalled tip deposits nothing -- and far below anything
// individually visible, so structure emerges only from accumulation.

//!include rhiz_common.wgsl

struct Tip {
    pos: vec2<f32>,
    heading: f32,
    rng: u32,
    // bits 0-1 order; bit 2 side (set = -1); bit 3 alive; bit 4 spent
    // (has lived at least once); bits 8-31 the parent's running count of
    // children handed out -- 24 bits, which at a lateral every half minute
    // wraps in years rather than the days a 16-bit count would.
    flags: u32,
    age: f32,
    since_branch: f32,
    // Which life this slot is on. Slots recycle (§15.11 step 4): a child's
    // n-th life begins when its parent's counter passes
    // `index + n * capacity`, so a parent that keeps growing re-uses its
    // block forever without any slot ever being written by another thread.
    generation: f32,
};

@group(0) @binding(0) var<storage, read> rp: RhizParams;
@group(0) @binding(1) var<storage, read> tips_src: array<Tip>;
@group(0) @binding(2) var<storage, read_write> tips_dst: array<Tip>;
@group(0) @binding(3) var moisture_tex: texture_2d<f32>;
@group(0) @binding(4) var structure_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(6) var samp: sampler;
// Two words the front controller reads back rarely: the deepest living row
// (fixed point, atomicMax) and the count of living tips. Derived state,
// zeroed by the host every tick, never checkpointed.
@group(0) @binding(7) var<storage, read_write> front_buf: array<atomic<u32>>;

const DEPOSIT_SCALE: f32 = 1048576.0;
const FRONT_SCALE: f32 = 64.0;
const DOWN: f32 = 1.5707963267948966;  // +pi/2: rows grow downward

fn order_of(flags: u32) -> u32 { return flags & 3u; }
fn side_of(flags: u32) -> f32 { return select(1.0, -1.0, (flags & 4u) != 0u); }
fn is_alive(flags: u32) -> bool { return (flags & 8u) != 0u; }
fn is_spent(flags: u32) -> bool { return (flags & 16u) != 0u; }
fn children_of(flags: u32) -> u32 { return flags >> 8u; }
fn with_children(flags: u32, n: u32) -> u32 {
    return (flags & 0x000000FFu) | (n << 8u);
}

fn wrap_angle(a: f32) -> f32 {
    return a - TAU * floor((a + PI) / TAU);
}

// Elongation by order, decelerating with the tip's age toward a floor: a
// young apex sprints, a mature one advances at roughly the descent's own
// pace, so the front hovers in the window instead of racing off its bottom
// edge, which is what it did when elongation was a flat rate.
fn elong_for(order: u32, age: f32) -> f32 {
    var base = rp.elong_fine;
    switch order {
        case 0u: { base = rp.elong_axis; }
        case 1u: { base = rp.elong_lateral; }
        default: { base = rp.elong_fine; }
    }
    let factor = rp.elong_floor
        + (1.0 - rp.elong_floor) * exp(-max(age, 0.0) * rp.elong_slow);
    return base * factor;
}

fn gsa_for(order: u32) -> f32 {
    switch order {
        case 0u: { return 0.0; }
        case 1u: { return rp.gsa_lateral; }
        default: { return rp.gsa_fine; }
    }
}

fn gsa_gain_for(order: u32) -> f32 {
    switch order {
        case 0u: { return rp.gsa_gain_axis; }
        case 1u: { return rp.gsa_gain_lateral; }
        default: { return rp.gsa_gain_fine; }
    }
}

fn splat_sigma_for(order: u32) -> f32 {
    switch order {
        case 0u: { return rp.splat_axis; }
        case 1u: { return rp.splat_lateral; }
        default: { return rp.splat_fine; }
    }
}

// Where this slot sits in the tree: (parent slot, own index among the
// parent's children), or parent == slot for an axis, which has no parent.
fn parent_of(slot: u32) -> vec2<u32> {
    let a = rp.max_axes;
    let l = rp.laterals_per_axis;
    if (slot < a) {
        return vec2<u32>(slot, 0u);
    }
    if (slot < a + a * l) {
        let li = slot - a;
        return vec2<u32>(li / l, li % l);  // parent axis, child index
    }
    let fi = slot - a - a * l;
    let f = max(rp.fines_per_lateral, 1u);
    return vec2<u32>(a + fi / f, fi % f);  // parent lateral, child index
}

// The tropism score of a probe point: what makes ground worth growing into.
//
// The textures this samples are still in the pre-shift frame (the tips run
// first in the tick), so texture reads use the tip's own y; the procedural
// soil is keyed to the *post*-shift origin the tick's parameters carry, so
// its row is the tip's y minus this tick's scroll. Two frames, one texel of
// difference at most, but exactness here is what the determinism tests hold
// the whole pass to.
fn probe(pos: vec2<f32>) -> f32 {
    let fdims = vec2<f32>(f32(rp.dims_x), f32(rp.dims_y));
    let uv = vec2<f32>(fract(pos.x / fdims.x), clamp(pos.y / fdims.y, 0.0, 1.0));
    let wet = textureSampleLevel(moisture_tex, samp, uv, 0.0).x;
    let field = textureSampleLevel(structure_tex, samp, uv, 0.0);
    let soil = soil_at(rp, uv.x, pos.y - f32(rp.scroll_rows));
    // The tropism sum, chemotropism included: richness pulls (the nutrient
    // channel rides the same sample the avoidance already takes).
    return rp.hydro_gain * wet
        + rp.chemo_gain * clamp(field.w, 0.0, 1.2)
        - rp.thigmo_gain * soil.imped
        - rp.avoid_gain * min(field.x, 1.0);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= rp.tips_total) {
        return;
    }
    var tip = tips_src[slot];
    tip.pos = finite_or2(tip.pos, 0.0);
    tip.heading = finite_or(tip.heading, DOWN);

    // --- Birth and rebirth ---------------------------------------------------
    // A slot not currently living is a candidate for its next life. For a
    // child, that life begins when its parent's counter passes
    // `index + generation * capacity` -- reading the *previous* tick's
    // counter, so the decision this depends on is already made and
    // immutable, and slots recycle forever as the counter climbs. For an
    // axis, it is germination (§15.4): the seed rests out its delay, then
    // wakes at a hashed site -- eagerly where the soil is wet, so rain
    // brings a pulse of new plants, and at a small unconditional floor so a
    // drought cannot be an absorbing state.
    if (!is_alive(tip.flags) && slot >= rp.max_axes) {
        let parent = parent_of(slot);
        let parent_tip = tips_src[parent.x];
        let capacity = select(
            max(rp.fines_per_lateral, 1u), rp.laterals_per_axis,
            slot < rp.max_axes + rp.max_axes * rp.laterals_per_axis);
        let number = parent.y + u32(max(tip.generation, 0.0)) * capacity;
        if (is_spent(parent_tip.flags) && is_alive(parent_tip.flags)
            && children_of(parent_tip.flags) > number) {
            var seed = pcg3(
                slot, rp.seed ^ u32(tip.generation) * 0x85EBCA6Bu, 0xB1127u);
            let side = select(4u, 0u, rnd(&seed) < 0.5);
            let order = order_of(parent_tip.flags) + 1u;
            tip.pos = parent_tip.pos;
            tip.heading = parent_tip.heading
                + side_of(side) * (rp.branch_angle + rp.branch_jitter * gauss(&seed));
            tip.rng = pcg(seed);
            tip.flags = order | side | 8u | 16u;  // alive, spent
            tip.age = 0.0;
            tip.since_branch = 0.0;
            tip.generation = tip.generation + 1.0;
        } else {
            tips_dst[slot] = tip;
            return;
        }
    } else if (!is_alive(tip.flags) && slot < rp.max_axes) {
        // A spent axis rests, then may wake as a new plant; an axis slot the
        // first seeding never used is a seed in the bank, resting from the
        // field's first tick, so the community fills toward the pool over
        // time as the weather allows. A waking axis's child counter is
        // deliberately *not* reset: the laterals keep cycling against the
        // continuing count, so no other slot needs touching.
        tip.age = tip.age + 1.0;
        var seed = tip.rng ^ pcg2(rp.tick, rp.seed);
        if (tip.age > rp.regerm_delay) {
            let gen = u32(max(tip.generation, 0.0));
            var site_seed = pcg3(slot, gen * 2654435761u, rp.seed ^ 0x5EEDu);
            let fdims = vec2<f32>(f32(rp.dims_x), f32(rp.dims_y));
            // Seeds wake just below the soil line (§17.4) -- crowns start at
            // the surface, where the rain arrives. With no surface in the
            // pane (the sentinel row, or a §15 column sunk past it) the old
            // upper-window band stands in.
            var germ_top = rp.surface_row + 1.0;
            if (germ_top < f32(rp.margin_top) || germ_top >= f32(rp.dims_y)) {
                germ_top = f32(rp.margin_top) + 0.06 * f32(rp.view_rows);
            }
            let site = vec2<f32>(
                rnd(&site_seed) * fdims.x,
                germ_top + (0.005 + 0.055 * rnd(&site_seed)) * f32(rp.view_rows),
            );
            let wet = textureSampleLevel(
                moisture_tex, samp,
                vec2<f32>(site.x / fdims.x, site.y / fdims.y), 0.0).x;
            let eager = smoothstep(
                rp.germ_moisture * 0.5, rp.germ_moisture, wet);
            if (rnd(&seed) < rp.germ_floor + rp.germ_prob * eager) {
                tip.pos = site;
                tip.heading = DOWN + 0.2 * gauss(&site_seed);
                tip.flags = (tip.flags & 0xFFFFFF00u) | 8u | 16u;  // axis, alive
                tip.age = 0.0;
                tip.since_branch = 0.0;
                tip.generation = tip.generation + 1.0;
                tip.rng = pcg(seed ^ site_seed);
            }
        }
        if (!is_alive(tip.flags)) {
            tip.rng = seed;
            tips_dst[slot] = tip;
            return;
        }
    } else if (!is_alive(tip.flags)) {
        tips_dst[slot] = tip;
        return;
    }

    // --- The scroll --------------------------------------------------------
    // This tick's source textures are in the pre-shift frame, so the tip
    // senses and moves there; the shift is applied on the way out. Carried
    // off the top, it is done.
    let order = order_of(tip.flags);
    var seed = tip.rng ^ pcg2(rp.tick, rp.seed);
    let fdims = vec2<f32>(f32(rp.dims_x), f32(rp.dims_y));

    // --- Steering ----------------------------------------------------------
    // Flank probes, exactly the Physarum trio -- but scored by the tropism
    // sum, and the structure term *negative*: the §15.1 sign flip.
    let ahead = vec2<f32>(cos(tip.heading), sin(tip.heading));
    let left = tip.heading + rp.sense_angle;
    let right = tip.heading - rp.sense_angle;
    let s_f = probe(tip.pos + ahead * rp.sense_dist);
    let s_l = probe(tip.pos + vec2<f32>(cos(left), sin(left)) * rp.sense_dist);
    let s_r = probe(tip.pos + vec2<f32>(cos(right), sin(right)) * rp.sense_dist);

    var turn = 0.0;
    if (s_f >= s_l && s_f >= s_r) {
        turn = 0.0;
    } else if (s_l > s_r) {
        turn = rp.tip_turn;
    } else if (s_r > s_l) {
        turn = -rp.tip_turn;
    } else {
        turn = rp.tip_turn * rnd_signed(&seed);
    }

    var heading = tip.heading + turn + rp.tip_jitter * rnd_signed(&seed);

    // What the tip is driving into, on its post-steering heading. Needed
    // before gravity, because gravity must know about it: a strong
    // gravitropism that re-points a blocked axis straight back into the
    // obstacle every tick is a deadlock -- the flank steering can never
    // accumulate enough angle to get around, and the tip stalls forever
    // (found by the foraging test: two of three axes pinned at their seed
    // rows). So the pull *releases* in proportion to the blockage -- a
    // stalled tip searches sideways, which is what a real root apex does at
    // hardpan -- and re-asserts as soon as the way is clear.
    let steered = vec2<f32>(cos(heading), sin(heading));
    let probe_ahead = tip.pos + steered * elong_for(order, tip.age);
    let blocked = soil_at(
        rp, fract(probe_ahead.x / fdims.x),
        probe_ahead.y - f32(rp.scroll_rows)).imped;
    var slow = 1.0 - smoothstep(0.30, 0.75, blocked);

    // Gravitropism: an angular relaxation toward the order's setpoint. Not a
    // steering vote like the probes -- gravity is not sensed at a distance,
    // it simply always pulls (except into a wall, above).
    let setpoint = DOWN + side_of(tip.flags) * gsa_for(order);
    heading = heading + gsa_gain_for(order) * (0.15 + 0.85 * slow)
        * wrap_angle(setpoint - heading);

    // --- Movement ----------------------------------------------------------
    // Stones are not walls but resistance: the tip slows into them (and the
    // probes are already steering it around), so deflection is emergent
    // rather than a bounce. The bottom margin is where growth waits for the
    // window (§15.4).
    let dir = vec2<f32>(cos(heading), sin(heading));
    let next = tip.pos + dir * elong_for(order, tip.age);
    let stone_ahead = soil_at(
        rp, fract(next.x / fdims.x), next.y - f32(rp.scroll_rows)).imped;
    // Floored, not zeroed: a tip that lands *inside* rock -- a seed hashed
    // onto a pebble, say -- would otherwise be trapped whatever it does,
    // since every direction is blocked and steering cannot help. The creep
    // is far below anything visible (twenty-odd seconds to force a pebble)
    // and it is the difference between a root delayed and a slot dead.
    slow = max(1.0 - smoothstep(0.30, 0.75, stone_ahead), 0.05);
    if (next.y > fdims.y - 2.5) {
        // The bottom margin is a true wait: the window will come.
        slow = 0.0;
    }
    let travelled = elong_for(order, tip.age) * slow;
    var pos = tip.pos + dir * travelled;
    pos.x = pos.x - floor(pos.x / fdims.x) * fdims.x;

    // --- Deposit -----------------------------------------------------------
    // A 3x3 gaussian stamp whose sigma is the order's width, *peak*
    // normalised rather than sum normalised: the stamp's centre lays down
    // `tip_deposit` per cell of travel whatever the width, so the density a
    // path settles at is one number for every order and the sigma carries
    // only the breadth. Sum-normalised (the first build) an axis spread the
    // same mass over more texels and rendered *fainter* than the hairline
    // fines -- the width hierarchy, upside down. Fixed-point atomics,
    // order-independent and so deterministic.
    let amount = rp.tip_deposit * travelled;
    if (amount > 0.0) {
        let sigma = max(splat_sigma_for(order), 0.2);
        let base = vec2<i32>(floor(pos - 1.0));
        let order_weight = f32(order) * 0.5;
        for (var j = 0; j < 3; j = j + 1) {
            for (var i = 0; i < 3; i = i + 1) {
                let t = vec2<i32>(
                    (((base.x + i) % i32(rp.dims_x)) + i32(rp.dims_x))
                        % i32(rp.dims_x),
                    base.y + j,
                );
                if (t.y < 0 || t.y >= i32(rp.dims_y)) {
                    continue;
                }
                let centre = vec2<f32>(base) + vec2<f32>(f32(i), f32(j)) + 0.5;
                let d = centre - pos;
                let share = amount * exp(-dot(d, d) / (2.0 * sigma * sigma));
                let index = 2u * (u32(t.y) * rp.dims_x + u32(t.x));
                let quantum = u32(max(share * DEPOSIT_SCALE, 0.0));
                if (quantum > 0u) {
                    atomicAdd(&deposit_buf[index], quantum);
                    atomicAdd(&deposit_buf[index + 1u],
                              u32(share * order_weight * DEPOSIT_SCALE));
                }
            }
        }
    }

    // --- Branching ---------------------------------------------------------
    // Distance-gated, then stochastic: real inter-branch spacing with real
    // irregularity. The parent only moves its own counter; the child builds
    // itself next tick. The counter is unbounded and the block recycles by
    // generation, so a long-lived axis keeps throwing laterals forever with
    // never more than its block's worth alive at once.
    var flags = tip.flags;
    var since_branch = tip.since_branch + travelled;
    let spacing = select(rp.spacing_lateral, rp.spacing_axis, order == 0u);
    // Foraging (§15.3): local richness modulates the branching -- the
    // documented proliferation response, and what turns a buried cache into
    // a burst of fuzz when a root finds it. Depleted ground grows sparse.
    let richness = clamp(textureSampleLevel(
        structure_tex, samp,
        vec2<f32>(fract(pos.x / fdims.x), clamp(pos.y / fdims.y, 0.0, 1.0)),
        0.0).w, 0.0, 1.5);
    let forage = clamp(rp.forage_gain * 0.5, 0.0, 1.0);
    let eagerness = clamp(
        rp.branch_prob * mix(1.0, richness, forage), 0.0, 1.0);
    if (order < 2u && since_branch > spacing && rnd(&seed) < eagerness) {
        flags = with_children(flags, children_of(flags) + 1u);
        since_branch = 0.0;
    }

    // --- Ageing and the ends of things --------------------------------------
    // Every order's growth is determinate: fines are ephemeral, laterals
    // stop in minutes, and even an axis spends its life in a quarter hour --
    // then rests as a seed and returns as the next plant (§15.4's
    // succession). The other death is the window's top edge.
    var age = tip.age + 1.0;
    var alive = true;
    if (order == 2u && age > rp.fine_life) {
        alive = false;
    }
    if (order == 1u && age > rp.lateral_life) {
        alive = false;
    }
    if (order == 0u && age > rp.axis_life) {
        alive = false;
        age = 0.0;  // the rest starts now; rebirth waits out regerm_delay
    }
    // The shift out, and the top of the window.
    pos.y = pos.y - f32(rp.scroll_rows);
    if (pos.y < 1.0) {
        alive = false;
        if (order == 0u) {
            age = 0.0;
        }
    }
    if (!alive) {
        flags = flags & ~8u;
    } else {
        // The front controller's two numbers: the deepest living tip, and
        // how many are living at all. Fixed point for the atomicMax.
        atomicMax(&front_buf[0], u32(max(pos.y, 0.0) * FRONT_SCALE));
        atomicAdd(&front_buf[1], 1u);
    }

    tip.pos = pos;
    tip.heading = wrap_angle(heading);
    tip.rng = seed;
    tip.flags = flags;
    tip.age = age;
    tip.since_branch = since_branch;
    tips_dst[slot] = tip;
}
