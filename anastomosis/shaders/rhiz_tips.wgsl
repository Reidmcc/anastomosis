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
    // (born at least once -- a dead tip must not be reborn); bits 8-15 the
    // parent's count of children handed out.
    flags: u32,
    age: f32,
    since_branch: f32,
    vigor: f32,
};

@group(0) @binding(0) var<storage, read> rp: RhizParams;
@group(0) @binding(1) var<storage, read> tips_src: array<Tip>;
@group(0) @binding(2) var<storage, read_write> tips_dst: array<Tip>;
@group(0) @binding(3) var moisture_tex: texture_2d<f32>;
@group(0) @binding(4) var structure_tex: texture_2d<f32>;
@group(0) @binding(5) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(6) var samp: sampler;

const DEPOSIT_SCALE: f32 = 1048576.0;
const DOWN: f32 = 1.5707963267948966;  // +pi/2: rows grow downward

fn order_of(flags: u32) -> u32 { return flags & 3u; }
fn side_of(flags: u32) -> f32 { return select(1.0, -1.0, (flags & 4u) != 0u); }
fn is_alive(flags: u32) -> bool { return (flags & 8u) != 0u; }
fn is_spent(flags: u32) -> bool { return (flags & 16u) != 0u; }
fn children_of(flags: u32) -> u32 { return (flags >> 8u) & 255u; }
fn with_children(flags: u32, n: u32) -> u32 {
    return (flags & 0xFFFF00FFu) | ((n & 255u) << 8u);
}

fn wrap_angle(a: f32) -> f32 {
    return a - TAU * floor((a + PI) / TAU);
}

fn elong_for(order: u32) -> f32 {
    switch order {
        case 0u: { return rp.elong_axis; }
        case 1u: { return rp.elong_lateral; }
        default: { return rp.elong_fine; }
    }
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
    let structure = min(
        textureSampleLevel(structure_tex, samp, uv, 0.0).x, 1.0);
    let soil = soil_at(rp, uv.x, pos.y - f32(rp.scroll_rows));
    return rp.hydro_gain * wet
        - rp.thigmo_gain * soil.stone
        - rp.avoid_gain * structure;
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

    // --- Birth -------------------------------------------------------------
    // A slot that has never lived checks whether its parent has handed its
    // index out. The parent's counter is read from the *previous* tick's
    // state, so the decision this depends on is already made and immutable.
    if (!is_alive(tip.flags) && !is_spent(tip.flags) && slot >= rp.max_axes) {
        let parent = parent_of(slot);
        let parent_tip = tips_src[parent.x];
        if (is_spent(parent_tip.flags) && children_of(parent_tip.flags) > parent.y) {
            var seed = pcg3(slot, rp.seed, 0xB1127u);
            let side = select(4u, 0u, rnd(&seed) < 0.5);
            let order = order_of(parent_tip.flags) + 1u;
            tip.pos = parent_tip.pos;
            tip.heading = parent_tip.heading
                + side_of(side) * (rp.branch_angle + rp.branch_jitter * gauss(&seed));
            tip.rng = pcg(seed);
            tip.flags = order | side | 8u | 16u;  // alive, spent
            tip.age = 0.0;
            tip.since_branch = 0.0;
            tip.vigor = 1.0;
        } else {
            tips_dst[slot] = tip;
            return;
        }
    } else if (!is_alive(tip.flags)) {
        // Dead or waiting: carried across unchanged (the scroll no longer
        // concerns it; nothing reads a dead tip's position).
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

    // Gravitropism: an angular relaxation toward the order's setpoint. Not a
    // steering vote like the probes -- gravity is not sensed at a distance,
    // it simply always pulls.
    let setpoint = DOWN + side_of(tip.flags) * gsa_for(order);
    heading = heading + gsa_gain_for(order) * wrap_angle(setpoint - heading);

    // --- Movement ----------------------------------------------------------
    // Stones are not walls but resistance: the tip slows into them (and the
    // probes are already steering it around), so deflection is emergent
    // rather than a bounce. The bottom margin is where growth waits for the
    // window (§15.4; the front controller arrives with step 4).
    let dir = vec2<f32>(cos(heading), sin(heading));
    let next = tip.pos + dir * elong_for(order);
    let stone_ahead = soil_at(
        rp, fract(next.x / fdims.x), next.y - f32(rp.scroll_rows)).stone;
    var slow = 1.0 - smoothstep(0.30, 0.75, stone_ahead);
    if (next.y > fdims.y - 2.5) {
        slow = 0.0;
    }
    let travelled = elong_for(order) * slow;
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
    // itself next tick. A spent block is determinate growth, not an error.
    var flags = tip.flags;
    var since_branch = tip.since_branch + travelled;
    let capacity = select(
        rp.fines_per_lateral, rp.laterals_per_axis, order == 0u);
    let spacing = select(rp.spacing_lateral, rp.spacing_axis, order == 0u);
    if (order < 2u && children_of(flags) < capacity
        && since_branch > spacing && rnd(&seed) < rp.branch_prob) {
        flags = with_children(flags, children_of(flags) + 1u);
        since_branch = 0.0;
    }

    // --- Ageing and the ends of things --------------------------------------
    var age = tip.age + 1.0;
    var alive = true;
    if (order == 2u && age > rp.fine_life) {
        alive = false;
    }
    if (order == 1u && age > rp.lateral_life) {
        alive = false;
    }
    // The shift out, and the top of the window.
    pos.y = pos.y - f32(rp.scroll_rows);
    if (pos.y < 1.0) {
        alive = false;
    }
    if (!alive) {
        flags = flags & ~8u;
    }

    tip.pos = pos;
    tip.heading = wrap_angle(heading);
    tip.rng = seed;
    tip.flags = flags;
    tip.age = age;
    tip.since_branch = since_branch;
    tips_dst[slot] = tip;
}
