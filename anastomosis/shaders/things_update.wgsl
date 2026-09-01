// The Things' lives, one tick -- DESIGN.md §18.3 pass 1.
//
// One invocation per population slot, double-buffered: every invocation
// reads any slot's *previous* state and writes only its own next state, so
// nothing here depends on GPU scheduling -- the property the bit-identical
// resume test holds every mechanism to. Because nothing dies (§18.1 soul
// 5), a slot is an identity for the life of the world: alive slots age and
// wander, empty slots decide for themselves whether this is the tick they
// are born.
//
// Randomness is the house counter PRNG keyed on (slot, tick, seed) -- no
// stored state, no wall-clock anywhere (soul 10). The founding file's
// behaviour is conserved quirk by quirk; where a line below looks naive,
// check docs/founding/small_strange_thing.html before "fixing" it.

//!include things_common.wgsl

@group(0) @binding(0) var<storage, read> params: ThingsParams;
@group(0) @binding(1) var<storage, read> things_in: array<Thing>;
@group(0) @binding(2) var<storage, read_write> things_out: array<Thing>;

fn click_pos(index: u32) -> vec2<f32> {
    switch index {
        case 0u: { return vec2<f32>(params.click0_x, params.click0_y); }
        case 1u: { return vec2<f32>(params.click1_x, params.click1_y); }
        case 2u: { return vec2<f32>(params.click2_x, params.click2_y); }
        default: { return vec2<f32>(params.click3_x, params.click3_y); }
    }
}

// A birth: the five traits rolled in the founding constructor's order
// (size, speed, hue, curiosity, shyness), fixed for life from here on.
fn born_at(pos: vec2<f32>, seed: ptr<function, u32>) -> Thing {
    var t: Thing;
    t.x = clamp(finite_or(pos.x, 0.0), 0.0, f32(params.dims_x));
    t.y = clamp(finite_or(pos.y, 0.0), 0.0, f32(params.dims_y));
    t.size = rnd(seed) * 3.0 + 1.0;
    t.speed = rnd(seed) * 0.5 + 0.1;
    t.hue = rnd(seed) * 360.0;
    t.curiosity = rnd(seed);
    t.shyness = rnd(seed);
    t.age = 0u;
    t.flags = THING_ALIVE;
    t.friend0 = NO_FRIEND;
    t.friend1 = NO_FRIEND;
    t.friend2 = NO_FRIEND;
    t.friend_count = 0u;
    t.spare = 0u;
    return t;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.capacity) {
        return;
    }
    var t = things_in[i];
    var seed = pcg3(i, params.tick, params.seed);

    if (thing_alive(t)) {
        t.age = t.age + 1u;

        // Wander: the founding Brownian step, variance-matched per second
        // (step_scale packs the sqrt(60*dt) -- see gpu_params.py).
        t.x = t.x + rnd_signed(&seed) * 0.5 * t.speed * params.step_scale;
        t.y = t.y + rnd_signed(&seed) * 0.5 * t.speed * params.step_scale;

        // "Stay in bounds, sort of" -- the founding wrap, verbatim: an edge
        // crossing teleports to the far edge rather than taking a modulo.
        // Part of the artifact; the deposit pass wraps texels regardless.
        if (t.x < 0.0) { t.x = f32(params.dims_x); }
        if (t.x > f32(params.dims_x)) { t.x = 0.0; }
        if (t.y < 0.0) { t.y = f32(params.dims_y); }
        if (t.y > f32(params.dims_y)) { t.y = 0.0; }
        t.x = clamp(finite_or(t.x, 0.0), 0.0, f32(params.dims_x));
        t.y = clamp(finite_or(t.y, 0.0), 0.0, f32(params.dims_y));

        // Look for friends (soul 2): only if curiosity outweighs shyness,
        // at the founding rate, scanning every Thing in index order and
        // appending while under the cap -- including, as in the founding
        // file, a neighbour already befriended on an earlier scan. Bonds
        // are never removed, by omission: no code path below unsets one.
        if (t.curiosity > t.shyness && rnd(&seed) < params.friend_prob) {
            let r2 = params.friend_radius * params.friend_radius;
            for (var j = 0u; j < params.capacity; j = j + 1u) {
                if (j == i || t.friend_count >= MAX_FRIENDS) {
                    continue;
                }
                let other = things_in[j];
                if (!thing_alive(other)) {
                    continue;
                }
                let d = thing_pos(other) - thing_pos(t);
                if (dot(d, d) < r2) {
                    switch t.friend_count {
                        case 0u: { t.friend0 = j; }
                        case 1u: { t.friend1 = j; }
                        default: { t.friend2 = j; }
                    }
                    t.friend_count = t.friend_count + 1u;
                }
            }
        }
        things_out[i] = t;
        return;
    }

    // An empty slot. Its rank among empty slots (previous state, so every
    // invocation agrees) decides whether it claims a pending click; the
    // rest run the spawn lottery. Either way it writes only itself.
    var rank = 0u;
    for (var j = 0u; j < i; j = j + 1u) {
        if (!thing_alive(things_in[j])) {
            rank = rank + 1u;
        }
    }

    // Click-to-add (soul 9): click k spawns per_click Things, claimed by
    // the first empty ranks, scattered as the founding handler scattered.
    let click_births = params.click_count * params.per_click;
    if (rank < click_births) {
        let at = click_pos(rank / max(params.per_click, 1u));
        let off = vec2<f32>(rnd_signed(&seed), rnd_signed(&seed))
            * params.click_scatter;
        things_out[i] = born_at(at + off, &seed);
        return;
    }

    // The spawn lottery (soul 4): pick one candidate parent
    // deterministically; if it is alive and mature and the roll passes,
    // a child is born nearby. Expected births per tick at a young village
    // match the founding `mature * 0.005` per frame (see ThingsParams);
    // near the cap the rate eases as empty slots grow scarce -- the cap
    // arriving as a softness rather than a wall.
    let cand = pcg3(i, params.tick, params.seed ^ 0x51A17u) % params.capacity;
    let parent = things_in[cand];
    if (thing_alive(parent)
        && f32(parent.age) > params.mature_ticks
        && rnd(&seed) < params.spawn_prob) {
        let off = vec2<f32>(rnd_signed(&seed), rnd_signed(&seed))
            * params.spawn_radius;
        things_out[i] = born_at(thing_pos(parent) + off, &seed);
        return;
    }

    things_out[i] = t;
}
