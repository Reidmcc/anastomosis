// Drawing the Things into the accumulator -- DESIGN.md §18.3 pass 2.
//
// Two entry points, both writing integer fixed-point atomics: addition is
// order-independent, so however the GPU schedules overlapping bodies and
// crossing bonds, the accumulated tick is bit-identical -- the same
// discipline as the rhizotron's deposit buffer.
//
// The accumulator is drained into the canvas field by things_canvas.wgsl,
// and the canvas *is* the image: the founding file drew everything onto
// one fading canvas, so bodies, bonds and sparkles here are simply the
// freshest stratum of their own ghost (§18.1 soul 7).
//
// What the engine buys them is light (§18.2): each body is a bright core
// inside a wider dim glow skirt, bonds have real width, and everything is
// deposited as linear light for the HDR chain -- but every colour is the
// founding HSL, verbatim: bodies hsl(hue, 60%, 50%) at 0.7 alpha through
// the birth fade-in, bonds hsl(mean hue, 50%, 40%) at 0.2, sparkles
// hsl(hue + 60, 80%, 80%) at 0.8.

//!include things_common.wgsl

@group(0) @binding(0) var<storage, read> params: ThingsParams;
@group(0) @binding(1) var<storage, read> things: array<Thing>;
@group(0) @binding(2) var<storage, read_write> deposit: array<atomic<u32>>;

fn stamp(texel: vec2<i32>, rgb: vec3<f32>) {
    let dims = vec2<i32>(i32(params.dims_x), i32(params.dims_y));
    let p = wrap_texel(texel, dims);
    let idx = (u32(p.y) * params.dims_x + u32(p.x)) * 4u;
    let fixed = vec3<u32>(max(rgb, vec3<f32>(0.0)) * DEPOSIT_SCALE + 0.5);
    if (fixed.r > 0u) { atomicAdd(&deposit[idx + 0u], fixed.r); }
    if (fixed.g > 0u) { atomicAdd(&deposit[idx + 1u], fixed.g); }
    if (fixed.b > 0u) { atomicAdd(&deposit[idx + 2u], fixed.b); }
}

// --- Bodies and sparkles ---------------------------------------------------
// One invocation per Thing: the pulsing disc with its glow skirt, then the
// occasional sparkle. A few hundred texels per Thing per tick; the
// population is village-sized, so the whole pass is small.

@compute @workgroup_size(64, 1, 1)
fn bodies(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.capacity) {
        return;
    }
    let t = things[i];
    if (!thing_alive(t)) {
        return;
    }
    var seed = pcg3(i, params.tick, params.seed ^ 0xB0D1E5u);
    let pos = thing_pos(t);

    // The birth fade-in and the breath, both the founding file's:
    // alpha = min(1, age / 50 frames) * 0.7,
    // radius = size + sin(time * 0.05 + x * 0.01) * 0.5 -- the phase
    // accumulated host-side so a resumed world breathes on, mid-breath.
    let alpha = min(1.0, f32(t.age) / max(params.fadein_ticks, 1.0)) * 0.7;
    // The trait size is a world unit; pulse_amp arrives pre-scaled and
    // pulse_x pre-divided, so the breath rides the world too (§18's
    // same-beings-at-every-resolution law).
    let radius = max(
        t.size * params.world_scale
            + sin(params.pulse_phase + t.x * params.pulse_x)
                * params.pulse_amp,
        0.5);
    let colour = hsl_to_linear(t.hue, 0.6, 0.5) * alpha * params.body_emit;

    let r_glow = radius * max(params.glow_mult, 1.0);
    let reach = i32(ceil(r_glow)) + 1;
    for (var dy = -reach; dy <= reach; dy = dy + 1) {
        for (var dx = -reach; dx <= reach; dx = dx + 1) {
            let texel = vec2<i32>(floor(pos)) + vec2<i32>(dx, dy);
            let centre = vec2<f32>(texel) + 0.5;
            let d = distance(centre, pos);
            // A crisp half-texel edge: the founding disc is confident,
            // opaque candy, not a soft light (the round-2 register
            // ruling -- moodiness belongs to the field, never to them).
            let core = 1.0 - smoothstep(radius - 0.5, radius + 0.5, d);
            let skirt = 1.0 - smoothstep(0.0, r_glow, d);
            let w = core + skirt * skirt * params.glow_gain;
            if (w > 1e-3) {
                stamp(texel, colour * w);
            }
        }
    }

    // Little sparkle sometimes (§18.1 soul 6): a one-tick deposit at the
    // founding offset and hue shift; the slew limiter rounds the attack,
    // the canvas fade carries it away.
    if (rnd(&seed) < params.sparkle_prob) {
        let off = vec2<f32>(rnd_signed(&seed), rnd_signed(&seed))
            * params.sparkle_offset;
        let at = pos + off;
        let sparkle = hsl_to_linear(t.hue + 60.0, 0.8, 0.8)
            * 0.8 * params.sparkle_amp;
        // Small at every resolution means small relative to the world.
        let sr = max(1.5 * params.world_scale, 1.0);
        let reach = i32(ceil(sr)) + 1;
        for (var dy = -reach; dy <= reach; dy = dy + 1) {
            for (var dx = -reach; dx <= reach; dx = dx + 1) {
                let texel = vec2<i32>(floor(at)) + vec2<i32>(dx, dy);
                let centre = vec2<f32>(texel) + 0.5;
                let w = 1.0 - smoothstep(0.0, sr, distance(centre, at));
                if (w > 1e-3) {
                    stamp(texel, sparkle * w);
                }
            }
        }
    }
}

// --- Bonds -----------------------------------------------------------------
// One workgroup per (Thing, friend slot). The segment is walked at
// sub-texel steps shared across the workgroup's threads, each stamp
// weighted by the step length so the line's deposit per texel is
// independent of how finely it happened to be sampled.
//
// The line is drawn straight between the two positions -- never the
// shortcut through the wrap seam -- because that is what the founding file
// did, and the long taut lines across the world are friendships that
// survived emigration (§18.1 souls 2 and 8). Bond colour is the plain
// arithmetic mean of the two hues (soul 3), quirk conserved.

@compute @workgroup_size(64, 1, 1)
fn bonds(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let bond = wgid.x;
    let i = bond / MAX_FRIENDS;
    let slot = bond % MAX_FRIENDS;
    if (i >= params.capacity) {
        return;
    }
    let t = things[i];
    if (!thing_alive(t) || slot >= t.friend_count) {
        return;
    }
    let j = thing_friend(t, slot);
    if (j >= params.capacity) {
        return;
    }
    let f = things[j];

    let a = thing_pos(t);
    let b = thing_pos(f);
    let len = distance(a, b);

    // Span differentiation (the round-2 ruling): intra-village bonds are
    // background hum -- near the formation scale they approach the
    // founding hairline -- while the long emigrant lines keep their
    // earned width and full presence. The ramp runs from the friendship
    // radius out to three of them: past that, a bond has left home.
    let span = smoothstep(
        params.friend_radius, params.friend_radius * 3.0, len);
    let colour = hsl_to_linear((t.hue + f.hue) * 0.5, 0.5, 0.4)
        * 0.2 * params.bond_emit
        * mix(params.bond_near_gain, 1.0, span);

    let steps = max(u32(len / 0.7), 1u);
    let step_len = len / f32(steps);
    let width = max(
        mix(params.bond_near_width, params.bond_width, span), 0.3);
    for (var k = lid.x; k < steps; k = k + 64u) {
        let p = mix(a, b, (f32(k) + 0.5) / f32(steps));
        let reach = i32(ceil(width)) + 1;
        for (var dy = -reach; dy <= reach; dy = dy + 1) {
            for (var dx = -reach; dx <= reach; dx = dx + 1) {
                let texel = vec2<i32>(floor(p)) + vec2<i32>(dx, dy);
                let centre = vec2<f32>(texel) + 0.5;
                let w = (1.0 - smoothstep(0.0, width + 0.8,
                                          distance(centre, p))) * step_len;
                if (w > 1e-3) {
                    stamp(texel, colour * w);
                }
            }
        }
    }
}
