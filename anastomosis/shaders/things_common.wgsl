// Shared definitions for the Small Strange Things port -- DESIGN.md §18.
//
// The Thing record mirrors THING_DTYPE in things.py word for word. Every
// field is a 4-byte scalar, so std430 packing is "one word per field, in
// order" -- the same discipline as the parameter blocks.

//!include common.wgsl

//!struct ThingsParams

struct Thing {
    x: f32,
    y: f32,
    // The five birth traits (§18.1 soul 1): rolled once, fixed for life.
    hue: f32,        // degrees, the founding file's HSL wheel
    size: f32,
    speed: f32,
    curiosity: f32,
    shyness: f32,
    // Age in ticks, in a u32: the §17 f16-age stall lesson, applied from
    // birth. At 30 Hz a u32 lasts four and a half years of continuous
    // running before anything would need thinking about.
    age: u32,
    flags: u32,
    // Friendship (soul 2): at most three, never removed, one-directional,
    // duplicates possible -- all exactly as the founding file has it.
    friend0: u32,
    friend1: u32,
    friend2: u32,
    friend_count: u32,
    spare: u32,
};

const THING_ALIVE: u32 = 1u;
const MAX_FRIENDS: u32 = 3u;   // a law, not a knob (§18.1 soul 2)
const NO_FRIEND: u32 = 0xffffffffu;

// Fixed-point scale for the deposit accumulator. Deposits are small
// fractions; 1024 keeps quantisation under a couple of percent of the
// smallest sustained deposit while leaving five decades of headroom in a
// u32 for overlap.
const DEPOSIT_SCALE: f32 = 1024.0;

fn thing_alive(t: Thing) -> bool {
    return (t.flags & THING_ALIVE) != 0u;
}

fn thing_pos(t: Thing) -> vec2<f32> {
    return vec2<f32>(t.x, t.y);
}

fn thing_friend(t: Thing, slot: u32) -> u32 {
    switch slot {
        case 0u: { return t.friend0; }
        case 1u: { return t.friend1; }
        default: { return t.friend2; }
    }
}

// The founding file's colours are HSL; a trait hue is a position on that
// wheel and stays one (§18.2). Standard HSL -> sRGB, then sRGB -> linear,
// because the canvas field accumulates linear light.
fn hsl_to_linear(h_deg: f32, s: f32, l: f32) -> vec3<f32> {
    let h = fract(h_deg / 360.0) * 6.0;
    let c = (1.0 - abs(2.0 * l - 1.0)) * s;
    let x = c * (1.0 - abs(fract(h * 0.5) * 2.0 - 1.0));
    var rgb = vec3<f32>(0.0);
    if (h < 1.0)      { rgb = vec3<f32>(c, x, 0.0); }
    else if (h < 2.0) { rgb = vec3<f32>(x, c, 0.0); }
    else if (h < 3.0) { rgb = vec3<f32>(0.0, c, x); }
    else if (h < 4.0) { rgb = vec3<f32>(0.0, x, c); }
    else if (h < 5.0) { rgb = vec3<f32>(x, 0.0, c); }
    else              { rgb = vec3<f32>(c, 0.0, x); }
    let srgb = rgb + vec3<f32>(l - c * 0.5);
    // sRGB EOTF (the exact inverse of common.wgsl's linear_to_srgb).
    let lo = srgb / 12.92;
    let hi = pow(max((srgb + 0.055) / 1.055, vec3<f32>(0.0)), vec3<f32>(2.4));
    return select(hi, lo, srgb <= vec3<f32>(0.04045));
}
