// Shared helpers: counter-based RNG, stateful noise, Oklab colour, sampling.
//
// Nothing in here takes wall-clock time as an input. Slow variation comes from
// integrating white increments into a stored field (an OU process), never from
// evaluating noise(x, y, t) -- see DESIGN.md §3.

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;
const U32_TO_UNIT: f32 = 2.3283064365386963e-10; // 1 / 2^32

// ---------------------------------------------------------------------------
// Counter-based PRNG (PCG). Stateless hash of (id, tick, stream), so any
// invocation can produce reproducible randomness without stored state, and the
// sequence has no practical period.
// ---------------------------------------------------------------------------

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn pcg2(a: u32, b: u32) -> u32 {
    return pcg(a ^ pcg(b));
}

fn pcg3(a: u32, b: u32, c: u32) -> u32 {
    return pcg(a ^ pcg(b ^ pcg(c)));
}

// Advances `seed` in place and returns a uniform value in [0, 1).
fn rnd(seed: ptr<function, u32>) -> f32 {
    *seed = pcg(*seed);
    return f32(*seed) * U32_TO_UNIT;
}

fn rnd_signed(seed: ptr<function, u32>) -> f32 {
    return rnd(seed) * 2.0 - 1.0;
}

// Box-Muller. Used for OU increments, so the tail behaviour matters more than
// speed -- a uniform would give the drift a subtly boxy character.
fn gauss(seed: ptr<function, u32>) -> f32 {
    let u1 = max(rnd(seed), 1e-7);
    let u2 = rnd(seed);
    return sqrt(-2.0 * log(u1)) * cos(TAU * u2);
}

// ---------------------------------------------------------------------------
// Smooth spatial noise, used only as the *increment* to a stored field.
// White in time, smooth in space; the field's own integration supplies temporal
// smoothness.
// ---------------------------------------------------------------------------

fn hash_grid(ix: i32, iy: i32, s: u32) -> f32 {
    let h = pcg3(u32(ix) * 374761393u, u32(iy) * 668265263u, s);
    return f32(h) * U32_TO_UNIT * 2.0 - 1.0;
}

fn value_noise(p: vec2<f32>, s: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    // Quintic fade: C2 continuous, so the derivative (which becomes velocity
    // via curl) has no visible creases.
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let ix = i32(i.x);
    let iy = i32(i.y);
    let a = hash_grid(ix, iy, s);
    let b = hash_grid(ix + 1, iy, s);
    let c = hash_grid(ix, iy + 1, s);
    let d = hash_grid(ix + 1, iy + 1, s);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

fn value_noise_octaves(p: vec2<f32>, s: u32) -> f32 {
    return value_noise(p, s) * 0.62
         + value_noise(p * 2.03, s ^ 0x9e3779b9u) * 0.26
         + value_noise(p * 4.11, s ^ 0x85ebca6bu) * 0.12;
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

// The simulation domain is a torus: wrapping avoids the boundary accumulation
// that reflecting edges produce over long runs.
fn wrap_uv(uv: vec2<f32>) -> vec2<f32> {
    return fract(fract(uv) + 1.0);
}

fn wrap_texel(p: vec2<i32>, dims: vec2<i32>) -> vec2<i32> {
    return ((p % dims) + dims) % dims;
}

// ---------------------------------------------------------------------------
// Oklab. All perceptual work happens here rather than in sRGB or HSV, because
// lightness must be independently boundable for the safety stage to mean
// anything (DESIGN.md §6-7).
// ---------------------------------------------------------------------------

fn cbrt_safe(x: f32) -> f32 {
    return sign(x) * pow(abs(x) + 1e-12, 0.3333333333);
}

fn linear_srgb_to_oklab(c: vec3<f32>) -> vec3<f32> {
    let l = 0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b;
    let m = 0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b;
    let s = 0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b;
    let l_ = cbrt_safe(l);
    let m_ = cbrt_safe(m);
    let s_ = cbrt_safe(s);
    return vec3<f32>(
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    );
}

fn oklab_to_linear_srgb(lab: vec3<f32>) -> vec3<f32> {
    let l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
    let m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
    let s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;
    return vec3<f32>(
         4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    );
}

fn oklch_to_oklab(lch: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(lch.x, lch.y * cos(lch.z), lch.y * sin(lch.z));
}

fn oklab_to_oklch(lab: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(lab.x, length(lab.yz), atan2(lab.z, lab.y));
}

fn in_gamut(c: vec3<f32>) -> bool {
    return c.r >= -0.0005 && c.g >= -0.0005 && c.b >= -0.0005
        && c.r <= 1.0005 && c.g <= 1.0005 && c.b <= 1.0005;
}

// Gamut-map by reducing chroma at constant L and hue. Clipping RGB directly
// would change perceived lightness, which is exactly the kind of uncommanded
// brightness change the safety stage exists to prevent.
fn gamut_map_oklab(lab: vec3<f32>) -> vec3<f32> {
    var rgb = oklab_to_linear_srgb(lab);
    if (in_gamut(rgb)) {
        return rgb;
    }
    var lo = 0.0;
    var hi = 1.0;
    // 8 bisection steps: chroma error below ~0.4%, imperceptible.
    for (var i = 0u; i < 8u; i = i + 1u) {
        let mid = (lo + hi) * 0.5;
        let trial = vec3<f32>(lab.x, lab.y * mid, lab.z * mid);
        if (in_gamut(oklab_to_linear_srgb(trial))) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    rgb = oklab_to_linear_srgb(vec3<f32>(lab.x, lab.y * lo, lab.z * lo));
    return clamp(rgb, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

// ---------------------------------------------------------------------------
// Numerical hygiene. A single non-finite value propagates through diffusion and
// destroys the whole field within seconds, and a multi-day session cannot
// recover from that on its own (DESIGN.md §4.4).
// ---------------------------------------------------------------------------

fn finite_or(x: f32, fallback: f32) -> f32 {
    // NaN fails every comparison, so this catches NaN and both infinities.
    if (x > -3.0e38 && x < 3.0e38) {
        return x;
    }
    return fallback;
}

fn finite_or2(v: vec2<f32>, fallback: f32) -> vec2<f32> {
    return vec2<f32>(finite_or(v.x, fallback), finite_or(v.y, fallback));
}

fn finite_or4(v: vec4<f32>, fallback: f32) -> vec4<f32> {
    return vec4<f32>(
        finite_or(v.x, fallback),
        finite_or(v.y, fallback),
        finite_or(v.z, fallback),
        finite_or(v.w, fallback),
    );
}

// Circular quantities (hue) are carried as a unit vector so that advection and
// blending interpolate along the shortest arc automatically.
fn hue_to_vec(h: f32) -> vec2<f32> {
    return vec2<f32>(cos(h), sin(h));
}

fn vec_to_hue(v: vec2<f32>) -> f32 {
    return atan2(v.y, v.x);
}

fn normalize_hue_vec(v: vec2<f32>) -> vec2<f32> {
    let len = length(v);
    if (len < 1e-5) {
        return vec2<f32>(1.0, 0.0);
    }
    return v / len;
}
