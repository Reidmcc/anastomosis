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
// Smooth spatial noise, used only as the *increment* to a stored field.
// White in time, smooth in space; the field's own integration supplies temporal
// smoothness.
//
// The noise *tiles*: its lattice wraps at a whole number of cells across the
// domain, so the field it produces is exactly periodic. That is not a nicety.
// Every field this drives is toroidal, and a forcing that does not close on
// the domain puts a permanent discontinuity along the wrap seam of whatever
// integrates it -- see psi.wgsl, where the untiled version left a line of
// steep gradient at u=0 and v=0 that curl turned into a jet.
// ---------------------------------------------------------------------------

fn hash_grid(ix: i32, iy: i32, s: u32) -> f32 {
    let h = pcg3(u32(ix) * 374761393u, u32(iy) * 668265263u, s);
    return f32(h) * U32_TO_UNIT * 2.0 - 1.0;
}

// `period` is the lattice size the index wraps at, so sampling p over any
// interval of length `period` gives a field with f(p + period) == f(p),
// interpolation across the join included: the two lattice rows either side of
// it are the same row. The identity is exact in the maths and holds to the
// rounding of the sample coordinate in floats, which is ~1e-7 here against
// texel-to-texel steps of ~1e-2 in the field itself.
fn value_noise_tiled(p: vec2<f32>, period: vec2<i32>, s: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    // Quintic fade: C2 continuous, so the derivative (which becomes velocity
    // via curl) has no visible creases.
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let lo = wrap_texel(vec2<i32>(i), period);
    let hi = wrap_texel(vec2<i32>(i) + vec2<i32>(1, 1), period);
    let a = hash_grid(lo.x, lo.y, s);
    let b = hash_grid(hi.x, lo.y, s);
    let c = hash_grid(lo.x, hi.y, s);
    let d = hash_grid(hi.x, hi.y, s);
    return mix(mix(a, b, w.x), mix(c, d, w.x), w.y);
}

// Three octaves over `uv` in [0, 1], at `period` cells across the domain.
//
// The frequency ratios are integers, which is what tiling requires: a 2.03x
// octave has no period on the domain at all. What those non-integer ratios
// were avoiding -- every octave's lattice landing on the same grid lines, which
// leaves a visible square signature -- is dealt with by offsetting each octave
// instead. A constant offset cannot affect the period, since f(p + c) repeats
// exactly where f does.
fn value_noise_octaves_tiled(uv: vec2<f32>, period: i32, s: u32) -> f32 {
    let n = max(period, 1);
    // The period is folded into the stream because `hash_grid` does not take
    // it: two calls at neighbouring periods walk overlapping lattice indices
    // and would otherwise draw the same lattice *values*, leaving the two
    // correlated. psi.wgsl crossfades neighbouring periods and normalises the
    // weights in quadrature, which is only right if they are independent --
    // measured, sharing the stream left the forcing 15% strong halfway between
    // two periods.
    let stream = s ^ (u32(n) * 0x27d4eb2fu);
    return value_noise_tiled(
               uv * f32(n) + vec2<f32>(0.37, 0.11),
               vec2<i32>(n), stream) * 0.62
         + value_noise_tiled(
               uv * f32(2 * n) + vec2<f32>(4.19, 7.53),
               vec2<i32>(2 * n), stream ^ 0x9e3779b9u) * 0.26
         + value_noise_tiled(
               uv * f32(4 * n) + vec2<f32>(2.71, 5.09),
               vec2<i32>(4 * n), stream ^ 0x85ebca6bu) * 0.12;
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

// The tolerance here is deliberately tiny rather than the ~1e-3 that would be
// visually harmless. The safety stage stores its output and reads it back as
// the next frame's history, so any component allowed through out of range gets
// clamped on a *later* frame -- enlarging that frame's step after the limiter
// has already bounded it. Near black, a 5e-4 absolute change in one channel
// moves Oklab L by ~1.6e-3, which is a sixth of the entire per-frame budget.
fn in_gamut(c: vec3<f32>) -> bool {
    return c.r >= -1e-6 && c.g >= -1e-6 && c.b >= -1e-6
        && c.r <= 1.0 + 1e-6 && c.g <= 1.0 + 1e-6 && c.b <= 1.0 + 1e-6;
}

// Gamut-map by reducing chroma at constant L and hue. Clipping RGB directly
// would change perceived lightness, which is exactly the kind of uncommanded
// brightness change the safety stage exists to prevent.
//
// Chroma zero is always in gamut for any L in [0, 1] (it is a neutral grey), so
// the bisection is guaranteed to find a solution and the final clamp is a
// no-op rather than a correction that could move L.
fn gamut_map_oklab(lab: vec3<f32>) -> vec3<f32> {
    let direct = oklab_to_linear_srgb(lab);
    if (in_gamut(direct)) {
        return clamp(direct, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    var lo = 0.0;
    var hi = 1.0;
    // 10 steps: chroma resolution ~1e-4, so the residual clamp is negligible.
    for (var i = 0u; i < 10u; i = i + 1u) {
        let mid = (lo + hi) * 0.5;
        let trial = vec3<f32>(lab.x, lab.y * mid, lab.z * mid);
        if (in_gamut(oklab_to_linear_srgb(trial))) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let mapped = oklab_to_linear_srgb(vec3<f32>(lab.x, lab.y * lo, lab.z * lo));
    return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

// WCAG relative luminance, which is the quantity the photosensitivity
// thresholds are actually defined in -- and which Oklab `L` is not. `L` is
// roughly the cube root of this, so `dY/dL = 3L^2` and a step that is bounded
// in `L` is bounded in `Y` by a factor that grows with lightness: 1.22 at the
// default `l_max` of 0.62, 2.55 at the ceiling of 0.9. Chroma is worse, since
// `L` is not luminance at all for a chromatic colour: `|dY/da|` reaches 0.315
// inside the default gamut. DESIGN.md 14.3 has the measurements. The safety
// stage bounds this quantity directly rather than inferring it.
//
// Linear-light input, so this is a plain dot product and, crucially, *linear*
// in the colour -- which is what lets the limiter hit its budget exactly with
// one interpolation instead of a search.
fn relative_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
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
