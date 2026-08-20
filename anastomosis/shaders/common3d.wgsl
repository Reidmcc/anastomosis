// Shared helpers for the volumetric slab -- DESIGN.md §5.1.
//
// The slab is a **3-torus**, not a box with faces. Every argument for a
// toroidal domain in two dimensions (common.wgsl, psi.wgsl) applies unchanged
// to the third: a reflecting or no-flux boundary accumulates material against
// itself over a run measured in days, and agents that bounce off a wall pile
// up along it. So x, y and z all wrap, exactly alike, and no pass anywhere in
// the volumetric backend has a special case for an edge -- because there
// isn't one.
//
// What that costs is one thing, and it is paid in the renderer rather than
// here: material leaving the near face reappears at the far one, so the march
// applies a smooth window in depth (raymarch.wgsl) and the arrival and
// departure are fades instead of steps.

//!include common.wgsl

fn wrap_uvw(p: vec3<f32>) -> vec3<f32> {
    return fract(fract(p) + 1.0);
}

// Lifted before the reduction for the same reason as wrap_texel; see there.
fn wrap_texel3(p: vec3<i32>, dims: vec3<i32>) -> vec3<i32> {
    let d = max(dims, vec3<i32>(1, 1, 1));
    let lifted = p + d * (abs(p) / d + vec3<i32>(1, 1, 1));
    return vec3<i32>(vec3<u32>(lifted) % vec3<u32>(d));
}

// ---------------------------------------------------------------------------
// Tiling 3D value noise, used only as the *increment* to a stored field.
//
// Same contract as the 2D version: white in time, smooth in space, and exactly
// periodic on the domain so that integrating it cannot build a ridge along a
// wrap seam. See common.wgsl for why that mattered enough to be rebuilt.
// ---------------------------------------------------------------------------

fn hash_grid3(ix: i32, iy: i32, iz: i32, s: u32) -> f32 {
    let h = pcg3(
        u32(ix) * 374761393u,
        u32(iy) * 668265263u,
        pcg2(u32(iz) * 2246822519u, s),
    );
    return f32(h) * U32_TO_UNIT * 2.0 - 1.0;
}

fn value_noise_tiled3(p: vec3<f32>, period: vec3<i32>, s: u32) -> f32 {
    let i = floor(p);
    let f = p - i;
    // Quintic fade: C2 continuous, so the curl of what this drives has no
    // visible creases.
    let w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let lo = wrap_texel3(vec3<i32>(i), period);
    let hi = wrap_texel3(vec3<i32>(i) + vec3<i32>(1, 1, 1), period);

    let c000 = hash_grid3(lo.x, lo.y, lo.z, s);
    let c100 = hash_grid3(hi.x, lo.y, lo.z, s);
    let c010 = hash_grid3(lo.x, hi.y, lo.z, s);
    let c110 = hash_grid3(hi.x, hi.y, lo.z, s);
    let c001 = hash_grid3(lo.x, lo.y, hi.z, s);
    let c101 = hash_grid3(hi.x, lo.y, hi.z, s);
    let c011 = hash_grid3(lo.x, hi.y, hi.z, s);
    let c111 = hash_grid3(hi.x, hi.y, hi.z, s);

    let x00 = mix(c000, c100, w.x);
    let x10 = mix(c010, c110, w.x);
    let x01 = mix(c001, c101, w.x);
    let x11 = mix(c011, c111, w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

// Three octaves, at `period` lattice cells across the domain on each axis.
//
// Integer frequency ratios, because that is what tiling requires; the square
// signature that non-integer ratios were avoiding is dealt with by offsetting
// each octave instead, which cannot affect the period.
fn value_noise_octaves_tiled3(uvw: vec3<f32>, period: vec3<i32>, s: u32) -> f32 {
    let n = max(period, vec3<i32>(1));
    // The period is folded into the stream for the same reason as in 2D: two
    // calls at neighbouring periods walk overlapping lattice indices and would
    // otherwise draw the same lattice values, leaving them correlated -- which
    // breaks the quadrature crossfade in vol_psi.wgsl.
    let stream = s ^ (u32(n.x) * 0x27d4eb2fu) ^ (u32(n.z) * 0x165667b1u);
    return value_noise_tiled3(
               vec3<f32>(n) * uvw + vec3<f32>(0.37, 0.11, 0.53),
               n, stream) * 0.62
         + value_noise_tiled3(
               vec3<f32>(2 * n) * uvw + vec3<f32>(4.19, 7.53, 1.87),
               2 * n, stream ^ 0x9e3779b9u) * 0.26
         + value_noise_tiled3(
               vec3<f32>(4 * n) * uvw + vec3<f32>(2.71, 5.09, 8.31),
               4 * n, stream ^ 0x85ebca6bu) * 0.12;
}

// The lattice period on each axis for `n` cells across the slab's *width*.
//
// Voxels are cubic, so a slab of w x h x d voxels is a box of world extent
// (1, h/w, d/w): asking for n cells across the width and n across the depth
// would make the noise cells long and thin, and the eddies with them.
//
// The floor of two on the depth axis is the one departure, and it is
// load-bearing. At the default 512 x 288 x 48 the honest answer is one cell
// through the thickness, and a lattice period of one is *constant* on that
// axis -- the two rows it interpolates between are the same row. A potential
// with no variation in z has a velocity with no variation in z, so every
// depth moves identically and the slab reads as one sheet seen through fog.
// Two is the smallest period that puts the front and back of the slab on
// opposite phases, which is exactly the differential motion depth is made of.
fn slab_periods(n: i32, dims: vec3<u32>) -> vec3<i32> {
    let f = vec3<f32>(dims);
    let base = max(f32(n), 1.0);
    return vec3<i32>(
        max(n, 1),
        max(i32(round(base * f.y / max(f.x, 1.0))), 1),
        max(i32(round(base * f.z / max(f.x, 1.0))), 2),
    );
}
