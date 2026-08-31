// Shared helpers for the rhizotron backend -- DESIGN.md §15.
//
// The soil is not a stored field: it is a pure function of (seed, world
// position), evaluated wherever a pass needs it. The world's vertical
// coordinate is an unbounded u64 row counter -- §3's pattern exactly: an
// integer, used only as hash input, so the soil arriving from below can never
// repeat and never quantises, however long the descent runs. WGSL has no
// 64-bit integers, so the counter travels as a (hi, lo) pair and the lattice
// arithmetic below is shifts, masks and adds-with-carry -- all exact.
//
// Horizontally the domain is a cylinder: x wraps, rows do not. Every lattice
// here tiles the x axis at a whole number of cells for the same reason the
// fungal noise tiles the torus (common.wgsl): a forcing that does not close
// on the domain leaves a seam.

//!include common.wgsl

// ---------------------------------------------------------------------------
// u64 arithmetic on (hi, lo) pairs.
// ---------------------------------------------------------------------------

// origin + n for a small signed offset n. Two's complement: the u32 add
// supplies the low word and the unsigned carry; a negative n also owes the
// high word a borrow, which is the sign extension.
fn offset64(hi: u32, lo: u32, n: i32) -> vec2<u32> {
    let u = u32(n);
    let l = lo + u;
    var h = hi + select(0u, 1u, l < lo);
    h = h + select(0u, 0xFFFFFFFFu, n < 0);
    return vec2<u32>(h, l);
}

fn shr64(v: vec2<u32>, k: u32) -> vec2<u32> {
    if (k == 0u) {
        return v;
    }
    if (k >= 32u) {
        return vec2<u32>(0u, v.x >> (k - 32u));
    }
    return vec2<u32>(v.x >> k, (v.y >> k) | (v.x << (32u - k)));
}

// ---------------------------------------------------------------------------
// Value noise on a cylinder-tiling lattice with a u64 vertical axis.
//
// `ux` is the wrapped horizontal coordinate in [0, 1); `cells_x` the lattice
// cells around it. The vertical cell is 2^row_shift world rows: found by
// shifting the u64 row, so a lattice cell index far beyond f32 precision is
// still exact, and the fractional position inside the cell is small enough
// that f32 is exact there too.
// ---------------------------------------------------------------------------

fn hash_soil(x: u32, cell: vec2<u32>, s: u32) -> f32 {
    let h = pcg3(
        x * 374761393u,
        cell.y * 668265263u ^ pcg(cell.x * 2654435761u),
        s,
    );
    return f32(h) * U32_TO_UNIT * 2.0 - 1.0;
}

fn soil_noise(
    ux: f32,
    cells_x: u32,
    origin: vec2<u32>,
    rel_row: f32,
    row_shift: u32,
    s: u32,
) -> f32 {
    let n = max(i32(cells_x), 1);
    let gx = fract(ux) * f32(n);
    let fx = gx - floor(gx);
    let x0 = u32((i32(floor(gx)) + n) % n);
    let x1 = u32((i32(x0) + 1) % n);

    let whole = i32(floor(rel_row));
    let fr = rel_row - floor(rel_row);
    let row = offset64(origin.x, origin.y, whole);
    let cell0 = shr64(row, row_shift);
    let cell1 = offset64(cell0.x, cell0.y, 1);
    let mask = (1u << row_shift) - 1u;
    let fy = (f32(row.y & mask) + fr) / f32(1u << row_shift);

    // Quintic fade, as everywhere else: C2, so nothing derived from this has
    // a crease.
    let wx = fx * fx * fx * (fx * (fx * 6.0 - 15.0) + 10.0);
    let wy = fy * fy * fy * (fy * (fy * 6.0 - 15.0) + 10.0);

    // The stream folds in the lattice sizes so lattices of different scales
    // walking overlapping indices stay independent (see common.wgsl on why).
    let stream = s ^ (cells_x * 0x27d4eb2fu) ^ (row_shift * 0x9e3779b9u);
    let a = hash_soil(x0, cell0, stream);
    let b = hash_soil(x1, cell0, stream);
    let c = hash_soil(x0, cell1, stream);
    let d = hash_soil(x1, cell1, stream);
    return mix(mix(a, b, wx), mix(c, d, wx), wy);
}

// ---------------------------------------------------------------------------
// The soil itself.
// ---------------------------------------------------------------------------

// At most one pebble per lattice cell, drawn by hash; a texel takes the
// strongest of the nine candidates around it.
fn pebbles(
    rp: RhizParams,
    ux: f32,
    rel_row: f32,
    cells_x: u32,
    shift: u32,
    gate: f32,
    s: u32,
) -> f32 {
    let origin = vec2<u32>(rp.origin_hi, rp.origin_lo);
    let n = max(i32(cells_x), 2);
    let gx = fract(ux) * f32(n);
    let fx = gx - floor(gx);
    let ix = i32(floor(gx));

    let whole = i32(floor(rel_row));
    let fr = rel_row - floor(rel_row);
    let row = offset64(origin.x, origin.y, whole);
    let cell = shr64(row, shift);
    let mask = (1u << shift) - 1u;
    let fy = (f32(row.y & mask) + fr) / f32(1u << shift);

    var strongest = 0.0;
    for (var dj = -1; dj <= 1; dj = dj + 1) {
        let cell_j = offset64(cell.x, cell.y, dj);
        for (var di = -1; di <= 1; di = di + 1) {
            let cx = u32((ix + di + n) % n);
            var h = pcg3(
                cx * 374761393u,
                cell_j.y * 668265263u ^ pcg(cell_j.x * 2654435761u),
                s,
            );
            let px = f32(h) * U32_TO_UNIT * 0.7 + 0.15;
            h = pcg(h);
            let py = f32(h) * U32_TO_UNIT * 0.7 + 0.15;
            h = pcg(h);
            // A wide size spread on purpose: a field of same-sized discs is
            // the §4.7 geometry, and it does not stop being that geometry by
            // being made of stone.
            let draw = f32(h) * U32_TO_UNIT;
            let radius = 0.14 + 0.34 * draw * draw;
            h = pcg(h);
            // More of the hashed pebbles exist where the stratum is stonier.
            let presence = 1.0 - smoothstep(
                0.55, 0.90, f32(h) * U32_TO_UNIT / max(gate, 1e-4));
            let d = length(vec2<f32>(
                fx - f32(di) - px,
                fy - f32(dj) - py,
            ));
            strongest = max(
                strongest,
                smoothstep(radius, radius * 0.62, d) * presence,
            );
        }
    }
    return strongest;
}

// The stone content of a texel: a common lattice of small pebbles and a much
// rarer one, four times the scale, of large stones -- two populations, so the
// sizes span an order of magnitude instead of clustering at one.
fn stones_at(rp: RhizParams, ux: f32, rel_row: f32, stoniness: f32) -> f32 {
    let gate = smoothstep(0.10, 0.32, stoniness);
    if (gate <= 0.0) {
        return 0.0;
    }
    let small = pebbles(
        rp, ux, rel_row, rp.stone_cells_x, rp.stone_shift,
        gate, rp.seed ^ 0x510Eu);
    let large = pebbles(
        rp, ux, rel_row, max(rp.stone_cells_x / 4u, 2u), rp.stone_shift + 2u,
        gate * 0.35, rp.seed ^ 0x51F0u);
    return max(small, large);
}

struct Soil {
    // Stratum character in [-1, 1]: which kind of band this row sits in.
    strat: f32,
    // Stone presence in [0, 1] -- the visual material.
    stone: f32,
    // Fine mineral grain in [-1, 1].
    grain: f32,
    // Hydraulic conductivity in (0, 1]: how readily water passes.
    cond: f32,
    // Dry-soil ramp position in [0, 1]: where between the family's dark and
    // pale chips this texel sits.
    light: f32,
    // What the tips push against, in [0, 1]: the stones plus most of the
    // hardpan -- which is felt as resistance but must not be *painted* as
    // stone, so it is its own number.
    imped: f32,
};

//!struct RhizParams

fn soil_at(rp: RhizParams, ux: f32, rel_row: f32) -> Soil {
    let origin = vec2<u32>(rp.origin_hi, rp.origin_lo);

    // The surface (§17.4). `depth` is rows below the soil line; the topsoil
    // weight eases in over the first tenth of the view -- the humus horizon:
    // organic-dark, stone-free, well-draining. With no surface configured
    // the sentinel row puts every texel at full depth and nothing here fires.
    let depth = rel_row - rp.surface_row;
    let topsoil = 1.0 - smoothstep(0.0, 0.10 * f32(rp.view_rows), depth);
    // Air, for the tips: everything above the line is a wall (§17.4 -- the
    // bottom margin's mirror), eased over a root-width so nothing steps.
    let air = 1.0 - smoothstep(-1.5, 0.5, depth);

    // Strata undulate: the row every stratum-scale lattice is evaluated at is
    // tilted by a smooth periodic function of x, so the bands dip and rise
    // instead of ruling straight lines. The tilt is a fraction of the stratum
    // thickness, so thin strata wobble finely and thick ones broadly.
    let tilt = value_noise_octaves_tiled(vec2<f32>(ux, 0.37), 3, rp.seed ^ 0x51A7u);
    let srow = rel_row + tilt * rp.strata_tilt * f32(1u << rp.strata_shift);

    // Two independent stratum-scale channels: what the band looks like, and
    // how it drains. Sandy-bright bands that pass water and dense dark ones
    // that hold it are different axes of real soil, so they get different
    // streams rather than one number wearing two hats.
    let strat = soil_noise(ux, 2u, origin, srow, rp.strata_shift, rp.seed ^ 0x0501u)
        * 0.72
        + soil_noise(ux, 5u, origin, srow, rp.strata_shift - 1u, rp.seed ^ 0x0502u)
        * 0.28;
    let perm = soil_noise(ux, 3u, origin, srow, rp.strata_shift, rp.seed ^ 0x0503u);

    // Stones: a bombing pattern rather than sliced noise. Each lattice cell
    // hashes at most one pebble -- a centre, a radius and a presence draw --
    // and a texel takes the strongest of the nine pebbles whose cells touch
    // it. Thresholded value noise was tried first and produces angular,
    // axis-ridged blobs (the lattice shows through the slice); a disc with a
    // soft rim is round by construction, which is what a pebble is. The rim
    // is a smoothstep over a third of the radius -- a spatial shape on static
    // material, not a temporal threshold -- and the stoniness of the stratum
    // gates how many of the hashed pebbles actually exist, so a fine-earth
    // band is genuinely stone-free.
    let stoniness = clamp(0.5 + 0.5 * perm, 0.0, 1.0) * rp.stone_amount
        * (1.0 - 0.9 * topsoil);
    let stone = stones_at(rp, ux, rel_row, stoniness);

    // Grain: one octave of fine speckle, static in world space so it scrolls
    // with the material rather than shimmering.
    let grain = soil_noise(ux, rp.grain_cells_x, origin, rel_row,
                           rp.grain_shift, rp.seed ^ 0x0506u);

    // Hardpan (§15.5, as a hashed feature rather than an event): rare
    // near-horizontal bands of compacted clay, flatter than the strata they
    // cut through, that pass almost no water. The percolation does the rest
    // unprompted -- water pools above them and races through their gaps --
    // and the thigmotropic slowdown reads them too, so roots run along them
    // the way excavated roots actually do.
    // The gate sits where the *interpolated* noise actually reaches:
    // bilinear value noise contracts toward zero between lattice corners, so
    // a gate up at 0.6 fires almost nowhere and the bands existed mostly in
    // theory (measured: no retention change at all on a test column).
    let hp = soil_noise(ux, 2u, origin, rel_row + tilt * 0.35
                        * f32(1u << rp.strata_shift),
                        rp.strata_shift + 1u, rp.seed ^ 0x0AD0u);
    let hardpan = smoothstep(0.38, 0.58, hp) * rp.hardpan_amount
        * (1.0 - topsoil);

    // Biopores: rare near-vertical old worm channels, tall thin lattices
    // wobbled slightly in x, that water races down and roots follow -- the
    // hydrotropism finds the wet channel on its own.
    let wobble = value_noise_octaves_tiled(
        vec2<f32>(ux * 0.5 + 0.11, 0.73), 5, rp.seed ^ 0x0B10u) * 0.004;
    let bp = soil_noise(ux + wobble, max(rp.stone_cells_x, 4u), origin,
                        rel_row, rp.stone_shift + 5u, rp.seed ^ 0x0B11u);
    let pore = smoothstep(0.55, 0.75, bp) * rp.biopore_amount;

    // Conductivity: permeable bands pass water, stones nearly block it,
    // hardpan more so, pores open it right up, and the floor keeps any of it
    // from damming the column forever.
    var cond = clamp(0.5 + 0.42 * perm, 0.0, 1.0) * (1.0 - 0.92 * stone);
    cond = cond * (1.0 - 0.88 * hardpan);
    cond = min(cond * (1.0 + 5.0 * pore), 1.0);
    // Topsoil takes rain readily -- organic crumb structure -- which is what
    // lets a shower reach the crowns instead of sheeting off the first row.
    cond = min(cond * (1.0 + 0.8 * topsoil), 1.0);
    cond = max(cond, rp.cond_floor);

    // Dry lightness: the stratum carries it, the grain only textures it --
    // grain much above a tenth of the travel reads as broadcast noise rather
    // than as material (measured by eye on the first build, and the kind of
    // judgement §15.13 says this section will keep needing). Hardpan reads
    // as its own material: a shade darker and denser than the band it cuts.
    // The humus horizon is the darkest earth in the column: topsoil pulls
    // the ramp position down hard, and the grain stays so the darkness still
    // reads as material rather than as a painted band.
    let light = clamp(
        (0.5 + 0.50 * strat + 0.10 * grain - 0.16 * hardpan)
            * (1.0 - 0.62 * topsoil) + 0.06 * topsoil * grain,
        0.0, 1.0);

    // The impedance the tips feel is the stone plus most of the hardpan --
    // roots do force hardpan eventually, which is why it is not a wall --
    // and the air, which is: nothing grows above the soil line.
    let impedance = min(stone + 0.7 * hardpan + air, 1.0);

    return Soil(strat, stone, grain, cond, light, impedance);
}

// The nutrient richness fresh soil arrives with (§15.11 step 5): a broad
// stratum-scale variation around the baseline, plus rare buried caches --
// something died here, and the roots that find it will say so. Pure function
// of (seed, world row) like everything else the generator makes; what
// happens to it afterwards -- uptake, recycling, the whisper of diffusion --
// is the structure pass's, because from arrival on it is *state*.
fn nutrient_seed(rp: RhizParams, ux: f32, rel_row: f32) -> f32 {
    let origin = vec2<u32>(rp.origin_hi, rp.origin_lo);
    let broad = soil_noise(ux, 3u, origin, rel_row,
                           rp.strata_shift, rp.seed ^ 0x0507u);
    let cache = pebbles(
        rp, ux, rel_row, max(rp.stone_cells_x / 3u, 2u), rp.stone_shift + 2u,
        clamp(rp.nutrient_cache, 0.0, 1.0) * 0.45, rp.seed ^ 0x0C4Eu);
    return clamp(
        rp.nutrient_baseline * (1.0 + 0.4 * broad)
            + rp.nutrient_cache * 1.6 * cache,
        0.002, 2.0);
}

// The lateral profile of an event, on the wrapping x axis.
fn event_lateral(ux: f32, ex: f32, radius: f32) -> f32 {
    var dx = abs(ux - ex);
    dx = min(dx, 1.0 - dx);
    let r = max(radius, 1e-4);
    return exp(-0.5 * (dx / r) * (dx / r));
}
