// The rhizotron's image: soil and roots, seen through the pane -- §15.2.
//
// Every pixel is material. Where the fungal compositors shade luminous
// density against a void, this pass paints the matrix itself: stratum colour
// from Munsell soil-chart ramps, stones, fine static grain, and moisture
// darkening the whole of it -- wet soil is darker and slightly more
// saturated, which is the most familiar material appearance there is. Over
// the matrix, the roots: coverage from the structure field through a
// steep-but-C1 transfer whose crispness is a measured licence (§15.7.2),
// coloured pale ivory at the living front and browning toward the soil with
// age -- the eye led to the growing edge by the same gradient the biology
// provides.
//
// The descent arrives here as `base_row`: where the top of the window sits in
// the texture this frame, in fractional rows, interpolated between the last
// two ticks. The simulation scrolls in exact integer rows (rhiz_moisture);
// the presentation glides over it at the sub-row remainder, so the sinking is
// continuous at any frame rate. The vertical axis never wraps -- the margins
// above and below the view are real generated soil, which is what makes the
// sub-row sampling at the extremes honest rather than a wraparound artefact.
//
// Every mapping is a smooth transfer, as everywhere else: no thresholds, and
// nothing anywhere is a function of the clock.

//!include rhiz_common.wgsl

//!struct RenderParams

@group(0) @binding(0) var<storage, read> rp: RhizParams;
@group(0) @binding(1) var<storage, read> render: RenderParams;
@group(0) @binding(2) var moisture_tex: texture_2d<f32>;
@group(0) @binding(3) var structure_tex: texture_2d<f32>;
@group(0) @binding(4) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var record_tex: texture_2d<f32>;
// The strata atlas (§17.6): the current one, and the one before the last
// ceremony, blended toward the current while the reveal climbs.
@group(0) @binding(7) var strata_tex: texture_2d<f32>;
@group(0) @binding(8) var strata_prev: texture_2d<f32>;

// One buried layer's (silhouette, halo) pair under a view position, read
// bilinearly through the atlas: two layers to a texel, tiles stacked, each
// tile spanning the whole column at half resolution (rhiz_strata.wgsl).
fn stratum(atlas: texture_2d<f32>, layer: u32, ux: f32, row: f32) -> vec2<f32> {
    let span = f32(rp.strata_h) * 2.0;
    let v = (f32(layer / 2u) + clamp(row / span, 0.0, 1.0))
        / f32(max(rp.strata_tiles, 1u));
    let s = textureSampleLevel(atlas, samp, vec2<f32>(ux, v), 0.0);
    if ((layer & 1u) == 0u) {
        return max(vec2<f32>(finite_or(s.x, 0.0), finite_or(s.y, 0.0)),
                   vec2<f32>(0.0));
    }
    return max(vec2<f32>(finite_or(s.z, 0.0), finite_or(s.w, 0.0)),
               vec2<f32>(0.0));
}

// A stroke one atlas texel wide keeps half its peak per softening step
// (rhiz_strata.wgsl's tent); this gives most of that back, so a softened
// stroke reads as the same mark out of focus more than as a fainter one
// -- most, not all, because a spread mesh that kept its whole weight is a
// dark wash over the entire ground.
const SPREAD_GAIN: f32 = 1.6;

// The ladder (§17.6): how much darker than the soil, and how far toward
// stone, the buried seasons under this texel make it. Each generation is
// one fixed rung fainter and one rung cooler than the one over it, the
// nearest at full strength, and the layers compose by *maximum* -- the
// nearest one never hides, and overlapping skeletons stack into a form
// rather than piling up into black. Bedrock, the uncounted wash, sits
// under all of them at its own fixed strength.
fn strata_shade(atlas: texture_2d<f32>, ux: f32, row: f32) -> vec2<f32> {
    var dark = 0.0;
    var cool = 0.0;
    let count = rp.strata_count;
    for (var g = 0u; g < count; g = g + 1u) {
        let s = stratum(atlas, g, ux, row);
        let rung = pow(rp.strata_step, f32(g));
        // The silhouette is stored as coverage (rhiz_strata.wgsl) and has
        // been softened once per generation; the spread gain gives a
        // spread stroke its weight back, so a generation's recession in
        // focus and its recession in salience (the rung) are two ladders
        // and not one compounded collapse. The halo goes through a gentle
        // transfer: a fan is shading, not a stroke.
        let spread = pow(SPREAD_GAIN, f32(g));
        let crisp_cov = clamp(s.x * spread, 0.0, 1.0);
        let soft_cov = smoothstep(0.08, 0.70, s.y);
        // The stronger of the two marks by magnitude; the sign is the
        // ladder's (dark or pale, see below).
        let dc = rp.strata_crisp * crisp_cov;
        let ds = rp.strata_soft * soft_cov;
        let d = rung * select(ds, dc, abs(dc) >= abs(ds));
        // Stone is the skeleton's colour; the halo stays earth, darker.
        // (Cooling the halo turned the grain inside it into blue speckle
        // on the first buried trial.)
        let c = min(rp.strata_cool
                    + (1.0 - rp.strata_cool) * f32(g) / f32(max(count, 1u)),
                    1.0) * crisp_cov;
        if (abs(d) > abs(dark)) {
            dark = d;
            cool = c;
        }
    }
    let bed_cov = smoothstep(0.02, 0.80, stratum(atlas, count, ux, row).y);
    let d_bed = rp.strata_bedrock * bed_cov;
    if (abs(d_bed) > abs(dark)) {
        dark = d_bed;
        cool = 0.0;
    }
    return vec2<f32>(dark, cool);
}

// The soil families, as ramps between two Munsell soil-chart chips in Oklab.
// Generated by tests/soil_palette.py (illuminant C -> D65, Bradford) -- the
// gamut is inherited from the referent rather than tuned by hand (§15.2).
// The chart chips are far brighter than this application's ground, so the
// shading keeps each chip's *chromatic identity* and re-anchors its lightness
// into the dark-earth envelope below.
const FAMILY_COUNT: u32 = 6u;

fn family_dark(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return vec3<f32>(0.3132, 0.0073, 0.0139); } // humus 10YR 2/1
        case 1u: { return vec3<f32>(0.4014, 0.0123, 0.0304); } // loam 10YR 3/2
        case 2u: { return vec3<f32>(0.4920, 0.0204, 0.0617); } // ochre 10YR 4/4
        case 3u: { return vec3<f32>(0.4057, 0.0474, 0.0410); } // laterite 2.5YR 3/4
        case 4u: { return vec3<f32>(0.4065, 0.0548, 0.0343); } // red clay 10R 3/4
        default: { return vec3<f32>(0.4891, 0.0005, 0.0187); } // podzol 5Y 4/1
    }
}

fn family_light(i: u32) -> vec3<f32> {
    switch i {
        case 0u: { return vec3<f32>(0.4906, 0.0120, 0.0334); } // 10YR 4/2
        case 1u: { return vec3<f32>(0.5796, 0.0158, 0.0471); } // 10YR 5/3
        case 2u: { return vec3<f32>(0.6653, 0.0071, 0.0672); } // 2.5Y 6/4
        case 3u: { return vec3<f32>(0.4972, 0.0635, 0.0618); } // 2.5YR 4/6
        case 4u: { return vec3<f32>(0.4982, 0.0732, 0.0521); } // 10R 4/6
        default: { return vec3<f32>(0.6641, -0.0005, 0.0194); } // 5Y 6/1
    }
}

// Stone: leached ash-grey, 5Y 4/1 -> 5Y 6/1.
const STONE_DARK: vec3<f32> = vec3<f32>(0.4891, 0.0005, 0.0187);
const STONE_LIGHT: vec3<f32> = vec3<f32>(0.6641, -0.0005, 0.0194);

// The Munsell chips above span roughly this much Oklab L; a chip's position
// in that span is what survives the re-anchoring into the dark ground.
const CHIP_L_LO: f32 = 0.30;
const CHIP_L_SPAN: f32 = 0.37;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= render.out_w || gid.y >= render.out_h) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5)
        / vec2<f32>(f32(render.out_w), f32(render.out_h));

    // Lateral aspect correction samples more (or less) of the wrapping x
    // axis; the vertical axis maps the view rows exactly, margins excluded.
    let ux = fract((uv.x - 0.5) * rp.x_span + 0.5);
    let row = rp.base_row + uv.y * f32(rp.view_rows);
    let suv = vec2<f32>(ux, row / f32(rp.dims_y));

    // The wetness the eye sees is the EMA channel: the moisture field's own
    // dynamics, lowpassed once more on the way to luminance.
    let wet_raw = textureSampleLevel(moisture_tex, samp, suv, 0.0).y;
    let wet = 1.0 - exp(-1.6 * clamp(finite_or(wet_raw, 0.0), 0.0, 1.5));

    let soil = soil_at(rp, ux, row);

    // --- Which soil this is ------------------------------------------------
    // The palette macro walks the family ring: the anchor (plus the same slow
    // autonomous drift the fungal hue has) picks a position among the six
    // families, and neighbouring families blend. The stratum then moves the
    // texel along its family's dark-to-pale ramp.
    let fpos = fract(render.hue_global / TAU) * f32(FAMILY_COUNT);
    let f0 = u32(floor(fpos)) % FAMILY_COUNT;
    let f1 = (f0 + 1u) % FAMILY_COUNT;
    let ft = fract(fpos);
    let dark = mix(family_dark(f0), family_dark(f1), ft);
    let light = mix(family_light(f0), family_light(f1), ft);
    var chip = mix(dark, light, soil.light);

    // Stones sit in the matrix as their own material -- but half-buried in
    // it, not resting on top: the chip is pulled a third of the way toward
    // the surrounding soil and its chroma dulled, because full-strength
    // olive-grey discs against warm earth read as a field of pale dots --
    // and a field of dots is §4.7's geometry whatever it is made of.
    var stone_chip = mix(STONE_DARK, STONE_LIGHT, 0.5 + 0.5 * soil.grain);
    stone_chip = vec3<f32>(stone_chip.x, stone_chip.yz * 0.8);
    stone_chip = mix(stone_chip, chip, 0.62);
    chip = mix(chip, stone_chip, clamp(soil.stone, 0.0, 1.0));

    // --- Into the visible-earth envelope (§17.5) ----------------------------
    // The chip's position in the chart's lightness span becomes its position
    // in [background + floor, background + floor + range]; its chromatic
    // direction comes with it, eased down as the lightness drops so a dim
    // texel is a dimmer version of the same soil rather than a neon one.
    // The floor is §17.1's fix: every soil texel clears the void's black, so
    // the ground is the mid of this image and the sky above it is the dark.
    let t_chip = clamp((chip.x - CHIP_L_LO) / CHIP_L_SPAN, 0.0, 1.0);
    var l = render.background_luma + rp.soil_l_floor + rp.soil_l_range * t_chip;
    var ab = chip.yz * pow(max(l, 1e-4) / max(chip.x, 1e-4), 0.7);

    // Moisture: darker and slightly richer. This is the weather made visible,
    // and the slew limiter downstream is what ultimately bounds its pace.
    // Floored: with the governor attenuation-only (§17.5), nothing re-lifts
    // a broadly wet pane any more, so the wetting bands may darken the
    // ground toward -- but never through -- the visible-earth floor.
    l = max(
        l * (1.0 - rp.wet_darken * wet),
        render.background_luma + rp.soil_l_floor * 0.55);
    ab = ab * (1.0 + rp.wet_chroma * wet);

    // Fine grain, static in world space: texture, not noise -- and quieter
    // inside stones, which are smoother material than the earth around them.
    l = l + soil.grain * rp.grain_amount * 0.009 * (1.0 - 0.8 * soil.stone);

    // The strata (§17.6, the fossil rethink): the buried seasons
    // themselves, standing in the ground under the living layer -- each
    // one the silhouette its fossil had, with the halo its fans left, a
    // fixed rung darker than the soil and cooler than wood, the way stone
    // is. Read before the roots so the living and the wood shade *over*
    // their ancestors: the plant grows through the strata. The ladder's
    // rungs are what make the seasons countable (the count test), and
    // the deepest of them dip toward, never through, the visible-earth
    // floor. Ghosts are grey, not brown: the chroma is pulled toward a
    // cool stone so no buried trunk competes with a living one for the
    // wood's red-brown. During the few seconds after a ceremony the
    // previous atlas is blended toward the new one -- the data changed on
    // one tick, the appearance never steps (§17.10(5)).
    var shade = strata_shade(strata_tex, ux, row);
    if (rp.strata_reveal < 1.0) {
        let t = clamp(rp.strata_reveal, 0.0, 1.0);
        shade = mix(strata_shade(strata_prev, ux, row), shade,
                    t * t * (3.0 - 2.0 * t));
    }
    // Signed: a positive ladder sinks the strata into the ground as dark
    // stone, a negative one lifts them as pale bone. Either way they stay
    // inside the earth's own envelope -- never through the visible-earth
    // floor, never up into the living material's rungs.
    l = clamp(
        l - shade.x,
        render.background_luma + rp.soil_l_floor * 0.35,
        render.background_luma + rp.soil_l_floor + rp.soil_l_range + 0.06);
    // Stone is the earth's own chroma mostly spent, not a colour of its
    // own: at the ground's lightness a cool tint reads as blue speckle in
    // the grain (the first buried trials), while plain desaturation reads
    // as grey against the wood's red-brown, which is the separation
    // wanted.
    ab = mix(ab, ab * 0.3, clamp(shade.y, 0.0, 1.0));

    // --- The roots: the record beneath, the living over it (§17.5-6) ------
    // Two figures, composited in depth order -- soil, then wood, then the
    // living sheath -- because the fourth viewing convicted every blend
    // that mixed them along a synthetic axis. `wood_frac`, the
    // lignin:living ratio, is not a clock: it leaps when senescence empties
    // the denominator (colour skipped straight to dark) and stalls where
    // fine traffic keeps re-depositing (colour parked at the soil's own
    // value -- and, being soil-derived, *tracked the soil's hue* as the
    // palette drifted). The only clock in the field is `bio_age`, and it
    // alone drives the wood's colour; the pale-to-dark transition is
    // occlusion -- the living sheath thinning over wood already coloured
    // by its age -- and the wood's hue is anchored mostly absolute, a
    // red-brown no soil family shares, so no stage of a root's life can
    // inherit the ground's colour and vanish into it. The two materials
    // share ONE silhouette, carried by their combined mass, so commitment
    // moves nothing at the outline: a root never thins on its way into
    // the record (the width report's finding, below).
    let root = textureSampleLevel(structure_tex, samp, suv, 0.0);
    let rec = textureSampleLevel(record_tex, samp, suv, 0.0);
    let density = max(finite_or(root.x, 0.0), 0.0);
    let lignin = max(finite_or(rec.x, 0.0), 0.0);
    let age = max(finite_or(root.y, 0.0), 0.0);
    let young_f = exp(-age / max(rp.root_age_scale * 0.4, 1.0));
    let wood_age = max(finite_or(rec.y, 0.0), 0.0);
    let mature = 1.0 - exp(-wood_age / max(rp.wood_age_scale, 1.0));

    // The silhouette: ONE figure, carried by the combined mass, living
    // plus committed. The width report convicted the split transfers that
    // stood here: living coverage read density alone and wood read lignin
    // alone, so commitment -- a *transfer* between the two channels --
    // narrowed the first silhouette minutes before the second could widen,
    // and every dying shaft pinched to a hairline, then re-widened texel
    // by texel as the lignin tails straggled over their own knee. The sum
    // is invariant under commitment, so the outline holds while the sheath
    // dies back; only genuine loss -- senescence taking fuzz that never
    // commits -- narrows anything. The knee eases toward the wood
    // transfer's as the texel becomes wood (committed mass reads at half
    // the living knee, so a part-committed lateral keeps the width its
    // life had), and eases further with maturity: secondary thickening,
    // radial growth rendered rather than re-deposited.
    let total = density + lignin;
    let woodiness = lignin / max(total, 1e-5);
    let knee_eff = max(rp.root_knee, 1e-4)
        * mix(1.0, 0.5 - 0.2 * mature, woodiness);
    let edge_eff = mix(rp.root_edge, rp.wood_edge, woodiness);
    let tot_sat = total / (total + knee_eff);
    let cov = smoothstep(0.5 - edge_eff, 0.5 + edge_eff, tot_sat);
    // The woody core claims the whole silhouette once wood is a real share
    // of the texel's mass -- an occlusion weight, never a colour axis (the
    // colour stays on bio_age; the ratio-is-not-a-clock ruling stands).
    // Under an opaque sheath the claim is invisible; it is what shows,
    // full-width, the moment the sheath thins -- so the reveal widens
    // nothing and the red arrives at the width the pale figure held.
    let wood_cov = cov * smoothstep(0.02, 0.25, woodiness);
    // Living coverage: absolute density through the living transfer -- the
    // sheath's own opacity over the wood, and the hairs' driver.
    let live_sat = density / (density + max(rp.root_knee, 1e-4));
    let live_cov = smoothstep(
        0.5 - rp.root_edge, 0.5 + rp.root_edge, live_sat);

    // A pallor boost over the shared macro: the visible-earth envelope
    // (§17.5) lifted the ground, and the living material keeps its two
    // rungs of headroom above the palest stratum.
    let l_young = clamp(
        render.background_luma + render.filament_luma * 1.55, 0.0, render.l_max);

    // Root hairs, and the mycorrhizal accent (§15.2): the living
    // transfer's soft skirt, re-admitted only around *young* material -- a
    // halo of pale fuzz at the growing front that fades as the root
    // lignifies. Where the material is young *and fine*, the same skirt
    // carries the faint cool shimmer: the hyphae, the one cool accent in a
    // warm field, spending chroma and no luminance beyond the hairs'.
    let skirt = smoothstep(0.04, 0.5, live_sat) * (1.0 - live_cov);
    let hair = skirt * young_f;
    l = mix(l, l_young, hair * rp.root_hair * 0.5);
    let fineness = clamp(finite_or(root.z, 0.0), 0.0, 1.0);
    let shimmer = hair * fineness * rp.mycorrhiza;
    ab = ab + vec2<f32>(-0.010, -0.024) * shimmer;

    // The record first: wood by biographical age alone, red-brown into
    // dark umber, floors keeping it above the sky whatever the ground.
    if (wood_cov > 1e-4) {
        let wood_new = vec3<f32>(
            max(render.background_luma + 0.07, l - 0.10),
            mix(ab, vec2<f32>(0.058, 0.046), 0.75));
        let umber = vec3<f32>(
            max(render.background_luma + 0.05, l - 0.18),
            mix(ab, vec2<f32>(0.020, 0.018), 0.75));
        let sweep = mature * mature * (3.0 - 2.0 * mature);
        let wood_lab = mix(wood_new, umber, sweep);
        l = mix(l, wood_lab.x, wood_cov);
        ab = mix(ab, wood_lab.yz, wood_cov);
    }

    // The living sheath over it: pale for as long as it lives, tanning
    // with recency age, never sinking toward the ground -- when it thins,
    // what shows through is the wood, already the colour of its years.
    if (live_cov > 1e-4) {
        let age_scale = max(rp.root_age_scale, 1.0)
            * (1.0 - 0.55 * fineness * fineness);
        let brown = (1.0 - exp(-age / age_scale)) * rp.root_brown;
        let young = vec3<f32>(
            l_young, 0.010 * l_young / 0.4, 0.045 * l_young / 0.4);
        let tan = vec3<f32>(
            l + 0.075, ab + vec2<f32>(0.008, 0.018));
        let living_lab = mix(young, tan, brown);
        l = mix(l, living_lab.x, live_cov);
        ab = mix(ab, living_lab.yz, live_cov);
    }

    // --- The sky (§17.4) ----------------------------------------------------
    // Above the soil line the pane is open air: the darkest thing in the
    // image, faintly cool so the earth below reads warm. The blend eases
    // over a couple of rows -- the litter line, where humus meets light --
    // and takes the root layer with it (hairs may skirt a row above the
    // line; nothing may glow in the sky).
    let depth = row - rp.surface_row;
    let airf = 1.0 - smoothstep(-0.5, 2.0, depth);
    // The litter line: a thin pale seam of surface debris where humus meets
    // air, textured by the grain so it reads as material, not as a rule.
    let seam = smoothstep(-0.3, 0.6, depth) * (1.0 - smoothstep(0.6, 3.0, depth));
    l = l + seam * (0.045 + 0.03 * soil.grain);
    let sky_l = render.background_luma * 0.85;
    l = mix(l, sky_l, airf);
    ab = mix(ab, vec2<f32>(-0.002, -0.008), airf);

    // The same perceptual bounds the fungal compositors respect; the safety
    // stage re-applies both to its target, so these are shaping, not the
    // guarantee.
    l = clamp(l, 0.0, render.l_max);
    let chroma = length(ab);
    if (chroma > 1e-6) {
        let bounded = clamp(chroma, render.chroma_floor, render.c_max);
        ab = ab * (bounded / chroma);
    }

    let rgb = oklab_to_linear_srgb(vec3<f32>(l, ab));
    textureStore(
        hdr_out, vec2<i32>(gid.xy), finite_or4(vec4<f32>(rgb, 1.0), 0.0));
}
