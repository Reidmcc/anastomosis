// The ceremony: a completed season becomes ground -- DESIGN.md §17.6, the
// fossil rethink.
//
// Runs on exactly one tick per season, the tick after the interment drive
// commits (rhizotron.py arms it on the same readback that offers the
// fossil), and rewrites the whole strata atlas in one dispatch:
//
//   layer 0      <- the fossil: the standing skeleton's silhouette (lignin
//                   plus the living remnant), box-sampled to the atlas, and
//                   its halo -- the silhouette's own blur plus the season's
//                   fan record, which is where the fines stood while they
//                   lived (the fossil itself cannot say: they are gone by
//                   the time it is taken).
//   layer g >= 1 <- layer g-1 as it stood, its halo one softening step
//                   softer: a buried season recedes by ceremony, one rung
//                   per burial, in salience (the composite's ladder) and in
//                   focus (here), never by rate.
//   last layer   <- bedrock: what it held, stepped down by the configured
//                   fraction, plus the oldest countable stratum's halo. The
//                   uncounted wash every season eventually joins.
//
// Two layers share a texel -- xy and zw -- and the tiles stack vertically,
// at half the column's resolution in each axis (STRATA_SCALE). Everything
// a stroke laid down is gathered by the box sample, only spread; nothing
// here consults a clock, and nothing here runs on any other tick.

//!include rhiz_common.wgsl

@group(0) @binding(0) var<storage, read> rp: RhizParams;
@group(0) @binding(1) var structure_tex: texture_2d<f32>;
@group(0) @binding(2) var record_tex: texture_2d<f32>;
@group(0) @binding(3) var season_tex: texture_2d<f32>;
@group(0) @binding(4) var strata_src: texture_2d<f32>;
@group(0) @binding(5) var strata_dst: texture_storage_2d<rgba16float, write>;

const SCALE: i32 = 2;
const HALO_CAP: f32 = 2.0;
// The halo is a mark left by *substantial* strokes: the silhouette goes
// through a transfer with a knee a few times the crisp channel's before it
// is spread, so the sub-hundredth wash of every lateral ever laid stays a
// wash and only trunk-class mass darkens the ground around itself. (The
// first painted trial spread the raw mass and turned six buried seasons
// into one black tangle: the whole pane was inside every halo.)
const HALO_KNEE_MULT: f32 = 3.0;
const SOFT_FROM_CRISP: f32 = 0.35;

// One column texel's silhouette mass: the record plus whatever living
// material still stands over it, the same sum the composite's silhouette
// is carried by, so the stratum is the exact figure the eye watched grow.
fn mass_at(x: i32, y: i32) -> f32 {
    let sx = (x + i32(rp.dims_x)) % i32(rp.dims_x);
    let sy = clamp(y, 0, i32(rp.dims_y) - 1);
    let p = vec2<i32>(sx, sy);
    return max(finite_or(textureLoad(record_tex, p, 0).x, 0.0), 0.0)
        + max(finite_or(textureLoad(structure_tex, p, 0).x, 0.0), 0.0);
}

fn fan_at(x: i32, y: i32) -> f32 {
    let sx = (x + i32(rp.dims_x)) % i32(rp.dims_x);
    let sy = clamp(y, 0, i32(rp.dims_y) - 1);
    return clamp(finite_or(
        textureLoad(season_tex, vec2<i32>(sx, sy), 0).x, 0.0), 0.0, 1.0);
}

// The silhouette as *coverage*: the column's mass through the same kind of
// saturate-then-smoothstep transfer the wood's own silhouette uses, at the
// strata's knee, box-sampled onto one atlas texel. Coverage rather than
// mass, so that the softening a generation takes per burial (below) is a
// spreading of the figure -- sum conserved, peak eased -- and not a
// collapse of a thin stroke through the knee.
fn cover_box(i: i32, j: i32, knee: f32) -> f32 {
    var sum = 0.0;
    for (var dy = 0; dy < SCALE; dy = dy + 1) {
        for (var dx = 0; dx < SCALE; dx = dx + 1) {
            let m = mass_at(i * SCALE + dx, j * SCALE + dy);
            // The wood's own window (rhiz_composite.wgsl, wood_edge): the
            // record is a sub-hundredth wash over most of the pane with
            // the skeleton a few hundredths above it, and the wash must
            // stay ground here exactly as it does on screen.
            sum = sum + smoothstep(0.40, 0.60, m / (m + knee));
        }
    }
    return sum / f32(SCALE * SCALE);
}

fn crisp_box(i: i32, j: i32) -> f32 {
    return cover_box(i, j, max(rp.strata_knee, 1e-4));
}

// The same box through the trunk-class transfer: what counts as a stroke
// substantial enough to leave a halo.
fn trunk_box(i: i32, j: i32) -> f32 {
    return cover_box(i, j, max(rp.strata_knee, 1e-4) * HALO_KNEE_MULT);
}

fn fan_box(i: i32, j: i32) -> f32 {
    var sum = 0.0;
    for (var dy = 0; dy < SCALE; dy = dy + 1) {
        for (var dx = 0; dx < SCALE; dx = dx + 1) {
            sum = sum + fan_at(i * SCALE + dx, j * SCALE + dy);
        }
    }
    return sum / f32(SCALE * SCALE);
}

// An old layer's (crisp, soft) pair at an atlas texel, x wrapping and the
// rows clamped inside the layer's own tile -- a blur tap must never read a
// neighbouring generation.
fn old_layer(layer: u32, i: i32, j: i32) -> vec2<f32> {
    let tile = i32(layer / 2u);
    let w = i32(rp.strata_w);
    let h = i32(rp.strata_h);
    let p = vec2<i32>((i + w) % w, tile * h + clamp(j, 0, h - 1));
    let v = textureLoad(strata_src, p, 0);
    if ((layer & 1u) == 0u) {
        return vec2<f32>(
            max(finite_or(v.x, 0.0), 0.0), max(finite_or(v.y, 0.0), 0.0));
    }
    return vec2<f32>(
        max(finite_or(v.z, 0.0), 0.0), max(finite_or(v.w, 0.0), 0.0));
}

// The one softening step a generation takes per burial: a tent blur,
// radius one atlas texel, of both the silhouette and the halo. A stroke
// keeps its integral and spreads -- the composite restores most of a
// spread stroke's weight (SPREAD_GAIN there), so what a generation loses
// is *focus* more than presence: the nearest ghost is in focus, the one
// before it soft-edged, the one before that a band, the one before that
// a shadow. Depth of field as time -- the cue the count test actually
// leans on, since several whole-pane meshes at several darknesses are one
// tangle to the eye. Gentle on purpose: a wider kernel (tried at radius
// two) turned every generation but the nearest into a uniform wash by the
// second step, and a wash is bedrock, not a countable rung.
fn old_layer_softened(layer: u32, i: i32, j: i32) -> vec2<f32> {
    var sum = vec2<f32>(0.0);
    var weight = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let wgt = select(1.0, 2.0, dx == 0) * select(1.0, 2.0, dy == 0);
            sum = sum + wgt * old_layer(layer, i + dx, j + dy);
            weight = weight + wgt;
        }
    }
    return sum / weight;
}

// The new stratum's halo: the silhouette spread over a couple of atlas
// texels either way (a stroke is a mark in the ground, not a wire), and
// the fan record with a lighter spread, on one scale.
fn new_soft(i: i32, j: i32) -> f32 {
    var crisp_sum = 0.0;
    var crisp_w = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let wgt = (3.0 - f32(abs(dx))) * (3.0 - f32(abs(dy)));
            crisp_sum = crisp_sum + wgt * trunk_box(i + dx, j + dy);
            crisp_w = crisp_w + wgt;
        }
    }
    var fan_sum = 0.0;
    var fan_w = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let wgt = select(1.0, 2.0, dx == 0) * select(1.0, 2.0, dy == 0);
            fan_sum = fan_sum + wgt * fan_box(i + dx, j + dy);
            fan_w = fan_w + wgt;
        }
    }
    return min(
        SOFT_FROM_CRISP * crisp_sum / crisp_w + fan_sum / fan_w, HALO_CAP);
}

// The (crisp, soft) pair layer `layer` holds after this ceremony.
fn new_layer(layer: u32, i: i32, j: i32) -> vec2<f32> {
    let count = rp.strata_count;
    if (layer == 0u) {
        return vec2<f32>(crisp_box(i, j), new_soft(i, j));
    }
    if (layer < count) {
        return old_layer_softened(layer - 1u, i, j);
    }
    if (layer == count) {
        // Bedrock: what stood, one step down, plus what just aged out. A
        // per-ceremony fraction, never a per-second fade (§17.6).
        let held = old_layer(count, i, j).y;
        let merging = old_layer(count - 1u, i, j).y;
        return vec2<f32>(0.0, min(
            held * (1.0 - rp.bedrock_fade) + merging * rp.bedrock_gain,
            HALO_CAP));
    }
    // Padding half of the last tile when the layer count is even.
    return vec2<f32>(0.0);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = rp.strata_h * rp.strata_tiles;
    if (gid.x >= rp.strata_w || gid.y >= rows) {
        return;
    }
    let tile = gid.y / rp.strata_h;
    let i = i32(gid.x);
    let j = i32(gid.y - tile * rp.strata_h);
    let a = new_layer(2u * tile, i, j);
    let b = new_layer(2u * tile + 1u, i, j);
    textureStore(
        strata_dst, vec2<i32>(gid.xy),
        finite_or4(vec4<f32>(a.x, a.y, b.x, b.y), 0.0));
}
