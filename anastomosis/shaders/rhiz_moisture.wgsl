// Moisture percolation through the soil column -- DESIGN.md §15.3.
//
// The rhizotron's one flowing field, and deliberately *not* incompressible:
// pooling above hardpan and draining through gaps is the imagery, so water
// moves by gravity-biased nonlinear transport through the soil's conductivity
// rather than by a stream function. Stability comes from bounded flux -- no
// texel ever gives up more than a quarter of what it holds in a tick -- and
// from clamping, which together make the explicit update unconditionally
// tame at any parameter this pass can be handed.
//
// The descent (§15.4) is folded into the read: the pass writes texel (x, y)
// from source (x, y + scroll), so a scroll is an exact integer translation --
// no resampling, no accumulated error -- and a source row below the bottom of
// the texture is a row that did not exist last tick, initialised from the
// soil generator at its own world coordinates. Rain lands on the top row,
// which is where the unseen surface above the window drains to.

//!include rhiz_common.wgsl

//!struct Event

@group(0) @binding(0) var<storage, read> rp: RhizParams;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read> events: array<Event>;

fn baseline_at(ux: f32, rel_row: f32) -> f32 {
    let soil = soil_at(rp, ux, rel_row);
    // Permeable soil sits drier at rest; dense soil holds more against
    // gravity. This is what gives the column resting wetness *bands* for the
    // weather to move against, rather than a uniform grey to darken. Above
    // the soil line (§17.4) the resting moisture eases to zero: air is dry,
    // and the relaxation term is what keeps it that way.
    let ground = smoothstep(-0.5, 1.5, rel_row - rp.surface_row);
    return rp.moisture_baseline * max(1.45 - 0.9 * soil.cond, 0.2) * ground;
}

// The source state of the texel that will land at row `y` after this tick's
// scroll. (moisture, wet EMA); rows the texture did not hold last tick come
// from the generator.
fn src_at(x: i32, y: i32) -> vec2<f32> {
    let sy = y + i32(rp.scroll_rows);
    let sx = ((x % i32(rp.dims_x)) + i32(rp.dims_x)) % i32(rp.dims_x);
    if (sy >= i32(rp.dims_y)) {
        let ux = (f32(sx) + 0.5) / f32(rp.dims_x);
        let w = baseline_at(ux, f32(y) + 0.5);
        return vec2<f32>(w, w);
    }
    let v = textureLoad(src_tex, vec2<i32>(sx, sy), 0);
    return vec2<f32>(
        clamp(finite_or(v.x, 0.0), 0.0, 1.5),
        clamp(finite_or(v.y, 0.0), 0.0, 1.5),
    );
}

// The event conductivity multiplier at one position, shared by every flux
// evaluation that touches it.
fn cond_mult_at(ux: f32, vy: f32) -> f32 {
    var mult = 1.0;
    for (var i = 0u; i < rp.event_count; i = i + 1u) {
        let e = events[i];
        let lateral = event_lateral(ux, e.pos_x, e.radius);
        let dyv = (vy - e.pos_y) / max(e.radius * 2.0, 1e-4);
        let disc = lateral * exp(-0.5 * dyv * dyv);
        mult = mult * clamp(1.0 + e.chan_flow * e.strength * disc, 0.25, 4.0);
    }
    return mult;
}

fn view_v(y: f32) -> f32 {
    return (y - f32(rp.margin_top) + 0.5) / f32(rp.view_rows);
}

// Downward flux out of world row `y` into `y + 1`, from source state. Both
// the donor and the receiver evaluate this from the same inputs -- the event
// multiplier included, at the donor's own position -- which is what keeps the
// transport conservative without atomics.
fn flux_down(x: i32, y: i32) -> f32 {
    if (y < 0) {
        return 0.0;
    }
    let w = src_at(x, y).x;
    let ux = (f32(x) + 0.5) / f32(rp.dims_x);
    let c_here = soil_at(rp, ux, f32(y) + 0.5).cond;
    let c_below = soil_at(rp, ux, f32(y) + 1.5).cond;
    // The interface passes what its tighter side allows: water pools *above*
    // a stone, which is the emergent image this pass exists to produce.
    let c = max(min(c_here, c_below) * cond_mult_at(ux, view_v(f32(y))), 0.0);
    let q = rp.perc_rate * c * pow(max(w, 0.0), rp.perc_pow);
    return min(q, 0.25 * w);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= rp.dims_x || gid.y >= rp.dims_y) {
        return;
    }
    let x = i32(gid.x);
    let y = i32(gid.y);
    let ux = (f32(x) + 0.5) / f32(rp.dims_x);
    let vy = view_v(f32(y));

    // --- Events ------------------------------------------------------------
    // The rhizotron reads the shared channels in its own terms (§15.5): a
    // positive feed is rain arriving at the surface above this column, a
    // negative one is a drying spell, and flow moves the conductivity -- a
    // seep or a tightening (that one is folded into `flux_down`, so donor and
    // receiver agree on it). Channels it has no reading of yet are inert.
    var rain_extra = 0.0;
    var dry = 0.0;
    for (var i = 0u; i < rp.event_count; i = i + 1u) {
        let e = events[i];
        let lateral = event_lateral(ux, e.pos_x, e.radius);
        let dyv = (vy - e.pos_y) / max(e.radius * 2.0, 1e-4);
        let disc = lateral * exp(-0.5 * dyv * dyv);
        if (e.chan_feed > 0.0) {
            rain_extra = rain_extra + e.strength * e.chan_feed * lateral;
        } else {
            dry = dry + e.strength * (-e.chan_feed) * disc;
        }
    }

    // --- Transport ----------------------------------------------------------
    let state = src_at(x, y);
    var w = state.x;

    // Gravity: what leaves for the row below, what arrives from the row
    // above. Rain lands on the first row below the soil line (§17.4) -- or,
    // with no surface in the pane, on the top row exactly as the §15 column
    // always took it: the sentinel surface row floors to below zero and the
    // clamp hands the old behaviour back.
    let rain_row = clamp(
        i32(floor(rp.surface_row)) + 1, 0, i32(rp.dims_y) - 1);
    w = w - flux_down(x, y);
    if (y == rain_row) {
        w = w + rp.rain_base + rp.rain_event_gain * rain_extra;
    }
    if (y > 0) {
        w = w + flux_down(x, y - 1);
    }

    // Sideways seep, conductivity-weighted and bounded like the gravity term.
    let c_here = soil_at(rp, ux, f32(y) + 0.5).cond;
    let wl = src_at(x - 1, y).x;
    let wr = src_at(x + 1, y).x;
    let seep = rp.lat_spread * c_here * ((wl - state.x) + (wr - state.x));
    w = w + clamp(seep, -0.2 * state.x, 0.2 * (wl + wr));

    // Relaxation toward the local resting moisture: drainage where the rain
    // left it wet, capillary re-supply where a dry spell parched it. A drying
    // event lowers the target rather than removing water directly, so the
    // effect arrives through the same slow transport everything else does.
    let resting = baseline_at(ux, f32(y) + 0.5) * clamp(1.0 - 0.8 * dry, 0.1, 1.0);
    w = w + rp.drain_rate * (1.0 + 2.0 * dry) * (resting - w);

    w = clamp(finite_or(w, rp.moisture_baseline), 0.0, 1.5);

    // The wetness the shading reads: one more lowpass, so the visible
    // darkening spends the luminance budget at its own deliberate pace.
    let ema = clamp(
        state.y + (w - state.y) * rp.wet_ema_rate,
        0.0, 1.5,
    );

    textureStore(dst_tex, vec2<i32>(x, y), vec4<f32>(w, ema, 0.0, 0.0));
}
