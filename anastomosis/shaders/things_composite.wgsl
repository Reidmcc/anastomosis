// The Things' image: the canvas, lifted onto the void -- §18.3.
//
// The canvas field already *is* the picture (image and trail are one
// object, the founding law), so the compositor's whole job is sampling it
// onto the window: aspect correction that wraps both axes (the world is a
// torus, §18.1 soul 8), the house void underneath (the founding #0a0a0a,
// spoken as background_luma with the fungal fog's faint coolness, packed
// host-side into fog_r/g/b), and the perceptual bounds in Oklab. Those
// bounds are shaping; §7's stage downstream is the guarantee.

//!include things_common.wgsl

//!struct RenderParams

@group(0) @binding(0) var<storage, read> params: ThingsParams;
@group(0) @binding(1) var<storage, read> render: RenderParams;
@group(0) @binding(2) var canvas_tex: texture_2d<f32>;
@group(0) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var samp: sampler;
// The painter's order (§18 round 5): which body owns each field texel
// this tick, and the population to look the owner up in.
@group(0) @binding(5) var<storage, read> owner: array<u32>;
@group(0) @binding(6) var<storage, read> things: array<Thing>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= render.out_w || gid.y >= render.out_h) {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5)
        / vec2<f32>(f32(render.out_w), f32(render.out_h));

    // Both axes wrap: the axis that gained on the simulation's aspect
    // samples more of the torus, seamlessly (backend.aspect_correction).
    let suv = wrap_uv((uv - 0.5) * vec2<f32>(params.x_scale, params.y_scale)
                      + 0.5);

    let canvas = finite_or4(
        textureSampleLevel(canvas_tex, samp, suv, 0.0), 0.0);
    let lit = max(canvas.rgb, vec3<f32>(0.0));
    let fog = vec3<f32>(render.fog_r, render.fog_g, render.fog_b);

    // The breath under the villages (§18.3): the ghost channel, rendered
    // as a faint cool-grey wander-shadow -- present at converged
    // exposure, always quieter than any body. The moodiness of this
    // image lives in the field; the shadows are how the field remembers.
    //
    // THE GHOST MAY NOT TINT THE LIVING (§18, round 4). Additive grey
    // under a candy disc is milk: the felt pass found bodies pastel in
    // dense breath and vibrant on virgin ground -- settlement being
    // punished, which was never in any soul. The founding file's
    // source-over order (fade first, discs after) guaranteed the living
    // always sat on top of their own history, and shadows existed only
    // where no body currently stood; this occlusion is that order,
    // restated for an additive canvas. The knees are trail arithmetic,
    // not taste: trail older than about two seconds has decayed below
    // 0.02 (steady state times e^-fade_rate*t), so the breath keeps its
    // full jurisdiction over settled ground, while anything the living
    // are currently lighting -- cores, skirts, fresh wake -- occludes it
    // fully by 0.12, well under the dimmest bilinear-diluted body core.
    // A Thing over thick cloud and a Thing on virgin dark are identical
    // by construction, and the shadows show exactly where no body
    // currently stands.
    let ghost = clamp(canvas.a, 0.0, 1.0);
    let lit_max = max(max(lit.r, lit.g), lit.b);
    let occlusion = smoothstep(0.02, 0.12, lit_max);
    let breath = vec3<f32>(0.85, 0.92, 1.0)
        * (ghost * params.ghost_luma * (1.0 - occlusion));

    var rgb = fog + breath + lit * params.out_gain;

    // THE LIVING MAY NOT TINT EACH OTHER (§18 round 5). The additive
    // canvas has no painter's order, so in a crowded village the
    // overlapping bodies, skirts and trails sum: lightness pegs at its
    // ceiling, chroma cannot follow, and village cores whiten to pastel
    // while lone Things stay candy -- the felt pass's "dots behind the
    // trails", second verse. The founding file composited source-over:
    // every disc painted 0.7 over everything beneath it, so a body's
    // core was always mostly its own colour whatever the crowd did.
    // Restored here: the tick's ownership layer names the topmost body
    // on each texel (later index wins, the founding draw order), and the
    // compositor paints the owner's own colour at the founding 0.7 --
    // over trail, breath and neighbours alike -- at the trail's own
    // steady-state level so the ratified brightness holds. The 30% that
    // shows through is the founding's own translucency: history and
    // neighbours keep exactly the vote they always had.
    let texel = min(
        vec2<i32>(suv * vec2<f32>(f32(params.dims_x), f32(params.dims_y))),
        vec2<i32>(i32(params.dims_x) - 1, i32(params.dims_y) - 1));
    let own = owner[u32(texel.y) * params.dims_x + u32(texel.x)];
    if (own > 0u && own <= params.capacity) {
        let t = things[own - 1u];
        // The owner's disc, evaluated exactly as the deposit pass drew
        // it -- same pulse, same crisp half-texel edge -- with the
        // distance taken around the torus so an edge-straddling body
        // owns both of its halves.
        let field = suv * vec2<f32>(f32(params.dims_x), f32(params.dims_y));
        var delta = field - thing_pos(t);
        let span = vec2<f32>(f32(params.dims_x), f32(params.dims_y));
        delta = delta - round(delta / span) * span;
        let radius = max(
            t.size * params.world_scale
                + sin(params.pulse_phase + t.x * params.pulse_x)
                    * params.pulse_amp,
            0.5);
        let cov = 1.0 - smoothstep(radius - 0.5, radius + 0.5, length(delta));
        let fade_in = min(1.0, f32(t.age) / max(params.fadein_ticks, 1.0));
        let alpha = 0.7 * fade_in * cov;
        if (alpha > 1e-3) {
            let body = hsl_to_linear(t.hue, 0.6, 0.5) * params.body_level;
            rgb = mix(rgb, body, alpha);
        }
    }

    // Perceptual bounds, applied to the target as everywhere else.
    var lab = linear_srgb_to_oklab(rgb);
    lab.x = clamp(lab.x, 0.0, render.l_max);
    let chroma = length(lab.yz);
    if (chroma > render.c_max && chroma > 1e-6) {
        lab = vec3<f32>(lab.x, lab.yz * (render.c_max / chroma));
    }
    let out = oklab_to_linear_srgb(lab);
    textureStore(
        hdr_out, vec2<i32>(gid.xy), finite_or4(vec4<f32>(out, 1.0), 0.0));
}
