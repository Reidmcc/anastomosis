// The flash-safety stage -- DESIGN.md §7. This is a safety property, not a
// style choice, and it is enforced here by construction rather than avoided by
// taste upstream.
//
// The bound holds no matter what the simulation does. If a parameter is set
// absurdly, if the reaction blows up, if an upstream shader has a bug, the
// output still cannot change faster than the limit set here.
//
// WCAG 2.3.1 / PEAT define a flash as a pair of opposing relative-luminance
// changes of >=10% over >25% of the screen, and permit at most 3 per second.
// With max_luminance_delta = 0.01 at 30 FPS, a 10% excursion needs >=10 frames
// (333 ms) and an opposing pair >=667 ms -- a ceiling of 1.5 flashes/second,
// half the threshold. Because the limit is per-pixel, it bounds any area.
//
// Note *which* limit that argument runs on. There are two, in two different
// quantities, and they are both here on purpose: the Oklab clamps bound how
// fast the image can change perceptually, and the relative-luminance clamp at
// the end bounds the photic quantity the paragraph above is about. The second
// is not implied by the first -- Oklab L is roughly the cube root of relative
// luminance, and for a chromatic colour it is not luminance at all.
//
// The order of operations matters: exposure is applied *before* the limiter, so
// the governor can never introduce a step of its own; it can only ask, and the
// limiter decides how fast the request is honoured.

//!include common.wgsl

//!struct RenderParams
//!struct Stats

@group(0) @binding(0) var<storage, read> render: RenderParams;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var velocity_tex: texture_2d<f32>;
@group(0) @binding(4) var final_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var<storage, read> stats: Stats;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= render.out_w || gid.y >= render.out_h) {
        return;
    }

    let p = vec2<i32>(gid.xy);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(render.out_w), f32(render.out_h));

    // --- Target ------------------------------------------------------------
    let exposure = clamp(finite_or(stats.exposure, 1.0), 0.02, 20.0);
    let incoming = finite_or4(textureLoad(hdr_tex, p, 0), 0.0).rgb * exposure;
    var wanted = linear_srgb_to_oklab(incoming);

    // Perceptual bounds are applied to the *target*, never to the result.
    // Clamping the result would mean that lowering l_max mid-session forces an
    // immediate downward jump on every pixel above the new ceiling -- an
    // unbounded step, straight through the guarantee this pass exists to make.
    // Applied here instead, a ceiling change is just a new target that the
    // limiter walks toward at the permitted rate.
    wanted.x = clamp(wanted.x, 0.0, max(render.l_max, 0.0));
    let wanted_lch = oklab_to_oklch(wanted);
    wanted = oklch_to_oklab(vec3<f32>(
        wanted_lch.x, min(wanted_lch.y, max(render.c_max, 0.0)), wanted_lch.z));

    // --- Motion-compensated history ---------------------------------------
    // Comparing against the raw previous frame would smear anything that
    // translates across the screen, because honest motion would read to the
    // limiter as change. Reprojecting through the velocity field means the
    // limiter only sees genuine change and leaves motion alone.
    let vdims = vec2<f32>(textureDimensions(velocity_tex, 0));
    let velocity = finite_or4(textureSampleLevel(velocity_tex, samp, uv, 0.0), 0.0).rg;
    let displacement = velocity * render.reproject_scale / vdims;
    let history_uv = wrap_uv(uv - displacement);
    // Kept in linear light as well as in Oklab: the relative-luminance limit
    // below is measured against exactly these values, and bilinear filtering of
    // an in-gamut buffer is a convex combination, so this is in gamut too.
    let history_rgb =
        finite_or4(textureSampleLevel(history_tex, samp, history_uv, 0.0), 0.0).rgb;
    let previous = linear_srgb_to_oklab(history_rgb);

    // --- Slew limit --------------------------------------------------------
    // The IIR factor gives a smooth approach (no hard corner when the clamp
    // releases); the clamp supplies the actual guarantee.
    var step = (wanted - previous) * clamp(render.iir_alpha, 0.0, 1.0);
    let luma_limit = max(render.max_luma_delta, 0.0);
    let chroma_limit = max(render.max_chroma_delta, 0.0);
    step = vec3<f32>(
        clamp(step.x, -luma_limit, luma_limit),
        clamp(step.y, -chroma_limit, chroma_limit),
        clamp(step.z, -chroma_limit, chroma_limit),
    );

    // The output is exactly the previous frame plus a bounded step. Nothing
    // below may enlarge it.
    var result = previous + step;
    result.x = max(result.x, 0.0);

    // Gamut mapping reduces chroma at constant lightness and hue; clipping RGB
    // directly would change perceived brightness, which is the very thing this
    // pass exists to bound.
    let mapped = gamut_map_oklab(result);

    // --- Relative-luminance limit -----------------------------------------
    // The clamps above bound the step in Oklab, which is the perceptual
    // quantity; WCAG's thresholds are in relative luminance, which is a
    // different one (see `relative_luminance` and DESIGN.md 7). Bounding only
    // the first leaves the second to a conversion factor that grows with
    // lightness -- and leaves the chroma limiter, which is not
    // luminance-neutral for chromatic colours, contributing as much again
    // uncounted. So the photic quantity is bounded here, directly.
    //
    // `Y` is linear in linear-light RGB, so interpolating between the previous
    // frame and the mapped target scales the luminance difference by exactly
    // the same factor: one `mix` lands on the budget with no search, where
    // scaling the Oklab step by the same ratio would only approximate it
    // (measured: 0.0153 against a 0.010 budget). Both endpoints are in gamut,
    // so the result is, and the factor is at most 1 -- this can only ever
    // shorten a step that has already been bounded, never lengthen one.
    let y_previous = relative_luminance(history_rgb);
    let luminance_step = relative_luminance(mapped) - y_previous;
    let scale = min(
        1.0,
        max(render.max_luminance_delta, 0.0) / max(abs(luminance_step), 1.0e-9),
    );
    let rgb = mix(history_rgb, mapped, scale);

    textureStore(final_out, p, vec4<f32>(rgb, 1.0));
}
