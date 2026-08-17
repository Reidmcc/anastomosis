// Layer compositing and the Oklab colour mapping -- DESIGN.md §5-6.
//
// Layers are composited back to front with Beer-Lambert transmittance and
// per-layer atmospheric attenuation, so nearer material genuinely occludes and
// tints what is behind it. Depth reads mostly from the attenuation and the
// desaturation, not from the parallax.
//
// Colour is a function of simulation state, never of a clock:
//   lightness <- pigment density
//   chroma    <- heavily lowpassed local activity
//   hue       <- carried by the flow, seeded from field orientation and the
//                climate's regional hue, offset by the slowly drifting anchor.
//
// Every mapping here is a smooth transfer function. There are no thresholds
// anywhere: a threshold is an edge, and an edge that moves is punctuation.

//!include common.wgsl

//!struct RenderParams
//!struct LayerData

const MAX_LAYERS: u32 = 4u;

@group(0) @binding(0) var<storage, read> render: RenderParams;
@group(0) @binding(1) var<storage, read> layers: array<LayerData>;
@group(0) @binding(2) var layer0: texture_2d<f32>;
@group(0) @binding(3) var layer1: texture_2d<f32>;
@group(0) @binding(4) var layer2: texture_2d<f32>;
@group(0) @binding(5) var layer3: texture_2d<f32>;
@group(0) @binding(6) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var samp: sampler;

fn sample_layer(index: u32, uv: vec2<f32>) -> vec4<f32> {
    // WGSL has no core binding arrays, so the layer count is a fixed maximum
    // with unused slots bound to a 1x1 dummy.
    switch index {
        case 0u: { return textureSampleLevel(layer0, samp, uv, 0.0); }
        case 1u: { return textureSampleLevel(layer1, samp, uv, 0.0); }
        case 2u: { return textureSampleLevel(layer2, samp, uv, 0.0); }
        default: { return textureSampleLevel(layer3, samp, uv, 0.0); }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= render.out_w || gid.y >= render.out_h) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(render.out_w), f32(render.out_h));
    let fog_rgb = vec3<f32>(render.fog_r, render.fog_g, render.fog_b);

    // Ground: the darkest the image ever gets, expressed in Oklab so that the
    // "dark" end of the range is perceptually rather than numerically dark.
    var accum = oklab_to_linear_srgb(vec3<f32>(render.background_luma, 0.0, 0.0));

    let count = min(render.layer_count, MAX_LAYERS);
    var i = count;
    loop {
        if (i == 0u) { break; }
        i = i - 1u;  // back (highest index) to front

        let layer = layers[i];

        // Haze sits in front of everything already accumulated behind it.
        accum = mix(accum, fog_rgb, clamp(layer.fog, 0.0, 1.0));

        let layer_uv = wrap_uv(
            (uv - 0.5) * vec2<f32>(layer.scale_x, layer.scale_y) + 0.5
            + vec2<f32>(layer.parallax_x, layer.parallax_y)
        );
        let pigment = sample_layer(i, layer_uv);

        let density = max(pigment.x, 0.0);
        let activity = max(pigment.w, 0.0);
        let hue = vec_to_hue(pigment.yz) + render.hue_global;

        // Lightness. A compressive power curve lifts faint structure without
        // raising peaks, and the result is hard-capped at l_max.
        let shaped = pow(clamp(density, 0.0, 1.0), max(render.glow_gamma, 0.05));
        var lightness = render.background_luma + render.filament_luma * shaped;
        lightness = min(lightness, render.l_max) * clamp(layer.depth_dim, 0.0, 1.0);

        // Chroma from activity, saturating smoothly so that a burst of change
        // cannot produce a corresponding burst of saturation.
        let saturation = 1.0 - exp(-activity * max(render.chroma_activity_gain, 0.0));
        var chroma = render.chroma_floor
            + (render.c_max - render.chroma_floor) * saturation;
        chroma = min(chroma, render.c_max) * clamp(layer.depth_desat, 0.0, 1.0);

        let rgb = oklab_to_linear_srgb(
            oklch_to_oklab(vec3<f32>(lightness, max(chroma, 0.0), hue))
        );

        // Beer-Lambert: dense material occludes what is behind it.
        let alpha = (1.0 - exp(-density * max(render.extinction, 0.0)))
            * clamp(layer.opacity, 0.0, 1.0);
        accum = mix(accum, rgb, clamp(alpha, 0.0, 1.0));
    }

    // Left slightly out of gamut on purpose: a single gamut map at the end of
    // the safety stage is both cheaper and more correct than mapping each
    // layer and then compositing the clipped results.
    textureStore(hdr_out, vec2<i32>(gid.xy), finite_or4(vec4<f32>(accum, 1.0), 0.0));
}
