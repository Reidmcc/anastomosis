// The volumetric compositor -- DESIGN.md §5.1, and the counterpart of
// composite.wgsl.
//
// One ray per output pixel, marched front to back through the slab with
// Beer-Lambert transmittance. What that buys over the layered stack is stated
// in §5.1 and is worth restating as what the code below actually does:
//
//   * **Real attenuation through a continuous medium.** Density accumulates
//     optical depth along the ray, so what is behind dense material is
//     genuinely dimmed by it rather than alpha-blended against three fixed
//     sheets.
//   * **Self-shadowing from a single soft light.** A short secondary march
//     toward the light gives structure interior shading, which is most of what
//     makes a volume read as volume rather than as fog.
//   * **Structures that pass smoothly in front of and behind each other,**
//     because there is nothing discrete for them to pass between.
//
// The colour mapping is identical to the layered backend's, term for term:
// lightness from pigment density through a compressive curve and hard-capped,
// chroma from heavily lowpassed activity, hue carried by the flow and offset
// by the drifting anchor. Every mapping is a smooth transfer function; there
// are no thresholds anywhere, because a threshold is an edge and an edge that
// moves is punctuation.
//
// Two outputs. The HDR image, and a screen-space velocity field for the safety
// stage to reproject its history through (DESIGN.md §7). The second is
// accumulated exactly like the first -- each sample's lateral velocity
// weighted by how much that sample contributes to the pixel -- so the
// reprojection follows whatever material the pixel is actually showing rather
// than some arbitrary depth.

//!include common3d.wgsl

//!struct RenderParams

@group(0) @binding(0) var<storage, read> render: RenderParams;
@group(0) @binding(1) var pigment_vol: texture_3d<f32>;
@group(0) @binding(2) var velocity_vol: texture_3d<f32>;
@group(0) @binding(3) var hdr_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var motion_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var noise_tex: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;

// `extinction` reaches this shader as optical depth per unit density *per
// filament*, because that is the only reading under which the `filament_glow`
// macro means the same thing here as it does in the compositor: there,
// `1 - exp(-density * extinction)` is applied once to a whole layer, i.e. once
// to whatever structure the pixel is looking through. Here the ray crosses that
// structure in several steps, so the per-length coefficient has to be the one
// that accumulates to `extinction` across a filament's width.
//
// Normalising by the *slab's* thickness instead -- so that a ray crossing the
// whole depth at density one accumulates `extinction` -- is the obvious
// alternative and is the wrong reading. A filament occupies a few voxels of
// however many the slab is deep, so under that rule it would pick up a small
// fraction of the optical depth the same setting gives it in the compositor --
// an eighth of it at the default thickness -- and `filament_glow` would mean
// two different things depending on which backend was running, and a third
// thing again whenever the thickness moved. The exposure governor would hide
// most of the difference by lifting the whole image, ground included, which is
// the wrong correction applied to the wrong thing.
//
// Six voxels is the width of a Gray-Scott feature at the default `scale`,
// taken as the full width of a filament rather than as the trail's diffusion
// sigma. It is a single number standing in for something that varies with the
// `scale` macro and with the climate, so it is an approximation on purpose --
// the alternative is a per-voxel feature-size estimate, and nothing here needs
// that precision. Expressed in voxels rather than as a fraction of the depth
// so that changing `volume.depth` -- which the control panel does -- alters how
// much material a ray passes through, which is the entire point of moving it,
// without also altering how opaque any of that material is.
const FEATURE_VOXELS: f32 = 6.0;

// Once this little light is still getting through, everything left behind is
// bounded by it, and the ground is added with whatever transmittance remains --
// so what stopping early actually drops is the *difference* between the ground
// and the material still behind it, at half an 8-bit code value of weight. That
// is beneath the dither floor.
const MIN_TRANSMITTANCE: f32 = 0.002;

// Where a lateral blur is narrower than this many voxels, the four taps land
// inside the trilinear footprint of one and the difference is far below the
// dither floor. Skipping them there keeps the near half of the march at one
// tap per step, which is where the cost budget in DESIGN.md §8.1 assumes it.
const DOF_MIN_VOXELS: f32 = 0.35;

fn sample_volume(uvw: vec3<f32>) -> vec4<f32> {
    return textureSampleLevel(pigment_vol, samp, wrap_uvw(uvw), 0.0);
}

// Blue noise, read once per pixel and *not* animated. A per-frame mask would
// give better numerical results and is the usual practice; it is also
// per-pixel temporal noise, which is precisely the shimmer this application
// exists not to produce (see blit.wgsl). Held still it contributes exactly
// zero frame-to-frame change while still breaking the march's fixed step
// lattice into noise the eye integrates away.
fn pixel_noise(p: vec2<i32>, offset: vec2<i32>) -> f32 {
    let dims = vec2<i32>(textureDimensions(noise_tex, 0));
    return textureLoad(noise_tex, ((p + offset) % dims + dims) % dims, 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= render.out_w || gid.y >= render.out_h) {
        return;
    }

    let pixel = vec2<i32>(gid.xy);
    let out_dims = vec2<f32>(f32(render.out_w), f32(render.out_h));
    let uv = (vec2<f32>(gid.xy) + 0.5) / out_dims;
    let vol_dims = vec3<f32>(f32(render.vol_w), f32(render.vol_h), f32(render.vol_d));

    // Lateral position in the slab's own normalised coordinates. The zoom is
    // the aspect correction: the axis that gained relative to the simulation
    // samples *more* of the field rather than stretching it, and the domain
    // wraps, so the extra area is seamless.
    let centred = (uv - 0.5) * vec2<f32>(render.zoom_x, render.zoom_y);

    // The camera. `converge` spreads the rays slightly off orthographic, so
    // the slab is seen at an angle away from the centre of the image and
    // depths separate laterally across the frame; `cam_shear` is the slow
    // drift, and it is what parallax *is* in a volume -- the near and far
    // faces move by different amounts because the ray is tilted, not because
    // two sheets were offset by different distances.
    let shear = centred * render.converge
        + vec2<f32>(render.cam_shear_x, render.cam_shear_y);

    let steps = max(render.march_steps, 1u);
    let inv_steps = 1.0 / f32(steps);
    // World-space length of one step. Voxels are cubic, so the slab is
    // `slab_depth` thick in the same units the lateral extent measures 1 in,
    // and the tilt adds its own small amount of path length.
    let ray_world = vec3<f32>(shear, render.slab_depth);
    let step_len = length(ray_world) * inv_steps;
    // Optical depth per unit density per unit world length. Voxels are cubic,
    // so one voxel is `1 / vol_w` of world.
    let sigma = render.extinction * vol_dims.x / FEATURE_VOXELS;

    // World units per unit of normalised volume coordinate, for the shadow
    // ray: the lateral axes span 1 and h/w, the depth axis d/w.
    let world_to_uvw = vec3<f32>(
        1.0,
        vol_dims.x / max(vol_dims.y, 1.0),
        vol_dims.x / max(vol_dims.z, 1.0),
    );
    let light = normalize(
        vec3<f32>(render.light_x, render.light_y, render.light_z)
        + vec3<f32>(0.0, 0.0, 1e-5));
    let shadow_steps = render.shadow_steps;
    // `shadow_reach` arrives as a world length rather than as a fraction of
    // the slab's thickness, because what the shadow ray is probing is a
    // filament and not the slab: a few filament widths toward the light is
    // what gives structure interior shading, whether the slab is forty-eight
    // voxels deep or three hundred.
    let shadow_step = render.shadow_reach / f32(max(shadow_steps, 1u));

    // Depth of field: lateral blur radius in *voxels* at the far face. Kept in
    // voxels right up to the tap, and divided by each axis's own resolution
    // there, so the sampled disc is round -- dividing by one axis for both
    // would make it an ellipse stretched by the slab's aspect.
    let dof_voxels = max(render.dof_radius, 0.0);
    // A fixed rotation per pixel, so the four taps do not form a visible
    // axis-aligned cross. Static for the same reason the step jitter is.
    let dof_angle = pixel_noise(pixel, vec2<i32>(37, 17)) * TAU;
    let dof_a = vec2<f32>(cos(dof_angle), sin(dof_angle));
    let dof_b = vec2<f32>(-dof_a.y, dof_a.x);

    let jitter = pixel_noise(pixel, vec2<i32>(0, 0));
    let fog_rgb = vec3<f32>(render.fog_r, render.fog_g, render.fog_b);
    let edge = clamp(render.depth_window, 1e-3, 0.49);

    var accum = vec3<f32>(0.0);
    var motion = vec2<f32>(0.0);
    var transmittance = 1.0;

    for (var i = 0u; i < steps; i = i + 1u) {
        let t = (f32(i) + jitter) * inv_steps;
        let lateral = 0.5 + centred + shear * (t - 0.5);
        let here = vec3<f32>(lateral, t);

        // The soft window over the two faces. The slab is a 3-torus like the
        // rest of the domain, so material leaving the near face reappears at
        // the far one; without this that arrival and departure would be a step
        // at the depth extremes, and the near one -- where material is
        // brightest and least fogged -- would be plainly visible. Raised
        // cosine, C1 at both joins.
        let ramp_near = 0.5 - 0.5 * cos(PI * clamp(t / edge, 0.0, 1.0));
        let ramp_far = 0.5 - 0.5 * cos(PI * clamp((1.0 - t) / edge, 0.0, 1.0));
        let window = ramp_near * ramp_far;

        // Depth of field. The near half of the march is a single tap; past
        // that the four taps approximate a lateral disc whose radius grows
        // with depth. Blurring the volume rather than the image is the more
        // correct of the two: occlusion is blurred along with what occludes.
        let radius = dof_voxels * t;
        var pigment = vec4<f32>(0.0);
        if (radius > DOF_MIN_VOXELS) {
            let ra = dof_a * radius / vol_dims.xy;
            let rb = dof_b * radius / vol_dims.xy;
            pigment = 0.25 * (
                sample_volume(here + vec3<f32>(ra, 0.0))
                + sample_volume(here - vec3<f32>(ra, 0.0))
                + sample_volume(here + vec3<f32>(rb, 0.0))
                + sample_volume(here - vec3<f32>(rb, 0.0)));
        } else {
            pigment = sample_volume(here);
        }
        pigment = finite_or4(pigment, 0.0);

        let density = max(pigment.x, 0.0) * window;
        // Beer-Lambert over the step. Below the point where this is
        // meaningful there is nothing to shade, so the rest of the body is
        // skipped -- an optimisation, not a threshold: `alpha` has already
        // gone to zero smoothly by the time it bites.
        let alpha = 1.0 - exp(-density * sigma * step_len);
        let weight = transmittance * alpha;
        transmittance = transmittance * (1.0 - alpha);
        if (weight < 1e-4) {
            // Nothing here worth shading -- but a ray that went opaque a
            // moment ago and is now crossing empty space should still stop
            // rather than step out the rest of the march.
            if (transmittance < MIN_TRANSMITTANCE) {
                break;
            }
            continue;
        }

        // --- Self-shadowing ------------------------------------------------
        // A short secondary march toward the light. Cheap because it only runs
        // where a sample contributes, and short because a long one on a slab
        // this thin would leave the far side unlit rather than shaded.
        var shadow = 1.0;
        if (shadow_steps > 0u) {
            var tau = 0.0;
            for (var s = 1u; s <= shadow_steps; s = s + 1u) {
                let offset = light * (f32(s) * shadow_step);
                let probe = here + offset * world_to_uvw;
                let probe_t = probe.z;
                let probe_window =
                    (0.5 - 0.5 * cos(PI * clamp(fract(probe_t) / edge, 0.0, 1.0)))
                    * (0.5 - 0.5 * cos(PI * clamp((1.0 - fract(probe_t)) / edge, 0.0, 1.0)));
                let d = max(sample_volume(probe).x, 0.0) * probe_window;
                tau = tau + d * sigma * shadow_step;
            }
            shadow = exp(-tau * max(render.shadow_density, 0.0));
        }
        let lit = mix(clamp(render.light_ambient, 0.0, 1.0), 1.0, shadow);

        // --- Colour --------------------------------------------------------
        let activity = max(pigment.w, 0.0);
        let hue = vec_to_hue(pigment.yz) + render.hue_global;

        // A compressive power curve lifts faint structure without raising
        // peaks, and the result is hard-capped at l_max.
        let shaped = pow(clamp(density, 0.0, 1.0), max(render.glow_gamma, 0.05));
        var lightness = render.background_luma + render.filament_luma * shaped;
        lightness = min(lightness, render.l_max)
            * clamp(1.0 + (render.depth_dim - 1.0) * t, 0.0, 1.0)
            * lit;

        // Chroma from activity, saturating smoothly so that a burst of change
        // cannot produce a corresponding burst of saturation.
        let saturation = 1.0 - exp(-activity * max(render.chroma_activity_gain, 0.0));
        var chroma = render.chroma_floor
            + (render.c_max - render.chroma_floor) * saturation;
        chroma = min(chroma, render.c_max)
            * clamp(1.0 + (render.depth_desat - 1.0) * t, 0.0, 1.0);

        var rgb = oklab_to_linear_srgb(
            oklch_to_oklab(vec3<f32>(lightness, max(chroma, 0.0), hue)));
        // Atmospheric attenuation: distant material loses chroma and contrast
        // toward the background colour. This does most of the perceptual work
        // of depth, more than the parallax does.
        rgb = mix(rgb, fog_rgb, clamp(render.depth_fog * t, 0.0, 1.0));

        accum = accum + rgb * weight;

        // Lateral velocity of what this sample is made of, in output pixels
        // per simulation tick, so the safety stage's reprojection is in the
        // units it expects.
        //
        // Premultiplied and deliberately *not* normalised by the accumulated
        // weight. A pixel showing dense material ends with weights summing to
        // nearly one, so this is the weighted mean velocity of what it shows;
        // a pixel showing almost nothing ends near zero, which says "nothing
        // here is moving" -- true, and the right answer. Normalising would
        // instead hand the limiter a full-strength velocity read off whatever
        // faint material the ray happened to graze.
        let v = finite_or4(
            textureSampleLevel(velocity_vol, samp, wrap_uvw(here), 0.0), 0.0).xy;
        let to_pixels = out_dims
            / (vol_dims.xy * vec2<f32>(render.zoom_x, render.zoom_y));
        motion = motion + v * to_pixels * weight;

        if (transmittance < MIN_TRANSMITTANCE) {
            break;
        }
    }

    // Ground: the darkest the image ever gets, expressed in Oklab so that the
    // "dark" end of the range is perceptually rather than numerically dark.
    // Whatever light got all the way through the slab lands on it.
    let ground = oklab_to_linear_srgb(vec3<f32>(render.background_luma, 0.0, 0.0));
    accum = accum + ground * transmittance;

    // Left slightly out of gamut on purpose: one gamut map at the end of the
    // safety stage is both cheaper and more correct than mapping here and then
    // limiting the clipped result.
    textureStore(hdr_out, pixel, finite_or4(vec4<f32>(accum, 1.0), 0.0));
    textureStore(motion_out, pixel, finite_or4(vec4<f32>(motion, 0.0, 1.0), 0.0));
}
