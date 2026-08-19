// Climate field update for the slab -- DESIGN.md §4.1, in three dimensions.
//
// Identical in intent to climate.wgsl and identical in channel layout:
//
//   climate_a = (feed, kill, sensor_angle, sensor_distance)
//   climate_b = (deposit, decay, flow, hue)
//   climate_c = (scale, prune, repel, spare)
//
// What changes is that a regime is now a *region of the slab* rather than a
// column through it. That is not decoration: with a 2D climate stretched
// through depth, every structure at a given (x, y) would share one feed, one
// kill and one feature size however far apart in z they were, and the volume
// would read as a thick sheet. Giving the climate the third axis is what makes
// a coarse zone in front of a fine one a thing that can happen.
//
// The advecting velocity is the curl of the weather potential, so regimes
// migrate through the volume without piling up or draining anywhere.

//!include common3d.wgsl

//!struct SimParams
//!struct Event

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var clim_a_in: texture_3d<f32>;
@group(0) @binding(2) var clim_b_in: texture_3d<f32>;
@group(0) @binding(3) var clim_c_in: texture_3d<f32>;
@group(0) @binding(4) var clim_a_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(5) var clim_b_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var clim_c_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(7) var psi_tex: texture_3d<f32>;
@group(0) @binding(8) var samp: sampler;
@group(0) @binding(9) var<storage, read> events: array<Event>;

// v = curl(psi), divergence-free by construction with central differences --
// `div(curl(A))` cancels term for term because the difference operators
// commute -- so regimes migrate without accumulating anywhere.
fn psi_velocity(uvw: vec3<f32>) -> vec3<f32> {
    let step = 1.0 / vec3<f32>(f32(params.psi_w), f32(params.psi_h), f32(params.psi_d));
    let px = textureSampleLevel(psi_tex, samp, wrap_uvw(uvw + vec3<f32>(step.x, 0.0, 0.0)), 0.0).xyz;
    let mx = textureSampleLevel(psi_tex, samp, wrap_uvw(uvw - vec3<f32>(step.x, 0.0, 0.0)), 0.0).xyz;
    let py = textureSampleLevel(psi_tex, samp, wrap_uvw(uvw + vec3<f32>(0.0, step.y, 0.0)), 0.0).xyz;
    let my = textureSampleLevel(psi_tex, samp, wrap_uvw(uvw - vec3<f32>(0.0, step.y, 0.0)), 0.0).xyz;
    let pz = textureSampleLevel(psi_tex, samp, wrap_uvw(uvw + vec3<f32>(0.0, 0.0, step.z)), 0.0).xyz;
    let mz = textureSampleLevel(psi_tex, samp, wrap_uvw(uvw - vec3<f32>(0.0, 0.0, step.z)), 0.0).xyz;
    return 0.5 * vec3<f32>(
        (py.z - my.z) - (pz.y - mz.y),
        (pz.x - mz.x) - (px.z - mx.z),
        (px.y - mx.y) - (py.x - my.x),
    );
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.clim_w, params.clim_h, params.clim_d);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let fdims = vec3<f32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / fdims;
    let texel = 1.0 / fdims;

    // 1. Advect. Regimes arrive from elsewhere and move on, so no region of
    //    the slab keeps a stable character.
    let vel = psi_velocity(uvw) * params.clim_advect;
    let src = wrap_uvw(uvw - vel * texel);

    var a = textureSampleLevel(clim_a_in, samp, src, 0.0);
    var b = textureSampleLevel(clim_b_in, samp, src, 0.0);
    var c = textureSampleLevel(clim_c_in, samp, src, 0.0);

    // 2. Diffuse, so the field can never develop a hard edge.
    if (params.clim_diffuse > 0.0) {
        var acc_a = vec4<f32>(0.0);
        var acc_b = vec4<f32>(0.0);
        var acc_c = vec4<f32>(0.0);
        let offsets = array<vec3<f32>, 6>(
            vec3<f32>(texel.x, 0.0, 0.0), vec3<f32>(-texel.x, 0.0, 0.0),
            vec3<f32>(0.0, texel.y, 0.0), vec3<f32>(0.0, -texel.y, 0.0),
            vec3<f32>(0.0, 0.0, texel.z), vec3<f32>(0.0, 0.0, -texel.z),
        );
        for (var i = 0u; i < 6u; i = i + 1u) {
            let s = wrap_uvw(src + offsets[i]);
            acc_a = acc_a + textureSampleLevel(clim_a_in, samp, s, 0.0);
            acc_b = acc_b + textureSampleLevel(clim_b_in, samp, s, 0.0);
            acc_c = acc_c + textureSampleLevel(clim_c_in, samp, s, 0.0);
        }
        a = mix(a, acc_a / 6.0, params.clim_diffuse);
        b = mix(b, acc_b / 6.0, params.clim_diffuse);
        c = mix(c, acc_c / 6.0, params.clim_diffuse);
    }

    // 3. Ornstein-Uhlenbeck step: mean-reverting so it stays in a sane band,
    //    aperiodic so it never recurs.
    var seed = pcg3(gid.x, pcg2(gid.y, gid.z), params.tick ^ params.seed);
    let theta = params.clim_theta;
    let sigma = params.clim_sigma;
    a = a * (1.0 - theta) + vec4<f32>(
        gauss(&seed), gauss(&seed), gauss(&seed), gauss(&seed)) * sigma;
    b = b * (1.0 - theta) + vec4<f32>(
        gauss(&seed), gauss(&seed), gauss(&seed), gauss(&seed)) * sigma;
    c = c * (1.0 - theta) + vec4<f32>(
        gauss(&seed), gauss(&seed), gauss(&seed), gauss(&seed)) * sigma;

    // 4. Slow events (DESIGN.md §4.3), applied to climate rather than to
    //    pigment or luminance so nothing an event does can arrive as a step.
    //
    //    An event is a ball here rather than a disc, and its radius is the
    //    same fraction of the *lateral* extent it is under the layered
    //    backend. The slab is thinner than that radius at any usable setting,
    //    so in practice an event reaches right through the depth it occupies
    //    -- which is wanted: an event that faded out through the thickness
    //    would put its own edge inside the volume.
    let n_events = min(params.event_count, arrayLength(&events));
    // The *slab's* aspect, not the climate grid's: the climate is coarser in
    // depth than in the plane by a different factor, so measuring the ball
    // with this grid's own shape would stretch it along the depth axis.
    let world = vec3<f32>(
        1.0,
        f32(params.dims_y) / f32(max(params.dims_x, 1u)),
        f32(params.dims_z) / f32(max(params.dims_x, 1u)),
    );
    for (var i = 0u; i < n_events; i = i + 1u) {
        let ev = events[i];
        let centre = vec3<f32>(ev.pos_x, ev.pos_y, ev.pos_z);
        // Toroidal distance on every axis, matching the wrapped domain, and
        // measured in world units so the ball is round rather than stretched
        // along whichever axis has fewer voxels.
        var d = abs(uvw - centre);
        d = min(d, 1.0 - d) * world;
        let r = length(d) / max(ev.radius, 1e-4);
        if (r < 1.0) {
            // Raised cosine in space as well as time: no edge anywhere.
            let falloff = 0.5 + 0.5 * cos(PI * r);
            let amp = ev.strength * falloff;
            a.x = a.x + amp * ev.chan_feed;
            a.y = a.y + amp * ev.chan_kill;
            b.z = b.z + amp * ev.chan_flow;
            b.w = b.w + amp * ev.chan_hue;
            b.y = b.y + amp * ev.chan_decay;
            c.y = c.y + amp * ev.chan_prune;
            c.z = c.z + amp * ev.chan_repel;
        }
    }

    a = clamp(finite_or4(a, 0.0), vec4<f32>(-1.0), vec4<f32>(1.0));
    b = clamp(finite_or4(b, 0.0), vec4<f32>(-1.0), vec4<f32>(1.0));
    c = clamp(finite_or4(c, 0.0), vec4<f32>(-1.0), vec4<f32>(1.0));

    let p = vec3<i32>(gid);
    textureStore(clim_a_out, p, a);
    textureStore(clim_b_out, p, b);
    textureStore(clim_c_out, p, c);
}
