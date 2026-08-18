// Climate field update -- DESIGN.md §4.1.
//
// A small texture holding *deviations* (in [-1, 1]) of each governing parameter
// from its global base. Consumers combine base + homeostat correction + range *
// deviation; this pass only maintains the deviation field.
//
// Each tick the field is advected by the same weather that moves the pigment,
// diffused slightly, and given an Ornstein-Uhlenbeck increment. That makes it a
// stateful PDE rather than a function of time: it cannot repeat, and it cannot
// quantise as an f32 clock would after a day of running.
//
// Channel layout:
//   climate_a = (feed, kill, sensor_angle, sensor_distance)
//   climate_b = (deposit, decay, flow, hue)
//   climate_c = (scale, prune, repel, spare)
//
// The third pair carries morphology (DESIGN.md §4.7). `scale` is a deviation
// on the reaction's diffusion rate, and so on the characteristic feature size;
// driving it from here rather than globally is the whole point of the
// mechanism. A global breathing of feature size would be coordinated global
// change of exactly the kind §4.2 forbids, and it would leave every feature on
// screen the same size as every other -- and *uniformity* of size, not density,
// is what makes the texture a trypophobia trigger. Carried spatially, coarse
// and fine regions coexist and migrate past each other.
//
// `prune` scales the flux-pruning term in trail.wgsl, so the network comes
// apart in some regions while it holds together in others. `repel` is the
// agents' junction commitment (agents.wgsl): past the point where the steering
// term changes sign the agents veer away from filaments instead of committing
// to them, which is the anti-fusion half of §4.7 step 4. §4.7 calls that
// channel `fusion`; it is named for what it delivers, because the axis is not
// monotone in how much fusing happens -- peak fusion sits at the crossing
// point, and both sides of it fuse less.

//!include common.wgsl

//!struct SimParams
//!struct Event

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var clim_a_in: texture_2d<f32>;
@group(0) @binding(2) var clim_b_in: texture_2d<f32>;
@group(0) @binding(3) var clim_c_in: texture_2d<f32>;
@group(0) @binding(4) var clim_a_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var clim_b_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var clim_c_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var psi_tex: texture_2d<f32>;
@group(0) @binding(8) var samp: sampler;
@group(0) @binding(9) var<storage, read> events: array<Event>;

// Velocity from the vector potential: v = curl(psi), divergence-free by
// construction, so regimes migrate without piling up or draining anywhere.
fn psi_velocity(uv: vec2<f32>) -> vec2<f32> {
    let step = vec2<f32>(1.0 / f32(params.psi_w), 1.0 / f32(params.psi_h));
    let px = textureSampleLevel(psi_tex, samp, wrap_uv(uv + vec2<f32>(step.x, 0.0)), 0.0).r;
    let mx = textureSampleLevel(psi_tex, samp, wrap_uv(uv - vec2<f32>(step.x, 0.0)), 0.0).r;
    let py = textureSampleLevel(psi_tex, samp, wrap_uv(uv + vec2<f32>(0.0, step.y)), 0.0).r;
    let my = textureSampleLevel(psi_tex, samp, wrap_uv(uv - vec2<f32>(0.0, step.y)), 0.0).r;
    return vec2<f32>(py - my, -(px - mx)) * 0.5;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.clim_w, params.clim_h);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let texel = vec2<f32>(1.0 / f32(dims.x), 1.0 / f32(dims.y));

    // 1. Advect. Regimes arrive from elsewhere and move on, so no region of the
    //    screen keeps a stable character.
    let vel = psi_velocity(uv) * params.clim_advect;
    let src = wrap_uv(uv - vel * texel);

    var a = textureSampleLevel(clim_a_in, samp, src, 0.0);
    var b = textureSampleLevel(clim_b_in, samp, src, 0.0);
    var c = textureSampleLevel(clim_c_in, samp, src, 0.0);

    // 2. Diffuse, so the field can never develop a hard edge (it is sampled
    //    bilinearly at 40x lower resolution than the sim, which already makes
    //    it smooth; this keeps it smooth under repeated advection).
    if (params.clim_diffuse > 0.0) {
        var acc_a = vec4<f32>(0.0);
        var acc_b = vec4<f32>(0.0);
        var acc_c = vec4<f32>(0.0);
        let offsets = array<vec2<f32>, 4>(
            vec2<f32>(texel.x, 0.0), vec2<f32>(-texel.x, 0.0),
            vec2<f32>(0.0, texel.y), vec2<f32>(0.0, -texel.y),
        );
        for (var i = 0u; i < 4u; i = i + 1u) {
            let s = wrap_uv(src + offsets[i]);
            acc_a = acc_a + textureSampleLevel(clim_a_in, samp, s, 0.0);
            acc_b = acc_b + textureSampleLevel(clim_b_in, samp, s, 0.0);
            acc_c = acc_c + textureSampleLevel(clim_c_in, samp, s, 0.0);
        }
        a = mix(a, acc_a * 0.25, params.clim_diffuse);
        b = mix(b, acc_b * 0.25, params.clim_diffuse);
        c = mix(c, acc_c * 0.25, params.clim_diffuse);
    }

    // 3. Ornstein-Uhlenbeck step: mean-reverting so it stays in a sane band,
    //    aperiodic so it never recurs.
    var seed = pcg3(gid.x, gid.y, params.tick ^ params.seed);
    let theta = params.clim_theta;
    let sigma = params.clim_sigma;
    a = a * (1.0 - theta) + vec4<f32>(
        gauss(&seed), gauss(&seed), gauss(&seed), gauss(&seed)) * sigma;
    b = b * (1.0 - theta) + vec4<f32>(
        gauss(&seed), gauss(&seed), gauss(&seed), gauss(&seed)) * sigma;
    c = c * (1.0 - theta) + vec4<f32>(
        gauss(&seed), gauss(&seed), gauss(&seed), gauss(&seed)) * sigma;

    // 4. Slow events (DESIGN.md §4.3). Applied here, to climate, rather than to
    //    pigment or luminance: their effect reaches the image only after
    //    several stages of diffusion and temporal lowpass, so nothing an event
    //    does can arrive as a step.
    let n_events = min(params.event_count, arrayLength(&events));
    for (var i = 0u; i < n_events; i = i + 1u) {
        let ev = events[i];
        // Toroidal distance, matching the wrapped simulation domain.
        var d = abs(uv - vec2<f32>(ev.pos_x, ev.pos_y));
        d = min(d, 1.0 - d);
        let r = length(d) / max(ev.radius, 1e-4);
        if (r < 1.0) {
            // Raised cosine in space as well as time: no edge anywhere.
            let falloff = 0.5 + 0.5 * cos(PI * r);
            let amp = ev.strength * falloff;
            a.x = a.x + amp * ev.chan_feed;
            a.y = a.y + amp * ev.chan_kill;
            b.z = b.z + amp * ev.chan_flow;
            b.w = b.w + amp * ev.chan_hue;
            // Severance (§4.7 step 4). Before these three an event could thin
            // material but never sever anything: feed and kill move how much
            // is there, not how it is connected.
            b.y = b.y + amp * ev.chan_decay;
            c.y = c.y + amp * ev.chan_prune;
            c.z = c.z + amp * ev.chan_repel;
        }
    }

    a = clamp(finite_or4(a, 0.0), vec4<f32>(-1.0), vec4<f32>(1.0));
    b = clamp(finite_or4(b, 0.0), vec4<f32>(-1.0), vec4<f32>(1.0));
    c = clamp(finite_or4(c, 0.0), vec4<f32>(-1.0), vec4<f32>(1.0));

    textureStore(clim_a_out, vec2<i32>(gid.xy), a);
    textureStore(clim_b_out, vec2<i32>(gid.xy), b);
    textureStore(clim_c_out, vec2<i32>(gid.xy), c);
}
