// Physarum agents in the slab -- the filament layer, and the source of
// anastomosis, in three dimensions.
//
// DESIGN.md §5 says plainly that "3D Physarum needs different sensing and
// steering, and is much harder to tune", and this file is where that lands.
// Every property the 2D layer has is kept; how each is expressed changes:
//
//   * **Heading is a unit vector, not an angle.** There is no scalar turn in
//     3D. Steering is a rotation of the heading toward (or away from) a sensed
//     direction, done as a normalised interpolation, which is a rotation in
//     the plane the two directions span and needs no basis to express.
//
//   * **Sensing is a cone, not three points.** Four flank sensors on a cone of
//     half-angle `sensor_angle` around the heading, plus one straight ahead.
//     The cone is rolled by a per-agent random phase every tick: four fixed
//     flanks would give the population a preferred lattice of turn directions,
//     which in 2D the left/right pair cannot do.
//
//   * **Motion is anisotropic.** The slab is four or five feature-widths deep,
//     so an isotropic random-walking population crosses the whole thickness in
//     a couple of seconds and depth reads as churn. `depth_agent` flattens the
//     heading toward the plane -- the whole heading, so sensing follows the
//     motion rather than probing at angles the agent will not take.
//
// The fusion bias, the anti-fusion crossing at commitment 1.0, the flux-prune
// return, the starve/age respawn and the founding cohorts all carry over
// unchanged in meaning; see agents.wgsl for why each exists.

//!include common3d.wgsl

//!struct SimParams
//!struct Stats

struct Agent {
    px: f32,
    py: f32,
    pz: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    rng: u32,
    recent: f32,
    age: f32,
};

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var trail_tex: texture_3d<f32>;
@group(0) @binding(3) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(4) var clim_a: texture_3d<f32>;
@group(0) @binding(5) var clim_b: texture_3d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> stats: Stats;
@group(0) @binding(8) var clim_c: texture_3d<f32>;

// Deposits accumulate in fixed point because floating-point atomics are not
// available in core WebGPU. The scale is the same as in 2D; a voxel sees less
// traffic than a cell does, not more, so the headroom argument only improves.
const DEPOSIT_SCALE: f32 = 1048576.0;

// Sensing saturation, exactly as in agents.wgsl and for the same reason:
// unbounded sensed trail is winner-take-all and collapses the layer into
// knots. See the 2D shader for the full argument. Zero disables.
fn sample_trail(uvw: vec3<f32>) -> f32 {
    let value = textureSampleLevel(trail_tex, samp, wrap_uvw(uvw), 0.0).r;
    if (params.sense_cap > 0.0) {
        return min(value, params.sense_cap);
    }
    return value;
}

// Trilinear splat across the eight surrounding voxels. A hard single-voxel
// write is a one-voxel step change, i.e. exactly the visual punctuation the
// application must never produce.
fn splat(pos: vec3<f32>, dims: vec3<u32>, amount: f32) {
    let base = floor(pos - 0.5);
    let f = pos - 0.5 - base;
    let idims = vec3<i32>(dims);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let corner = vec3<i32>(
            i32(i & 1u), i32((i >> 1u) & 1u), i32((i >> 2u) & 1u));
        let w = mix(1.0 - f.x, f.x, f32(corner.x))
              * mix(1.0 - f.y, f.y, f32(corner.y))
              * mix(1.0 - f.z, f.z, f32(corner.z));
        let t = wrap_texel3(vec3<i32>(base) + corner, idims);
        let index = (u32(t.z) * dims.y + u32(t.y)) * dims.x + u32(t.x);
        let quantum = u32(max(amount * w * DEPOSIT_SCALE, 0.0));
        if (quantum > 0u) {
            atomicAdd(&deposit_buf[index], quantum);
        }
    }
}

// Any unit vector perpendicular to `d`. The helper axis is chosen away from
// `d` so the cross product is never degenerate.
fn perpendicular(d: vec3<f32>) -> vec3<f32> {
    let helper = select(
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(d.z) > 0.9);
    return normalize(cross(d, helper));
}

fn unit_gauss3(seed: ptr<function, u32>) -> vec3<f32> {
    return vec3<f32>(gauss(seed), gauss(seed), gauss(seed));
}

// Flatten a direction toward the slab's plane and renormalise. Applied to the
// heading rather than to the step so that sensing and motion agree.
fn flatten(d: vec3<f32>, factor: f32) -> vec3<f32> {
    let flat = vec3<f32>(d.x, d.y, d.z * clamp(factor, 0.0, 1.0));
    let len = length(flat);
    if (len < 1e-5) {
        return vec3<f32>(1.0, 0.0, 0.0);
    }
    return flat / len;
}

// Where this epoch's founding cohort lands, as in agents.wgsl: the barest of
// four candidates, because bare ground is the entire point -- a cohort landing
// on the existing network only reinforces it. Every agent respawning into the
// same (epoch, site) computes the same answer from the same hash, which is
// what makes a cohort out of a stream of unrelated respawns with no
// communication between invocations.
fn founding_site(epoch: u32, site: u32, fdims: vec3<f32>) -> vec3<f32> {
    var seed = pcg3(epoch, site, params.seed);
    var best = vec3<f32>(rnd(&seed), rnd(&seed), rnd(&seed)) * fdims;
    var bare = sample_trail(best / fdims);
    for (var i = 0u; i < 3u; i = i + 1u) {
        let candidate = vec3<f32>(rnd(&seed), rnd(&seed), rnd(&seed)) * fdims;
        let trail = sample_trail(candidate / fdims);
        if (trail < bare) {
            bare = trail;
            best = candidate;
        }
    }
    return best;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.agent_count) {
        return;
    }

    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    let fdims = vec3<f32>(dims);
    var agent = agents[index];

    // Recover from a corrupted agent rather than letting it wander off into
    // NaN and stop depositing forever.
    var pos = finite_or4(vec4<f32>(agent.px, agent.py, agent.pz, 0.0), 0.0).xyz;
    var dir = finite_or4(vec4<f32>(agent.dx, agent.dy, agent.dz, 0.0), 0.0).xyz;
    if (length(dir) < 1e-5) {
        dir = vec3<f32>(1.0, 0.0, 0.0);
    }
    dir = normalize(dir);

    var seed = agent.rng ^ pcg2(params.tick, params.seed);
    let uvw = pos / fdims;

    // Local parameters from the climate volume.
    let ca = textureSampleLevel(clim_a, samp, wrap_uvw(uvw), 0.0);
    let cb = textureSampleLevel(clim_b, samp, wrap_uvw(uvw), 0.0);
    let cc = textureSampleLevel(clim_c, samp, wrap_uvw(uvw), 0.0);

    let sensor_angle = max(0.02, params.sensor_angle + params.range_sensor_angle * ca.z);
    // Sensing reach, bounded against the width of what it senses (DESIGN.md
    // §4.9). Past roughly four times the trail's diffusion sigma an agent stops
    // following the strand it is on and starts steering at whatever it can see
    // from a distance; on a torus the straight runs that produces close on
    // themselves and reinforce every lap. Must stay in step with
    // config.clamp_sensor_distance.
    let sensor_dist = clamp(
        params.sensor_distance + params.range_sensor_distance * ca.w,
        1.0,
        max(params.sensor_distance_max, 1.0));
    // Flux pruning removes a fraction of the trail field's throughput from
    // strands traffic has abandoned (vol_trail.wgsl). It is handed back here so
    // pruning redistributes rather than drains: the mass reappears wherever
    // agents currently are instead of wherever mass already is.
    let deposit_amt = max(0.0, params.deposit + params.range_deposit * cb.x + stats.corr_deposit)
        * (1.0 + clamp(stats.prune_return, 0.0, 2.0)
           + clamp(stats.cap_return, 0.0, 3.0));

    // --- Sense ------------------------------------------------------------
    // One sensor ahead and four on a cone around it, rolled by a random phase
    // so the four flanks do not become a fixed lattice of turn directions.
    let u_axis = perpendicular(dir);
    let v_axis = cross(dir, u_axis);
    let roll = rnd(&seed) * TAU;
    let cone_c = cos(sensor_angle);
    let cone_s = sin(sensor_angle);

    let s_f = sample_trail((pos + dir * sensor_dist) / fdims);

    var best_dir = dir;
    var worst_dir = dir;
    var best = -1.0;
    var worst = 1e30;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let phi = roll + f32(k) * (PI * 0.5);
        let flank = normalize(
            dir * cone_c + (u_axis * cos(phi) + v_axis * sin(phi)) * cone_s);
        let value = sample_trail((pos + flank * sensor_dist) / fdims);
        if (value > best) {
            best = value;
            best_dir = flank;
        }
        if (value < worst) {
            worst = value;
            worst_dir = flank;
        }
    }

    // --- Steer ------------------------------------------------------------
    // `turn_rate` is an angle per tick and the sensed directions sit at
    // `sensor_angle` off the heading, so the fraction of the way to a sensed
    // direction that one tick's turn covers is their ratio, capped at 1.
    let turn_frac = clamp(params.turn_rate / max(sensor_angle, 1e-3), 0.0, 1.0);
    let strongest = max(s_f, best);
    let head_on = s_f >= best;

    // Fusion bias. `recent` is the agent's own running sense of ambient trail
    // strength; encountering something much stronger means another filament,
    // so commit to it instead of turning away.
    let excess = strongest / (agent.recent + 1e-4) - 1.0;
    // Anti-fusion. The clamp reaches past 1.0, where `1 - commitment` changes
    // sign and the interpolation runs *away* from what was sensed. The axis is
    // one of deflection toward the junction, running through fusion rather
    // than peaking at it -- which is why the climate channel is named for the
    // repulsion it buys. Additive, not geometric, so the fraction of the field
    // that is repelling can be read off the climate amplitude.
    let junction = params.fusion_bias + params.range_repel * cc.z;
    let commitment = clamp(
        junction * smoothstep(0.0, 2.0, excess), 0.0, params.fusion_max);

    var steer_to = select(best_dir, dir, head_on);
    var frac = turn_frac * (1.0 - commitment);
    if (commitment > 1.0 && head_on) {
        // Head-on the steering term is already zero -- the target *is* the
        // heading -- so scaling it cannot express avoidance, and the agent
        // would fuse with the thing it is supposed to be avoiding. That is the
        // common case, since an agent that has been committing to junctions is
        // by then pointed at one. Turn toward the weakest flank instead, at
        // the strength the sign-flipped term would have had.
        steer_to = worst_dir;
        frac = turn_frac * (commitment - 1.0);
    }

    var new_dir = mix(dir, steer_to, frac);
    // Stochastic steering, as an angular perturbation rather than a scalar
    // turn: a Gaussian nudge of the direction vector followed by
    // renormalisation is a rotation of roughly `jitter` radians, isotropic
    // about the heading.
    new_dir = new_dir + unit_gauss3(&seed) * params.jitter;
    if (length(new_dir) < 1e-5) {
        new_dir = dir;
    }
    new_dir = flatten(normalize(new_dir), params.depth_agent);

    var new_pos = pos + new_dir * params.speed;
    // Toroidal domain, on all three axes.
    new_pos = new_pos - floor(new_pos / fdims) * fdims;

    agent.recent = mix(agent.recent, strongest, 0.05);
    agent.age = agent.age + 1.0;

    // Respawn starved or elderly agents. Because a single deposit is far below
    // the visible threshold, a respawn cannot appear as an event on screen.
    let starving = agent.recent < params.starve_threshold;
    let elderly = agent.age > params.max_age;
    if (starving || elderly) {
        new_pos = vec3<f32>(rnd(&seed), rnd(&seed), rnd(&seed)) * fdims;
        if (rnd(&seed) < params.found_fraction) {
            // Integer division, so the epoch index is a counter and not a
            // clock: nothing here is ever a float function of the tick.
            let epoch = params.tick / max(params.found_period, 1u);
            let sites = max(
                (dims.x * dims.y * dims.z) / max(params.found_site_cells, 1u), 1u);
            let site = min(u32(rnd(&seed) * f32(sites)), sites - 1u);
            let centre = founding_site(epoch, site, fdims);
            // Gaussian scatter rather than a ball: a hard-edged cohort would
            // put a step in the trail field at its rim. Flattened like the
            // motion is, so a cohort is a patch in the plane rather than a
            // column through the slab it could not then find itself across.
            let scatter = unit_gauss3(&seed)
                * vec3<f32>(1.0, 1.0, clamp(params.depth_agent, 0.0, 1.0));
            new_pos = centre + scatter * params.found_radius;
            new_pos = new_pos - floor(new_pos / fdims) * fdims;
        }
        new_dir = flatten(
            normalize(unit_gauss3(&seed) + vec3<f32>(1e-4, 0.0, 0.0)),
            params.depth_agent);
        agent.age = 0.0;
        agent.recent = params.starve_threshold * 4.0;
    }

    agent.px = new_pos.x;
    agent.py = new_pos.y;
    agent.pz = new_pos.z;
    agent.dx = new_dir.x;
    agent.dy = new_dir.y;
    agent.dz = new_dir.z;
    agent.rng = seed;
    agents[index] = agent;

    splat(new_pos, dims, deposit_amt);
}
