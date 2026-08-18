// Physarum agents -- the filament layer, and the source of anastomosis.
//
// Two departures from textbook Physarum, both required by the brief:
//
//   * Deposits are bilinear splats, never single-texel writes. A hard write is
//     a one-pixel step change, i.e. exactly the visual punctuation the
//     application must never produce. Deposit magnitude is also kept far below
//     the trail decay rate, so no individual agent event is ever visible --
//     structure emerges only from thousands of reinforcements.
//
//   * A fusion bias: when an agent senses a trail much stronger than what it
//     has been laying down itself, it *reduces* its turn rate and commits to
//     the junction rather than glancing off. This is what turns a tangle into a
//     network with genuine anastomoses.
//
// Two additions from DESIGN.md 4.7 step 4, both of which exist because the
// layer was topologically one-way -- it could only ever add edges and mass to
// the structure it already had:
//
//   * Anti-fusion. The commitment clamp now reaches past 1.0, where
//     `turn * (1 - commitment)` changes sign and the agent veers *away* from
//     what it sensed. Driven per region from the climate `repel` channel, so
//     there are migrating zones where the network comes apart while it fuses
//     everywhere else, and reachable by a rift event.
//
//   * Founding respawn. A respawned agent used to land alone on uniform random
//     ground, where one agent's deposit is orders of magnitude below what can
//     hold against decay -- so respawn could never found anything and every
//     new structure had to grow off an old one. Respawns now mostly land
//     together at a shared site on bare ground. §4.7 wants this as the thing
//     that would turn flux pruning from concentration into turnover --
//     material resorbed here, reinvested there -- though measurement has not
//     yet shown that it does; pruning stays off.

//!include common.wgsl

//!struct SimParams
//!struct Stats

struct Agent {
    pos: vec2<f32>,
    heading: f32,
    rng: u32,
    recent: f32,
    age: f32,
};

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(2) var trail_tex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> deposit_buf: array<atomic<u32>>;
@group(0) @binding(4) var clim_a: texture_2d<f32>;
@group(0) @binding(5) var clim_b: texture_2d<f32>;
@group(0) @binding(6) var samp: sampler;
@group(0) @binding(7) var<storage, read> stats: Stats;
@group(0) @binding(8) var clim_c: texture_2d<f32>;

// Deposits accumulate in fixed point because floating-point atomics are not
// available in core WebGPU. 2^20 gives ample headroom: a heavily trafficked
// texel accumulating a thousand agents at the maximum deposit rate reaches
// ~3e7, well inside u32.
const DEPOSIT_SCALE: f32 = 1048576.0;

fn sample_trail(uv: vec2<f32>) -> f32 {
    return textureSampleLevel(trail_tex, samp, wrap_uv(uv), 0.0).r;
}

fn splat(pos: vec2<f32>, dims: vec2<u32>, amount: f32) {
    // Bilinear splat across the four surrounding texels.
    let base = floor(pos - 0.5);
    let f = pos - 0.5 - base;
    let w = array<f32, 4>(
        (1.0 - f.x) * (1.0 - f.y),
        f.x * (1.0 - f.y),
        (1.0 - f.x) * f.y,
        f.x * f.y,
    );
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>(0, 0), vec2<i32>(1, 0), vec2<i32>(0, 1), vec2<i32>(1, 1),
    );
    let idims = vec2<i32>(dims);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let t = wrap_texel(vec2<i32>(base) + offsets[i], idims);
        let index = u32(t.y) * dims.x + u32(t.x);
        let quantum = u32(max(amount * w[i] * DEPOSIT_SCALE, 0.0));
        if (quantum > 0u) {
            atomicAdd(&deposit_buf[index], quantum);
        }
    }
}

// Where this epoch's founding cohort lands.
//
// Every agent respawning into the same (epoch, site) computes the same answer
// from the same hash, which is what makes a cohort out of what is otherwise a
// stream of unrelated events -- there is no communication between invocations
// and none is needed. The number of sites scales with the field's area, so the
// arrival rate *per site* is the same on a 128-cell test layer as on a 1440p
// one; a single global site would concentrate a hundred times more traffic on
// the large layer and land as a visible flare rather than a founding.
//
// The barest of four candidates, because bare ground is the entire point: a
// cohort landing on the existing network only reinforces it, which is what
// uniform respawn already did. The four candidates are fixed for the epoch but
// the trail under them is not, so a site that fills up can hand off to another
// mid-epoch. That is wanted -- once something is established there, the next
// arrivals are better spent elsewhere.
fn founding_site(epoch: u32, site: u32, fdims: vec2<f32>) -> vec2<f32> {
    var seed = pcg3(epoch, site, params.seed);
    var best = vec2<f32>(rnd(&seed), rnd(&seed)) * fdims;
    var bare = sample_trail(best / fdims);
    for (var i = 0u; i < 3u; i = i + 1u) {
        let candidate = vec2<f32>(rnd(&seed), rnd(&seed)) * fdims;
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

    let dims = vec2<u32>(params.dims_x, params.dims_y);
    let fdims = vec2<f32>(dims);
    var agent = agents[index];

    // Recover from a corrupted agent rather than letting it wander off into
    // NaN and stop depositing forever.
    agent.pos = finite_or2(agent.pos, 0.0);
    agent.heading = finite_or(agent.heading, 0.0);

    var seed = agent.rng ^ pcg2(params.tick, params.seed);
    let uv = agent.pos / fdims;

    // Local parameters from the climate field.
    let ca = textureSampleLevel(clim_a, samp, wrap_uv(uv), 0.0);
    let cb = textureSampleLevel(clim_b, samp, wrap_uv(uv), 0.0);
    let cc = textureSampleLevel(clim_c, samp, wrap_uv(uv), 0.0);

    let sensor_angle = max(0.02, params.sensor_angle + params.range_sensor_angle * ca.z);
    let sensor_dist = max(1.0, params.sensor_distance + params.range_sensor_distance * ca.w);
    // Flux pruning removes a fraction of the trail field's throughput from
    // strands that traffic has abandoned (trail.wgsl). It is handed back here,
    // scaled onto the deposit, so pruning redistributes rather than drains: the
    // mass reappears wherever agents currently are instead of wherever mass
    // already is. Returning it through decay instead -- the obvious centring --
    // reinforces established structure and freezes the field; see trail.wgsl.
    let deposit_amt = max(0.0, params.deposit + params.range_deposit * cb.x + stats.corr_deposit)
        * (1.0 + clamp(stats.prune_return, 0.0, 2.0));

    // Sense: three points ahead, bilinear so there is no texel-grid bias.
    let heading = agent.heading;
    let ahead = vec2<f32>(cos(heading), sin(heading));
    let left_dir = vec2<f32>(cos(heading + sensor_angle), sin(heading + sensor_angle));
    let right_dir = vec2<f32>(cos(heading - sensor_angle), sin(heading - sensor_angle));

    let s_f = sample_trail((agent.pos + ahead * sensor_dist) / fdims);
    let s_l = sample_trail((agent.pos + left_dir * sensor_dist) / fdims);
    let s_r = sample_trail((agent.pos + right_dir * sensor_dist) / fdims);

    var turn = 0.0;
    if (s_f >= s_l && s_f >= s_r) {
        turn = 0.0;
    } else if (s_l > s_r) {
        turn = params.turn_rate;
    } else if (s_r > s_l) {
        turn = -params.turn_rate;
    } else {
        turn = params.turn_rate * rnd_signed(&seed);
    }

    // Fusion bias. `recent` is the agent's own running sense of ambient trail
    // strength; encountering something much stronger means another filament,
    // so commit to it instead of turning away.
    let best = max(s_f, max(s_l, s_r));
    let excess = best / (agent.recent + 1e-4) - 1.0;

    // Anti-fusion. The clamp reaches past 1.0, and the sign of the steering
    // term is what changes there: below 1 the agent turns toward the filament
    // and follows it, at 1 the turn is cancelled and it drives straight
    // through the junction and fuses, above 1 it turns away. So the axis is
    // one of *deflection toward the junction*, running through fusion rather
    // than peaking at it -- which is why the climate channel is named for the
    // repulsion it buys rather than for the fusion it passes through.
    //
    // Additive, not geometric like `range_du`: what matters here is the fixed
    // crossing point at 1.0, so the fraction of the field that is repelling
    // can be read straight off the climate amplitude.
    let junction = params.fusion_bias + params.range_repel * cc.z;
    let commitment = clamp(
        junction * smoothstep(0.0, 2.0, excess), 0.0, params.fusion_max);
    turn = turn * (1.0 - commitment);
    if (commitment > 1.0 && s_f >= s_l && s_f >= s_r) {
        // Head-on, the steering term is already zero, so scaling it cannot
        // express avoidance and the agent would fuse with the thing it is
        // supposed to be avoiding -- which is the common case, since an agent
        // that has been committing to junctions is by then pointed at one.
        // Turn toward the weaker flank instead, at the same strength the
        // sign-flipped term would have had.
        turn = turn + params.turn_rate * (commitment - 1.0)
            * select(-1.0, 1.0, s_l < s_r);
    }

    turn = turn + params.jitter * rnd_signed(&seed);

    let new_heading = heading + turn;
    let step = vec2<f32>(cos(new_heading), sin(new_heading)) * params.speed;
    var new_pos = agent.pos + step;
    // Toroidal domain.
    new_pos = new_pos - floor(new_pos / fdims) * fdims;

    agent.recent = mix(agent.recent, best, 0.05);
    agent.age = agent.age + 1.0;

    // Respawn starved or elderly agents. Because a single deposit is far below
    // the visible threshold, a respawn cannot appear as an event on screen.
    let starving = agent.recent < params.starve_threshold;
    let elderly = agent.age > params.max_age;
    if (starving || elderly) {
        new_pos = vec2<f32>(rnd(&seed), rnd(&seed)) * fdims;
        if (rnd(&seed) < params.found_fraction) {
            // Integer division, so the epoch index is a counter and not a
            // clock: nothing here is ever a float function of the tick.
            let epoch = params.tick / max(params.found_period, 1u);
            let sites = max(
                (dims.x * dims.y) / max(params.found_site_cells, 1u), 1u);
            let site = min(u32(rnd(&seed) * f32(sites)), sites - 1u);
            let centre = founding_site(epoch, site, fdims);
            // Gaussian scatter rather than a disc: a hard-edged cohort would
            // put a step in the trail field at its rim.
            new_pos = centre
                + vec2<f32>(gauss(&seed), gauss(&seed)) * params.found_radius;
            new_pos = new_pos - floor(new_pos / fdims) * fdims;
        }
        agent.heading = rnd(&seed) * TAU;
        agent.age = 0.0;
        agent.recent = params.starve_threshold * 4.0;
    } else {
        agent.heading = new_heading;
    }

    agent.pos = new_pos;
    agent.rng = seed;
    agents[index] = agent;

    splat(new_pos, dims, deposit_amt);
}
