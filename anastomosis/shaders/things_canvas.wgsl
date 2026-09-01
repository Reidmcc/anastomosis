// The canvas field: fade, then drink the tick's deposits -- §18.3 pass 3.
//
// The founding file's one mechanism, kept whole: every frame the canvas
// was covered with 5% black and everything was drawn on top, so the image
// and the trail were the same object and every village stood on the ghost
// of everywhere it had wandered. Here the fade is exponential per tick
// (rate-converted, so the tick rate cannot change how long a ghost lives)
// and the drawing arrives through the atomic accumulator, drained by
// exchange so it is empty between ticks -- the rhizotron's pattern.

//!include things_common.wgsl

@group(0) @binding(0) var<storage, read> params: ThingsParams;
@group(0) @binding(1) var canvas_in: texture_2d<f32>;
@group(0) @binding(2) var canvas_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<storage, read_write> deposit: array<atomic<u32>>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.dims_x || gid.y >= params.dims_y) {
        return;
    }
    let p = vec2<i32>(gid.xy);
    let idx = (gid.y * params.dims_x + gid.x) * 4u;

    let drained = vec3<f32>(
        f32(atomicExchange(&deposit[idx + 0u], 0u)),
        f32(atomicExchange(&deposit[idx + 1u], 0u)),
        f32(atomicExchange(&deposit[idx + 2u], 0u)),
    ) / DEPOSIT_SCALE;

    let old = finite_or4(textureLoad(canvas_in, p, 0), 0.0);
    let faded = max(old.rgb, vec3<f32>(0.0)) * (1.0 - params.fade);
    let value = clamp(faded + drained, vec3<f32>(0.0), vec3<f32>(64.0));

    // The breath layer (§18.1 soul 7, incarnated properly). The founding
    // wander-shadows were an 8-bit quantisation floor -- the fade stalled
    // below ~10/255 and every village stood permanently on the ghost of
    // everywhere it had been. Here the rounding error is a mechanism: the
    // alpha channel max-tracks the canvas's own brightness -- which is
    // rate-invariant, so the breath is too -- and lets go on a clock of
    // minutes rather than seconds. Breath on glass, with a memory.
    let lit = max(max(value.r, value.g), value.b);
    let ghost = clamp(
        max(max(old.a, 0.0) * (1.0 - params.ghost_fade),
            lit * params.ghost_gain),
        0.0, 1.0);

    textureStore(canvas_out, p, vec4<f32>(value, ghost));
}
