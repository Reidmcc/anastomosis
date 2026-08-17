// Periodic field sanitisation -- DESIGN.md §4.4.
//
// Every pass already NaN-guards its own output, so in normal operation this
// does nothing. It exists because the failure it prevents is unrecoverable and
// permanent: one non-finite texel propagates through diffusion and destroys the
// entire field within seconds, and a session that has been running for two days
// has no way back. Running it every N ticks costs a fraction of a percent.

//!include common.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec2<u32>(params.dims_x, params.dims_y);
    if (gid.x >= dims.x || gid.y >= dims.y) {
        return;
    }

    let p = vec2<i32>(gid.xy);
    let value = finite_or4(textureLoad(src, p, 0), params.sanitize_fallback);
    let clamped = clamp(
        value,
        vec4<f32>(params.sanitize_min),
        vec4<f32>(params.sanitize_max),
    );
    textureStore(dst, p, clamped);
}
