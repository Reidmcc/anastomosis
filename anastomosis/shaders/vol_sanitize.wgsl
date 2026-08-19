// Periodic field sanitisation for the slab -- DESIGN.md §4.4.
//
// Every pass already NaN-guards its own output, so in normal operation this
// does nothing. It exists because the failure it prevents is unrecoverable and
// permanent: one non-finite voxel propagates through diffusion and destroys the
// entire field within seconds, and a session that has been running for two days
// has no way back. Running it every N ticks costs a fraction of a percent.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var src: texture_3d<f32>;
@group(0) @binding(2) var dst: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let p = vec3<i32>(gid);
    let value = finite_or4(textureLoad(src, p, 0), params.sanitize_fallback);
    let clamped = clamp(
        value,
        vec4<f32>(params.sanitize_min),
        vec4<f32>(params.sanitize_max),
    );
    textureStore(dst, p, clamped);
}
