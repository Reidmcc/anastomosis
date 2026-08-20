// Semi-Lagrangian advection of the volumetric trail -- the slab's twin of
// trail_advect.wgsl, and the same reasoning throughout: its own pass after
// the velocity is written (same-tick read keeps the checkpoint honest),
// skipped entirely at gain zero, all four channels carried together. The
// velocity here already carries the depth anisotropy of DESIGN.md §5.1, so
// the trail drifts laterally faster than it churns through the thickness,
// exactly as the pigment does.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var trail_in: texture_3d<f32>;
@group(0) @binding(2) var trail_out: texture_storage_3d<rgba16float, write>;
@group(0) @binding(3) var velocity_tex: texture_3d<f32>;
@group(0) @binding(4) var samp: sampler;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let fdims = vec3<f32>(dims);
    let uvw = (vec3<f32>(gid) + 0.5) / fdims;

    let velocity = textureLoad(velocity_tex, vec3<i32>(gid), 0).xyz;
    let src = wrap_uvw(
        uvw - velocity * params.trail_advect * params.advect_dt / fdims);
    let carried = finite_or4(textureSampleLevel(trail_in, samp, src, 0.0), 0.0);

    textureStore(trail_out, vec3<i32>(gid), vec4<f32>(
        clamp(carried.x, 0.0, 8.0),
        clamp(carried.y, 0.0, 8.0),
        clamp(carried.z, 0.0, 8.0),
        1.0,
    ));
}
