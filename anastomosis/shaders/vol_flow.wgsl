// Velocity field: the curl of the potential assembled in vol_potential.wgsl.
//
// Central differences, which is what makes the result *exactly*
// divergence-free rather than approximately so:
//
//   div(curl(A)) = DxDyAz - DxDzAy + DyDzAx - DyDxAz + DzDxAy - DzDyAx
//
// and the difference operators commute, so every term cancels against another
// term identically -- not to within a discretisation error, but to the last
// bit the arithmetic can carry. Pigment advected through this field is
// therefore neither compressed into bright accumulations nor drained into
// holes, which is the property that makes the motion read as fluid instead of
// as material appearing and disappearing.

//!include common3d.wgsl

//!struct SimParams

@group(0) @binding(0) var<storage, read> params: SimParams;
@group(0) @binding(1) var potential_tex: texture_3d<f32>;
@group(0) @binding(2) var velocity_out: texture_storage_3d<rgba16float, write>;

fn potential_at(p: vec3<i32>, idims: vec3<i32>) -> vec3<f32> {
    return textureLoad(potential_tex, wrap_texel3(p, idims), 0).xyz;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = vec3<u32>(params.dims_x, params.dims_y, params.dims_z);
    if (gid.x >= dims.x || gid.y >= dims.y || gid.z >= dims.z) {
        return;
    }

    let p = vec3<i32>(gid);
    let idims = vec3<i32>(dims);

    let px = potential_at(p + vec3<i32>(1, 0, 0), idims);
    let mx = potential_at(p - vec3<i32>(1, 0, 0), idims);
    let py = potential_at(p + vec3<i32>(0, 1, 0), idims);
    let my = potential_at(p - vec3<i32>(0, 1, 0), idims);
    let pz = potential_at(p + vec3<i32>(0, 0, 1), idims);
    let mz = potential_at(p - vec3<i32>(0, 0, 1), idims);

    var velocity = 0.5 * vec3<f32>(
        (py.z - my.z) - (pz.y - mz.y),
        (pz.x - mz.x) - (px.z - mx.z),
        (px.y - mx.y) - (py.x - my.x),
    );

    // Clamped to a few voxels per tick: semi-Lagrangian advection is stable at
    // any step, but very long steps would sample uncorrelated regions and read
    // as tearing rather than as flow.
    velocity = clamp(
        finite_or4(vec4<f32>(velocity, 0.0), 0.0).xyz,
        vec3<f32>(-8.0), vec3<f32>(8.0));

    textureStore(velocity_out, p, vec4<f32>(velocity, 1.0));
}
