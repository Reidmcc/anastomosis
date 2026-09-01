// Wiping yesterday's painter's order -- §18 round 5, pass 0 of the tick.
//
// The ownership layer is derived state: every tick the bodies re-claim
// their own ground (things_deposit.wgsl), so it starts from nobody. Its
// own tiny module because auto pipeline layouts carry only the bindings
// an entry point actually uses, and this one uses almost nothing.

//!include things_common.wgsl

@group(0) @binding(0) var<storage, read> params: ThingsParams;
@group(0) @binding(1) var<storage, read_write> owner: array<atomic<u32>>;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.dims_x * params.dims_y) {
        return;
    }
    atomicStore(&owner[idx], 0u);
}
