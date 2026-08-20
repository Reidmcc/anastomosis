> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 10. Testing a thing like this

Conventional unit tests cover little of the risk here. The real QA is:

- **Flash-safety assertion.** Headless offscreen render; record per-frame max `ΔL`
  per pixel and the area fraction exceeding thresholds; assert the WCAG criterion of
  §7 is never met, across a sweep of extreme and adversarial parameter settings.
  This is the test that matters most.
- **Soak test.** Run the simulation headless at accelerated tick rate for a
  simulated 24–72 h; log field mass, variance, activity, agent survival, NaN count,
  luminance stats. Assert: no NaNs, no field death, no saturation, activity stays in
  band. Shortened version in CI, long version run manually.
- **Non-repetition check.** Autocorrelation of the field statistics time series over
  long lags — assert no periodic component above noise.
- **Numeric parity.** NumPy reference implementations of the RD step and the
  semi-Lagrangian advection, checked against the WGSL to a tolerance. Catches shader
  bugs that otherwise present only as "it looks a bit wrong".
- **Morphology check.** Feature count, characteristic length and hole count over a
  long run (`tests/morphology.py`, asserted in `tests/test_morphology.py`);
  assert the arrangement is *not* stationary, and that feature size is not
  uniform *across* the field — uniformity is what makes the texture a
  trypophobia trigger, so a churning field of identically-sized features would
  pass a non-stationarity check and still fail the requirement. This is the
  counterpart to the soak test: the soak test asserts the field is alive, and a
  field can be alive and yet look identical for hours (§4.7).

  Three things were added to this once §4.7 step 5 landed, and the split between
  them is the useful part. The characteristic length the controller is closed on
  is checked against the numpy reference in `test_parity.py`, along with the sign
  and size of the exponent relating it to `du` — a loop whose plant gain measured
  negative would be a positive feedback. The loop itself is asserted against a
  running engine in `test_soak.py`, paired by restoring one mature field into
  arms that differ only in what they ask for. And the shading stage gets its own
  assertion in `test_morphology.py`, because a field of soft varied bumps still
  reaches the eye as a lattice of hard round holes if the last stage before
  colour clips every one of them.
- **No-allocation check.** Assert steady-state buffer/texture count and process RSS
  are flat over a long run.

### 10.1 What the suite needs from an adapter

All of the above assumes the adapter under the tests behaves. One does not, and
it is the one you get by default on a machine with no GPU and no Vulkan driver:
wgpu falls back to OpenGL, and Mesa's GL backend binds a 3D storage texture as a
single non-layered slice, so every `textureStore` lands at `z = 0`. The slab
collapses onto its near face, the depth axis stops wrapping, `div(curl(psi))`
stops cancelling, and the threads that should have written different voxels race
for one — which also costs repeatability, so a checkpoint no longer restores to
the field it captured.

Most of the suite passes anyway, on a simulation that is no longer three
dimensional. Three tests notice — the divergence-free residual, the three-axis
wrap, and the checkpoint-resume comparison — and what they report is a number,
several layers from the cause. Read on its own, three unrelated-looking numeric
failures in an otherwise green run look like tests that have gone stale, and
they are not: they are the only ones telling the truth.

So the condition is asserted directly rather than inferred. `tests/gpucaps.py`
writes `z` into a small 3D texture and reads it back, sharing no code with the
shaders it protects; `gpu_device` skips the session with a sentence naming the
fix when that fails, and CI asserts the same thing as a hard failure so a run
that skipped everything cannot come back green. The fix is
`apt-get install mesa-vulkan-drivers`, which is what CI itself runs on.

One real bug came out of the same investigation, and it is not
adapter-specific: `wrap_texel` reduced with `((p % dims) + dims) % dims`, and
`%` on a negative operand is undefined in GLSL. Naga's GL backend lowers WGSL's
`i32` remainder straight onto it, so on that backend `-1 % 48` came back as 15
and the single step across the seam landed in the interior. Both wrap helpers
now lift into the non-negative range and reduce in `u32`, which is well defined
everywhere; Vulkan, Metal and DX12 were always right and pay only the shift.
