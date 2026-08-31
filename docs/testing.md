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
- **Adapters run different fields, and long tests must expect it.** Integer
  arithmetic can be made identical across drivers (§4.8's postscript: after the
  signed-modulo miscompile, by construction). Float arithmetic cannot: f16
  store rounding, FMA contraction and transcendental precision are all legal
  degrees of freedom, and the dynamics are chaotic, so those ulp-scale
  differences compound until two adapters running the same seed hold entirely
  decorrelated fields — measured, the rift test's fixed seed put 0.59 of trail
  under its disc on lavapipe and 0.18 on an RTX 3080 by tick 800. The
  consequence for tests: anything calibrated against one adapter's
  *trajectory* — a fixed seed's ground, the magnitude of a peak-to-median
  ratio on a nearly-bare field — is a property of that adapter, not of the
  code. Long-horizon tests assert statistics that hold across trajectories,
  set thresholds inside the measured gap between healthy and broken rather
  than at either cluster's edge, and measure their preconditions at runtime
  instead of trusting a seed to reproduce them (the rift test's ground scan).
- **Core-limits check** (`test_integrated.py`, §8.3). Every binding, workgroup
  shape and byte of workgroup memory counted out of the WGSL and asserted
  against core WebGPU's guaranteed minima. This one is unusual in that it
  cannot be written against a device at all: the property it protects — that
  the application builds on an adapter reporting only the specification's
  minima — is invisible from every adapter available to anyone working on it,
  since the software adapter and every discrete card report far higher. A
  single extra storage texture in a single pass would stop it opening on
  integrated GPUs and pass every other test in this file.
