> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

### 4.8 The wrap seam — an edge in a domain that has none

Reported from a real viewing: line-like vertical and horizontal structures near
the edges of the window. Not display artefacts — they split, rejoin and
partially disappear, and they behave like part of the field. Enlarging the
window moved them *further in*, as though the image had zoomed out. The
reporter's reading was that the agents were treating the window edge as an edge
of the environment.

The agents were not, and there is no edge in the simulation to treat as one:
every pass wraps, `wrap_uv` and `wrap_texel` are used consistently, and the
sampler's address mode is `repeat`. But the diagnosis was the right shape. There
*was* an edge, one layer down from the one suspected — in the noise that drives
the flow, not in the domain.

**Mechanism.** `psi.wgsl` forces the vector potential with an OU increment drawn
from coarse value noise, sampled at `uv * psi_noise_scale`. That noise had no
period on the domain: `uv = 0` and `uv = 1` land on unrelated lattice cells, and
the octave multipliers (2.03, 4.11) put the finer octaves out of phase with any
period at all. So every tick injected a discontinuity along `u = 0` and
`v = 0`.

A discontinuous *forcing* would be harmless if it were transient. It is not,
because psi integrates:

- psi's own diffusion keeps it continuous across the seam, so what accumulates
  is not a jump but a **kink** — a permanent line of steep gradient;
- velocity is `curl(psi)`, i.e. a derivative, so the kink becomes a **jet**.
  Measured on the shipped configuration at a 320×180 psi grid, the mean
  `|∂psi/∂x|` across the seam was **9.0×** the interior value, and
  `|∂psi/∂y|` **7.5×**;
- everything downstream rides that velocity field. Pigment is advected through
  it (`advect.wgsl`), the climate field is advected through it
  (`climate.wgsl`), and the agents follow the structure the other two leave. A
  standing line of ~9× shear draws material out along itself, which is exactly
  the reported behaviour: lines that are part of the simulation, that split and
  rejoin as the flow either side of them changes.

**Why near the edges rather than at them.** The seam is at the *field's* `u = 0`,
which need not be at the window's. §5's compositor samples each layer at
`(uv − 0.5) · scale + 0.5`, where `scale` carries two factors: the depth term
(`1 + 0.06·depth`, so back layers read as further away) and the aspect
correction (`engine.aspect_correction`, which samples a wider area rather than
stretching the field when the window is reshaped). Both push `scale` above 1,
and the seam then lands at `0.5 − 0.5/scale` of the way in from each edge.
A back layer is 2.8% in at any window shape; a 1280×720 field shown in a
2560×1080 window puts the front layer's seam 12.5% in. Resize the window and
the lines move inward — the "zoomed out" reading was literally correct, on one
axis, by the aspect ratio.

That the compositor deliberately samples outside `[0, 1]` is not the bug. It is
allowed to, because the domain is a torus; the bug is that the torus had a
visible join, and the compositor's job was to bring it into view.

What the compositor does *not* stop being is a tiler. With the join invisible
the wrap is now seamless, but at a large aspect mismatch the strip it wraps in
is still the same material seen twice, on opposite sides of the image — 33% of
the width duplicated in a 1280×720 field shown at 2560×1080. That is inherent to
covering a reshaped window from a fixed toroidal field, and the remedy is the
one the README already gives: launch at the size you mean to run at, rather
than resizing a long way into it.

**Fix.** The noise tiles. `value_noise_tiled` wraps the lattice index at a
period, so the field it produces closes on the domain exactly — the lattice row
either side of the join is the same row. Tiling requires integer frequency
ratios between octaves, so the octaves are at 1×, 2× and 4× the base period
rather than 1×, 2.03× and 4.11×; what the non-integer ratios were avoiding — the
three lattices sharing grid lines, which leaves a square signature — is bought
back with a constant offset per octave, which cannot affect a period.

Two second-order details, both measured rather than assumed:

- `psi_noise_scale` is continuous (it is what the `scale` macro drives) and
  tiling needs whole cells, so psi crossfades the two bracketing periods. The
  weights are normalised **in quadrature**, not to one, because the two fields
  are incoherent: linear weights thin the forcing by 30% halfway between
  periods, which would move flow speed from a knob that is only supposed to
  move feature size.
- The two periods must also be **independent**. `hash_grid` does not take the
  period as an input, so neighbouring periods walk overlapping lattice indices
  and draw the same lattice values; sharing a stream left them correlated and
  the quadrature normalisation then overshot, making the forcing 15% *strong*
  mid-crossfade. Folding the period into the stream fixes it: measured RMS is
  flat to within 8% across `psi_noise_scale` 2 → 5, matching the untiled
  baseline's own trend.

After the fix the same measurement gives 1.3× and 0.65× rather than 9.0× and
7.5×, and the per-column profile across the seam is flat. The crossfade doubles the noise
evaluations in `psi.wgsl`, which does not register against §8.1: psi runs at a
quarter of the simulation's linear resolution, so the whole pass is ~0.3 M
texels across all layers.

**This does not weaken §3.** The tiling is spatial. The noise is still re-drawn
every tick from a hash of `(lattice cell, tick, seed)`, so the increment is as
white in time as it ever was, and it is still only an increment to a stored
field. What §3 rules out is a *temporal* period; a spatial one is the shape of
the domain.

**Tests.** `tests/test_seam.py`, in three layers: that the noise closes on the
domain to floating-point rounding (~1e-15 against texel-to-texel steps of
~1e-2), including through the crossfade at non-integer scales; that psi driven
by it does not bend at the seam, with the pre-fix forcing run through the same
integrator as a positive control, so the measurement is known to be able to see
the defect; and the same on the GPU, against `psi.wgsl` itself, which is also
checked for parity with the numpy port. The measurement is curvature rather than
gradient, because the artefact is a kink and the first difference is dominated
by the field's own slope. Pre-fix, the GPU test reports 21× and 8.7×; post-fix,
under 1.5×.
