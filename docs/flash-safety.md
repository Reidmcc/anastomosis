> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 7. The flash-safety stage (non-negotiable, enforced by construction)

Applied to the final composited image every frame, after all colour work:

1. **Motion-compensated reprojection.** The previous output frame is reprojected
   through the velocity field before comparison. Without this, a slew limiter smears
   any structure that translates across the screen; with it, the limiter only sees
   genuine change and leaves motion alone. This reuses the velocity field the
   simulation already computes, so it is nearly free.
2. **Per-pixel slew limit in Oklab.** `ΔL` per frame is hard-clamped to
   `max_luma_delta` (default **0.01**, i.e. 1% of range). `Δa`, `Δb` clamped more
   loosely (chroma change is far less provocative than luminance change). All
   perceptual ceilings (`l_max`, `c_max`) are applied to the *target*, never to the
   result — clamping the result would mean that lowering a ceiling mid-session
   forces an immediate unbounded jump on every pixel above it.
3. **Exposure governor.** Mean and 95th-percentile luminance are computed by mip
   reduction; global exposure is corrected with asymmetric slow attack/release so
   the overall level is stable and cannot drift bright.
4. **Temporal IIR** `out = mix(prev, new, α)`, α ≈ 0.2, as a final backstop.

### Exactly what is guaranteed

> Per pixel, `|ΔL|` per frame is bounded by `max_luma_delta`, measured against the
> **motion-compensated** previous frame.

The reprojection qualifier is load-bearing and worth stating plainly. At a *fixed*
screen pixel, a filament translating past can produce a larger change than the
limit. That is honest motion, not a flash, and suppressing it would be the wrong
behaviour — it is precisely what the reprojection exists to permit. So the safety
argument rests on two separate claims, and the test suite asserts them separately:

1. **The limiter is exact.** With flow disabled, reprojection is the identity and
   the per-pixel bound holds to within f16 quantisation (measured: 0.010043 against
   a 0.010 limit) — including under adversarial parameter stepping.
2. **Large correlated change is impossible.** WCAG 2.3.1 / the PEAT general flash
   threshold defines a flash as a pair of opposing relative-luminance changes of
   ≥10% covering >25% of the screen, at ≥3 per second. The suite asserts that fewer
   than 25% of pixels ever change by ≥10% in a single frame, so the area criterion
   cannot be met regardless of timing. Mean screen lightness is held to the
   per-frame limit as well.

### Why the default is 1%, and why the ceiling is 1.2%

At `ΔL ≤ 0.01` and 30 FPS, a 10% excursion requires ≥10 frames = **333 ms** and a
full opposing pair ≥667 ms — **1.5 flashes/second**, half the threshold, with no
assumption whatsoever about the simulation's behaviour. The bound holds if the
reaction blows up, if a parameter is set absurdly, if an upstream shader has a bug.

The user-settable ceiling is **0.012** (1.8 flashes/s). An earlier draft of this
document specified 0.03 and claimed it still yielded ≤2 pairs/s. That was an
arithmetic error: 0.03 permits 4.5 flashes/second, *above* the WCAG limit rather
than below it. The test that encodes this criterion caught it, which is the entire
reason for expressing the property numerically instead of describing it in prose.

### One non-obvious implementation constraint

The safety stage stores its output and reads it back as the next frame's history,
so **gamut mapping must not let out-of-range values into the buffer**. Allowing a
component through with the usual ~1e-3 tolerance means it gets clamped on a *later*
frame — enlarging that frame's step after the limiter has already bounded it. Near
black this is not a rounding detail: a 5×10⁻⁴ change in one channel moves Oklab `L`
by ~1.6×10⁻³, a sixth of the entire per-frame budget. Measured, this leaked the
per-pixel bound to 0.0161 against a 0.010 limit until the tolerance was tightened
and both gamut-mapping paths were made to clamp.
