# Anastomosis — Design

A long-running generative visual field for self-regulation and stimming. Built on
`wgpu-py` + WGSL. Designed to run for days on a secondary display while the machine
is used for other work.

The name comes from hyphal anastomosis: the fusion of fungal filaments into a
network. That is the target behaviour — filaments that grow, seek, touch, and fuse,
inside a slowly breathing medium.

---

## 1. Design constraints, restated as engineering requirements

| Stated need | Engineering requirement |
|---|---|
| Fluid continuous motion with depth | Velocity-field advection (not just cellular update); multi-layer composite with parallax + depth attenuation |
| No visual punctuation or flashing | **Hard** per-pixel slew limit on the final image, motion-compensated; exposure governor; no thresholds anywhere in shading |
| Unpredictable, never loops | All slow variation driven by *stateful random walks*, never by a function of wall-clock time; counter-based PRNG with unbounded period |
| Slow, reactive colour change | Colour derived from simulation state in Oklab, with a drifting palette anchor; heavy temporal lowpass |
| Cap 30 FPS | `rendercanvas` `update_mode="continuous", max_fps=30`, vsync on |
| Leave GPU headroom | Sim decoupled from render (sim ~15 Hz, render 30 Hz, motion-compensated interpolation); sim at fraction of display resolution; explicit frame budget governor |
| Adjustable parameters | TOML config as source of truth, hot-reloaded; ~6 macro knobs over ~40 primitives; presets |

**Target hardware:** RTX 3080, 2560×1440. Sized in §8.1; 4K is explicitly not a
requirement, which is what makes a native-resolution front layer affordable.

Two requirements dominate everything else and deserve to be called out before the
architecture, because they are the ones that are *hard*:

**(a) Not looping is easy. Not settling is hard.** Reaction-diffusion, Physarum, and
Lenia all have attractors. Left alone, every one of them either dies, saturates, or
reaches a quasi-static texture within minutes to hours. An application that must be
interesting for eight hours cannot rely on the simulation's own dynamics. The
architectural answer is §4: the governing parameters are themselves a slowly
drifting spatial field, so the system is never solving the same equation twice.

**(b) "No flashing" is a safety property, not a style.** It should be *enforced by
construction at the output stage*, not merely avoided by taste in the simulation
stage. A parameter regime nobody tested, a numerical blow-up, a NaN — any of these
could otherwise produce exactly the thing the application must never do. §7 makes
it a bounded, testable invariant that holds regardless of what the simulation does.

---

## 2. Substrate: a three-system hybrid

None of the three named systems alone hits the brief.

- **Physarum** gives literal anastomosis — filaments that seek and fuse — but its
  agent deposits are point-like and produce fine-grained shimmer (visual
  punctuation), and its networks stabilise or die.
- **Reaction–diffusion** gives organic texture and self-maintaining structure, but
  crawls rather than flows, and Gray–Scott settles into a steady state across most
  of its parameter space.
- **Lenia** gives beautiful smooth morphology, but its interesting regimes are
  narrow and metastable — it dies or explodes on long horizons.

The design uses each for what it is good at, in a stack of coupled fields:

```
climate field  (64×36, very slow)     ── governs every parameter below, per-region
      │
      ├─► agents (Physarum)           ── filament seeking, fusion, network topology
      │        │ soft deposit
      │        ▼
      ├─► trail field  T              ── hyphal density
      │        │ feeds
      │        ▼
      ├─► reaction field (U,V)        ── Gray–Scott-ish, gives texture *within* filaments
      │        │
      │        ▼
      └─► velocity field  v = ∇×ψ     ── incompressible flow; ψ from climate + blurred field
               │ advects
               ▼
           pigment field  P            ── what is actually shaded; carries colour history
```

The key structural choice is the **pigment field advected by a divergence-free
velocity field**. This is what produces "fluid continuous motion" as opposed to the
crawling, twitchy quality that raw RD and raw Physarum both have. Structures are
*carried* rather than recomputed. Because `v = curl(ψ)` it is incompressible by
construction, so pigment neither piles up nor drains — no bright accumulation
spots, no washing out. Semi-Lagrangian advection with bilinear (`textureSampleLevel`
works in compute shaders) is unconditionally stable at any timestep, which matters
for a process that must never blow up.

`ψ = a·curl_noise(climate) + b·blur(V)` — so the flow is partly imposed weather and
partly the structure's own field pushing itself around. That feedback is a
significant source of the non-predictability in requirement (a).

### Anastomosis specifically

Fusion is an emergent property of Physarum sensing, but it can be encouraged
explicitly, which makes the visual signature much stronger:

- Agents sense `T` at three points ahead; standard.
- Add a **fusion bias**: when the sensed value exceeds the agent's own recent
  deposit history, reduce the turn angle sharply (the filament commits to the
  junction rather than glancing off). Cheap, and it is what turns a tangle into a
  network.
- Agent deposits are **soft splats** (a small Gaussian, or bilinear-weighted into 4
  texels), never a single-texel write. This is a flashing-safety measure as much as
  an aesthetic one: a hard write is a one-pixel step change.
- Deposit magnitude is kept well below the field's decay rate per tick, so no single
  agent event is individually visible. Structure emerges from thousands of
  reinforcements, which is inherently gradual.

---

## 3. Never loops, and never *can* loop

The naive approach — `noise(x, y, t)` for slow variation — is wrong here for two
reasons. It is periodic in practice (any tileable noise repeats; any non-tileable
one drifts into float precision loss), and at `t = 86400 s` an `f32` has ~0.008
resolution, so after one day the animation quantises visibly.

Instead, **every slow-varying quantity is a stateful process, integrated forward**:

- **Ornstein–Uhlenbeck random walk** for each global scalar:
  `x ← x + θ(μ − x)·dt + σ·√dt·N(0,1)`, computed on-GPU in a single-workgroup pass.
  Mean-reverting, so it stays in a sane band; aperiodic by construction; bounded
  variance; no dependence on absolute time.
- The **climate field** (§4) is itself advected and diffused each tick, so it is a
  stateful PDE, not a function of `t`.
- Randomness comes from a **counter-based PRNG** (PCG-family hash of
  `(pixel_id, frame_counter, stream_id)`) seeded from OS entropy at launch. The
  counter is `u64` split across two `u32`s; at 30 Hz the period exceeds the age of
  the universe.
- `frame_counter` is a `u32`/`u64` integer, never a float, and is used only as hash
  input — never as a phase. Nothing anywhere is `sin(t)`.

The consequence is stronger than "does not loop": there is no periodic component in
the system at all, and no state that recurs, because the state space is being
explored by a diffusion process rather than traversed by a trajectory.

---

## 4. Homeostasis and the climate field — the long-duration core

This is the part that decides whether the application is good for ten minutes or for
ten hours.

### 4.1 Climate field

A small texture (64×36 per layer) whose channels are the *local* values of the
simulation's governing parameters: feed rate, kill rate, agent sensor angle, sensor
distance, deposit rate, decay rate, flow strength, hue anchor. Each tick it is:

1. advected by its own very slow flow field,
2. diffused slightly,
3. perturbed by a small OU noise increment.

Effects, all of which serve requirement (a):

- **Different regions of the screen are in different regimes at the same time** —
  one area making dense network, another dissolving into wisps, another nearly
  still. This alone removes most of the "same-y" quality that kills these
  simulations over long viewing.
- **Regimes migrate.** Because the climate is advected, a region's character
  arrives from elsewhere and moves on. The viewer never gets a stable mental model
  of "this corner does X".
- **Regime boundaries are where the most interesting structure forms** — filaments
  growing from a productive zone into a dissolving one.

The climate field is sampled with bilinear interpolation and is 40× lower resolution
than the sim, so it is essentially free (2.3k texels) and inherently smooth — it can
never introduce a hard edge.

### 4.2 Homeostat

A PI controller, running **entirely on the GPU** (a reduction pass writes to a small
storage buffer; the climate pass reads it next tick — no CPU readback, no pipeline
stall). It measures, per tick:

- total field mass `Σ V`
- field variance (proxy for "is there structure, or is it flat mush?")
- mean activity `Σ|∂V/∂t|` (proxy for "is anything still happening?")
- live agent fraction

and gently corrects the climate field's *mean* parameters to hold each measure
inside a band. Critically:

- **A band, not a setpoint.** A tight controller makes the output feel regulated and
  monotonous — it actively fights the variety we want. Deadband is wide (±30%).
- **Long time constant** (minutes, τ ≈ 120 s). The controller must be far slower
  than anything visible, or it becomes a source of coordinated global change, which
  is precisely the "punctuation" the brief forbids.
- Corrections are applied to the *mean* the OU process reverts toward, not to the
  state directly, so the controller can never cause a step.

Without this, the system dies or saturates on a timescale of hours. With it, the
intended behaviour is indefinite.

### 4.3 Slow events

Statistical stationarity is its own kind of predictability. After an hour, a
perfectly homeostatic system is boring even though it never repeats — the *texture*
of change becomes known. So: a Poisson-arrival scheduler (mean inter-arrival ~8
minutes, tunable) triggers localised, long-enveloped perturbations — a nutrient
bloom, a slow die-back, a shift in flow direction across one region.

Every event is constrained to be non-punctuating:

- raised-cosine envelope, 30–180 s attack and release, never a step;
- spatially localised with a smooth radial falloff, and capped at ~25% of screen
  area (this is also the WCAG flash-area threshold — see §7);
- applied to *climate*, never directly to pigment or luminance, so its effect
  reaches the image only through several stages of diffusion and lowpass.

### 4.4 Numerical survival

Long runs fail in specific, known ways. Each gets an explicit countermeasure:

- **NaN quarantine.** A single NaN propagates through diffusion and kills the whole
  field within seconds. A sanitise pass (`select(fallback, x, isfinite(x))` plus a
  clamp) runs every 60 ticks on every field. Cheap insurance against a permanent,
  unrecoverable failure mode.
- **Slow drift to saturation or zero.** Handled by the homeostat, plus hard clamps
  on all field values.
- **Precision.** Fields are `rgba16float`/`r16float` (fine for bounded values with
  slow dynamics, and halves bandwidth). Anything *accumulating* — climate state, OU
  state, counters — stays `f32`/`u32` in storage buffers, where `f16`'s 10-bit
  mantissa would visibly quantise the drift.
- **Device loss** (driver reset, GPU hang, external monitor unplug, sleep/wake).
  Detect, tear down, rebuild all resources, resume from the last checkpoint.
- **Checkpointing.** Field textures + climate state written to disk every ~5 min so
  a crash or a reboot resumes a mature simulation rather than restarting from noise.
  A three-hour-old field looks materially different from a fresh one; this is worth
  the small complexity.
- **Zero per-frame allocation.** All buffers, bind groups, and pipelines are created
  at startup. A run of `10^7` frames will find any leak.

---

## 5. Depth

On the target hardware (§8.1) the GPU budget does *not* rule out true volumetric
raymarching — a 1440p 48-step march is ~5 Gsamples/s, which is a few percent of an
RTX 3080's texture throughput. The reasons to start with **layered 2.5D** are
implementation risk and lateral resolution, not cost:

- a 2D grid at 1440p has far finer filament detail than any affordable 3D grid;
- 3D Physarum needs different sensing and steering, and is much harder to tune;
- the layered path validates the colour, safety, and pacing stages, which are
  identical under either depth backend.

So: layered 2.5D first, volumetric slab as a **planned alternate depth backend**
once the rest is proven (§5.1), not as a speculative stretch goal.

- 3 independent simulation layers at different spatial scales and tempos (back layer
  large/slow, front layer fine/quicker).
- **Resolution follows depth of field.** Back layers are blurred by DOF, so
  simulating them at full resolution computes detail that is then discarded. Front
  layer at native 1440p, mid at 1/2 linear, back at 1/4 — total ~4.9 M cells instead
  of 11.1 M, with no visible loss. Agent counts scale with layer resolution the same
  way.
- Composited back-to-front with Beer–Lambert transmittance, so nearer material
  genuinely occludes and tints what is behind it, rather than just alpha-blending.
- **Parallax** offset per layer driven by an extremely slow drift (and optionally by
  cursor position, if the cursor is on that display — probably not, since the point
  is the user is working elsewhere).
- **Depth-of-field**: per-layer blur radius increases with distance. Because the
  layers are separate render targets this is one cheap separable blur each, not a
  gather.
- Atmospheric attenuation: distant layers lose chroma and contrast toward a
  background colour. This is doing most of the perceptual work.
- **Weak cross-layer coupling**: layer *k*'s field slightly perturbs layer *k+1*'s
  climate. The layers therefore aren't independent processes that happen to be
  stacked — structures loosely echo through depth, which reads as a single volume
  rather than three sheets.

### 5.1 Volumetric slab (alternate backend)

A thin slab — `512 × 288 × 48` ≈ 7.1 M voxels — keeps usable lateral resolution while
giving genuine volume. 48 depth slices is ample for parallax and occlusion given that
DOF blurs the far field anyway.

- Sim: 7.1 M voxels × 6 passes × 20 Hz ≈ 850 M voxel-ops/s ≈ **20 GB/s** — ~3% of a
  3080's bandwidth.
- Render: 3.7 M px × 48 steps × 30 Hz ≈ **5.3 Gsamples/s** of 3D trilinear. Ray
  coherence is excellent (near-orthographic camera, no secondary rays), so this sits
  around 2% of texture throughput.

What it buys over layers: real Beer–Lambert attenuation through a continuous medium,
self-shadowing from a single soft light, and structures that pass smoothly in front
of and behind each other rather than living in three discrete sheets. For "fluid
continuous motion with depth" it is materially better — but only once the 2D system
is tuned and proven, since it makes every parameter harder to reason about.

The output stages (§6, §7) are unchanged between backends, so this is a clean swap
rather than a fork.

---

## 6. Colour

All colour work happens in **Oklab / OkLCh**, not sRGB or HSV. This is not
fastidiousness — it is a requirement of the brief. Interpolating a hue rotation in
sRGB or HSV swings through large *perceived* lightness excursions (the classic
blue→yellow brightness surge), which is exactly the punctuation the application must
not produce. In Oklab, lightness is separable and can be capped independently.

**Colour is a function of simulation state, not of a clock:**

| Perceptual channel | Driven by |
|---|---|
| Lightness `L` | pigment density, with layer depth attenuation |
| Chroma `C` | heavily lowpassed local activity — busy regions saturate, quiet regions desaturate toward the background |
| Hue `h` | local field orientation (`atan2` of `∇V`) + reaction-species ratio `U/V`, offset by a global drifting anchor |

The hue anchor is one channel of the climate field, so hue varies *spatially* as
well as drifting globally — different regions sit in different parts of the palette
and those regions migrate. Global hue rotation defaults to one full turn per ~45
minutes (tunable), slow enough to be imperceptible moment-to-moment while making a
glance ten minutes later clearly different.

Constraints applied after mapping and before output:

- `L` and `C` clamped to configured ranges — a hard bound on both brightness and
  saturation, enforced at the last stage.
- Gamut-mapped back to sRGB by chroma reduction at constant `L` and `h`, so clipping
  can never change perceived brightness.
- **Blue-noise dithering before quantisation.** This matters more than it sounds:
  an 8-bit display showing a very slowly drifting smooth gradient produces visible
  banding, and worse, *crawling* band boundaries as the gradient moves — a moving
  hard edge, which is precisely a form of visual punctuation. A void-and-cluster
  blue-noise mask, animated per-frame, removes it. If a 10-bit or HDR surface is
  available, use it and reduce dither amplitude accordingly.

---

## 7. The flash-safety stage (non-negotiable, enforced by construction)

Applied to the final composited image every frame, after all colour work:

1. **Motion-compensated reprojection.** The previous output frame is reprojected
   through the velocity field before comparison. Without this, a slew limiter smears
   any structure that translates across the screen; with it, the limiter only sees
   genuine change and leaves motion alone. This reuses the velocity field the
   simulation already computes, so it is nearly free.
2. **Per-pixel slew limit in Oklab.** `ΔL` per frame is hard-clamped to
   `max_luma_delta` (default **0.01**, i.e. 1% of range). `Δa`, `Δb` clamped more
   loosely (chroma change is far less provocative than luminance change).
3. **Exposure governor.** Mean and 95th-percentile luminance are computed by mip
   reduction; global exposure is corrected with asymmetric slow attack/release so
   the overall level is stable and cannot drift bright.
4. **Temporal IIR** `out = mix(prev, new, α)`, α ≈ 0.2, as a final backstop.

### Why the default value is 1%

WCAG 2.3.1 / the PEAT general flash threshold defines a flash as a pair of opposing
relative-luminance changes of ≥10%, over >25% of the screen, and permits at most 3
per second.

With `ΔL ≤ 0.01` per frame at 30 FPS, a 10% excursion requires ≥10 frames = **333
ms**, and a full opposing pair ≥667 ms — a ceiling of **1.5 flashes per second**,
half the threshold, with no assumption whatsoever about the simulation's behaviour.
The bound holds if the simulation blows up, if a parameter is set absurdly, if a
shader has a bug. The config validator enforces a hard ceiling on `max_luma_delta`
(0.03, which still yields ≤2 pairs/s) so no user setting can defeat it.

This property is asserted directly in the test suite (§10), which is the point of
expressing it numerically.

---

## 8. Frame pacing and GPU budget

**Simulation is decoupled from presentation.** Sim runs at ~15 Hz (tunable 8–30);
render at 30 Hz. Between sim ticks, the renderer **motion-compensated interpolates**:
rather than a naive lerp of two states (which is mushy and reintroduces crawl), both
states are advected by the fractional velocity toward the intermediate time and
blended. Motion looks *smoother* than running the sim at 30 Hz, at half the cost.

This decoupling is also what makes the frame budget adaptive without visible
artefacts: if a frame exceeds `gpu_budget_ms`, the governor lowers the **sim tick
rate**, which the interpolator hides completely. It never changes resolution at
runtime — that would be a visible discontinuity.

### 8.1 Target profile — RTX 3080, 2560×1440

Reference numbers: ~760 GB/s memory bandwidth, ~465 Gtexel/s bilinear fill, 5 MB L2.
These passes are bandwidth-bound stencil and gather work, so bandwidth is the
binding constraint, not FLOPs.

| Stage | Work | Cost |
|---|---|---|
| Sim | 4.9 M cells (1440p + 1/2 + 1/4) × 6 passes × 20 Hz = 588 M cell-updates/s, ~24 B effective each | **~14 GB/s** (1.9% of bandwidth) |
| Agents | 1.55 M agents × 20 Hz = 31 M steps/s → ~93 M sensor samples, ~124 M atomic adds/s | negligible |
| Render | 3.7 M px × ~30 taps × 30 Hz = 3.3 G taps/s bilinear `rgba16f` | ~0.7% of texture fill |

Total steady state lands **under 10% of the card**, including present and driver
overhead. That is comfortably inside "leave the machine usable" — normal desktop
work, video, and an IDE will not notice it. It is *not* sized to coexist with a
game, which matches the stated requirement.

Because the headroom is large, it is spent on quality rather than banked:

- front layer simulated at **native 1440p** (no upscale) for fine filament detail;
- sim tick at **20 Hz** rather than 15, so the interpolator extrapolates less;
- ~1.5 M agents total, enough for dense network structure without individual
  deposits ever being visible;
- wider, higher-quality separable diffusion kernels rather than minimal 5-tap
  stencils — smoother fields, which directly serves the no-punctuation goal;
- multiple RD substeps per tick for finer temporal resolution in the reaction term.

The budget governor is still worth building: the user may be running other GPU work,
and a 3080 driving a second display through the compositor has variable overhead. It
throttles the **sim tick rate** only, which the interpolator hides completely.

**Headroom check:** even the volumetric slab backend (§5.1) fits inside ~10% on this
card, so the depth-backend decision can be made on aesthetics rather than cost.

**Secondary-display specifics:**

- Borderless windowed fullscreen, never exclusive fullscreen — exclusive mode can
  stall the compositor on the other display and steal focus.
- Never take input focus or capture the cursor.
- `PresentMode::Fifo` (vsync). At 60 Hz, 30 FPS is exactly every other vsync. At 144
  Hz, 30 is not an integer divisor (144/30 = 4.8) — `rendercanvas`'s `max_fps` will
  pace to the nearest vsync, giving slight cadence jitter. Because the renderer
  interpolates to the *actual* elapsed time rather than assuming a fixed step, this
  jitter is not visible.
- Survive the display sleeping, waking, or being unplugged: reconfigure the surface
  rather than treating it as fatal.
- Optionally drop to a lower sim rate when the window is not visible.

---

## 9. Parameters and control surface

~40 primitive parameters, but exposing 40 sliders is a worse interface than exposing
6 good ones. Two tiers:

**Macros** (the normal interface):

| Macro | Effect |
|---|---|
| Intensity | overall activity, contrast, agent count, event rate |
| Scale | feature size across all layers |
| Tempo | sim rate, flow strength, drift rates |
| Palette | hue anchor, hue rotation rate, chroma cap |
| Brightness | luminance ceiling and exposure target |
| Depth | parallax strength, layer separation, DOF, atmospheric falloff |

Each macro drives many primitives through a curve defined in the config. Presets
(named macro settings) are first-class — this is a regulation tool, so *quickly
getting back to the one that worked* matters more than fine-grained tweaking.

**Primitives** available in the config file for anyone who wants them.

**Mechanism:** TOML file as the single source of truth, hot-reloaded on change
(watchdog); every parameter change is **ramped, never stepped** (250 ms–5 s
depending on the parameter) so adjusting a slider can't itself cause punctuation.
Invalid values are clamped with a logged warning rather than crashing a
long-running session.

The control UI itself is an open question — see the questions below.

---

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
- **No-allocation check.** Assert steady-state buffer/texture count and process RSS
  are flat over a long run.

---

## 11. Module layout

```
anastomosis/
  __main__.py           entry point
  app.py                canvas, event loop, pacing, hot-reload
  device.py             adapter selection, feature detection, device-lost recovery
  config.py             dataclasses, TOML load/save, validation, safety ceilings
  macros.py             macro → primitive curves, parameter ramping
  sim/
    scheduler.py        tick pacing, substeps, interpolation state
    layers.py           per-layer resource sets
    passes.py           pipeline + bind group construction
    homeostat.py        PI controller config, telemetry readback ring
    events.py           Poisson slow-event scheduler
  gfx/
    composite.py        layer compositing, parallax, DOF
    grade.py            Oklab colour mapping
    safety.py           slew limiter, exposure governor, dither
  shaders/
    common/             rng.wgsl, noise.wgsl, oklab.wgsl, sampling.wgsl
    climate.wgsl  agents.wgsl  reaction.wgsl  advect.wgsl  curl.wgsl
    blur.wgsl  couple.wgsl  reduce.wgsl  sanitize.wgsl
    interpolate.wgsl  composite.wgsl  grade.wgsl  safety.wgsl
  state/checkpoint.py   periodic save/restore
  ui/                   control surface (TBD)
tests/
  test_flash_safety.py  test_soak.py  test_parity.py  test_config.py
```

**Dependencies:** `wgpu>=0.32`, `rendercanvas>=2.7`, `glfw`, `numpy`, `tomlkit`,
`watchdog`. Python ≥3.11 (wgpu-py requirement). No heavy frameworks.

Ping-pong texture pairs throughout (sampled read + storage write) rather than
read-write storage textures, which are an optional WebGPU feature — keeps the whole
thing on core WebGPU and portable across Vulkan/Metal/DX12.

---

## 12. Build order

1. Skeleton: canvas at 30 FPS, device management, config load + hot reload, one
   full-screen pass. Verify pacing and GPU load on the target machine early.
2. Single-layer Physarum + trail decay. Confirm agent cost and visual character.
3. Velocity field + semi-Lagrangian pigment advection. **This is the step that
   determines whether the "fluid" requirement is met** — worth evaluating before
   building on top of it.
4. Reaction–diffusion coupling.
5. Climate field + OU drift + homeostat. First point at which a long soak test is
   meaningful.
6. Oklab grading + full safety stage + flash-safety test.
7. Multi-layer depth compositing.
8. Sim/render decoupling + motion-compensated interpolation + budget governor.
9. Macros, presets, control UI.
10. Checkpointing, device-loss recovery, long soak.

Steps 1–6 produce something already usable for its purpose.
