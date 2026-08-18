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

### 4.4 Staying on the live band — three findings from implementation

Everything above was designed before any of it ran. Building it surfaced three
liveness problems that the architecture as drawn did not address, all found by
sweeping the Gray–Scott map in NumPy (`tests/reference.py`) rather than by
watching the output. They are recorded here because each is invisible on a
timescale of minutes and fatal on a timescale of hours.

**The default regime was wrong.** The familiar `F=0.038, K=0.062` sits in a weak
corner next to the dead zone and *settles* — measured activity decayed 45× below
target within 30 seconds. The persistently-live ridge is at much lower values;
`F=0.018, K=0.051` holds mean V ≈ 0.12 with variance ≈ 0.009 and does not settle.
Those happen to be almost exactly the homeostat targets originally guessed, which
is reassuring about the targets and damning about the regime.

**Kill, not feed, is the control lever.** Mean V and activity both respond
monotonically to `−kill`, so one control serves both objectives. Feed cannot do
the job: its effect on activity is non-monotonic and collapses abruptly at the top
of its range (activity falls to 2×10⁻⁶ at `F=0.030`). The homeostat therefore
steers with kill and uses feed only for mass and structure.

**The live region is a diagonal strip, not a rectangle.** This is the important
one. Holding kill fixed while the climate varies feed walks regions clean off the
map: an uncorrelated `−0.008` excursion in feed **kills the field outright**, and
it does not come back. So kill now *follows* feed along the band (slope 0.55), and
is additionally clamped relative to the band centre as well as absolutely — a
fixed box in `(F, K)` admits dead corners at both ends, which is exactly what the
first two attempts did. `config.clamp_reaction` mirrors the shader so the tests
exercise the real logic.

### 4.5 The absorbing state

`V = 0` is absorbing: `dV/dt = 0` when `V = 0`, so Gray–Scott can never restart on
ground where it has been fully extinguished. No amount of feed or kill correction
helps, because the homeostat's levers all multiply through `V`.

This is a genuine long-duration hazard rather than a theoretical one. Over days,
*something* will eventually extinguish some region — a bad excursion, a sanitised
NaN, a driver glitch — and the result would be permanent: a black screen with no
path back.

The fix is a direct injection path that the fiction wanted anyway. Agents seed `V`
where they run, not merely fertilise it via `feed`:

```wgsl
let seed_room = clamp(1.0 - v * trail_seed_falloff, 0.0, 1.0);
v += trail_seed_gain * (trail / (1.0 + trail)) * seed_room;
```

The falloff means established structure is untouched and only empty ground is
reseeded, so the normal regime is undisturbed (verified: all three homeostat
targets still converge). Hyphae colonising bare substrate is what the piece is
about, so it is a better model as well as a safer one.

### 4.6 Numerical survival

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
  the small complexity. **Not yet implemented** — see §13.
- **Zero per-frame allocation.** All buffers, bind groups, and pipelines are created
  at startup. A run of `10^7` frames will find any leak.

### 4.7 Morphological monotony — the failure mode the homeostat cannot see

Everything above is about staying *alive*. The first real viewing found a
different failure: after a few minutes the field reaches a texture whose
**character** never changes again. A dense population of small, similar-sized,
round features, holding steady indefinitely. The simulation is doing exactly what
§4.2 asks — mass, variance and activity all in band, never settling, never
repeating — and it is still the same picture it was an hour ago.

This also has an accessibility dimension the brief did not anticipate. A regular
lattice of similar-sized, high-contrast, round holes is a common trypophobia
trigger, and anastomosis is a process that walks straight into that geometry.
The application is meant to be a regulation aid, so a texture a proportion of
viewers find repellent is a functional defect, not a matter of taste.

**The reaction layer is a monodisperse spot field with a pinned length scale.**
Measured offline (`tests/morphology.py`) at the §4.4 regime, 160² torus, 6000
ticks after warm-up:

| quantity | value |
|---|---|
| component count | 255–297 (mean 270, s.d. 10 — ±4%) |
| length scale ℓ = mean V / mean\|∇V\| | 2.20–2.31 cells |
| holes (components − Euler characteristic) | 2 — it is a spot field, not a mesh |

Both the count and the size are constant to within a few percent for the whole
run. On the 1440p front layer that is ~250 features across the screen, all the
same size. The regularity is the problem, not the density.

**Three causes, and they are independent.**

1. *Nothing drives the length scale.* `du`/`dv` are constant except for the
   `scale` macro. A Gray–Scott regime at fixed diffusion has one characteristic
   wavelength, so the feature size is pinned by construction.
2. *The homeostat is blind to arrangement.* Mass, variance and activity are all
   invariant under rearrangement (`homeostat.wgsl`). A field can be perfectly
   on-band and morphologically frozen; the controller has no term that objects,
   and in defending its three measures it defends the texture along with them.
3. *The agent layer is topologically one-way.* `commitment` in `agents.wgsl` is
   clamped to `[0, 0.92]`, so it only ever *reduces* the turn — there is
   attraction and no repulsion. Trail decay is uniform and traffic-independent,
   so nothing can remove a strand; decay hits trunk and twig alike while agents
   preferentially reinforce whatever is already strong. Every fusion adds a
   cycle and no mechanism destroys one. This is anastomosis without autolysis,
   and it is only half of what real hyphal networks do: fungi resorb unused
   hyphae, and Physarum prunes low-flux tubes. Two lesser contributors sit
   alongside it — respawn is uniform and solitary (`agents.wgsl`), so a
   respawned agent can never found anything and all growth accretes onto the
   existing network; and only pigment is advected, so the fluid motion is in the
   colour carrier while the structure sits still and never experiences shear.

**The lever is `du`, and the reason is that it is nearly orthogonal to
everything the controller measures.** Static sweep at `feed=0.018, kill=0.051`,
ratio `dv/du` held at 0.50:

| du | features | ℓ | mean V | area | activity |
|---|---|---|---|---|---|
| 0.12 | 712 | 1.69 | 0.129 | 0.170 | 0.00046 |
| 0.21 (shipped) | 266 | 2.30 | 0.114 | 0.116 | 0.00139 |
| 0.26 | 177 | 2.56 | 0.108 | 0.104 | 0.00155 |
| 0.32 | 157 | 2.80 | 0.114 | 0.122 | 0.00153 |
| 0.40 | 112 | 3.05 | 0.102 | 0.114 | 0.00166 |
| 0.50 | 83 | 3.44 | 0.096 | 0.098 | 0.00170 |

An 8.6× change in feature count for a 26% change in mass. Across the usable
band below (du 0.17–0.40) it is tighter still: mass 0.102–0.114 and covered
area 0.104–0.122, both inside the noise of the fixed-`du` control. The two obvious alternatives are both worse: `kill`
0.046→0.056 gives ℓ 5.80→1.67 but drags mean V 0.178→0.084, and the homeostat
already owns kill; the diffusion *ratio* `dv/du` 0.36→0.64 gives 598→105
components but moves mass 0.144→0.084. Both fight the controller, and anything
that moves mass also moves the exposure governor, which turns a morphology
change into a slow global luminance swing — precisely the coordinated global
change §4.3 forbids.

Walking `du` between 0.16 and 0.34 over 3000 ticks, against a fixed-`du` control:

| | components | ℓ | mean V | activity |
|---|---|---|---|---|
| fixed | 255–297 (s.d. 10) | 2.20–2.31 | 0.1141 | 0.00135 |
| drifting | 147–395 (s.d. 68) | 2.02–2.85 | 0.1120 | 0.00142 |

A 2.7× swing in feature count — structures genuinely merging and splitting
throughout — while mass and activity stay within a few percent of the control
and inside both homeostat deadbands (mass [0.083, 0.153], activity
[0.00084, 0.00156]). Coarsening merges adjacent cells; refinement splits them.

The usable band is **du ∈ [0.17, 0.40]** at fixed ratio. Below 0.17 activity
falls under the homeostat's floor (0.00044 at du = 0.12) and the controller
starts fighting the drift with kill; above ~0.42 the explicit-diffusion headroom
starts being spent, although du = 0.5 still ran clean (`dt·du` = 0.43 against a
limit near 1.0 for this averaging-form Laplacian).

**It must be spatial, not global.** A globally coherent breathing of feature size
is coordinated global change of exactly the kind §4.2 warns about. Driven through
the climate field instead, coarse and fine regions coexist and migrate — which
additionally destroys the *uniformity of size*, and uniformity is the actual
trigger. This is worth stating plainly because it corrects the obvious framing:
breakup alone does not fix the texture. A churning field of uniformly-sized
holes is still a field of uniformly-sized holes. Varying the size fixes the
texture; breakup fixes the monotony. One lever happens to serve both.

Both `climate_a` and `climate_b` are fully allocated, so this wants a third
climate pair — 64×36 `rgba16f` is ~9 KB and free — with channels
`(scale, prune, fusion, spare)`, which is enough for every mechanism below.

**The missing half of anastomosis: flux-based pruning.** Give the trail field the
ability to lose an edge, by storing an income EMA in `trail.g` — the trail
texture is `rgba16float` and only `.r` is used, so this costs no new texture and
no extra bandwidth — and raising decay where income falls short of expenditure:

```wgsl
let income  = mix(prev_income, deposited, income_rate);        // trail.g
let deficit = clamp(1.0 - income / (decay * previous + 1e-6), 0.0, 1.0);
let decay_eff = decay * (1.0 + prune_gain * (deficit - 0.5));  // note the centring
```

Once a strand thins below sensing range agents stop finding it, which starves it
further: a positive feedback that *severs* the edge and merges two cells into
one. That is the coarsening half of a foam, and with fusion still running the
result is stationary churn rather than monotone refinement.

The centring is load-bearing and was not obvious. Uncentred at `prune_gain = 6`
the term removed 68% of trail mass; centred on the mean deficit it removed 19%.
An uncentred version is a net mass sink, so the homeostat cancels it through
`corr_decay` — yielding a globally weaker network and no severance, the exact
opposite of the intent. Use 0.5 as the reference, or carry a mean-deficit term
in the stats buffer if it needs to be exact. The trail blur smooths all four
channels, so the income channel is diffused for free, which is wanted: per-texel
agent arrivals are Poisson-spiky.

**Three smaller additions.** *Anti-fusion*: widen the `commitment` clamp past 1.0
so `turn * (1.0 - commitment)` goes negative and the agent turns *away* from a
junction, driven per-region from the new climate channel — migrating zones where
the network visibly comes apart while it fuses elsewhere. *Rift events*:
`EVENT_FIELDS` reaches only feed, kill, flow and hue, so `dieback` can thin
material but cannot sever anything; adding `chan_decay` → `climate_b.y` and
`chan_prune`/`chan_fusion` → the new pair buys an event kind that raises decay
and prune and negates fusion across a region under the usual long envelope.
*Trail advection*: the largest payoff for dynamism and the largest risk — shear
would stretch and pinch filaments, and the semi-Lagrangian pass already exists,
but it changes the agent↔trail feedback qualitatively and can push the reaction
into stripe instabilities. Last, behind a gain that starts at zero.

**The homeostat needs a morphological input**, or it remains unable to
distinguish a live pattern from a frozen one. `ℓ = mean V / mean|∇V|` costs one
term in `reduce.wgsl` (partials go from one `vec4` per tile to two, so the buffer
stride goes 16 → 32 bytes). Then either hold ℓ in a band using the global `du`,
or — better, and more in keeping with §4.2 — let the ℓ *setpoint* be a slow
bounded OU walk, τ ≈ 5–15 min, so coarsening and refinement become a continuous
cycle rather than a defended equilibrium. Measured ℓ spans 1.7–3.5 across
du 0.12–0.5, so a setpoint band of ℓ ∈ [2.0, 3.0] maps onto du ∈ [0.17, 0.40].
Split it the way feed and kill already are: the controller owns the global mean,
the climate field carries the deviation.

**What this does to the safety argument: nothing, by construction.** All of it is
upstream of §7, which bounds the output regardless. Two things still need
watching. The WCAG *area* criterion in `test_flash_safety.py` — the fraction of
pixels changing by ≥10% in one frame — will rise as churn increases, and trail
advection is the change most likely to move it. And the exposure governor is the
real interaction risk: any lever that moves mass produces a slow global
brightness cycle, which is the concrete reason to prefer `du` over `kill` or the
diffusion ratio.

**Testing.** The complaint should be encoded numerically for the same reason the
flash threshold was (§7): `holes = components − χ`, where the Euler
characteristic is a vectorised count of 2×2 pixel patterns — pure NumPy, no
scipy. Assert that hole count is non-monotone over a long run, and that feature
count has a coefficient of variation above a floor. The fixed-`du` control fails
that at s.d./mean = 0.04; the drifting one passes at 0.30. The drifting figures
move a little between runs — the reaction is chaotic, so a change of 1e-5 in
`dv` reshuffles which structures merge — but the statistics are stable, which is
what a test can assert. `gray_scott_step`
already takes `du`/`dv` as scalars and should widen to arrays, so a
climate-varying `du` stays covered by `test_parity.py`.

**Build order.** Each step is useful on its own and the first is throwaway:

1. A global OU on `du` in `_sim_values`, as a spike — confirms the visual before
   any plumbing is paid for.
2. The third climate pair, with `du` deviation per region. *This is the step that
   addresses the texture itself.*
3. Flux pruning in `trail.g`, centred.
4. Rift events and anti-fusion, both riding the channels from (2).
5. ℓ in the reduce pass, with a drifting setpoint.
6. Trail advection, behind a knob, once the rest is tuned.

Steps 1–3 should carry most of the value: polydisperse, migrating feature sizes
plus genuine edge severance.

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
- **A window resize rebuilds the presentation chain only.** Simulation resolution is
  fixed when the session starts, for the same reason the governor never touches it:
  re-resolving a running field is a visible discontinuity, and rebuilding the layers
  would additionally discard every field, every agent and the tick count — a hard
  restart in the middle of a session meant to run for days. Only the HDR target, the
  final ping-pong and the exposure partials follow the window; the compositor samples
  layers in normalised coordinates, so they need not match it. Two details make the
  seam invisible: the frame on screen is resampled into the new history buffer (the
  slew limiter emits `history + bounded step`, so an empty history would fade up from
  black over about a second), and a shape change is absorbed by sampling *more* of the
  toroidal field along the axis that grew, rather than by stretching it. Sizes are
  applied once they have held for ~150 ms, so dragging an edge reallocates once.
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
- **Morphology check.** Feature count, characteristic length and hole count over a
  long run (`tests/morphology.py`); assert the arrangement is *not* stationary.
  This is the counterpart to the soak test: the soak test asserts the field is
  alive, and a field can be alive and yet look identical for hours (§4.7).
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


---

## 13. Implementation status

Built and verified headless against a software adapter (Mesa lavapipe), so every
shader compiles and the full tick/render sequence runs in CI without a GPU. 80
tests pass in about a minute.

**Complete:** all 17 WGSL modules; the three-system substrate with agents, trail,
reaction, curl-noise flow and pigment advection; the climate field and the
GPU-resident homeostat; slow events; layered compositing with parallax, DOF and
atmosphere; the Oklab colour pipeline; the full safety stage with blue-noise
dither; sim/render decoupling with motion-compensated interpolation and the
budget governor; the parameter system with macros, presets, hot reload and
ramping; the Qt control panel; CLI.

**Not implemented:**

- **Checkpointing** (§4.4). Restarts begin from a fresh field rather than
  resuming a mature one. Everything else about long-duration survival is in
  place; this only affects what happens *after* a crash or reboot.
- **The morphology work in §4.7.** The field is alive but its texture never
  changes character, which is both a monotony problem and an accessibility one.
  The diagnosis and the measured levers are recorded; none of the six steps in
  that section are built yet.
- **The volumetric slab backend** (§5.1), which was always positioned as the
  second step after the layered path is proven.
- **Device-loss recovery** is scaffolded in `device.py` but the rebuild path is
  untested, since a software adapter offers no way to provoke a device loss.

**Not yet possible to assess here:** how it actually looks, and whether the
defaults sit in the right place perceptually. The software adapter renders
correct pixels far too slowly to watch. The numbers say the simulation is alive,
structured, and stable; whether it is *pleasant* is a judgement that needs the
real GPU and a pair of eyes.
