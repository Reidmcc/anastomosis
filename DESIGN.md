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
| Adjustable parameters | TOML config as source of truth, hot-reloaded; ~8 macro knobs over ~40 primitives; presets |

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

The prohibition is on periodicity **in time**. Noise that tiles in *space* is a
different matter, and is in fact required: the domain is a torus, and a spatial
increment that does not close on it leaves a seam in whatever integrates it
(§4.8). The spatial lattice tiles; the value drawn from it is still re-hashed
every tick, so the increment is as white in time as it ever was.

The consequence is stronger than "does not loop": there is no periodic component in
the system at all, and no state that recurs, because the state space is being
explored by a diffusion process rather than traversed by a trajectory.

---

## 4. Homeostasis and the climate field — the long-duration core

This is the part that decides whether the application is good for ten minutes or for
ten hours.

### 4.1 Climate field

A small texture (64×36 per layer, three `rgba16f` pairs) whose channels are the
*local* values of the simulation's governing parameters: feed rate, kill rate,
agent sensor angle, sensor distance, deposit rate, decay rate, flow strength,
hue anchor, and — added later, for the reasons in §4.7 — feature size, trail
pruning and fusion bias. Each tick it is:

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

One calibration note, found while tuning §4.7 and applying to every channel: the
field is clamped to [-1, 1] and each `range_*` parameter is the deviation at 1,
but the field never gets near 1. The OU drive is spatially white and the
per-tick diffusion removes almost all of it immediately, so the realised
deviation settles at **s.d. ≈ 0.11, extremes ≈ ±0.44**. Every range is
delivering about a tenth of its nominal amplitude. The existing values are tuned
against what they actually produce and are staying as they are; the point is
that anything *measuring* the effect of a range has to use the realised
amplitude, not the nominal one.

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

Events can also be **asked for** from the control panel, one button per kind.
This is arrival-time only: the request goes through the same `_spawn` the
scheduler's own arrivals do, so the event that results is jittered, localised,
enveloped, capped in radius and counted against `max_concurrent` exactly as a
sampled one is — there is no privileged path and nothing a button can produce
that the simulation could not have produced by itself. A request that would
exceed the concurrency cap is refused rather than queued, and it never
reschedules the next automatic arrival: exponential inter-arrivals are
memoryless, and letting a click move the next one would make the stream
predictable from the user's own actions, which is the property this section
exists to prevent.

**Arrival rate is adjustable** from the same panel, as the `event_rate` macro
(§9) — 0.5 to 20 events/hour, roughly one every two hours to one every three
minutes, with the centre of the travel at the ~8-minute mean this section
specifies. It moves `events.rate_per_hour` and nothing else, which is what makes
it safe to expose without a ceiling of its own: amplitude, radius, envelope
timings and `max_concurrent` are all untouched, so the fast end of the knob is a
field that spends more of its time inside an event rather than one perturbed
harder. Every non-punctuation constraint above holds identically at both ends,
and the concurrency cap still bounds what can overlap. The readout is in
minutes between arrivals rather than events per hour, and hedged — "about one
every 8 min" — because the mean is the only thing a Poisson stream lets you
promise.

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
- **Checkpointing.** Field textures + climate state written to disk every ~5 min, and
  again when the window closes, so a crash, a reboot or an ordinary quit resumes a
  mature simulation rather than restarting from noise. A three-hour-old field looks
  materially different from a fresh one; this is worth the small complexity.

  What is *not* in the snapshot is not there on purpose — at 1440p the fields add
  up to ~230 MB and this is written every five minutes for days. `velocity` and
  `reaction_prev` are rewritten unconditionally at the start of every tick, before
  anything reads them, so they carry nothing between ticks; skipping them takes the
  file to ~150 MB. The deposit accumulator is drained by `atomicExchange` every
  tick, so between ticks it is already empty. The
  output history the safety stage slew-limits against is left cold, so a resumed
  session grows up from black over a couple of seconds instead of cutting — the one
  way resuming could produce punctuation. And the event scheduler's RNG stream is
  not saved, because exponential arrivals are memoryless and a fresh stream is
  statistically identical; the *in-flight* events are saved, since those are
  mid-envelope.

  The test of what belongs in the snapshot is not size but whether the next tick
  reads something the last one left behind — which includes state that does not
  live in a texture at all. Both halves of §4.7's feature-size mechanism are in
  it on that basis: the per-region morphology climate (`climate_c`), and the
  global OU walk on the diffusion rate, whose value *and* noise-stream position
  are a hundred bytes of metadata. `test_checkpoint.py` advances a restored
  engine alongside the one it was captured from and compares every field
  bit-for-bit, which is what turns "everything stateful is saved" into something
  that fails loudly when a new mechanism forgets to say so.

  Geometry is saved, not required. Resolution, layer count, agent counts and
  climate size all follow from the window size and the config, so treating them as
  a compatibility key meant that opening the window at a different size — or
  editing any config value that touched them — silently discarded a field that had
  taken hours to grow, over a mismatch that only ever concerned the presentation.
  The snapshot therefore records the geometry it was captured at, and the launch
  builds its engine in *that* shape before loading the field into it. This costs
  nothing visually, because the simulation's resolution is already independent of
  the window's: a resize only rebuilds the presentation chain (§8), and the
  compositor corrects the aspect difference. The consequence is that a resumed
  field keeps the geometry it was grown at, so the structural config values land on
  the next *new* field — which is what the reset is for.

  What is still refused is a file this build cannot use at all: a foreign format
  version, missing or wrongly-shaped arrays, or a geometry no engine could be built
  at — bounds-checked before allocation, since that number comes off disk and
  decides how much memory the launch asks for. Every failure path here degrades to
  "start fresh"; nothing on disk can stop the application opening. Verified by
  resuming into a second engine and asserting it evolves bit-identically for
  further ticks — the only check that catches a piece of state quietly left out,
  and the one that keeps `velocity` and `reaction_prev` honest about being derived.
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

1. ~~A global OU on `du` in `_sim_values`, as a spike~~ — **built**, and kept
   rather than thrown away; see below.
2. ~~The third climate pair, with `du` deviation per region.~~ **Built.** *This
   is the step that addresses the texture itself.*
3. ~~Flux pruning in `trail.g`, centred.~~ **Built, and off by default** — it
   does not do what this section predicted. See below.
4. ~~Rift events and anti-fusion, both riding the channels from (2).~~
   **Built**, together with the founding respawn step 3's postmortem asked
   for; see below.
5. ℓ in the reduce pass, with a drifting setpoint.
6. Trail advection, behind a knob, once the rest is tuned.

~~Steps 1–3 should carry most of the value: polydisperse, migrating feature
sizes plus genuine edge severance.~~ Step 2 carries the value. Step 3 works
mechanically and delivers no visible benefit; the reasons are worth keeping and
are recorded below.

**What steps 1 and 2 turned out to be.** They are not two mechanisms but one
split the way feed and kill already are — a global mean and a per-region
deviation around it. The spike was worth keeping in that role: a unit-variance
OU walk on the mean (`Engine._advance_du_walk`, τ = 7 min, ±7% per standard
deviation), with the climate deviation on top of it. The walk alone is
explicitly *not* the fix, for the reason given above — it moves every feature on
screen the same way at the same time and leaves them all the same size as each
other — but as the carrier of the mean it is what step 5 will eventually hand
over to the controller, so the plumbing is the same plumbing.

**The climate field realises about a tenth of its nominal range, and this had to
be measured before anything could be calibrated.** Every `range_*` parameter is
the deviation at a climate value of 1, and the field is clamped to [-1, 1], so
the natural reading is that `range_feed = 0.008` means ±0.008. It does not. The
OU drive is spatially white and the per-tick diffusion (§4.1) removes almost all
of the injected power immediately, so the field settles at **s.d. ≈ 0.11 with
extremes near ±0.44** — measured off a running engine over ticks 1200–3600, and
the same for every channel. Every existing range is therefore delivering roughly
a tenth of its apparent amplitude. That is not being changed here: those values
were tuned by eye against the behaviour they actually produce, and rescaling
them would change the shipped look for no benefit. But `range_du` had to be
calibrated against the realised amplitude rather than the nominal one, and any
future measurement of a climate range has to do the same or it overstates the
effect by an order of magnitude.

**The deviation is geometric, not additive.** `du` is driven by the `scale`
macro, so a fixed offset would mean a different spread at each end of that knob.
More importantly the survivable band is not symmetric around the base — a factor
of two of headroom above, and 0.57 below — so an additive deviation spends its
downside against the floor while its upside is still unused. Measured, at
matched spread of feature size: additive puts 13% of texels on a clamp,
geometric 4%. Both terms are geometric, so the walk and the regional deviation
compose exactly rather than interacting. `dv` rides on `du` throughout; holding
the ratio is what keeps the lever off the homeostat's measures, and varying the
two independently would reintroduce the mass movement that ruled the ratio out
as a lever in the first place.

**The floor is measured, not chosen.** The sweep in the table above stops at du
= 0.12; below that, at fixed feed and kill: 0.10 gives mean V 0.128 with
activity 1.5×10⁻⁴, 0.08 gives 0.097 and 1.1×10⁻⁴, and **0.06 collapses** — mean
V 0.015 and activity indistinguishable from zero. `du_min = 0.12` sits a factor
of two above that collapse and is the lowest point with any real measurement
behind it. The ceiling, 0.42, has more headroom than it needs: du = 0.50 still
ran clean.

**One thing the analysis did not anticipate: the region has to be big enough to
hold a wavelength.** Measured at a climate-texel-to-cell ratio of 10 the effect
largely cancels — regions a few features across cannot establish their own
length scale and average back to one. At 24 it is clean, and the engine's real
ratio is ~40 (64×36 over a 1440p layer), so this is comfortable rather than
marginal. It does mean an offline measurement has to be run at a realistic ratio
or it understates the effect.

Measured against a fixed-`du` control, isolated reaction (`tests/morphology.py`,
192² torus, 4000 ticks, one climate snapshot at the realised amplitude):

| | local ℓ spread (c.v.) | corr(local ℓ, local `du`) | mean V | activity |
|---|---|---|---|---|
| fixed | 0.081 | — | 0.1116 | 0.00135 |
| climate-varying | 0.118 | 0.88 | 0.1175 | 0.00116 |

and with the climate drifting under the field (160², 4000 ticks after warm-up):

| | components | c.v. |
|---|---|---|
| fixed | 263–302 | 0.037 |
| drifting | 241–388 | 0.157 |

The correlation is the number that says the mechanism is doing what it claims
rather than merely adding noise: where the climate asks for coarse structure,
coarse structure appears. Mass and activity both stay inside the homeostat
deadbands and within a few percent of the control, which is the property the
whole choice of `du` rests on — and they stay there at both extremes of the
walk (at ±2 s.d.: mean V 0.121 and 0.113, activity 1.01×10⁻³ and 1.32×10⁻³).
That last check is what set the walk's amplitude: at ±20% rather than ±7%, the
low end takes activity to 9.0×10⁻⁴ against a deadband floor of 8.4×10⁻⁴, and a
homeostat that starts correcting for the drift would cancel it.

The spatial swing in feature count is smaller than the 2.7× the global drift
produced, and that is expected — a coarsening region and a refining one partly
cancel in a whole-field count. The count is the weaker of the two measures here;
the spread of the *local* length scale is the one that speaks to the complaint,
since uniformity is the trigger.

Two caveats worth recording. About 5% of texels sit on a `du` clamp at any
moment, which is intended — the band is meant to be reachable — but those
regions have no local variation left, so the figure is worth watching if the
range is ever widened. And in the *full* engine, rather than the isolated
reaction, the local length scale already varies (c.v. 0.15) from the agent and
feed/kill machinery, so this lever is one contributor among several and the
effect is not cleanly separable there. That is why the measurements above are
made on the reaction alone, as §4.7's original analysis was.

The third channel of the new pair is read by the agent layer as of step 4,
under the name `repel` rather than `fusion`; the section below says why the
name had to change.

### What step 3 actually did

The flux pruning above is built: an income EMA in `trail.g`, decay raised where
income falls short of expenditure, the whole thing measurable from `trail.b` and
gated per region by the `prune` climate channel. It is off by default
(`agents.prune_gain = 0`), because measuring it changed the conclusion three
times. All three corrections are worth recording, because the reasoning that
produced the original prescription looks sound and is wrong in specific ways.

**The income EMA has to be faster than the trail's own decay, not slower.** The
first implementation used `income_rate = 0.05` against a trail decay of 0.055
and the deficit came out at **0.02 across the entire field** — the term was
inert and would have shipped looking like it worked. The reason is structural:
the trail *is* a decaying integral of deposits, so an EMA with the same time
constant reproduces it exactly and `income / (decay · trail)` is 1 by
construction. The deficit only carries information when income is measured over
a window shorter than the trail's memory, so that a strand's *recent* traffic
can be compared against its *accumulated* size. At 0.15 the deficit spreads
across the full range, median 0.51.

**The centring must not be local, and 0.5 is not the reference.** The
mass-weighted mean deficit is 0.21 at the default intensity — but 0.07 at the
top of the intensity macro, where the network concentrates two thirds of its
mass into 2.6% of texels. So the reference is not a constant and cannot be one;
it is now measured in the reduce pass, which is why the per-tile partials went
from one `vec4` to two (the stride change §4.7 anticipated for step 5's ℓ term —
that half of the plumbing is done).

That fixed the accounting and broke something worse. Centring the term locally —
raising decay on starved strands and lowering it on well-fed ones, exactly as
prescribed above — **froze the field solid**. Well-fed strands have a deficit
near zero, so centring hands them several times the memory of everything else,
and the network locks into whatever shape it happened to hold. Measured, the
trail's autocorrelation at a 1050-tick lag went from 0.11 unpruned to **0.72**
centred. That is a far worse failure than the monotony this section exists to
fix, and it is not a tuning problem: any mass-neutral multiplicative scheme has
to give back what it takes, and giving it back through decay means giving it to
whatever is already strongest.

The fix is to return the mass somewhere else. Pruning is now one-sided — decay
is only ever raised — and the removed fraction of the field's throughput is
measured and handed back through the *agent deposit*, so it reappears wherever
traffic currently is rather than wherever mass already is. That is also the
better model: fungi translocate what they resorb to the growing tips. Persistence
at a 1050-tick lag comes back to 0.27.

**And with all of that right, it still does not deliver churn.** Measured at
256², against a `prune_gain = 0` control:

| | trail mass | mass / area | trail autocorr, lag 1050 | reaction autocorr, lag 600 |
|---|---|---|---|---|
| off | 0.097 | 1.12 | 0.11 | 0.03 |
| gain 3 | 0.098 | 1.44 | 0.27 | 0.02 |

Mass is preserved and `corr_decay` does not move at all between the two, so the
accounting works and the homeostat genuinely does not fight it. The network
concentrates — the same mass in a third less area, which is a real coarsening.
But the trail gets *more* persistent, not less, and the reaction field, which is
what the texture complaint was actually about, does not change at all: mean V,
feature count, ℓ and the local-ℓ spread are all within noise of the control.

The reason is a selection effect rather than a mechanism failure. Pruning
removes the parts of the field that were changing, so what survives is by
definition the persistent part. The prediction above — "with fusion still
running the result is stationary churn rather than monotone refinement" —
assumed a cellular network where severing an edge merges two cells and fusion
creates new ones. The trail field is not cellular: measured across thresholds
and at 160², 256² and 384², the web has essentially no closed loops (holes 0–2),
so there are no cells to merge and severance just deletes a wisp.

There is also a stability cost. One run in four at `prune_gain = 1.5` fell into
a sparse state — trail mass around 0.045 against a normal 0.095 — and stayed
there for the rest of the run. Gains of 3 and 5 were stable across four seeds
each, but a rare one-way transition is precisely what §4.5 and §4.6 exist to
prevent over a multi-day run.

**What would make it earn its place.** The returned mass currently goes through
the agent deposit, and agents live on the network, so the mass cycles within the
structure that is already there. The accretion bias identified as a lesser
contributor above — "respawn is uniform and solitary, so a respawned agent can
never found anything" — is what closes that loop. If a founding respawn let the
returned mass start new structure on bare ground, pruning would become genuine
turnover rather than concentration: material resorbed here, reinvested there.
That makes step 4 the interesting one, and it makes step 3 a component of step 4
rather than a step in its own right. The mechanism stays in the tree, tested and
inert, waiting for that.

### What step 4 actually did

Three mechanisms rather than two: anti-fusion and rift events as prescribed,
plus the founding respawn that step 3's postmortem identified as the thing
standing between flux pruning and genuine turnover. All three are on by
default; flux pruning still is not.

**Anti-fusion, and the axis it turned out to sit on.** The prescription is what
shipped — the `commitment` clamp reaches past 1.0, at `fusion_max = 1.85`, so
that `turn * (1.0 - commitment)` changes sign — with two corrections found in
the building.

The first is that this is not a *fusion* axis, and a channel named for fusion
would have been named for the wrong thing. Commitment runs from 0, where the
agent turns toward what it sensed and follows it, through 1, where the turn is
cancelled and it drives straight through the junction and fuses, to above 1,
where the turn reverses and it veers away. Fusion is a *point* on that axis
rather than an end of it, and both sides of the point fuse less. A single
signed channel has to pick a monotone direction, so the channel carries
deflection toward the junction and is named `repel`. Note also that the old
0.92 ceiling was never reached in the first place — the bias tops out at 0.72
at the far end of the intensity macro — so widening the clamp does nothing on
its own. What it does is make the region past the crossing reachable at all.

The deviation is additive, unlike `range_du` and `range_prune`, and for a
reason specific to this axis: what matters is a fixed crossing at 1.0 rather
than a ratio, so an additive deviation puts that crossing at a fixed distance
in climate units and the fraction of the field that repels can be read off the
amplitude. At `range_repel = 2.6` against the realised climate amplitude of
0.11 (measured for step 2, and the same for every channel) the crossing sits
1.6 s.d. out: about 6% of the field is coming apart at any moment, and those
zones migrate with the climate.

The second correction is that the sign flip alone does not cover the case that
matters most. When the forward sensor is the strongest reading the steering
term is *already* zero — that is what committing to a junction means — so
multiplying it by a negative number leaves it zero and the agent drives
straight into the thing it is supposed to be avoiding. And an agent that has
been committing to junctions is, by construction, pointed at one: head-on is
the common case here, not the corner case. A repelling region therefore breaks
the agent toward the weaker flank explicitly, at the strength the sign-flipped
term would have had.

**Founding respawn.** Respawns now land in cohorts — 55% of them — at a site
shared by everything respawning in the same 240-tick epoch, chosen as the
barest of four hashed candidates. There is no communication between
invocations and none is needed: every agent hashes the same (epoch, site) and
gets the same answer. The one thing that needed care is that the number of
sites is a *density* (cells per site) rather than a count. Agent count scales
with layer area, so a fixed number of sites would put a hundred times more
traffic on each one at 1440p than on a 128-cell test layer, and a founding
would arrive as a flare rather than as a founding.

**Rift events, and what an event amplitude actually means.** Three channels
added to the event record — `chan_decay` into `climate_b.y`, `chan_prune` and
`chan_repel` into the new pair — and one kind that drives all three.

Measuring it turned up something about the *whole* event system that had not
been noticed: **an event pins every channel it names**. The climate is
mean-reverting at `theta = 0.0016` and an event *adds* its amplitude every
tick, so a sustained contribution settles at `amplitude / theta` — hundreds of
times the clamp. Every channel a kind names therefore sits at ±1 through the
whole hold, whatever coefficient it was given; the coefficient shapes the ramp
and nothing else. (The spatial raised cosine survives only as a rim, though the
climate's own diffusion softens it; the temporal envelope is unaffected, since
the plateau is approached over the ~625-tick reversion time.)

That turned a tidy-looking choice into a wrong one. The kind originally carried
a small feed and kill "so the reaction is not what is being taken apart" —
which, pinned, meant feed at its clamp for the length of the event. Measured
against a no-event control at 96²: with feed and kill, the reaction inside the
disc fell to 0.65 of its surroundings and the whole field's mass fell 16%; with
the identical severance channels and no feed or kill, the reaction sat at 0.93
against a control's 0.97 and the effect on the trail was *exactly the same*.
So the kind now names the three severance channels and nothing else. That is
what makes it structural: it changes how the material is connected, not how
much of it there is, which is also what keeps it clear of the exposure governor
— the interaction §4.7 flagged as the real risk.

Traced at 128² through a shipped-length envelope, as the disc's middle over
ground the event never reached, against a no-event control on the same seed:

| | trail inside / outside | V inside / outside | global mean V |
|---|---|---|---|
| control | 0.10 – 0.13 | 0.9 – 1.2 | 0.115 – 0.125 |
| dieback | 0.10 – 0.13 | 0.13 – 0.37 | 0.085 – 0.104 |
| rift | 0.024 – 0.06 | 0.9 – 1.2 | 0.115 – 0.125 |

(The rift's two right-hand columns are the control's, which is the point of the
row: measured against the control run's *own* disc, the reaction inside a rift
sits at 0.97 and 1.00 of it on the two seeds tried, while the trail falls to
0.32 and 0.77.)

The dieback row for the trail is not merely similar to the control, it is
*identical at every sample*, and that is the sharpest possible statement of the
gap this kind fills. Nothing flows from the reaction back to the trail, so an
event reaching only feed and kill cannot touch the network at all, whatever it
does to the material. The two kinds are now orthogonal: dieback moves how much
there is, rift moves how it is joined up.

Decomposed at 96², against the same control (trail inside/outside 3.92, V
0.97): the raised decay does most of the work (3.22), the repulsion adds to it
(3.63 alone, 3.08 together), and neither moves the reaction outside the
run-to-run spread (1.25 and 1.11). The `prune` channel is inert, since
`prune_gain` is zero — it is wired so that switching pruning on would sharpen
rifts rather than needing a second change.

Severance is strongest where the network is thinnest, by a factor of two or
more. In a sparse region the raised decay thins a strand, agents stop finding
it, and that thins it further — the feedback §4.7 predicted — and the trail
inside falls to 0.32 of the control's. Where the disc lands on a dense hub the
same event takes off 0.77, and at 96² about a fifth. That is the right shape
for the mechanism: it takes apart what is weakly held and leaves trunks alone.
It also means the effect needs on the order of a thousand ticks at strength to
develop, since the direct effect of the decay term alone is only that fifth.
The reaction sat at 0.97 and 1.00 of the control in both, which is the claim
that does not depend on where the disc lands.

And it heals. Traced for 4000 ticks past the release, the reaction inside the
disc is back to parity with its surroundings about 1700 ticks after the hold
ends — before the envelope has finished releasing — and the trail comes back to
0.53 and 0.92 of the control's on the two seeds tried, the network being the
slower of the two to return. This half is not optional: V = 0 is an absorbing
state for Gray–Scott, and an event that could empty a quarter of the screen and
leave it empty would be a worse failure than the monotony this section exists
to fix, because it would be permanent. It is asserted in `test_soak.py`.

**What could not be measured, which is most of what step 4 claims.** The
mechanisms are verified individually — that agents in a repelling region turn
away from a junction and break off a head-on approach, that respawns land
together and on bare ground, that a rift severs and heals — and every invariant
they had to respect is unmoved: across every configuration and seed run here,
trail mass sits at 0.088–0.093, mean V at 0.123–0.144, activity at
0.0014–0.0015, and the homeostat's `corr_decay` does not move.

The system-level claim — that this makes the trail layer churn — did not
survive its own measurement. Every statistic tried (structural turnover at a
1000-tick lag, per-texel change rate, threshold-crossing rate, field
autocorrelation) has a run-to-run spread comparable to or larger than the
difference between configurations, in *both* directions. At 160² over 4000
sampled ticks, four seeds gave structural turnover 0.058 → 0.090 with the
mechanisms on, which looks like the doubling one would want; the spread between
seeds is 0.049, so it is not distinguishable from nothing. Nine seeds per arm
at 128², using a per-texel rate rather than a pattern correlation, gave a mean
change of nothing at all: 0.127 against 0.130, with individual runs spanning
0.04 to 0.22 in both arms. Nor can the runs be genuinely paired to
recover power: founding respawn draws extra random numbers per agent, so
turning it on shifts every subsequent draw and two runs of "the same seed"
share only their first few ticks. The honest summary is that the field's
aggregate behaviour on a 128–160 cell test layer is dominated by which
dynamical state a given run wandered into, and that resolving an effect of this
size would need tens of independent runs per arm at a resolution the software
adapter cannot reach in reasonable time. What the tests assert is therefore the
mechanism and the invariants, not the aggregate.

They are on by default anyway, and that is a deliberate call rather than an
oversight: the argument for them is structural rather than statistical — the
agent layer could previously only add edges and could only grow onto what it
already had — and nothing measured says they do harm, across nine seeds per
arm, with the field neither collapsing nor freezing and the homeostat never
reaching for a correction. Whether they earn their place perceptually is one
more question for the first viewing on real hardware (§13), alongside the ones
already waiting there.

---

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

---

### 4.9 Sensing reach — the condensation the layer was sitting inside

§4.8 fixed a seam that *nucleated* lines at the field's edge. It did not fix the
lines. What makes a line is one step further in: at the sensing reach the layer
shipped with, the agent population condenses onto a **single axis-aligned strand
that wraps the torus**, and everything else goes bare.

Measured at 256×160 with one layer, seed 5, over 2000 ticks — `axis` is
`|mean exp(4iθ)|` over agent headings (1.0 = every agent running along ±x or
±y), `band` is the share of agents inside the busiest eight-cell row or column,
and `peak` is the ratio of the strongest trail row to the median row:

| `sensor_distance` | `trail_diffuse` | ratio | axis | band | peak | outcome |
|---|---|---|---|---|---|---|
| 8.0 *(shipped)* | 1.375 | 5.8 | **0.87** | **0.97** | 340 | one strand, everything else bare |
| 5.5 | 1.375 | 4.0 | 0.35 | 0.24 | 23 | holding, but concentration still climbing |
| 4.5 | 1.375 | 3.3 | 0.30 | 0.35 | 49 | distributed network |
| 3.5 | 1.375 | 2.5 | 0.01 | 0.37 | 16 | distributed network |
| 7.0 | 1.90 | 3.7 | 0.01 | 0.41 | 43 | distributed network |
| 12.0 | 1.90 | 6.3 | 0.88 | 0.06 | 1.2 | axis-locked, no structure at all |

The controlling quantity is not the sensing distance but its **ratio to the
width of what it senses**. Above ~5 the layer condenses; at 3.7 and below it
holds a network. Both of the last two rows have the same absolute reach as rows
above them and behave in opposite ways, which is what identifies the ratio
rather than the distance.

**Why the ratio.** An agent whose sensors reach much further than a strand is
wide stops resolving the strand it is standing on and starts steering at
whatever it can see from a distance. The paths that produces are straight — and
a straight path on a torus is the one path that closes on itself, after exactly
one lap. A closed path is re-walked, and re-walking deposits, so it reinforces
every lap while nothing else does. From there it is winner-take-all: trail
following has no capacity limit and no exit except `max_age`, so a strand that
starts ahead captures every agent that crosses it, the ambient falls away
beneath the starve threshold, and within ~1500 ticks the whole population is on
one strand. Straight and axis-aligned because those are the shortest closed
geodesics; horizontal on one seed, vertical on another, and that is the only
difference between them.

This is the mechanism behind the reported artefact: vertical and horizontal
line-like structures that split, rejoin and partially disappear but never go
away. They behaved like part of the simulation because they *were* the
simulation — by the end, all of it.

**What it is not.** Four candidates were ruled out by measurement rather than by
argument, each a 2000-tick run at the same size and seed:

- *the §4.8 seam*: the pre-fix and post-fix builds condense identically
  (axis 0.869 both, at tick 2000), differing only in which axis wins. The seam
  biased *where* a line nucleated, near the field edge, which is why the
  artefact first read as an edge effect; it had nothing to do with whether one
  formed;
- *jitter*: tripled to 0.30, the layer still condenses (axis 0.68, band 0.94);
- *the anti-fusion channel*: `range_repel` moved either way changes nothing;
- *commitment persisting after arrival*: gating the fusion `excess` on the trail
  underfoot, so the commitment releases the moment an agent reaches what it
  sensed, leaves the collapse untouched (axis 0.86, band 0.95). Worth recording
  because it is the plausible-sounding fix, and it is wrong: the straight runs
  are not the approach, they are the strand.

`fusion_bias` does move it — at zero the axis alignment goes to 0.014 — but it
does so by removing anastomosis, which is the application. It is the reach that
has to change.

**The threshold moves with the field.** A smaller torus condenses more easily —
the closing lap is shorter, there are fewer agents, and fewer strands fit — so
the ratio that is comfortable at 256×160 is not automatically comfortable
below it. At 192×128 a ratio of 3.2 still ended half-condensed (axis 0.72, band
0.78) where 2.5 did not (axis 0.25, band 0.28). The operating point is set from
the smaller field, since that is the harder case and the real one is larger
still.

**Fix.** Bound the sensing reach against the width of what it senses, at every
place that can set it:

- the `scale` macro moves `sensor_distance` and `trail_diffuse` *together*
  (2.2–4.9 against 0.85–1.90), holding the ratio at ~2.6 along its whole length.
  It used to sweep 4.0–12.0 against the same diffusion, so the ratio ran 4.7 at
  one end to 6.3 at the other and the macro was over the threshold everywhere,
  the default included;
- `range_sensor_distance` drops from ±3.0 cells to ±0.8, which is the spread the
  climate can carry inside the safe band rather than one that leaves it;
- `agents.wgsl` clamps the result against `sensor_reach_max × trail_diffuse`
  (3.2×), mirrored by `config.clamp_sensor_distance` in the same way
  `clamp_du` mirrors the reaction's. The clamp is what makes it safe rather than
  merely likely: the climate reaches its own bounds in the tails, and a region
  that wandered over the threshold for a few hundred ticks would nucleate a line
  that outlives the excursion.

With that, the reach spans 1.4–5.7 cells across the whole macro range and the
whole climate excursion, and the ratio stays inside 1.7–3.2 throughout.

**A second length had to follow it.** Shortening the reach broke the founding
respawn of §4.7 step 4, and silently: a cohort is scattered around its site with
a Gaussian radius of `found_radius`, and at 6.0 cells against a reach of 3.6 its
members land outside each other's sensing. A cohort that cannot find itself
never establishes, so bare ground stopped being recolonised at all — a rifted
disc sat at 1–3% of the control's trail for 5400 ticks after the envelope
released, which is precisely the "permanently bare ground" the rift test exists
to catch. `found_radius` now moves with the `scale` macro alongside the reach
(1.6–3.5 against 2.2–4.9, i.e. ~0.72× throughout) and defaults to 2.55, which
heals the same disc back to 80% and, as a second-order benefit, disperses the
population better than the wide cohort did: on the seed that concentrated worst,
the busiest band falls from 0.77 of the agents to 0.26.

**After.** Two seeds at 192×128 — the harder of the two sizes — over 2000 ticks:
axis alignment sits at **0.003–0.06** against 0.87 before, i.e. the headings are
no longer aligned to anything, and the trail's peak-to-median rises to ~58 and
comes back down to ~14. One seed still gathers 0.6–0.77 of the agents into a
band for a while, but with no axis alignment under it: that is a dense *region*
of network, not a line, and it disperses over the run rather than tightening.
Structure forms, concentrates, and comes apart again, which is the behaviour
§4.7 spent its length arguing for and could not get from the morphology levers
alone.

**Cost.** Shorter sensing makes the network finer and more reticulate, and the
`scale` macro still moves feature size — through `du`, `trail_diffuse` and the
flow's noise scale — over its full range. What it no longer does is move the
layer across a bifurcation.

**Tests.** `test_config.py` asserts the ratio invariant across the whole macro
range and the whole climate deviation, including the clamp, which costs nothing
and is the guard that actually holds the line. `test_soak.py` runs the layer to
1100 ticks and asserts the population has neither aligned to the axes nor
emptied the field into one strand, with the shipped-at-the-time reach as a
positive control in the same test, because a collapse metric that has never seen
a collapse is not evidence.

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
once the rest is proven (§5.1), not as a speculative stretch goal. Both are now
built, and the choice between them is the user's -- a selector in the control
panel, a `backend` key in the config, a `--backend` flag. The default is still
the layered path.

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

  Two things about that drift were wrong for as long as it existed, and both are
  worth recording because neither is visible as a fault — a viewpoint that does
  not move produces a perfectly good-looking image. It was an Ornstein–Uhlenbeck
  walk with a fixed *per-frame* decay of 0.02 against a `sqrt(dt)` noise term,
  which gave it a stationary spread of 3e-4 across a travel of 1: at
  `render.parallax = 0.02` that is a viewpoint displaced by a few hundredths of a
  pixel, and over two hours of simulation it never left a thousandth of its
  range. Parallax was, in effect, switched off in both backends — including the
  volumetric one, where it is the *whole* of the camera's motion. It is now
  written in the same form as the feature-size walk (`_advance_du_walk`):
  unit-variance, `sqrt(1 - (1-θ)²)` noise, θ from `dt/τ`, so the excursion is a
  property of the walk and not of the frame rate.

  The second correction is that an OU process is smooth in its envelope and
  *white* in its increments. That is fine for an amplitude and disastrous for a
  position: the raw walk moves the image about half a pixel per frame, at
  random, which is precisely the per-pixel temporal noise §7 and the dither
  design exist to avoid. The walk therefore reaches the viewpoint through one
  first-order lag at a sixth of its own time constant, which makes the position
  C¹ and costs almost none of the travel — measured over an hour at 1440p, the
  frame-to-frame step falls from 0.54 px to 0.020 px while the peak excursion
  stays at 49 px of 50. What is left is about 0.6 px/s, which is what the
  parallax actually is.

  What that fix did *not* settle is whether the amount of parallax on offer was
  ever enough to see, since the range it moved over — 0.006 to 0.038 of the
  screen's width — was chosen against a mechanism that did not work and is
  therefore not evidence of anything. It is now its own macro, split out of
  `depth` exactly as `event_rate` was split out of `intensity` and for the same
  reason: the two answer different questions. Everything left under `depth` is
  a shading trick applied to a *normalised* depth and says the same thing about
  the far face however far away it is; parallax is the only cue that comes from
  the scene moving. The new knob reaches a quarter of the screen's width at
  about 8 px/s, and a config written before the split takes the new default
  rather than inheriting the old one — there is nothing to carry across from a
  setting that did nothing.

  **And the volumetric backend holds itself to what its slab has earned.**
  Parallax is thickness times the tangent of the viewing angle, which is the
  geometry underneath the whole complaint that a thin slab reads flat: 48 voxels
  against 512 is a sheet of paper, and there is not much depth to be found by
  walking around a sheet of paper. Swinging further does not buy more depth, it
  buys the same slab seen edge-on — a long oblique smear through it, with a path
  length (and so an optical depth) that grows with the swing. So the drift's
  amplitude is scaled — not clipped, which would put a corner in the viewpoint's
  path — to `PARALLAX_MAX_TANGENT` (0.8, about 39°) times the thickness. At the
  default slab that caps the travel at about 8% of the width; at 144 voxels deep
  it is 22%. The thickness knob and the parallax knob therefore compound, and
  the control panel's readout reports the *effective* travel so that a knob
  which has stopped doing anything visibly stops moving.

  The safety stage is deliberately not told about the viewpoint. It reprojects
  its history through the screen-space velocity of the *material* (§7), so a
  moving camera leaves an uncompensated residual — 0.02 px per frame against
  material moving several, three orders of magnitude below anything the limiter
  responds to. At the top of the parallax knob it is 0.3 px per frame, which is
  the same argument with less margin — and still not a visible artefact, because
  a rate limiter tracking a steadily moving edge settles at a constant lag
  rather than suppressing the motion: the displayed edge trails the true one by
  the sub-pixel offset at which demand equals budget. What is bounded here is
  the *rate* of change, and a smooth pan is exactly the case that costs it
  nothing.
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

A thin slab — `512 × 288 × 48` ≈ 7.1 M voxels at the default thickness — keeps usable
lateral resolution while giving genuine volume. 48 depth slices is enough for parallax
and occlusion given that DOF blurs the far field anyway, and it is where the thickness
starts rather than where it stays: see *How thick* below.

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

#### What building it actually settled

Five decisions were not obvious from the sketch above, and each is the answer to
a question the layered path never had to ask.

**The slab is a 3-torus, not a box.** Every argument for a toroidal domain in
two dimensions applies unchanged to the third: a reflecting or no-flux face
accumulates material against itself over a run measured in days, and agents that
bounce off a wall pile up along it. So all three axes wrap and no pass anywhere
has a boundary case. The cost is paid once, in the renderer: material leaving the
near face reappears at the far one, so the march applies a raised-cosine window
over both faces and the arrival and departure are fades rather than steps. That
window is also the reason the toroidal choice is free rather than merely cheap —
the far face is the fogged, dimmed, blurred end of the volume anyway.

**Divergence-free flow needs a stored potential, not a sum of velocities.** In
the plane, `v = curl(ψ)` for a scalar ψ, and the two components — imposed weather
and structure-following — can be built as velocities and added, because the sum
of two curls is a curl. In three dimensions the structure term is not the curl of
anything obvious: "flow along the iso-surfaces of V" wants `∇V × ê`, which is
divergence-free only when `ê` is constant — and a constant `ê` gives the slab a
permanent preferred plane with every filament in it sliding the same way. So the
vector potential `A` is assembled into a texture at full resolution and
differentiated in a second pass. `div(curl(A))` then cancels term for term under
central differences, because the difference operators commute — exactly zero, not
zero to within a discretisation error. Measured on the running field, the residual
divergence is 0.04% of the velocity magnitude, which is the `float16` quantisation
of the velocity texture and nothing else.

Two things ride on that licence, and neither would be exactly divergence-free
applied to a velocity: the climate's per-region flow gain (multiplying a velocity
by a varying gain adds `∇g·v` to the divergence, and material piles up wherever
the gain falls off), and the depth anisotropy below.

**Motion has to be anisotropic, and the anisotropy belongs on the potential.**
The slab is four or five feature-widths deep. Isotropic flow at a speed the agent
layer needs carries material through the entire thickness in a couple of seconds,
and the depth axis reads as churn rather than as depth. Weighting the *lateral*
components of `A` by a factor scales `v_z = ∂ₓA_y − ∂_yA_x` by roughly that factor
while leaving the lateral velocity dominated by the terms in `A_z` — and leaves
the field a curl, so it stays exactly divergence-free. Scaling `v_z` directly does
not. Agents get the same treatment applied to their heading rather than their
step, so that sensing follows motion instead of probing at angles they will not
take.

**3D Physarum sensing is a rolled cone.** Heading is a unit vector, since there is
no scalar turn in three dimensions, and steering is a normalised interpolation
toward (or, past the anti-fusion crossing, away from) a sensed direction — which
is a rotation in the plane the two span and needs no basis to express. Four flank
sensors sit on a cone of half-angle `sensor_angle`, rolled by a per-agent random
phase every tick: four *fixed* flanks give the population a preferred lattice of
turn directions, which the left/right pair in 2D cannot do. The fusion bias, the
sign change at commitment 1.0, and the head-on special case all carry over
unchanged in meaning.

**The climate needs the third axis too.** A 2D climate stretched through depth
would give every structure at a given (x, y) the same feed, the same kill and the
same feature size however far apart in z, and the volume would read as a thick
sheet. A regime is a region of the slab.

#### How thick

Forty-eight voxels is a slab a ray crosses one or two filaments of, and that is
about as subtle as genuine volume gets: the mechanisms are all working, and there
is not much material for them to work *on*. So the thickness is a knob — the only
piece of the slab's geometry the control panel moves — running from 8 voxels up to
the shorter lateral axis, which is 288 on a 16:9 display. It is structural, so it
grows a new field; and unlike the backend switch it cannot keep the old one, since
a slab of a different depth is a differently shaped array and nothing resamples one
into the other.

What it costs is memory, linearly and almost exactly: ~92 bytes per voxel, so ~13 MB
per slice at 512 × 288, from ~650 MB at the default to ~3.9 GB at the ceiling. The
panel quotes the figure beside the slider rather than leaving it to be discovered.

What it buys stops arriving before the ceiling does, and for a reason worth stating:
extinction is calibrated *per filament* (see `raymarch.wgsl`), so a ray's optical
depth grows with the number of filaments it crosses, which grows with the thickness.
Six times the depth is roughly six times the accumulated optical depth, and past some
thickness — dependent on the `intensity` macro, since that sets how much material
there is — the near structure is opaque enough that the far face contributes nothing
the eye can find. Beyond that point the extra voxels are memory spent on material
the camera cannot see. The ceiling is where the shape stops being a slab; the useful
range ends somewhere below it, and where exactly is a judgement for a real GPU.

Making the thickness a knob also forced a re-expression that was overdue. Three of
the march's quantities were fractions of the slab's depth, which is harmless while
the depth is fixed and wrong once it moves: held as fractions, a slab six times
deeper would fade six times as much of its crisp near face at the toroidal seam, and
send its shadow rays six times as far on the same six steps — blotches rather than
shading. Both were calibrated against a *filament*, not against the slab, so both are
now lengths in voxels (`depth_window_voxels`, `shadow_voxels`) and come out the same
absolute size at every thickness. The third is the march itself, which takes one step
per slice up to a ceiling (`volume.steps`, 160), because a thick slab marched at a
thin slab's step count is a ray stepping over whole filaments. Only the amount of
material a ray passes through is left to vary with the knob, which is the whole point
of moving it.

The one thing genuinely lost is lateral resolution, and it is the reason the
layered path remains the default: 512 voxels across against 2560, so filaments are
about five display pixels wide rather than one. Motion is unaffected in the way
that matters — features are five times larger *and* move five times faster in
absolute terms, so feature-widths per second, which is what the eye reads at this
timescale, is the same under both. Memory is the other real cost: ~650 MB of
`rgba16float` against ~90 MB for the 1440p stack, and up to ~3.9 GB if the
thickness is taken to its ceiling.

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
  stall the compositor on the other display and steal focus. `rendercanvas` has no
  fullscreen API of its own, so this reaches past it to the native window
  (`anastomosis/window.py`): Qt's `showFullScreen`, which is a window state and never
  a mode change, and for glfw an undecorated window sized to the monitor it is
  already on — deliberately *not* `set_window_monitor`, which is the exclusive path.
  Toggled with **F11**, from the render window or the control panel, and exactly
  reversible: leaving restores the frame the window had, maximised included.
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
| Intensity | overall activity, contrast, agent count |
| Scale | feature size across all layers |
| Tempo | sim rate, flow strength, drift rates |
| Palette | hue anchor, hue rotation rate, chroma cap |
| Brightness | luminance ceiling and exposure target |
| Depth | layer separation, DOF, atmospheric falloff |
| Parallax | how far the viewpoint drifts, and how briskly |
| Event rate | mean arrival interval of the slow events, and nothing else (§4.3) |

Event rate was originally folded into intensity, and separating them is the one
change to this table worth arguing for. The two answer different questions. How
much material is on screen and how often it is disturbed are independent things
to want — a dense field left alone for an hour at a time is coherent, and so is
a sparse one that keeps being interrupted — and while they were one knob, nobody
could ask for either. It is also the macro most likely to be adjusted *for* a
state rather than for a look: "not right now" is a thing to be able to say to
the events without also dimming the network. Presets carry the rate their
intensity used to imply, so the split moved nothing; configs written before it
have their old rate recovered from their intensity on load, for the same reason.

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
  long run (`tests/morphology.py`, asserted in `tests/test_morphology.py`);
  assert the arrangement is *not* stationary, and that feature size is not
  uniform *across* the field — uniformity is what makes the texture a
  trypophobia trigger, so a churning field of identically-sized features would
  pass a non-stationarity check and still fail the requirement. This is the
  counterpart to the soak test: the soak test asserts the field is alive, and a
  field can be alive and yet look identical for hours (§4.7).
- **No-allocation check.** Assert steady-state buffer/texture count and process RSS
  are flat over a long run.

---

## 11. Module layout

The sketch below is the shape this was planned in; the built layout is flatter
(no `sim/` and `gfx/` packages) and has one addition the sketch does not name.
The two depth backends live in `engine.py` (layered) and `volume.py` (slab), and
everything they share — the output chain from the exposure governor to the
present blit, the flash-safety stage, the parameter mapping, and the device-side
plumbing — lives in `backend.py`. That module is what makes "a clean swap rather
than a fork" (§5.1) true rather than merely intended: the safety stage in
particular is a guarantee enforced by construction, and two copies of it free to
drift apart would be the most expensive duplication in the application.

```
anastomosis/
  __main__.py           entry point
  app.py                canvas, event loop, pacing, hot-reload
  backend.py            shared: output chain, safety stage, parameter mapping
  volume.py             the volumetric slab backend (§5.1)
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
  checkpoint.py         periodic save/restore of simulation state
  ui/                   control surface (TBD)
tests/
  test_flash_safety.py  test_soak.py  test_parity.py  test_config.py
  test_regime.py  test_morphology.py  test_agents.py  test_resize.py
  test_ui_backend.py
  reference.py  morphology.py        numpy reference + measurement, not tests
  test_checkpoint.py
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
shader compiles and the full tick/render sequence runs in CI without a GPU. The
suite is 259 tests and takes about eight minutes there: 248 pass and 11 skip for
want of a display. The checkpoint-determinism check that this section previously
recorded as failing on that adapter passes on the llvmpipe build measured here;
it was never explained, so treat that as an observation about one adapter build
rather than as a fix.

**Complete:** all 30 WGSL modules; the three-system substrate with agents, trail,
reaction, curl-noise flow and pigment advection; the climate field and the
GPU-resident homeostat; slow events; layered compositing with parallax, DOF and
atmosphere; the Oklab colour pipeline; the full safety stage with blue-noise
dither; sim/render decoupling with motion-compensated interpolation and the
budget governor; the parameter system with macros, presets, hot reload and
ramping; the Qt control panel, including asking for an event of a given kind on
demand; CLI; checkpointing on a five-minute interval and
on close, resuming by default, with an explicit reset in the control panel;
shutdown as a single idempotent path reached from the window closing, a signal,
or the loop ending, so closing the window saves the field and ends the process;
and **both depth backends** -- the layered 2.5D stack and the volumetric slab of
§5.1 -- selectable from the config, the command line or the control panel, with
one saved field each so switching between them is not destructive. The slab's
thickness is a control panel knob as well, from 8 voxels to the shorter lateral
axis, priced in graphics memory beside the slider.

**Not implemented:**

- **The morphology work in §4.7**, steps 5–6. Feature size is now polydisperse
  and migrating — the third climate pair drives the reaction's diffusion rate
  per region, over a global mean that walks — which addresses the texture
  itself. Step 4 is in: agents repel from junctions where the climate asks them
  to, respawns land in founding cohorts on bare ground, and a `rift` event
  takes a region's network apart and lets it heal. Its individual mechanisms
  are asserted and its invariants hold, but the aggregate churn it was meant to
  buy could not be resolved above run-to-run variance at test resolution — see
  §4.7. Flux pruning (step 3) is still switched off; the founding respawn it
  was waiting for exists now, but nothing measured says it has earned being
  switched on. Outstanding: the ℓ setpoint (step 5, though the reduce pass
  already carries the wider partials it needs) and trail advection (step 6).
- **Device-loss recovery** is scaffolded in `device.py` but the rebuild path is
  untested, since a software adapter offers no way to provoke a device loss.

**Not yet possible to assess here:** how it actually looks, and whether the
defaults sit in the right place perceptually. The software adapter renders
correct pixels far too slowly to watch. The numbers say the simulation is alive,
structured, and stable; whether it is *pleasant* is a judgement that needs the
real GPU and a pair of eyes.

That caveat is heavier for the volumetric backend than for the layered one, and
worth being explicit about. Its invariants are checked and hold -- the flow is
divergence-free to the storage precision, the slab wraps on all three axes with
no accumulation at the faces, the depth axis carries structure of its own, the
homeostat keeps mean V in band over a long run, and the flash-safety bound holds
through the ray march exactly as it does through the compositor. One statement
about the *image* survives too, and it is the only one: over 700 frames the
exposure governor settles the mean image lightness on its target under the slab
as it does under the stack (0.153 against a target of 0.156), with the exposure
multiplier well inside its bounds -- so the march is handing the output stage
something it can work with, rather than a field too sparse or too dense for the
knobs the two backends share to mean the same thing. But §5.1's own warning that
a volume "makes every parameter harder to reason about" is unaddressed by any of
that. The numbers a slab needs are not the numbers a sheet
needs, and the ones most likely to want moving once someone has watched it are
the agent density (a filament network occupies a much smaller fraction of a
volume than of a plane), the depth anisotropy, the light's ambient floor, and
now the thickness -- which has a defensible range and a cost curve but no
measured answer for where inside that range the image stops improving, since
that is exactly the judgement a software adapter cannot make.
The layered path stays the default until that judgement has been made.
