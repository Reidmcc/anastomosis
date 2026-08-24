> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

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
  up to ~230 MB and this is written every fifteen minutes for days. `velocity` and
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
