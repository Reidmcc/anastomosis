> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 14. Sensory activation — a second mode

Everything above serves down-regulation: slow, dark, quiet, an input to settle
against. Sensory seeking is the other half of stimming — wanting *more* input,
not less: movement, colour, novelty, things happening. The question this
section answers is whether the same engine can serve it, and the answer is yes,
for a structural reason worth stating before any of the tuning:

**The design already separates "never punctuate" from "be calm", and only the
first is enforced.** The safety stage (§7) bounds the output whatever the
simulation does; the no-punctuation discipline — ramped parameters, enveloped
events, no thresholds, no clock — is enforced at stages a second mode would
not touch. The *calmness*, by contrast, lives entirely in tuning: macro curve
endpoints chosen gently, presets that are "nowhere near energetic" on purpose,
a palette that keeps neighbourhoods in one hue family. An activation mode is
therefore a retuning inside an unchanged safety envelope, not a fork — the
same relationship to the safety argument that the volumetric backend has to
the output chain.

### 14.1 What activation is, in this engine's terms

Not flashing. That is excluded by construction and stays excluded; the point
of building the limiter downstream of everything (§1b) was exactly that a
regime nobody has tuned yet cannot produce the thing the application must
never do. The energy has to come from the channels the safety argument
deliberately leaves free, and it is worth listing them because together they
are most of what "stimulating" means:

1. **Motion.** The limiter measures change against the motion-compensated
   previous frame (§7), so honest translation is exempt *by design* — a
   filament sweeping across the screen costs the luminance budget almost
   nothing. Faster flow, faster agents, brisker parallax: more stimulation at
   near-zero spend against the bound.
2. **Chroma and hue.** `safety.max_chroma_delta` ships at 0.030 against a
   ceiling of 0.100 — the design's own judgement that chroma change is far
   less provocative than luminance change, and a budget that is mostly
   unspent. Colour variety is the request's first ask, and it is the cheapest
   thing on this list.
3. **Density of incident.** Event rate, event overlap, shorter envelopes:
   more of the field's time spent inside something happening (§4.3 already
   frames the rate knob this way).
4. **Novelty rate.** Faster morphology drift, faster climate migration —
   regimes arriving and moving on in minutes rather than tens of minutes.

And one channel it must not come from: **luminance dynamics**. The
`max_luma_delta` ceiling stays 0.012 in both modes. Activation cannot buy
energy with brightness swings, and that constraint is what makes the mode
shippable at all — it is the difference between "energetic" and the thing the
README promises the application will not do.

### 14.2 Mechanism: per-mode curve tables, not a second engine

Three shapes considered:

- **Just presets.** Insufficient. Presets move within the existing macro
  ranges, and those ranges were capped gently on purpose (`presets.py`: "even
  the top preset is gentle"; `tempo` tops out at `sim_hz = 26`,
  `hue_turns_per_hour = 2.6`). Activation lives beyond the top of several of
  them, and simply extending the shared ranges would spend most of every
  slider's travel on territory the regulation user never wants — worse
  resolution for the primary use to serve the secondary one.
- **A separate simulation or substrate.** Overkill. Nothing about activation
  wants different *mechanisms* — it wants the same field moving faster and
  coloured more variously. Two engines would also mean two of everything §11
  says must never fork.
- **`MACRO_CURVES` keyed by mode.** The same eight macros, the same meanings,
  different endpoints. This is the right shape, and it is small: `resolve()`
  looks up the active mode's table; `curve_value()` (the slider readouts)
  takes the mode; everything else is unchanged.

So: a `mode = "regulation" | "activation"` key on `Config`, beside `backend`
— but with the opposite character, and the difference is the point.
`backend` is structural: nothing ramps it, and it decides what the engine
*is*. `mode` is **not structural**: it changes no geometry, allocates
nothing, and every value it moves is one the ramp already smooths. Switching
modes is therefore a live transition on the running field, exactly like a
preset switch — the field you grew stays on screen and changes character over
seconds. That matters for the use case: the person reaching for activation is
switching *states*, and being made to regrow a field from seeds at that
moment would be the tool refusing its purpose.

Presets carry a mode. The existing seven are regulation presets and do not
move; activation gets its own small set (three to start), and the panel's
preset list shows the active mode's. One selector, near **Depth** at the top
of the panel, since it is the other "which application is this" control —
but unlike Depth it acts on the field you already have.

### 14.3 Where the activation curves go — a first cut

Endpoints to be tuned on real hardware, but the shape and the reasoning can
be fixed now. Low ends stay at or near the regulation table's — the modes
overlap rather than abut, so the bottom of activation is recognisably the
same instrument.

| Macro | Regulation top | Activation top (proposed) | Why / bound |
|---|---|---|---|
| tempo → `sim_hz` | 26 | 30 | Budget: sim is ~14 GB/s at 20 Hz (§8.1), so 30 Hz is ~21 GB/s — still ~3% of bandwidth. **Not** `max_fps`: see 14.5. |
| tempo → `agents.speed` | 1.35 | ~2.2 | Perceptual; bounded by the WCAG-area sweep (14.5). |
| tempo → `flow.psi_gain` / `field_gain` | 2.10 / 1.30 | ~4.0 / ~2.4 | Same sweep. Advection is unconditionally stable at any speed (§2), so the bound is perceptual and WCAG-area, not numerical. |
| tempo → `hue_turns_per_hour` | 2.60 | ~8 | A full turn in 7–8 minutes: visible drift rather than a spin. Ramp tau 8 s already smooths changes to it. |
| palette → `render.hue_spread` | 1.25 | ~2.8 | Most of the circle in play at once; see 14.4. |
| intensity → `render.chroma_activity_gain` | 8.0 | ~11 | With `c_max` raised toward its 0.22 ceiling (shipped 0.145) and `chroma_floor` up, so quiet regions stay coloured instead of grey. |
| event_rate → `events.rate_per_hour` | 20 | ~40 | One every ~90 s at the top. Arrival-time only, as ever; envelope changes are separate and deliberate (14.6). |
| scale | 0.16–0.26 (`du`) | shifted ~15% finer | Busier texture. The §4.9 sensing-ratio invariant and clamp are untouched — the ratio stays ~2.6 across the activation range too, which `test_config.py` already knows how to assert. |
| brightness, filament_glow, depth, parallax | — | shared table | Nothing about activation wants different luminance architecture, and parallax already reaches a quarter of the screen. |

Two things the table deliberately does not touch: the reaction's `feed`/`kill`
ranges and clamps (§4.4 measured the live band; activation must not walk
regions off the map any more than regulation may), and every entry in
`SAFETY_CEILINGS`, which remains one table serving both modes.

The homeostat may need per-mode *targets* — a faster regime holds a higher
mean activity, and a controller centred on the regulation band would lean
against the mode with `corr_decay` for the whole session. Whether it actually
does is a measurement (run the activation endpoints against the existing
bands and watch the corrections), and the fix if needed is a small table:
targets per mode, same controller, same deadband philosophy. Corrections
still reach only the OU means, so a mode switch cannot make the controller
step anything. *(Measured in step 2: it does not lean — the candidate
endpoints sit nearer the targets than the regulation busy corner does, so
no per-mode targets exist. See the step 2 record in §14.8.)*

### 14.4 The one genuinely new mechanism: a polychrome palette

"More varied colours" is the part tuning alone cannot buy. Hue today is
`anchor + spread × (orientation, species ratio, climate deviation)` — one
anchor, so however wide the spread, the field is neighbourhoods of one hue
family with excursions. Activation wants simultaneous *contrasting* families
— a field that is teal here, ember there, violet in the region arriving from
the left.

The mechanism that fits the existing architecture: keep one drifting global
anchor, and let the **climate hue channel choose among K offsets from it**
(K = 2 or 3, e.g. 0° / +120° / −120°) — the per-region hue becomes
`anchor + offset(climate_hue) + spread × (local terms)`, where `offset()` is
a smooth periodic warp of the channel toward the K wells, not a selection.
No thresholds (the warp is C¹), spatially smooth by inheritance (the climate
is 64×36, bilinear, diffused — §4.1 "it can never introduce a hard edge"),
and temporally slow because the channel is the same OU-driven advected field
it always was. Regions of distinct colour family form, migrate and hand over
exactly the way regimes already do. In regulation mode the warp gain is zero
and the mapping is bit-identical to today's.

All of it happens in OkLCh with `L` untouched, so colour variety spends the
chroma budget and none of the luminance budget. Gamut mapping at constant
`L` and dither are downstream and unchanged. One thing to measure rather
than assume: hue *rotation rate* interacts with chroma — at `c_max = 0.145`
a 120° hue distance is a Δ(a,b) of ~0.25, and a region crossing between
wells too quickly could spend chroma delta faster than `max_chroma_delta`
allows, which would surface as the limiter visibly dragging colour. The
climate's own timescales make this unlikely (the channel moves over minutes),
but the flash-safety suite's chroma assertions should be run at the
activation endpoints before the endpoints are trusted.

The second candidate for new machinery already has a design, a build slot
and a warning label: **trail advection** (§4.7 step 6, "the largest payoff
for dynamism and the largest risk"). Its natural home is this mode — gain
zero in regulation, modest in activation — which also answers the question
§4.7 left open of how to ship it safely: behind a mode nobody is in while
seeking calm. It stays last in the build order, as §4.7 always said.

### 14.5 Safety analysis — what activation actually risks

The per-pixel bound is not at risk; it is enforced downstream and neither
mode can reach it. The honest risks are these, each with its measurement:

1. **The WCAG area criterion is the binding constraint on tempo.** The suite
   asserts fewer than 25% of pixels change by ≥10% in a single frame — at
   fixed pixels, as WCAG measures, so *honest motion counts against it* even
   though the limiter rightly permits it. §4.7 already flagged this metric
   as the one churn moves. This inverts the tuning workflow: the activation
   tempo endpoints are not chosen and then tested, they are **derived from
   the sweep** — run the tempo axis (speed, flow gains, sim rate) headless,
   measure the changing-area fraction, set the curve tops where the margin
   is still comfortable (say, area ≤ 15% sustained). §4.4-style: measure the
   map first, then pick the operating point. A translating filament changes
   pixels only at its edges, so the fraction scales with speed × perimeter
   density — which couples this sweep to `intensity` and `scale`, and means
   the sweep must run at the *busy* corner of the mode, not the default.
2. **The frame-rate arithmetic must be pinned before any of this.** The
   ceiling argument in §7 is per-frame at 30 FPS: 0.012 → 1.8 flashes/s. At
   `max_fps = 60` — which `SAFETY_CEILINGS` permits today — the same
   per-frame delta arithmetic yields 3.6/s, over the WCAG limit. This is a
   latent issue in the current table, not something activation introduces,
   but a mode whose whole character says "faster" is how someone finds it.
   Before the mode ships, the luma-delta ceiling should be expressed as a
   per-second budget divided by the actual frame rate (or `max_fps`'s
   ceiling dropped to 30, which the §1 design cap argues for anyway), and
   `test_ceiling_implies_wcag_margin` extended to assert the pair jointly.
   Activation itself does not raise the frame rate: rapidity comes from the
   field moving faster, not from more frames of it.
3. **The mode switch is a large coordinated parameter movement.** It rides
   the existing ramp (every path smoothed, hue along the shortest arc,
   `sim_hz` snapping being invisible by §8's decoupling), so the machinery
   is already the right machinery — but "switch modes back and forth while
   asserting the flash bound" belongs in the adversarial suite alongside the
   existing parameter-slamming tests, because it is precisely the kind of
   input those tests exist to distrust.
4. **Trypophobia (§4.7) tightens, not loosens.** Finer scale plus higher
   intensity is movement *toward* the dense-spot-field geometry. The
   morphology suite — non-stationarity, size non-uniformity — runs against
   the activation endpoints too, and the polychrome palette helps here
   (varied colour breaks the uniformity that makes the pattern a trigger)
   but is not the mitigation of record; the size-spread mechanisms are.
5. **Photosensitivity posture is unchanged and must be said.** Same bound,
   same enforcement, same README caveat, now stated for both modes: the
   activation mode is more stimulating *within* the same tested envelope,
   not a relaxation of it.

### 14.6 Events at activation tempo

The rate knob already goes to one-every-three-minutes and stays
arrival-time-only. What activation additionally wants is **shorter
envelopes** — at 40/hour with a 45 s attack and 90 s release, events cease
to be incidents and become weather. Proposal: the activation curve set may
shorten `attack_seconds` toward ~15 and `release_seconds` toward ~40. That
leaves the envelope raised-cosine (never a step), the radius under the 25%
area cap, amplitude untouched, and the effect still arriving through the
climate's own diffusion and the colour pipeline's lowpasses — every §4.3
constraint intact except the specific "30–180 s" attack figure, which this
section deliberately revises for one mode and the soak/heal tests re-verify
at the fast end (a rift that arrives in 15 s must still heal; §4.7's healing
measurements were envelope-length-independent in mechanism but should be
re-run). `max_concurrent` can rise 4 → 6; the cap exists to bound overlap,
and overlap is the point at this end of the knob.

### 14.7 What does not change — the invariants, gathered

- `SAFETY_CEILINGS`: one table, both modes; `max_luma_delta` ≤ 0.012.
- The output chain in `backend.py`: limiter, exposure governor, gamut
  mapping, dither — shared code, untouched, exactly as between the two
  depth backends.
- No functions of the clock anywhere new; all added variation is stateful
  walks and fields, per §3.
- §4.9's sensing-ratio clamp and §4.4's reaction-band clamps, at every
  activation endpoint.
- Ramping: no parameter change steps, mode switches included.
- Events: enveloped, localised, area-capped, concurrency-capped, applied to
  climate only, arrival-rate knob moves timing only.
- The checkpoint format: `mode` is not structural, so one field serves both
  modes per backend, nothing new is stateful outside existing fields, and a
  session checkpointed in one mode resumes cleanly into the other. (The
  polychrome warp is a pure mapping; trail advection adds no state the
  trail texture does not already carry.)

### 14.8 Build order

Each step lands something usable alone, and the first is deliberately
invisible:

1. ~~**Mode plumbing.** `mode` on `Config`, `MACRO_CURVES` keyed by mode with
   the activation table starting as a copy of regulation's, panel selector,
   preset mode-tagging, ramped switch. Tests: `resolve()` respects mode,
   ceilings clamp identically in both, switch-slamming holds the flash
   bound. No visual change yet.~~ **Built** — and the switch-slam test paid
   for the whole step on its first run; see below.
2. ~~**The two load-bearing measurements**, offline, before any endpoint is
   trusted: the tempo/WCAG-area sweep (which *sets* the tempo tops), and
   the frame-rate ceiling fix of 14.5(2) (which is due regardless).~~
   **Built** — and both measurements returned answers the plan did not
   predict; see below.
3. ~~**The activation curve set** for tempo, palette, intensity, scale and
   event_rate, endpoints from (2); homeostat per-mode targets if the
   measurement says the controller leans (14.3).~~ **Built**; see below.
4. ~~**The polychrome palette warp**, regulation-identical at gain zero, plus
   the first activation presets.~~ **Built**; see below.
5. ~~**Shorter event envelopes and higher concurrency**, with soak and heal
   tests at the fast end.~~ **Built**; see below.
6. ~~**Trail advection** behind an activation-only gain. Last, as §4.7
   always said.~~ **Superseded**: it shipped on `main` as a default-on fix
   while this branch was in flight, so what landed here is a harder shear
   under activation rather than the mechanism. See below.

Steps 1–4 are the mode; 5–6 are its depth.

### What step 1 actually did

The plumbing is as prescribed: `MODE_CURVES` keyed by mode with the
activation table an exact copy of the regulation one, `mode` on `Config`
beside `backend` but non-structural, `curve_value` taking the mode so the
panel's readouts quote the table that is actually driving, presets tagged
with the mode they were tuned in (`presets.PRESET_MODES` — a separate table
rather than a field on `Macros`, because a mode is not a knob and `Macros`
is the shape of the eight sliders), and a Mode selector at the top of the
panel that asks no question before acting, because unlike every other
selector up there it loses nothing. What the tests assert is the structure —
both tables driving the same macros and paths, one ceiling table serving
both modes — rather than the temporary equality of the values, which step 3
exists to break.

**The switch-slam test found a real hole in the safety stage on its first
run.** Stepping everything both modes' tables drive between opposite macro
extremes, un-ramped, flow off so the per-pixel bound is exact, produced a
lightness step of **0.0111 against the 0.0100 budget** — from a pixel whose
stored value was exact black and whose neighbour-frame had one channel at
exactly zero, the signature of a clamp. The mechanism is the §7 constraint
("gamut mapping must not let out-of-range values into the buffer") in a
corner the earlier tolerance-tightening missed: from a black history the
limiter permits a step of (+`max_luma_delta`, ±`max_chroma_delta`,
±`max_chroma_delta`), which at that lightness is far out of gamut; the
bisection in `gamut_map_oklab` accepted trials with channels down to −1e-6;
and the final clamp raised them to zero, which near black raises `L` — the
cube root's slope is unbounded there, so even a e-6 tolerance is worth
~1e-3 of `L`. Modelled in numpy: a maximal step off black stores `L` up to
**0.0125** under the −1e-6 acceptance and exactly **0.0100** with the low
side exact. The fix is that asymmetry, in `in_gamut`: no tolerance at all
below zero, tolerance kept above one, where clamping down at 1.0 moves `L`
by ~3e-7 and refusing it would send every bright pixel through the
bisection for nothing.

Two things about the test that carried the finding. The regression is
pinned by a *deterministic* test rather than the slam that found it: the
mode-slam's leak rode on which trajectory a chaotic field wandered into,
so the dedicated test manufactures the corner instead — chroma floor
raised to the chroma ceiling, so every pixel including black ones demands
full chroma from the first frame; the chroma slew limit at its
user-settable ceiling of 0.10, because the leak grows with the chroma
step; the hue anchor flipping by π each frame, so the demand stays a
*change* in chroma rather than a satisfied one. Under the pre-fix shader
that fails on ~1000 pixel-frames at 0.0107, not on one lucky pixel, and
every knob in it is a value a user can legitimately set. And the mode-slam
test itself stays in the suite unchanged: today the two modes contribute
identical values and the macro extremes do the work, and when step 3
retunes the activation endpoints their divergence rides into the same
assertions with no change to the test.

### What step 2 actually did

The ceiling fix went in as 14.5(2) preferred: the bound is now expressed
the way the arithmetic runs, as a per-second budget (`MAX_LUMA_PER_SECOND`
= 0.36/s, i.e. 0.012 × 30) that `validate` divides by the frame-rate cap
after both individual ceilings have been applied. §7 records the details.
The two measurements are the substance of the step, and each came back with
an answer the plan did not predict.

**The WCAG area criterion does not bind the tempo axis — anywhere the sweep
could reach.** `tests/tempo_sweep.py`, at the busy corner (intensity, glow
and brightness at 1.0, scale at its finest, two layers, simulation at 30 Hz
with one render per tick so a rendered frame is a display frame), 300
frames of warm-up and 90 measured, sweeping a single multiplier on the
three motion primitives the tempo macro drives — `agents.speed`,
`flow.psi_gain`, `flow.field_gain` — over their regulation tops:

| mult | area ≥10%/frame (p95) | area ≥5%/frame (p95) | peak per-pixel \|ΔL\|/frame |
|---|---|---|---|
| 1.0 | 0.0% | 0.0% | 0.018–0.019 |
| 2.0 | 0.0% | 0.0% | 0.026–0.027 |
| 4.0 | 0.0% | 0.0% | 0.035 |
| 6.0 | 0.0% | 0.0% | 0.036–0.043 |

(Ranges span the two sizes measured, 128×96 and 224×126; medians and
maxima are identical to the p95s at 0.0%.) The proposed activation tops of
§14.3 sit between 1.6× and 1.9× on this axis. Not a single pixel-frame
reached even *half* the 10% flash threshold at *six times* the regulation
top: the worst per-frame per-pixel change grows from 0.018 to 0.043 across
the whole sweep. The reason is structural rather than lucky. Per-frame
material motion at these speeds is sub-pixel, and the pigment field is
smooth by construction — soft deposits, incompressible advection, upstream
lowpasses (§2) — so a moving edge spends many frames crossing any one
pixel, and each frame's share of the crossing is small. The design's
anti-punctuation machinery, built for stillness, turns out to be what
licenses speed.

So §14.5(1) inverts back: the sweep was meant to *set* the tempo tops, and
instead it certifies that the criterion leaves them free — the activation
tempo endpoints are perceptual choices, to be judged on real hardware, with
their WCAG headroom now on record. Three caveats keep the conclusion
honest. A small field overstates the area fraction (features cover more of
the screen, so their moving edges weigh more) — and the overstated figure
is zero, so the error runs in the conclusion's favour. Trail advection
(§4.7 step 6) remains the change most likely to move this metric, which is
why it ships behind a gain and why the sweep script is committed rather
than run once and discarded. And the peak figures are honest motion that
the limiter's reprojection deliberately permits, not leaks — the per-pixel
bound against motion-compensated history held throughout, as it must.

**The homeostat does not lean on the activation endpoints.** Three
2400-tick runs at 128×96 (seed 9): regulation defaults, the regulation
busy corner, and the candidate activation endpoints (`speed` 2.2,
`psi_gain` 4.0, `field_gain` 2.4, `advect_gain` 0.55, 30 Hz, at the same
busy corner). Settled over the last 1200 ticks:

| | mean V | activity | corr_kill at 2400 |
|---|---|---|---|
| regulation defaults | 0.1160 | 0.001335 | −0.00071 |
| regulation busy corner | 0.1255 | 0.001118 | −0.00094 |
| activation candidate | 0.1111 | 0.001242 | −0.00078 |

The candidate sits *nearer* both targets (mass 0.118, activity 0.0012)
than the regulation busy corner does, every measure is comfortably inside
its ±30% deadband, and the controller's corrections are smaller under the
candidate than under the regulation corner — two orders of magnitude below
the integral limit, still carrying the shared grow-in transient. The
per-tick dynamics the reaction sees are barely moved by the tempo axis:
`speed` changes how far an agent walks per tick, but deposit, decay, feed
and kill are untouched, and the flow gains move pigment, which the
homeostat does not measure. **Step 3 therefore adds no per-mode homeostat
targets** — the bands of §4.2 serve both modes, which is one less way the
two tunings can drift apart. The caveat is horizon: 2400 ticks is minutes,
and the multi-hour answer belongs to the soak test once step 3 fixes the
real endpoints.

### What step 3 actually did

The endpoints landed essentially as §14.3 proposed, now that the
measurements license them: `ACTIVATION_CURVES` in `config.py` is a full
literal beside the regulation table rather than anything derived from it —
a tuned table should read like one — with the low ends at regulation's
throughout, so the bottom of activation is recognisably the same
instrument. The one addition the proposal's table only gestured at:
`c_max` rises to 0.205 (toward the 0.22 ceiling, not to it — gamut-mapping
pressure grows with chroma and the margin is deliberate) and
`chroma_floor` to 0.035, both on the intensity macro. Driving them needed
a small structural pattern worth naming: the two tables must drive the
same paths (the structure test forbids a slider going dead across modes),
so regulation's intensity curve now carries those two paths as
*constant* entries, pinned at the defaults its look was tuned with. A flat
curve is the honest way for one mode to say "not this lever" while the
other uses it.

Three tests hold the endpoints to their evidence rather than merely to
themselves. The tempo tops must stay inside the swept certificate — at or
under 6× the regulation motion tops, which is as far as step 2's sweep
measured and no further; a retune past that fails the test until
`tempo_sweep.py` is re-run. The activation scale curve must not dig the
`du` floor below regulation's (§4.7's activity-collapse edge) and must
hold `dv/du` at 0.50 at both ends. And the §4.9 sensing-ratio assertion
now runs per mode, since each mode's scale curve sweeps its own range and
the bifurcation does not care which tuning walks over it (the activation
ratio spans 2.58–2.59). At the GPU level, the WCAG area criterion is
pinned at the shipped activation top over the fresh-start frames, and the
soak suite gains the activation twin of the long-run liveness test — same
thresholds, same homeostat-convergence assertion, at the busy corner of
the mode. All pass, the mode-slam test of step 1 now slamming genuinely
divergent tables.

What did *not* change is as load-bearing as what did: no per-mode
homeostat targets (measured unnecessary in step 2), no event envelope
changes (step 5's, deliberate), no touched `feed`/`kill` ranges or
clamps, and the luminance architecture — brightness, glow, depth,
parallax — shared verbatim. The chroma budget and the motion budget carry
the mode, exactly as §14.1 argued they could.

### What step 4 actually did

**The warp is a C∞ three-plateau staircase, applied at hue injection.**
`polychrome_offset` in `common.wgsl` (mirrored in `tests/reference.py`,
property-tested rather than merely ported): plateaus at −2π/3, 0 and
+2π/3, tanh transitions whose steepness is tied to the threshold
(k = 2.5/t) so one value moves the wells and their ramps together, gain
scaling the whole triad so the parameter ramp is a smooth widening rather
than a gate. It rides the climate hue channel at the *injection* site in
both backends — the one place hue enters pigment — so material carries its
family with it and the families migrate exactly as regimes do. The
threshold defaults to 0.06 in the channel's realised units (§4.1's
σ ≈ 0.11), placing roughly two fifths of the field in the middle family.

**The gain lives on the activation intensity curve, not on palette.** The
palette macro says *where* the families sit (they all ride the anchor);
polychrome says *how much* colour contrast the field carries, which is an
intensity question — and its low end at zero keeps the bottom of the
travel the same instrument as regulation, like every other activation
curve. Regulation pins the gain at zero through a constant curve entry,
the same pattern `c_max` and `chroma_floor` use.

**Measured on the rendered image**, with injected hue isolated
(`hue_inject_mix` 1, orientation and spread zero, 500 ticks): at gain 0
the output hue concentration R = 1.000 — one hue, the regulation mapping
exactly; at gain 1 it falls to 0.650. The output triad is softer than the
injected one (the tri-modal statistic is low): bilinear sampling mixes hue
vectors across family boundaries, the interpolator blends them again, and
the chroma slew limiter lags migrating families — at the default threshold
about a third of the field sits between plateaus at any moment. Whether
that reads as marbled richness or as mud, and whether the threshold should
tighten (fewer, crisper transitions), is precisely a §14.9 judgement for
real hardware; the conservative default errs toward smoothness, which is
the safe direction to err.

**Presets: `prism`, `cascade`, `spark`** — colour-forward, motion-forward,
and busiest-texture respectively, each a step short of the measured corner
on every axis, all keeping the dark ground. First cuts for the judgement
§13 defers to eyes, not its record. A preset now *brings its mode with
it*: the CLI's `--preset` writes the preset's mode into the config
(`presets.mode_of`), because macro positions only mean something through
the table they were tuned against — `spark` resolved through the
regulation table would be a picture nobody has judged under a name that
promises one somebody has. The panel needed no such change: its preset
list was mode-filtered from step 1.

Nothing stateful was added: the warp is a pure mapping, the checkpoint
format is untouched, and a field saved under either mode still resumes
into the other.

### What step 5 actually did

**The envelope rides the tempo macro, not the event-rate one.** §14.6 said
"the activation curve set may shorten the envelopes" without saying which
knob carries them, and the choice matters more than the values: §4.3
promises — in the code, the panel tooltip and the README — that the rate
knob moves *when* events come and nothing about what they do, and hanging
the envelope on it would have quietly broken that promise in one mode.
How briskly a perturbation builds is a tempo question, so activation's
tempo curve takes `attack_seconds` 45 → 15 and `release_seconds` 90 → 40
across its travel, with the slow end shared — a low-tempo activation
field keeps the minute-long arrivals, and only the fast end gets the
brisk ones. Regulation's tempo curve pins both at 45/90 as constants,
the same paired-constant pattern as `c_max`.

**The concurrency cap is a per-mode value wearing a curve entry.** A
constant is not a coupling: `max_concurrent` sits on the event-rate curve
at 4 in regulation and 6 in activation, flat in both, so the knob itself
still moves only the rate — the parametrised rate-isolation test is what
keeps that honest — and the *mode* sets the cap. Six because at one event
every ~90 s with ~2-minute envelopes, overlap is the fast end's normal
condition, and a cap that refused it would turn the top of the knob into
a queue of refusals.

**The safety claims at the fast end were mostly already held.** The
envelope-step test now runs per mode at the worst case — the 0.75 jitter
floor of the 15 s attack, ticked at 30 Hz — and the raised cosine moves
~0.005 of its range per tick there, a quarter of the existing bound. And
the heal requirement ("a rift that arrives in 15 s must still heal")
turned out to be *already verified*: the rift recovery test has always
run its attack compressed to 200 ticks — 10 s at 20 Hz, faster than
anything the activation table can ask for — because the severance
feedback needs the hold, not the ramp. That is recorded here instead of
duplicating a multi-minute GPU test to re-prove it.

One small honesty fix rode along: the panel's "it comes up over the next
minute or two" note now reads the live attack time, since under a fast
activation tempo that sentence would describe a fault rather than the
event.

### What step 6 actually did

**It was built twice, independently, and the other one won.** This branch
implemented trail advection as a separate pass behind an activation-only
gain — and while it was in flight, `main` shipped the same mechanism from
the other direction: not as an activation flourish but as the fix that
dissolves the trail hubs, on by default at `trail_advect = 0.5`, folded
into the trail pass, measured against blob-carry and mass-conservation
tests (see §4.7). The two arrived at the same non-obvious architectural
conclusion for the same reason — **the velocity pass must move ahead of
the trail pass**, so that `velocity` stays a derived field rather than
becoming checkpoint state (§4.6) — which is worth recording as evidence
that the constraint is real and not a matter of taste.

The merge kept main's, entirely, and this is the more interesting half of
the record. A duplicate mechanism is easy to spot; what nearly slipped
through is that this branch's *curve entry* pinned `agents.trail_advect`
to **0.0** in regulation, written when the mechanism was believed not to
exist. Merged naively that would have silently switched off a shipped fix
for a real visual defect — the persistent light dots — in the mode that is
the application's default. The regulation curve therefore now pins the
value at its shipped **0.5**: a constant *at the default*, not a disable,
and `test_activation_shears_the_structure_harder_without_disabling_it`
exists to catch exactly that mistake being made again.

**So activation's contribution is no longer the mechanism but more of
it**: `trail_advect` rises to **0.8** at the top of the activation tempo
curve, against the 0.5 both modes share at the bottom. Deliberately under
1.0 — at parity with the pigment the network would ride the flow exactly
as its colour does, and the shear that stretches and pinches filaments is
precisely the *difference* between the two rates, so parity would remove
the effect being asked for. §4.7's WCAG-area worry was measured on this
branch before the merge and stands: at the activation top, both sizes,
the area fraction is 0.0% at the 10% *and* 5% thresholds with the gain on
or off, and the worst per-pixel per-frame ΔL moves only 0.024–0.026 to
0.023–0.028 — structure carried at a fraction of pigment speed is slower
than motion already certified to 6× (§14.8 step 2). Whether 0.8 is the
right amount of shear is, as ever, §14.9's question for eyes.

The same caveat as §13, sharpened: every endpoint above is an argument, not
a judgement. Specifically open: whether activation keeps the dark ground
(the README calls it an identity; a brighter ground is more activating and
also more fatiguing — the guess here is the ground stays dark and the
*chroma* carries the mode); whether the volumetric backend wants its own
activation deltas (a faster slab probably wants more thickness and more
anisotropy before more speed); and where "energetic" tips into "stressful",
which is the whole game and needs eyes. The presets are where that judgement
gets recorded once made.
