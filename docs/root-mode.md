> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 15. Rhizotron — a plant-root backend (proposal)

Everything above grows one organism's picture: a mycelial network, luminous
material floating in a dark isotropic medium, seeking itself and fusing. This
section proposes a second *metaphor* — the plant root — and the first thing to
say about it is what kind of thing it is, because §14 just spent a section
establishing that the activation mode is **not** structural, and this one is
the opposite in every particular. A root system is not the fungal field moving
differently. It is a different substrate, a different geometry, a different
topology and a different way of drawing, behind the same output chain. In the
taxonomy the activation work sharpened — `mode` changes tuning on the running
field, `backend` decides what the engine *is* — this is a backend: a third
engine beside `engine.py` and `volume.py`, sharing `backend.py`'s output chain,
safety stage and plumbing exactly as the volumetric slab does, keeping its own
checkpoint, switchable without losing either field. The proposed key is
`backend = "rhizotron"`, after the instrument the look is modelled on: a pane
of glass pressed against living soil, which is how root scientists actually
watch roots grow.

The visual brief from the request is *very different from the fungal mode*,
and the design gets that difference structurally rather than by palette swap.
Three inversions carry it.

### 15.1 The metaphor inverted — what a root is, in this engine's terms

**Fusion becomes ramification.** Anastomosis is convergence: filaments seek
each other, commit to junctions, and the signature image is a mesh — a network
with no root node, where every strand is a peer. A root system is the opposite
figure: divergence from an origin, one axis ramifying into ever finer ends
that never rejoin. The agent rule at the heart of the fungal mode — the fusion
bias, which *sharpens* the turn toward sensed structure so a filament commits
to the junction (§2) — inverts almost literally: a root tip that senses
existing structure ahead turns *away*, because root systems space themselves
out to partition soil rather than piling onto their own strongest strand. One
mode's attraction is the other's avoidance, and everything else follows from
that sign. The fungal image is reticulate; the root image is arborescent, with
a hierarchy the fungal mode structurally cannot produce: axes, laterals,
second-order laterals, fine fuzz — widths spanning an order of magnitude *by
construction*. (Worth noting what that buys against an old enemy: §4.7's
uniformity hazard, the field of same-sized features, is fought in the fungal
mode by measuring and steering feature size. A branching hierarchy carries its
size spread in its topology — a picture of a root system cannot be a field of
identical dots without ceasing to be a picture of a root system.)

**The torus grows an up.** The fungal domain is isotropic — no direction means
anything, and both backends wrap every axis precisely so that nothing
accumulates anywhere (§4.8, §5.1). Soil has a broken symmetry: gravity. The
rhizotron domain wraps laterally (a cylinder — no seam, same §4.8 reasoning)
but is oriented vertically, and *everything* refers to that axis: root growth
biases downward, moisture percolates downward, strata run across, the soil
gets denser and rockier with depth. The vertical axis is not a wall — §15.4
explains why there is no bottom to hit — but it is finally a *direction*, and
an image with a direction in it is already a different image than any the
fungal mode can make.

**The void becomes a matrix.** The fungal image is figure on ground in the
strictest sense: luminous material against darkness, and where there is no
material there is nothing. Soil is a plenum — every pixel of a rhizotron pane
is *something*: mineral grain, organic matter, stone, moisture, a root. The
root mode is therefore full-field in a way the fungal mode never is: the
ground itself has texture, strata, wetness and history, and the roots read
against a material world rather than against a void. This is the largest
single contributor to the two modes not being mistakable for each other at a
glance.

And one thing deliberately does not invert: the discipline. No clocks, no
thresholds, no steps, every change ramped, and the output chain — exposure
governor, slew limiter, gamut mapping, dither — shared verbatim, with
`SAFETY_CEILINGS` remaining one table. The §14 sentence generalises: a new
metaphor is a new simulation upstream of an unchanged safety argument.

### 15.2 The look — a window pressed against soil

The style target, stated as a picture: **a rhizotron pane**. A vertical
cross-section of living soil, seen face-on from a hand's width away. Warm,
granular, stratified ground; pale roots descending through it; the whole
column settling almost imperceptibly downward; after rain, a dark front of
moisture soaking down through the strata. Where the fungal mode is cold light
in a void, this is warm material in a matrix.

**The soil is the palette, and the palette is Munsell.** Soil science has a
canonical colour system — Munsell soil charts, the 10YR and 7.5YR and 2.5Y
pages that field pedologists match samples against — and it is a gift to this
design: a curated, perceptually-spaced gamut of exactly the colours real soil
can be, from ash-grey podzol through ochre and umber to laterite rust and
chernozem near-black. The proposal is to derive the soil ramps from Munsell
notations offline (§15.10) and commit them as Oklab literals, so the ground's
gamut is *inherited from the referent* rather than tuned by hand. The
**Palette** macro then slides along the soil families — grey-brown, ochre,
rust, dark humus — rather than around the full hue circle: the mode trades the
fungal palette's breadth for depth in one earthy register, and that trade is
its colour identity. Chroma stays modest (soil is never saturated), lightness
stays low — the dark-ground identity the README promises holds here as dark
*earth* rather than dark void — and the variation budget is spent on texture
and moisture instead of hue excursion.

**Roots are pale, and age is a colour.** Young roots are white — living root
tips are among the palest things in nature — and older roots suberise and
brown toward the soil. So the root's colour is its age: the growing front is
bright ivory with a translucent tip and a fuzz of root hairs, and material
darkens and dulls behind it, sinking toward the ground colour as it
lignifies. The eye is led to the living edge by the same gradient the biology
provides, and the **Filament glow** macro keeps its meaning exactly — how
luminous the pale living material is against the ground — driving root pallor
instead of hyphal glow.

**Moisture is the weather made visible.** Wet soil is darker and slightly
more saturated than dry soil — one of the most familiar material appearances
there is — and the moisture field (§15.3) drives exactly that mapping:
lightness eased down, chroma eased up, heavily lowpassed. A rain event is a
darkened band soaking down from the surface over minutes; a drought is the
whole column slowly paling. All of it reaches the image through the colour
pipeline's existing lowpasses, and its luminance spend is measured, not
assumed (§15.7).

**Motion is growth, not flow.** The fungal mode's character is that
everything flows — pigment rides a velocity field always and everywhere. The
rhizotron's character is that the world is still and *alive things move
through it*: root tips extend at a few pixels per second, the moisture front
soaks, the column descends at hour-hand speed (§15.4), and nothing else
moves at all. Stillness with purposeful exceptions, against flow without
exceptions — the two modes differ in motion *character*, not just rate, and
the difference is legible in a two-second glance.

**Depth is soil haze between panes.** Three planes of root growth — near pane
native-resolution and sharp, two farther panes at 1/2 and 1/4 like the
layered stack (§5) — separated by translucent soil rather than by atmosphere:
farther roots are dimmed, warmed toward the ground colour and slightly
blurred, as if seen through centimetres of earth. Parallax between panes
comes from the existing viewpoint-drift machinery and keeps its knob and its
meaning. The compositing, DOF and drift code paths are the layered backend's,
reused; only the attenuation's colour target changes (toward soil, not toward
darkness).

**And one grace note.** At the top of **Intensity**, the finest root tips
grow a faint cool shimmer of hyphal thread around them — mycorrhizae, the
symbiosis that connects this metaphor to the application's first one. A few
percent of the image at most, and the one cool accent in a warm field; the
mode remembering where it came from.

### 15.3 The substrate — what replaces the three-system hybrid

The fungal stack (§2) is agents → trail → reaction → flow → pigment. The
rhizotron's stack has the same architecture — a slow climate governing local
parameters, agents depositing into fields, fields shading the image — with
the systems swapped for the rhizosphere's own:

```
climate field  (64×36-ish, slow)      ── per-region parameters, as ever (§4.1)
      │
      ├─► soil matrix  S              ── generated, quasi-static: mineral grain,
      │                                  impedance, stones, strata, organic matter
      ├─► moisture  W                 ── percolates down through S; rain feeds it,
      │        │                         evaporation and root uptake drain it
      │        ▼
      ├─► nutrient  N                 ── patchy; consumed by roots, replenished by
      │        │                         decay and by fresh soil arriving from below
      │        ▼
      ├─► root tips (agents)          ── a few thousand individuals; tropism steering,
      │        │ deposit                 branching, senescence, per-plant character
      │        ▼
      └─► structure  R                ── the root map: density + age + order;
                                         near-zero decay while alive, senescence
                                         returns it to N when it dies
```

**The tips are individuals, not a population.** The fungal agent layer is on
the order of a million anonymous walkers whose aggregate is the picture. Root
tips number in the low thousands, each one visible and consequential: it has
a position, a heading, a branching order, an age, a parent plant, and a
vigour (its share of the carbon budget, below). Steering is a weighted sum of
tropisms, which is not a flourish of naming — it is the actual botanical
control vocabulary, and each term is one sensed gradient:

- **Gravitropism**, toward the tip's *gravitropic setpoint angle* — the angle
  from vertical a root of a given order holds, which is real root biology and
  the single most shape-giving parameter: axes plunge near 0°, first-order
  laterals hold 40–70°, fines wander nearly agravitropic. Per-order setpoint
  distributions, drifted per-region by the climate, are what make a root
  system look like a root system rather than a diffusion-limited aggregate.
- **Hydrotropism**, up `∇W` — roots steer toward moisture, which couples the
  architecture to the weather: growth chases the wetting fronts.
- **Chemotropism**, up `∇N` — and, on high local N, a raised branching rate:
  the documented foraging response, roots proliferating into a rich patch.
- **Thigmotropism**, deflection along impedance gradients in S — tips slide
  around stones and along hardpan rather than stopping, which produces the
  characteristic kinks and runs of real excavated roots.
- **Self-avoidance**, away from sensed R — the inverted fusion bias of §15.1,
  the sign flip that makes a tree instead of a mesh.

Branching is a per-tick probability (climate- and vigour-modulated, never a
threshold) that spawns a lateral of the next order behind the tip, at the
next order's setpoint angle. Senescence is the same shape in reverse: fine
roots carry a mortality hazard that drought and carbon shortage raise, and a
dead segment's R decays over minutes into N — the recycling loop that makes
old growth *fertilise* future growth, a slow spatial memory the fungal mode
has no analogue of.

**Structure is a field, not a polyline list.** Tips deposit soft splats into
R exactly as fungal agents deposit trail — the same §2 argument applies
unchanged: accumulation from many small reinforcements is inherently gradual,
and no single deposit is visible. Width comes from deposit kernel scaling
with order and age (axes broad, fines narrow), and slow secondary thickening
falls out of re-deposit along still-active axes. Crispness — a root's edge
against soil, sharper than anything in the fungal image — is applied at
shading time by a steep-but-C¹ transfer on R, never a threshold, and how
steep that transfer may be is a measured question, not a taste question
(§15.7). R carries (density, age, order-weight) channels so the shading can
do pallor-by-age locally.

**Moisture is the mode's one flowing field, and it is not incompressible.**
The fungal flow is divergence-free by construction because pigment must
neither pool nor drain (§2). Percolation is the opposite: water *should*
pool above hardpan and drain through gaps — that is the imagery. So W moves
by gravity-biased nonlinear diffusion through S's conductivity, with
evaporation near the surface and uptake where living R is, and its
stability comes from bounded flux and clamping rather than from a stream
function. It is a scalar transport pass, unconditionally stable at the small
per-tick steps involved, and everything it does is slow by construction.

**The homeostat is the canopy.** The regulating fiction writes itself: above
the window, unseen, is the shoot — and the root system grows on its carbon
budget. The homeostat measures live tip count, live R mass by order,
front-advance rate and fine:coarse ratio, and steers the *means* of
elongation rate, branching rate and senescence hazard — same PI controller
pattern, same wide deadbands, same minutes-long τ, same rule that
corrections touch OU means and can never step anything (§4.2). A lean budget
grows sparse and deep; a rich one grows dense and fine; neither is ever
allowed to finish.

### 15.4 The descent — how a growing thing never settles

This is the section that decides whether the mode survives its own metaphor,
so it gets the §1(a) treatment. The fungal substrate is a *dynamics* — it
never finishes because nothing accumulates. A root system is a *history* —
growth is cumulative, and a bounded window of it fills, completes, and
becomes a still image. Turnover alone cannot fix that: fine roots can churn
forever, but the axes must persist (they are the picture's skeleton), and a
window whose skeleton is finished is settled in the exact sense §1 forbids.

The answer: **the window sinks.** The view tracks the growing front
downward, forever. Concretely:

- The domain is a **ring buffer over rows**. The world scrolls upward through
  it in whole-row steps; a row that exits the top is retired and a fresh row
  of soil is generated at the bottom. Integer-row shifts mean no resampling,
  no accumulation of interpolation error, and an exactly-representable
  motion vector for the safety stage (§15.7).
- Fresh soil is hashed from `(column, absolute_row, stream)` with the
  existing counter-based PRNG, where `absolute_row` is a **u64 depth
  counter** — §3's pattern precisely: an integer counter as hash input, never
  a phase. Two consequences fall out free: the soil can never repeat (the
  counter does not wrap on any human timescale), and the checkpoint needs
  only the counter and the ring to resume bit-identically.
- The scroll rate is **driven by the front, not by a clock**: a slow
  controller (deadbanded, minutes-long τ, like everything else here) holds
  the deepest active growth around two-thirds of window height. Growth
  surges, the descent quickens; drought stalls the tips, the descent pauses
  with them. The rate is a property of the simulation's state, so §3's
  prohibition is intact — and the pace is glacial, on the order of a window
  per hour at default tempo, a fraction of a pixel per second: the hour hand,
  not the second hand. You do not see it move; you notice, ten minutes
  later, that it has moved.

What the descent buys, beyond survival:

- **Perpetual novelty with zero repetition** — the soil below is always new:
  new strata, new stones, new buried caches, new seeds, drawn from an
  unbounded counter.
- **A narrative axis.** Old axes exit the top thickened, browned, done; new
  soil arrives below carrying the next things to find. The image acquires a
  past above and a future below, which no isotropic mode can have.
- **Deep-time legibility.** Strata crossing the window over hours are the
  slowest visible structure in either mode — a glance an hour apart shows a
  different *geology*, not just a different arrangement.

**Succession, not a specimen.** One immortal plant descending forever would
strain the fiction and concentrate all structure into one ageless skeleton.
Instead the rhizotron grows a community: **seeds** are hashed rarely into
arriving soil, and germinate — smoothly gated, never a threshold — when
moisture reaches them and the carbon budget has room. Each plant draws its
character (setpoint angles, branching density, root tint, tempo) from its
seed's hash: per-plant variation doing the job §4.1's per-region regimes do
in the fungal mode. Plants age; whole systems senesce, decay into N, and
rise out of the frame; new ones are always arriving. The field is a
succession, and succession is the botanical answer to "never settles."

Germination also closes the **absorbing-state analogue** (§4.5): all tips
dead would otherwise be permanent, since roots cannot re-seed themselves
from nothing. Seeds arriving in fresh soil are this mode's `trail_seed_gain`
— the direct injection path that the fiction wanted anyway — and the
homeostat's live-mass band governs how eagerly they wake.

### 15.5 Events in the rhizosphere

Same Poisson scheduler, same envelopes, same concurrency caps, same
"applied to climate and generators, never to pigment or luminance" rule
(§4.3). The kinds are the mode's own:

- **Rain** — the signature event: a moisture pulse across the surface with a
  smooth lateral profile, whose wetting front then takes minutes to soak
  down. Doubly enveloped in effect: the event envelope shapes the input, and
  percolation physics shapes everything after it.
- **Drought spell** — a climate excursion: conductivity down, evaporation
  up, senescence hazard up. Fine roots retreat, the column pales, the
  descent slows. The dieback's counterpart.
- **Nutrient cache** — a buried richness in arriving soil (something died
  here, the fiction says); roots find it and proliferate into it — the bloom,
  as foraging.
- **Hardpan** — a band of high impedance written into *future* rows by the
  soil generator: it arrives by descent, roots pool and run laterally along
  it, find the gaps, and pour through. Non-punctuating by construction,
  since it enters the window at the descent's own speed.
- **Burrow** — a low-impedance channel through coming soil; roots exploit it
  and trace it out, the way real roots follow biopores.
- **Germination** — the manual button's natural kind: ask for a new plant
  now, subject to the same gates a hashed seed faces.

### 15.6 The eight knobs keep their meanings

Per §9, macros are meanings, not parameter lists — and every meaning maps:

| Macro | In the rhizotron |
|---|---|
| Intensity | how much the community invests: germination pressure, branching density, fine-root mass, moisture contrast — and the mycorrhizal shimmer at the top |
| Scale | which orders dominate: left is fine, fibrous, grass-like fuzz; right is coarse, sparse, taprooted |
| Tempo | elongation and percolation rates; the descent follows the front, so it inherits tempo rather than being driven by it |
| Palette | which soil family the column lives in (§15.2); root tint follows for contrast |
| Brightness | unchanged — shared luminance architecture |
| Filament glow | root pallor: how bright living material reads against the ground |
| Depth | soil haze: attenuation and blur between the panes |
| Parallax | unchanged — the same viewpoint drift, between panes |
| Event rate | unchanged — arrival timing only, as ever |

**Scale** deserves a note: in the fungal mode it moves feature size inside
one kind of picture; here it effectively chooses a *flora*, which makes it
the most character-changing knob in the mode and the first thing presets
will differ on. Structurally this lands as curve tables keyed by backend
family as well as mode — `MODE_CURVES` already established tables that share
meanings while diverging in paths and endpoints, and the structure tests
extend to assert every macro drives real paths in every table.

The regulation/activation axis is orthogonal in principle — the rhizotron
would ship tuned for regulation (root growth is inherently the calm half of
this application), with an activation retune as a later, separate exercise
against §14's checklist (§15.12).

### 15.7 Safety analysis — what the root mode actually risks

The per-pixel bound is not at risk, for the standing reason: the limiter is
downstream of everything and neither backend can reach it. The honest risks,
each with its measurement:

1. **The descent must be visible to the reprojection.** §7's limiter
   measures change against motion-compensated history, and the scroll is
   bulk motion of every pixel. Left uncompensated it would be charged as
   change — wrongly, and expensively. Because the sim scrolls in integer
   rows, its motion vector is exact (no resampling error), and the smooth
   sub-row presentation offset rides the interpolator the same way §8's
   motion compensation already works. The residual is then zero for the
   scroll component — better than the parallax residual §5 already accepts.
   Test: the §7 suite run with descent forced fast, asserting the bound with
   and without compensation enabled, so the compensation's correctness is
   what the test pins.
2. **Crisp edges spend more than soft ones, and the sweep decides how crisp.**
   §14 step 2 found the WCAG area criterion toothless against the fungal
   mode's motion — *because* its fields are soft, so a moving edge spends
   many frames crossing a pixel. The rhizotron sharpens edges on purpose,
   which erodes exactly that argument, and tip extension (a bright front
   advancing a few px/s) is the mode's fastest luminance-bearing motion. So
   the §14 workflow applies unchanged: sweep tip speed × transfer steepness
   × tip density at the busy corner, measure the changing-area fraction and
   the per-pixel deltas, and let the sweep set the ceilings on steepness and
   elongation rate. The transfer's licensed steepness is this mode's
   equivalent of the tempo top: derived, not chosen.
3. **Moisture darkening is a real luminance actor.** Wet soil is darker —
   that is the point of it — so the wetting front is a slow, sustained,
   spatially smooth luminance change, the benign kind, but it must be
   *shown* benign: the albedo mapping's full swing is capped well inside
   what the slew limiter passes untouched at percolation speed, and the
   rain-at-maximum case joins the adversarial suite beside the mode-slam
   tests. Drought pallor is the same actor with the sign flipped and a far
   longer τ.
4. **Static contrast is not dynamic contrast, but growth converts one into
   the other.** Pale roots on dark soil is a high-contrast *still* — free
   under §7 — and a growing tip is that contrast arriving at new pixels.
   Deposit accumulation already makes arrival gradual (§15.3, the §2
   argument), and the grow-in ramp is shared machinery; the measurement in
   (2) covers the residual risk.
5. **The comb.** This mode's morphology failure is not the dot lattice —
   §15.1 explains why — but its own: parallel verticals, every axis
   plunging at 0° through homogeneous soil, a curtain of straight cords.
   The mitigations are the setpoint-angle *distributions* (per order, per
   plant, climate-drifted), soil heterogeneity (stones and strata exist to
   be deflected around), and the tropism weights never letting gravity win
   outright. The morphology suite gains an orientation-uniformity measure —
   the §4.7 pattern: name the failure, measure it, steer against it at the
   source.
6. **No thresholds, in new clothing.** Germination gates, senescence
   hazards, branching decisions, the shading transfer: every one is a
   smooth function of state with a C¹ floor, and the counter-based PRNG
   consumes streams, never wall-clock. The §3 audit applies to every
   mechanism in §15.3–§15.5.

### 15.8 What does not change — the invariants, gathered

- `SAFETY_CEILINGS`: one table, all backends, both modes; the per-second
  flash budget of §7 unmoved.
- The output chain in `backend.py`: exposure governor, slew limiter, gamut
  mapping, dither — shared code, untouched, exactly as between the existing
  backends.
- No functions of the clock: the descent is front-driven, seeds and soil are
  counter-hashed, all slow variation is OU walks and stateful fields (§3).
- Events: Poisson arrivals, raised-cosine envelopes, localised, area-capped,
  concurrency-capped, applied to climate and generators only; the rate knob
  moves timing and nothing else (§4.3).
- Ramping: every parameter change slewed; backend switching is structural
  and keeps both saved fields, exactly as layered/volumetric do today.
- Checkpointing discipline (§4.6): everything the next tick reads is in the
  snapshot — R, W, N, S's ring, tips, per-plant state, climate, OU streams,
  and the depth counter — proven the only way that works, by the
  bit-identical-resume test. Its own file (`checkpoint-rhizotron.npz`),
  per-backend as today.
- Zero per-frame allocation; ping-pong texture policy; core WebGPU only
  (§11).

### 15.9 Cost

Cheaper than either existing backend, on the §8.1 envelope. The field stack
is a handful of 2D passes at the layered pyramid's resolutions — S, W, N, R
against the fungal backend's trail/reaction/flow/pigment — with W and N
comfortably at half resolution. The agent pass drops from ~10⁶ walkers to a
few thousand tips. The one addition, ring-buffer row generation, costs a few
rows of hashing per scroll step — noise against the budget. Memory sits near
the layered stack's ~90 MB, nowhere near the slab's gigabytes. No new
resolution, no new frame-pacing questions; §8 applies as written.

### 15.10 Libraries

None at runtime. The stack as it stands — wgpu/WGSL, numpy, the existing
noise and dither machinery — covers everything in this section; crisp
strands come from deposits and a shading transfer, not a geometry library,
and percolation is one more transport pass. The invitation to add
dependencies was considered and is better spent offline, on two design-time
instruments in the §4.4 tradition of *measure the referent, then commit the
numbers*:

- **`colour-science`**, to translate Munsell soil-chart notations into the
  Oklab ramps of §15.2. The Munsell renotation data is the ground truth for
  what soil colours are; the script runs once per palette revision, and what
  ships is a committed table of Oklab literals with the notations in
  comments. A dev-extra at most, imported by a script in `tests/`, never by
  the application.
- **CPlantBox** (the Jülich root-architecture model), to calibrate the tip
  agents' parameter distributions — setpoint angles by order, inter-branch
  distances, elongation rates — against published measurements of real
  species. Run offline to generate reference architectures; distil into the
  distributions the climate drifts around; commit the numbers with their
  provenance. The agents stay this engine's agents; the library's job is to
  keep "looks like a root system" anchored to something other than taste.

### 15.11 Build order

Each step lands something usable alone, and the first two are the risk
retirement:

1. **Soil and descent, no roots.** The ring buffer, the generator, strata,
   stones, the depth counter, the front-controller stub (driven by a fixed
   virtual front until roots exist), moisture with rain events, the Munsell
   ramps, and the scroll wired into the reprojection. Safety suite extended
   per §15.7(1) and (3). What ships is already a piece: a slow core sample,
   strata and weather drifting past forever — and every §15.4 mechanism is
   proven before a single root grows.
2. **The measurement, before the look is trusted:** the crispness/tip-speed
   sweep of §15.7(2), which sets the shading transfer's licensed steepness
   and the elongation ceiling. The §14 lesson applied in advance: derive the
   endpoints, then tune inside them.
3. **One plant.** Tips, tropisms, branching, deposits, pallor-by-age;
   gravitropism and thigmotropism first (they are the shape). An
   `test_agents.py`-style harness: dispatch tips against a manufactured
   soil with a known right answer — a stone to deflect around, a wet patch
   to find, existing structure to avoid.
4. **The long-duration core.** Senescence and recycling, the carbon
   homeostat, seeds and succession; the soak suite's liveness and
   non-repetition assertions against multi-hour runs, and the comb measure
   of §15.7(5) into the morphology suite.
5. **The rest of the rhizosphere.** Nutrients and foraging, the full event
   set, panes with haze and parallax, root hairs, the mycorrhizal accent —
   and the first presets, which are §13-style judgements for eyes once
   there is something to judge. Candidates: `meadow` (fine, fibrous, busy),
   `taproot` (sparse, deep, austere), `loam` (the balanced default).
6. **Later, separately:** an activation retune of the rhizotron against
   §14's checklist, and only if the mode earns it.

### 15.12 Open questions

- **Does the descent read as sinking or as rising?** The design intends "the
  window follows the roots down"; if the eye reads "soil moving up" instead,
  the cure is probably parallax phase and the strength of the strata cues,
  but it needs eyes on a real build.
- **How crisp can the roots actually be?** §15.7(2)'s sweep answers the
  safety half; whether the licensed crispness is *enough* for the vector-like
  look this section promises — or whether the mode's identity softens toward
  painterly — is the biggest aesthetic risk here, and it is exactly the kind
  §13 says needs a real GPU and a pair of eyes.
- **Succession pacing.** Plant lifetimes against the descent rate decide how
  often the field holds a mature system versus ruins and seedlings; pure
  judgement, preset territory.
- **One pane or three.** The rhizotron fiction is honestly 2D — a pane of
  glass — and the haze-separated panes may read as richer or may read as
  three fictions stacked. Build flat first (step 3 needs no panes), decide
  at step 5.
- **A volumetric rhizosphere** — raymarched soil, roots at all depths — is
  explicitly out of scope: it would inherit §5.1's lateral-resolution
  problem in the mode whose whole identity is fine pale strands. If it ever
  happens it is a fourth backend, by the same rules as the other three.
