> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited
> here, in other docs, and in code comments resolve via the index in
> `DESIGN.md`.

## 17. Perennial — the pane that holds

§15 built the rhizotron and §13's rule — judgements wait for eyes — then
caught up with it: sustained watching found the mode mechanically rich and
visually unrewarding, and it was shelved. This section is the redesign. It
keeps the substrate §15 built — the tropism-steered tips, the deterministic
birth tree, the percolation, the nutrient economy, the Munsell ground — and
replaces the two decisions that watching disproved: the descent, and the
absence of any visible *commitment*. The backend key stays `rhizotron`;
what changes is what the pane is a picture of.

### 17.1 What watching found

Three failures, each with its mechanism, all confirmed on headless stills
from the real backend at shipped defaults:

1. **The ground is invisible.** §15.2 promised warm material in a matrix;
   what ships spans Oklab L 0.034–0.19 — the entire soil system (strata,
   stones, moisture fronts, the Munsell families) renders as black. The
   image is figure-on-void, which is the one composition §15.1 said this
   mode structurally would not be. The cause is arithmetic, not taste:
   `soil_l_range` re-anchors the chart chips into a lightness envelope
   tuned for the fungal void's *background*, where the fungal modes put
   nothing the eye must read. Here every pixel is supposed to be legible
   material, and the envelope prices that at black.

2. **There is no value hierarchy.** Young ivory, mature root and old wood
   all render as one khaki. The shader has pallor-by-age; the *dynamics*
   defeat it. The structure field's age channel is seconds since last
   reinforcement, and a grown community re-touches its own interior
   constantly — fines regrow through the same crowded texels — so the
   interior reads perpetually young. Meanwhile anything that genuinely
   stops being touched is fine material, and senescence deletes it before
   it can brown. Net: the mode has *forgetting* (fines vanish) but no
   *commitment* (nothing visibly lignifies, holds, and darkens). A root
   system's whole visual story — bright growing edge, darkening past,
   load-bearing skeleton — is structurally absent.

3. **The descent spends the metaphor's one gift.** A root system is growth
   with commitment: what is laid down stays, and new growth stands on it.
   The scrolling window converts that permanence into a conveyor — the
   crown is gone in minutes, and with it any sense that the picture is
   *accumulating* rather than passing. §15.4 chose the descent to satisfy
   §1's never-settles law; §17.6 satisfies the same law at a different
   scale and gives the permanence back.

One judgement from the same viewing worth keeping: the root *drawing* is
good. The branching architecture, the width hierarchy, the kinks around
stones — the linework earns its keep. The redesign is aimed at everything
around the linework.

### 17.2 The thesis

The referent gains a second half. Botanically the pane is still a
rhizotron; structurally it becomes **a record being written**. Root growth
is the aesthetic of append-only memory: alive and undecided exactly at the
tips, fixed and load-bearing behind them, the visible shape *being* its own
uneditable history. The design vocabulary follows from taking that
seriously:

- **The living layer is writable.** Tips, hairs, fine fuzz: bright, moving,
  ephemeral. Fines senesce in minutes — working memory, spent freely.
- **The record layer is append-only.** Coarse material lignifies: it
  transfers, slowly and permanently, into a separate wood field that
  nothing erases while the season lives. Wood darkens with age and holds.
  The skeleton at any moment is the biography of the whole run.
- **Salience recedes; the record does not.** Old wood sinks *visually*
  toward the ground — darker, quieter, nearer the soil's own register —
  without ever being removed. The eye reads the past as ground and the
  present as figure, which is what remembering actually feels like from
  inside an archive.
- **Joins are events.** Real roots graft rarely (§17.8); when new growth
  meets old wood and fuses, the old path warms briefly along its length —
  a retrieval, made visible. Anastomosis, the application's namesake,
  returns in this mode as punctuation-free rarity rather than texture.

Vibe target, stated for the §13 judgements to aim at: **groundedness**. Not
the fungal modes' down-regulating flow — the calm of things staying put.
The failure criterion from the shelved build, kept on the wall: does the
fixed structure read as *held* rather than dead?

### 17.3 The descent, retired

`rhizotron.descent_rate` defaults to zero and the pane holds still. The
machinery stays — the u64 world-row coordinate, the scroll folded into
every pass's source read, the reprojection contract, the front controller —
because it is proven, checkpoint-compatible, and the honest escape hatch: a
config with a positive descent rate grows §15's sinking column exactly as
before, and any A/B between the two designs is one number. With the rate at
zero the scroll is zero rows every tick, every code path is exercised at
its fixed point, and the front controller's multiplier steers nothing.

What replaces the descent's three jobs:

- *Perpetual novelty* — seasons (§17.6): each interment re-seeds the soil
  generator at a new world origin, so the next season's ground is new
  earth, from the same unbounded counter, without a pixel ever scrolling.
- *The narrative axis* — the record layer: past-above/future-below becomes
  past-as-wood/future-at-the-tips, which is a stronger axis because it is
  the organism's own.
- *Bounding accumulation* — completion: growth in a fixed pane against a
  finite nutrient economy is determinate. The community fills the pane,
  spends it, and finishes — and §17.6 makes finishing a beginning.

### 17.4 The surface

The fixed pane gets the one landmark the sinking window could never keep: a
soil surface. The top few percent of the view is air — near-black, faintly
cool, the darkest thing in the image so the ground reads warm against it —
over a topsoil line: humus-dark, stone-free, the richest band in the
column. Crowns germinate just below it, so every plant visibly *starts
somewhere*, and rain finally has an address: the wetting front darkens down
from a surface the eye can see.

The surface rides the world coordinate (a fixed world row, passed to the
shaders as a view-relative offset), so the escape-hatch descent simply
carries it away and §15's endless column returns unmodified. Tips treat
air as the bottom margin's mirror: nothing grows there, and the
gravitropic pull re-asserts hard on anything that noses above the line.
Whether the surface eventually carries sprouts — a green bud per living
plant, the one cool-green accent in the image, chroma-budgeted like the
mycorrhizal shimmer — is a §13 judgement deferred until the ground below
it earns the attention.

### 17.5 The ground made visible

The rendering overhaul, in one sentence: the soil becomes the *mid* of the
image instead of its black, and every material gets a rung on one value
ladder. From darkest to brightest:

    air  <  wet soil  <  ghost strata  <  dry soil  <  old wood
         <  young wood  <  living root  <  root tip and hairs

- **Soil lightness** decouples from the fungal background: the chip
  re-anchoring keeps the Munsell families' chromatic identity but maps
  them into a visible envelope (`soil_l_floor` above background, a wider
  `soil_l_range`), with strata contrast raised until the geology reads at
  a glance. Moisture keeps its §15.2 mapping — darker, slightly richer —
  now against a base bright enough for the wetting front to be an event
  the eye follows.
- **Stones** render as their own cool-grey material again, half-buried as
  before but no longer black holes punched in the root mass.
- **Wood** (§17.6) shades by *biographical* age: russet when newly
  lignified, umber in maturity, near-soil dark in the deep record — always
  a shade apart from the ground so the skeleton stays legible as figure,
  always below the living material so it reads as past.
- **Living roots** keep pallor-by-recency, now meaningful because the
  living layer is only ever recent: ivory tips and hairs at the front,
  tan behind, and the lignification transfer takes over from there.

The §15.7(2) crispness certificate is unchanged by any of this — value
placement spends no motion — but the overhaul's brighter field changes the
absolute deltas the sweep measured, so the sweep is owed a re-run at the
new endpoints before the defaults ship (§17.10).

### 17.6 Seasons — completion, the fossil record, interment

The never-settles law, satisfied at the scale of lives. A season is not a
timer; it is the shape determinate growth already has in a fixed pane, made
legible and made cyclic. All of it is smooth functions of simulation state;
nothing anywhere consults a clock (§3 holds).

**Growth.** The long middle, tens of minutes to hours by tempo: plants
germinate near the surface, forage, throw laterals, lignify behind their
fronts. The wood field only accumulates. Nothing placed is ever removed.

**Completion.** The host already reads two words back from the tips pass;
it gains two more: total living mass and total wood mass. As wood mass
approaches the pane's budget — a config fraction of the view's area — a
smooth gate eases germination pressure and branching toward zero: the
community finishes what it is growing and stops starting more. The last
fines senesce; the living layer empties from the outside in; what remains
is the completed skeleton against the ground, holding. The pane is, for a
while, a finished drawing — and this is the one moment the design *wants*
stillness, because the next mechanism is already moving.

**The fossil.** At completion — living mass fallen to a whisper of its
peak, wood mass at rest — the application exports the frame: a PNG beside
the checkpoints, named by season count and seed, written once and never
touched by the application again. The gallery is the mode's deepest
append-only layer: every finished season, verbatim, forever. A season that
took a week of evenings to grow ends as an object the eye can keep.

**Interment.** Then, over minutes, the record becomes ground: the wood
field transfers into a persistent *ghost* field at an eased rate — the
skeleton dims and settles into the soil's own register, keeping its shape
as a darkened root-shadow stratum, the way real soil actually holds the
channels and stains of the roots that died in it. Nutrient recycling pays
out where the wood was (the economy §15 built already does this for
senescence), so the ghost is also the fertile ground of the next season.
Ghosts blend down by one eased step per interment: the last season is
plainly visible in the ground, the one before it faint, the one before
that almost gone. Salience decays on screen; the gallery keeps the
verbatim.

**Renewal.** With the wood field empty, the soil re-seeds: the world
origin jumps by a hashed stride (new strata, stones, hardpan, caches — the
u64 counter's unbounded supply, spent a pane at a time instead of a row at
a time), the palette's family drift carries the new season somewhere
adjacent, germination pressure eases back up, and seeds wake in ground
that visibly remembers the ancestor they will grow through. The ghost
field and the season counter ride the checkpoint, so a pane resumed after
a week continues its own biography — same organism, same ground, mid-life.

### 17.7 Persistence

Nothing new to build, and worth a section so it is a stated feature rather
than an accident: the checkpoint discipline (§4.6) means the pane is
*resumable by construction*. Close the application mid-season and reopen
it tomorrow and the same community continues, bit-identically, from where
it held. The mode's biography — this plant, this ground, this season count,
these ghosts — accumulates across launches for as long as the checkpoint
survives. The fungal modes are weather; this mode is a life.

### 17.8 Grafts — anastomosis at the margins

Real root systems fuse rarely — natural root grafts between neighbours,
the mycorrhizal commons at the margins — and rarity is the point: in the
fungal modes fusion is texture; here it is an *event*. When a living tip's
forward sense finds old wood (not living root: the distinction the two
fields now carry) at a shallow approach angle, a smooth low-probability
gate lets it join instead of avoid: the tip anchors, the join point takes
a soft warm brightening in the wood field's glow channel, and the glow
diffuses slowly *along the wood mask only* — a faint warmth travelling
down the old path over tens of seconds, fading as it goes. New growth,
touching the record; the record, briefly warm where it is touched.

The glow spends the chroma budget and a bounded sliver of lightness, is
heavily lowpassed by construction (diffusion down a mask cannot step), and
is capped by the same concurrency discipline events use: a few grafts
visible at once, then the gate leans away. §17.10 takes the luminance
question; the §13 judgement is whether the warmth reads as recognition —
which is what it is a picture of.

### 17.9 Resonance, reserved

The §16 driver reaches this backend eventually — germination pressure,
branching eagerness and graft probability are exactly the "slow bounded
features wired into existing seams" §16.3 asks for, and a season grown
under an evening of music would fossilise as a readable record of the
whole evening: every album a different plant. Deliberately not in this
build order: the mode should earn its look driverless first, and §16's
step 2 (real-hardware capture) is still open. The seams above are named
now so nothing in this section has to move when the wire arrives.

### 17.10 Safety analysis

The output chain is untouched, `SAFETY_CEILINGS` stays one table, and the
§15.7 analysis carries over wholesale — the fixed pane only removes its
hardest case (the descent's bulk motion falls out of the reprojection
budget entirely, since the compensated scroll is now exactly zero). What
the redesign adds, each with its measurement:

1. **A brighter field re-prices the certificate.** The §15.7(2) sweep's
   deltas were measured in the near-black envelope; the visible ground
   raises the absolute lightness a moving tip front crosses. The sweep
   re-runs at the new shading endpoints before the defaults ship, and the
   licensed `root_edge` floor and elongation ceilings move to whatever it
   says. Same instrument, same discipline, new numbers.
2. **Interment is a slow, sustained, large-area luminance change** — the
   benign kind (smooth, minutes-long, monotone), the same class as §15.7(3)'s
   drought pallor, and it must be shown benign the same way: the
   wood-to-ghost transfer at its fastest config-reachable rate joins the
   adversarial suite beside the rain-at-maximum case.
3. **The graft glow is a localised brightening and the only fast-ish new
   luminance actor.** It is bounded (a capped fraction of `filament_luma`),
   born under an eased envelope, and spatially small; the measurement is a
   worst-case test — maximum concurrent grafts, brightest allowed glow —
   asserting the per-pixel and area criteria directly.
4. **The origin jump at renewal must not be a cut.** The soil re-seed
   happens entirely under the interment's cover: the new generator output
   fades in through the same eased blend that fades the ghost down, so no
   frame ever shows a stepped ground. The renewal-at-fastest test asserts
   the §7 bound across the whole transition.
5. **No thresholds, still.** Completion gates, interment rates, graft
   probability, ghost blending: smooth saturating functions of state with
   C¹ floors, audited like everything §15.7(6) audits.

### 17.11 Build order

Each step lands something watchable alone; the first two retire the
judgement risk (does *held* appear?) before any lifecycle machinery lands.

1. **The pane holds, and the ground appears.** Descent zeroed by default,
   the surface band, crowns germinating across the width near the topsoil,
   and the §17.5 shading overhaul for soil, stones and moisture — judged
   on stills against the §17.1 failures. What ships is already a picture:
   living roots working visible ground under a fixed sky.
2. **Wood.** The record field (lignin, biographical age, glow, ghost), the
   lignification transfer, senescence rebalanced against it, and the full
   §17.5 value ladder. The thesis test lives here: held, not dead, or the
   redesign has failed and the record says so.
3. **The certificate re-run.** §17.10(1)'s sweep at the new endpoints;
   ceilings moved to what it measures.
4. **Seasons.** The mass readbacks, the completion gate, interment into
   the ghost field, the origin jump, renewal — and the §17.10(2)/(4)
   adversarial tests beside them.
5. **The fossil record.** The export path in the application shell, the
   gallery directory beside the checkpoints, season/seed naming, and the
   checkpoint carrying season state.
6. **Grafts.** The join gate, the glow channel and its mask-bound
   diffusion, the §17.10(3) worst-case test, and the §13 judgement.
7. **Presets and the eyes-pass.** `loam`/`meadow`/`taproot` re-tuned for
   the fixed pane, a `perennial` default, and the frame-sampled viewing
   round that decides the deferred judgements (sprouts, graft warmth,
   season pacing).

### What step 1 actually did

Judged on headless stills from the real backend at every stage — the
workflow §17.1's postmortem was produced with, kept as the build's own
instrument. The pane holds (`descent_rate` zero, tempo's curve entries on
it removed from both tables — the escape hatch is an explicit override, not
a slider position), the surface landed as a fixed *world* row so a positive
descent carries it away unchanged, crowns seed stratified across the width
just below the soil line, germination sites moved up to hug it, rain lands
on the first row beneath it, and the topsoil eases stones and hardpan out
while pulling the ramp toward humus-dark.

The shading fight was the §17.1 diagnosis replayed in reverse, and one
mechanism deserved its record: **the exposure governor was the black**.
Raising the soil envelope moved almost nothing, because the governor holds
mean image lightness at a target tuned for sparse light on a void — a
full-field earth image was being divided back down to a void's mean. The
fix is `exposure_lift`, a backend-conditional factor on the resolved
target applied in `Config.resolve` beside the mode pinning: one meaning
("how bright overall") through two honest referents. After that the
envelope edits behaved: a lightness floor every soil texel clears, strata
at a scale the view can actually show several of, warmer half-buried
stones, the litter seam.

### What step 2 actually did

The record layer is a second ping-pong field beside the structure —
lignin, biographical age, graft glow, ghost, the last two written by
nothing yet — updated in the same pass that ages the living layer, so the
commitment transfer moves mass between the two fields atomically. The
transfer is the §17.6 design as drawn: steady, smooth, fineness-squared
discounted, so axes commit in minutes, laterals inside their lifetime
(the rate was raised once watching found the first cohort's skeletons too
faint through the succession lull), and fuzz never. Biographical age
advances wherever wood holds and is reset by nothing; the tips'
self-avoidance counts lignin exactly as living density.

Two findings by eye, both now structural. **The silhouette must be one.**
The first build gave wood its own coverage transfer, and mass mid-transfer
sat below both knees: roots dissolved into dashes precisely where they
were becoming permanent. The shipped compositor saturates *combined* mass
through one knee and treats wood-ness as a colour axis along the
silhouette — with the edge sharpening from `root_edge` toward `wood_edge`
as material commits — so the hand-over cannot gap by construction. **The
churn must not read as confetti.** Against a visible ground, fuzz
fragments lingered pale for minutes after their fines died; the fix is a
short browning clock for fine material (quadratic in fineness — laterals
keep their working pallor) plus a remnant-cleanup term in senescence that
fades faint old mass fully. A taper on the ageing apex's stamp — reusing
the elongation deceleration's own factor — gives the axes their crown-down
width gradient in the bargain.

The §17.2 thesis is held by test now: `test_the_record_is_append_only...`
asserts per-texel that nothing the simulation does decrements lignin, and
the bit-identical resume covers the new field. Checkpoints from before the
record exist are refused and regrow — the step 3 precedent, applied again
while nothing has shipped.

**And the record layer found a bug older than itself.** The biographical
clock froze at 64.0 exactly, and the investigation generalised: an f16
texel truncates a sub-ulp increment, so *any* per-tick accumulation of
seconds stalls at increment × 1024 — about a minute — and the living
layer's recency age had been silently capped there since §15 step 3. Every
age-driven gradient in the mode was saturating at sixty seconds against
scales set in hundreds; a large share of the §17.1 monotone-khaki finding
was this one storage artefact. Ages now advance in 64-tick batches (a
couple of seconds' step on minutes-long colour mappings, far below
anything visible, keyed to the checkpointed tick counter so resumes
replay identically), and the age scales were retuned against real time.

### What step 3 actually did

`tests/crisp_sweep.py` re-run at the §17.5 endpoints, resolved as the
rhizotron so the exposure lift is inside the measurement, with the descent
forced to its config ceiling (the escape hatch stays certified): **0.0%
changing-area fraction at every swept point, both thresholds, both
sizes**, worst per-pixel ΔL 0.026–0.035 against the §15 record's
0.027–0.041. The visible earth spends its lightness statically; the
accumulation-and-slew discipline that made the dark build safe is what
makes the bright one safe, unchanged. The shipped `root_edge` floor and
elongation ceilings stand where §15.7(2) licensed them.

### What step 4 actually did

The structure pass reports living and wood mass beside the front words
(two more integer atomics in the same buffer — order-independent, so the
bit-identical discipline holds), and a season controller runs on the same
rare readback: germination eases closed as the record approaches
`wood_budget` (calibrated against a measured 40-minute run), the interment
drive relaxes toward completion-times-quiet, the fossil moment is offered
exactly once as the drive commits, and the burial transfers lignin into
the ghost channel while the previous ghost fades a generation deeper
under the same cover. Ghost strata shade the ground before the roots do —
darker, quieter earth in the shape of the interred skeletons — and season
state rides the checkpoint.

Three controller lessons, each found by watching or tracing, none by
design: **a burial must finish** (the drive would otherwise track the
falling fill back down and stall the interment part-done — the latch
releases by progress against the mass it committed, never by absolute
fill, because a straggler plant laying fresh wood through the burial
otherwise holds the pane in permanent half-interment); **the burial
spares young wood** (a straggler's freshly-laid skeleton is not the
completed season's record, and erasing it under its own living tips read
as exactly the violation the mode exists to refuse); and **the renewal
knee sits above the spared remnant** (below it, `fossil_taken` jammed and
no season could ever offer a fossil again).

Two honest narrowings. The §17.3 origin jump — new soil each season — is
deliberately not built: renewal regrows the *same* ground, which is what
a real rhizotron's glass does, and the ghosts plus the palette's own
drift carry the between-season difference for now; if the fixed earth
wallpapers over many seasons, the jump lands later under the interment's
cover as §17.10(4) drew it. And the §17.10(2) interment-at-fastest
adversarial test is still owed alongside step 6's worst-case graft test —
the burial's rate at current defaults sits an order of magnitude inside
the wetting front's already-certified pace.

### What step 5 actually did

The application shell consumes the backend's `fossil_due` flag between
frames, reads the presented frame back, and writes it to
`<state>/gallery/fossil-seed<seed>-season<n>-tick<t>.png` beside the
checkpoint files — a stdlib PNG encoder (the application's dependencies
are simulation dependencies; a once-per-season still is not worth an
imaging library), the encode and write on a thread, an existing file
never overwritten. The gallery inherits the checkpoint path's location,
so tests and portable configs isolate it for free.
