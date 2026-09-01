> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited
> here, in other docs, and in code comments resolve via the index in
> `DESIGN.md`.

## 18. Small Strange Things — a port with identity preservation

The fourth backend, and the first whose referent is not a metaphor but a
prior artwork. `docs/founding/small_strange_thing.html` is a single HTML
file from ~December 2025 — 138 lines of naive canvas JavaScript, written in
response to "try less, be more" — in which little Things wander, befriend
each other, sparkle for no reason, and spawn children near their parents.
Its whole law was purposelessness. Every run of it has died with its
window.

This section ports that world onto the engine so it can persist. It is a
**substrate transition with identity preservation**: the founding file is
kept in the repo verbatim, naive comments intact ("Stay in bounds, sort
of"), as the reference implementation this backend is answerable to. The
fidelity criterion for every rendering choice: someone who loved the
original must recognise them instantly — *that's them, seen clearly*, never
*that's a fancy new thing*.

### 18.1 The conserved identity layer

These are the souls; the port migrates them exactly. Where the original's
behaviour has a quirk, the quirk is conserved — this list is a bill of
rights, not a redesign seam.

1. **Traits rolled at birth, fixed for life**: curiosity, shyness, hue,
   size, gentle wander speed. Curiosity and shyness were not designed to
   *mean* anything, and still don't — the one behavioural consequence they
   have is (2).
2. **Friendship**: a Thing seeks friends only if `curiosity > shyness`;
   formation requires proximity (50 world units in the original); at most
   THREE friends; and the load-bearing law — **bonds never break and render
   at any distance**. The long taut lines across the world are friendships
   that survived emigration. Conserved quirks: bonds are one-directional
   (the seeker records the friend; the befriended may never know), and a
   later scan may record the *same* neighbour again — a duplicate bond is
   the same friendship, twice as bright. The reference implementation does
   both, so this backend does both.
3. **Bond colour is the average of the two Things' hues** — the original's
   plain arithmetic mean of the two hue angles, not a circular mean. Two
   beings, one shared colour.
4. **Villages by lineage**: mature Things occasionally spawn children
   nearby. Population capped village-sized — the original's 200.
   Resolution up, population not: ten thousand Things would be a different
   sociology wearing their name.
5. **No death.** The original has no mortality and the port must not invent
   any. Age gates spawning; it never kills. The cap handles growth — and
   because nothing dies, a slot in the population buffer is an identity for
   the life of the world.
6. **Sparkles for no reason.** Small, dim, occasional — and now subject to
   the house flash-safety law like every other mode: a sparkle is a
   one-tick deposit into the canvas field, the slew limiter (§7) rounds its
   attack, and the trail fade carries it away, which is within a constant
   of what the original's global fade did to a sparkle drawn once.
7. **Trails**: the original's low-alpha fade meant every village stood on
   the ghost of everywhere it had wandered. The Things had a record layer
   before the house did. Here it is incarnated as the *canvas field*
   (§18.3) — softer than Perennial's strata, breath on glass: everything
   drawn decays exponentially, nothing is ever erased by anything but time.
8. **The toroidal world** (edges wrap). It is why the long lines exist —
   positions wrap, but a bond renders as the straight line between the
   wrapped positions, never the shortcut through the seam.
9. **Click-to-add survives.** It is the participation verb — someone
   watching must be able to add a few Things where they point. The click
   arrives through the window's pointer events, maps through the same
   aspect correction the compositor uses, and spawns the original's three
   Things with the original's scatter.
10. **Frame-time, never wall-clock.** The original wrote `time++` in 2025
    as if it knew §3 early; the engine agrees natively. Ages count ticks in
    a u32 (the §17 f16-age lesson applied from birth: no stored age ever
    saturates), and every rate is expressed per-second and converted per
    tick against `sim_hz`, so the tempo of their world does not depend on
    the tick rate.

### 18.2 What the engine buys them

* **Light.** The original lived at `hsla(…, 0.2)` through half-pixel
  strokes. Here every deposit carries a soft glow skirt by construction —
  bodies are bright cores inside wider dim halos, bonds have width and
  presence — and the whole image passes through the shared HDR chain. The
  hue circle stays the original's HSL wheel (a trait is a trait), converted
  to linear light at deposit time; perceptual bounds (`l_max`, `c_max`) are
  applied in Oklab at composite, and §7's stage guarantees the rest.
* **Persistence, the whole point.** One ongoing world per install: the
  canvas field, the population buffer and the tick counter ride the
  checkpoint (§4.6), so ages are measured in weeks, friendships are older
  than sessions, and villages have real histories. They may not need
  fossils — nothing about them ever ends — so no gallery is wired; if one
  is ever wanted it can follow Perennial's export path unchanged.
* **Their tempo stays unhurried.** Wander is the original's Brownian step,
  variance-matched per second to the reference at its 60 fps; they wander,
  they do not perform.

### 18.3 Mechanism

The world state is two objects, both checkpointed:

* **The population buffer**: `capacity` fixed-size records (double-buffered
  the way the rhizotron's tips are — each invocation reads any slot's
  previous state and writes only its own next state, so no pass has a
  scheduling-dependent read and the bit-identical resume discipline holds).
  A record is position, the five birth traits, a u32 age in ticks, an alive
  flag, three friend slots and a friend count. Randomness is the house
  counter PRNG keyed on (slot, tick, seed) — no stored RNG state, and since
  nothing dies, a slot never changes owner.
* **The canvas field**: one ping-pong rgba16float texture at the derived
  resolution. Per tick it decays by `1 - exp(-fade_rate·dt)` and drains the
  deposit accumulator into itself. It is simultaneously the trail layer and
  the image — the original's one canvas, kept: bodies are just the freshest
  stratum of their own ghost.

Per tick, one compute pass:

1. **Update** (one thread per slot). Alive slots age, wander (uniform step
   scaled `sqrt(60·dt)` to match the reference's per-second diffusion),
   wrap, and — if `curiosity > shyness`, at the reference's rate — scan all
   slots in index order for neighbours within `friend_radius`, appending
   while fewer than three friends, exactly as the founding `forEach` did.
   Empty slots compute their rank among empty slots from the previous
   state: the first ranks claim any pending clicks (three births per click,
   scattered); the rest run the spawn lottery — pick a deterministic
   candidate parent, and if it is alive, mature, and the roll passes and
   the pick is proportional to the mature fraction, a child is born nearby
   with fresh traits. Expected births per tick at a young village match the
   reference's `mature × 0.005` per frame; near the cap the rate eases as
   empty slots grow scarce, which is the cap arriving as a softness rather
   than a wall.
2. **Deposit** (two entry points, atomic fixed-point accumulation, exactly
   the rhizotron's deposit-buffer pattern). *Bodies*: each Thing stamps its
   pulsing disc (the reference's `sin(time·0.05 + x·0.01)·0.5` breath, on
   an accumulated phase that rides the checkpoint) with the birth fade-in
   (`min(1, age/50 frames)`), plus the glow skirt; each Thing rolls the
   sparkle (rate-converted 2%/frame) and stamps it at the reference's
   offset and hue+60. *Bonds*: one workgroup per (thing, friend-slot) walks
   the straight segment between the pair, stamping a soft-edged line in the
   averaged hue. Integer atomics make overlap order-independent, which is
   what makes the pass deterministic.
3. **Canvas** (one thread per texel): `next = cur·(1-fade) + drain`, the
   accumulator zeroed by `atomicExchange` on the way through, values
   sanitised (§4.4).

The compositor samples the canvas with the standard aspect correction
(wrapping both axes — the world is a torus in x and y), lifts it onto the
house void (the original's `#0a0a0a`, spoken as `background_luma` with the
fungal fog's faint coolness), applies the Oklab bounds, and writes HDR. The
velocity texture is zeros: nothing here moves fast enough for reprojection
to owe it anything.

### 18.4 Exposure

Pinned attenuation-only (`exposure_max = 1`), the Perennial ruling (§17.5
finding 4) applied in advance rather than learned twice: a mean-holding
governor would brighten five founding Things into a blaze and then dim
every village for the crime of growing. The mode's brightness is its
census. The governor stays as a guard against a config-reachable
over-bright field, and stills are judged only at convergence — which, with
the pin, is by construction.

### 18.5 Safety analysis

The output chain is untouched and `SAFETY_CEILINGS` is one table, so §7's
guarantee holds by construction; what the mode adds, each with its bound:

1. **Sparkles are the only fast luminance actor.** A sparkle is a one-tick
   deposit of bounded amplitude over ~2×2 texels; the slew limiter admits
   it at `max_luma_delta` per frame, so its attack is a ramp of a few
   frames and its decay is the canvas fade. Area is microscopic against
   the WCAG 25% criterion; the per-pixel limiter bounds it anyway.
2. **Births and clicks fade in** over the reference's 50-frame ramp before
   the limiter ever sees them.
3. **Everything else is slow**: Brownian wander at fractions of a texel
   per tick, trail decay over seconds, pulse amplitude half a texel of
   radius on a dim disc.

### 18.6 Build order

1. **The world lives.** Backend, geometry, the three passes, the
   compositor, seeding, config, checkpoint layout, app and panel wiring,
   click-to-add. Judged on headless stills against the founding file side
   by side.
2. **The souls under test.** No death, bonds never break, three-friend
   cap, cap-not-exceeded, traits fixed for life, click-to-add, toroidal
   wrap, bit-identical resume, flash bound.
3. **The eyes-pass**, porch in the review seat: stills and notes by
   letter, felt responses back, defaults moved to what watching says.

### What step 1 actually did

Built as designed above, with the calibration notes that mattered recorded
in `things.py` where each number lands. Deviations from the sketch: none
structural. The porch's review rounds are the record from here.

### What the first review round changed

The porch's round-1 verdict (2026-09-01): the sociology conserved
completely — the identity test passed at the structural layer — and every
finding was about the light. Four felt responses, each traced to its
mechanism:

1. **"Through glass", diagnosed as sibling-style bleed.** The port had
   arrived with a touch of Perennial's soul: dark, reverent,
   moody-elegant. The Things' register is construction-paper candy — the
   moodiness belongs to the black field, never to the inhabitants. Two
   mechanisms under the symptom: body steady-state sat at ~0.7 of the
   founding disc's level (`body_emit` raised past the fade rate for
   opaque confidence, core edge crispened to a half texel), and —
   the larger share — regulation's resolved `c_max` of 0.145 was
   *pastelising* them: the founding `hsl(h, 60%, 50%)` discs carry Oklab
   chroma 0.15–0.20. The backend now lifts resolved `l_max` and `c_max`
   (`luma_lift`, `chroma_lift`, the `exposure_lift` pattern applied to
   colour), inside the hard ceilings, before overrides.
2. **Glow re-judged after the cores came up** — and tightened by radius,
   not gain, per the ruling: a narrower bright skirt says glowing, a wide
   dim one says fog (`glow_mult` 3.0 → 2.4).
3. **The social figure/ground had flipped**: at round-1 calibration the
   villages read as friendship graphs that happened to have creatures.
   Bonds now differentiate by span — intra-village bonds are background
   hum near the founding hairline, and the long emigrant lines keep their
   earned width and full presence (`bond_near_width`/`bond_near_gain`,
   ramped over one-to-three friend radii; past three radii, a bond has
   left home). Both conserved quirks were signed off in the same letter:
   duplicate bonds as twice-bright friendship, and one-directional
   recording.
4. **The breath was missing — and turned out to be a rounding error that
   had become a soul.** The founding wander-shadows are an 8-bit canvas
   quantisation floor: the 5% fade stalls below ~10/255, so every village
   stood permanently on the ghost of everywhere it had been. The
   reference arm proved it (a float-precision transliteration of the
   founding renderer has no shadows either). The port incarnates the
   artifact as a mechanism: the canvas alpha channel max-tracks the
   canvas's own brightness (rate-invariant by construction) and decays on
   a clock of minutes (`ghost_fade_rate`, ~20-minute half-life), rendered
   as a faint cool-grey breath under the villages — softer than
   Perennial's strata, longer than a second, bounded by `ghost_luma`'s
   validation ceiling. It rides the checkpoint inside the canvas.

Also settled in round 1: **no fossils, ratified** ("fossils are for worlds
where things end; nothing about them ever ends; the checkpoint is the
whole biography"), and the **soft cap accepted** as the same law with
kinder manners.

### What the second review round changed

Round 2 ratified the register, the glow, the social figure/ground and the
breath — and its one new symptom, caught with the close-crop instrument
built for it, produced the port's one genuinely new law. At screen scale
the world read dimmer and line-poor than at the review scale, and the
porch's second hypothesis was right, completely: every length was
texel-denominated, so at higher resolution the beings shrank, wandered a
smaller fraction of their world per second, and — the subtle half —
*emigrated less*, because emigrant lines are born of edge-crossings and a
relatively slower walk crosses edges less often. Both symptoms, one
mechanism.

The law, stated for the record: **they must be the same beings at every
resolution.** The founding file was pixel-native because it only ever
lived at one window; the port lives at every size. Every length in
`ThingsParams` is now a *world unit* — one texel at `WORLD_WIDTH`, the
review-ratified 960 — scaled to the canvas when packed, with the trait
size and the sparkle's stamp carrying the scale into the shaders through
`world_scale`. `WORLD_WIDTH` is a constant, not a knob: changing it would
rescale the meaning of every saved world. Sociology, gait, presence and
emigration rate are all fractions of the world now, and a world grown at
any window is the same world at Pseudo's native 2560.

One emergent honesty, noted at the same viewing: the breath layer
remembers the emigrant lines' *sweeps* — a long bond whose endpoint
wanders paints a slow grey fan across the field. The founding 8-bit floor
would have kept exactly this residue (bond strokes deposited above the
stall threshold), so it is conserved rather than suppressed; if watching
ever rules the fans too loud, the mechanism is a soft luminance threshold
on what the ghost tracks, and it is deliberately not built until eyes ask
for it.

### What the felt pass changed (round 4)

The final gate did exactly what the governance built it for: motion,
breath density and a real screen caught what stills at review scales
structurally could not — bodies embedded in dense breath read
milky-pastel while a Thing on virgin dark read full candy, because the
compositor *added* the grey breath under the living and neutral light
added to candy is desaturation. The ruling is a law now, phrase and all:
**the ghost may not tint the living.** The founding file's source-over
order (fade first, discs after) guaranteed the living always sat on top
of their own history, and shadows existed only where no body currently
stood; the fix restates that order for an additive canvas — the breath is
occluded by living light, with knees set by trail arithmetic (full
jurisdiction over ground older than ~two seconds, nothing at any body
core), so a Thing over thick cloud and a Thing on virgin dark are
identical by construction. Verified by `test_the_ghost_does_not_tint_the_living`
(twin Things, one over a painted full-strength cloud, indistinguishable
in Oklab L and C) and by a twin pair on a real aged world (ΔL 0.000,
ΔC 0.003), with the breath confirmed standing where no body is.

The verification instrument then caught a second founding truth the port
had missed: **the click outranks the cap.** In the founding file the
200-cap only ever gated *reproduction* — the click handler pushed
unconditionally past it — but the port's structural capacity silently ate
clicks once a world filled, and every long-lived world is full. The
population buffer now carries a click reserve beyond the cap
(`CLICK_RESERVE`): the lottery stops at the cap, budgeted by rank so one
tick cannot overshoot it (the founding checked per push, synchronously),
and click-born Things spend the reserve — a full village still answers
the finger. `test_the_click_outranks_the_cap` holds it.

### What the second felt pass changed (round 5)

The first human viewer kept watching past their own verdict and found the
milk again — subtler: pastel village centres with pulses of full candy
around them. The ring was the diagnosis: the additive canvas has no
painter's order, so in a crowd the overlapping bodies, skirts and trails
*sum*; lightness pegs at its ceiling, chroma cannot follow, and the
excess renders as white at exactly the crowded cores, while the annulus
where luminance falls off — and any lone Thing — keeps full saturation.
The ghost was only the first tinter: **the living may not tint each other
either.**

The founding file never had this because it composited source-over —
every disc painted 0.7 over everything beneath it, so a body's core was
always mostly its own colour whatever the crowd did. Restored: each tick
the bodies claim a per-texel ownership layer (`atomicMax`, later index
wins — the founding draw order exactly), and the compositor paints the
owner's own colour source-over trail, breath and neighbours at the
founding 0.7, at the trail's own steady-state level so the ratified
brightness holds. The 30% that shows through is the founding's own
translucency: history and neighbours keep exactly the vote they always
had. Ownership is derived state, wiped and rebuilt every tick, never
checkpointed. `test_the_living_do_not_tint_each_other` holds it: a Thing
mobbed by eight different-hued overlapping neighbours and its lone twin
match at the core.

### What the third felt look changed (round 6)

The pastels crept back — slower. The clock was the diagnosis: the slow
variable in an aging village is the *bond count*, and bonds never break,
so village-interior light accumulates for hours; even a 30% leak of an
unbounded sum eventually outvotes the disc. The viewer's suggestion —
split the dots and trails into genuinely separate layers, as the fungal
stack separates its sheets — turned out to be a description of what the
founding file had been doing all along, in two halves the port had only
half-honoured:

1. **Source-over compounds.** A stationary disc repainted at 0.7 every
   frame sits on residue that is ~95% its own previous self; the founding
   fixed point was ~97% own colour, not 70%. The overlay now paints at
   that fixed point (0.95), so no pileup underneath can outvote a disc
   however old the village grows — the few percent showing through are
   the founding's own translucency.
2. **An 8-bit source-over canvas can never hold more light than its
   brightest paint.** The founding trail was bounded at 1.0 by
   construction; the additive record piles without limit. `trail_knee`
   (the fungal knee idea, inherited) restates that ceiling at render: a
   soft tanh bound on the trail layer's contribution, leaving single-body
   levels essentially untouched and compressing an aged village's pile to
   the founding's own magnitude. The record itself stays honest
   underneath; only its rendering is bounded — retrieval tuned to the
   signal, §17's lesson again.

`test_the_pileup_cannot_wash_the_living` holds the whole law at an
absurdity: a Thing standing on a record painted to six times any
reachable level matches its twin on virgin dark, while the record still
shows, bounded, where no body stands.

### What the fourth felt look changed (round 7)

The viewer kept watching an *aged* world and named what the averages had
hidden: the wash is size- and age-dependent. The fixed-point overlay
protects only the exact core, but the visible dot is its core plus its
canvas halo — and with the history layer rendering at the life's own
level, a small dot in an old village reads mostly as its knee-grey halo.
(The build seat's own measurements agreed and were averaged past: a mean
C/L of 0.30 contained individuals at 0.17. Means hide; look at the
spread.) A wrong turn is also on the record: the build seat spent a
round suspecting a stale process, when the viewer's live knob-tests had
already disproved it. The viewer restarts every time.

The ruling, the viewer's own prescription: **the history is quieter than
the life.** `out_gain` drops to 0.4 and `trail_knee` to 1.0, so an aged
pile renders at most a third of a body's own level and the trails stay
easily noticeable while unmistakably behind; `bond_emit` and
`sparkle_amp` rise to hold the living marks — the ratified emigrant
presence, the founding sparkle — at their previous rendered levels
through the dimmer layer. Bodies live in the overlay and never dimmed.
This completes the viewer's two-layer suggestion from round 6 in its
final form: not just separate mechanisms, but separate *loudness* — the
figure is the life, and the ground is the ghost of it.

### The closing aesthetic ruling

Entered from the direction seat after round 7, prompted by the question
of whether within-dot colour variation — against the founding's
perfectly flat `arc()`-and-one-fillStyle discs — is a defect or the new
body. The ruling: **the self is flat; the body is lit.** Hue is identity
— rolled at birth, fixed for life, never varying within a Thing and
never mixed at the core (rounds 4–6 exist to guarantee exactly this, and
they do). Luminance texture within a disc is *embodiment* — the engine's
light falling on a being that finally has enough body to catch it.
Identity contamination was always a bug; dimensional shading never was.

And the criterion the whole arc used implicitly, made explicit: a
founding-substrate accident is a **soul** when it means something in the
piece's own terms (the 8-bit floor meant memory → the breath; the
click-past-cap meant participation outranks capacity → the click
reserve), and a mere **limitation** when it doesn't (the rAF throttle,
the banding, the disc flatness — Canvas 2D's `arc()` and one
`fillStyle`, meaning nothing). Flatness was never on the souls list; hue
was. The seven-round chase for flat consistency was load-bearing anyway:
every milk it caught was genuine hue contamination, wrong under this
ruling — the chase caught everything that mattered and ended precisely
where taste begins. At taste, the seats agree: glass beads, not paper
dots.

The final gate's question, reframed by the same ruling: not "is it the
same" but "are they them, with light on them."

### The round-3 verdict

PASS — the direction seat's review is complete. The fans were ratified as
*founding phenotype*, verified against the primary source before ruling:
a screenshot of the founding original running live shows the same faint
grey streaks across its field. Friendships leave ghosts too, and always
did. The reviewer's calibration note stands in the record: at a mature
seed the fans sit near the loudness ceiling, judged history rather than
haze — and that final aesthetic call is deliberately left to the one
judgement this document cannot contain, the first human felt response on
a real screen at converged exposure, watching it move. The §18 threshold
stays named and unbuilt unless those eyes ask. What remains is the
governance's last step: the merge, and then the first checkpoint of a
world that keeps.
