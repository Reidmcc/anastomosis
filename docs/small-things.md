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
