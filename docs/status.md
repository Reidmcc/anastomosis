> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 13. Implementation status

The application runs on real hardware, and has since early in the build: it is
watched on the RTX 3080 of §8.1, and those viewings are where §4.7's monotony,
§4.8's wrap seam, §4.9's line condensation, and the frozen frame loop behind
§8.2 all came from. What this section records is the other half of that loop.
It is written from a development environment with no GPU in it, and where it
says something cannot be assessed *here*, that is a statement about this
environment rather than about the application's history: the judgements left
open below are open because nobody has yet watched *these* defaults, not
because nothing has ever been watched.

Here, then: built and verified headless against a software adapter (Mesa
lavapipe), so every shader compiles and the full tick/render sequence runs in CI
without a GPU. The suite is 456 tests, split the way their costs are:
`.github/workflows/ci.yml` runs everything not marked `slow` on every push,
across three Python versions plus a leg with no PySide6 that holds the README's
promise that the panel is optional, and runs the `slow` marks -- drift,
morphology, regime occupancy, the long soaks -- nightly and on demand, where
minutes are affordable. The
checkpoint-determinism check that this section previously recorded as failing
on that adapter passes on the llvmpipe build measured here; it was never
explained, so treat that as an observation about one adapter build rather than
as a fix.

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
and **all three backends** -- the layered 2.5D stack, the volumetric slab of
§5.1, and the rhizotron of §15 -- selectable from the config, the command line
or the control panel, with one saved field each so switching between them is not
destructive. The slab's thickness is a control panel knob as well, from 8 voxels
to the shorter lateral axis, priced in graphics memory beside the slider.

The rhizotron is the newest and the only one on a second metaphor: a soil column
generated as a pure function of an unbounded depth counter, moisture percolating
through it, a community of root systems growing by tropism and branching, and a
window that sinks after the growing front. Steps 1-5 of §15.11 are built and
tested here; what it has not had is a long watching, so its endpoints are the
ones in this section's "wants eyes" list rather than settled ones.

Also complete: **both modes** (§14) -- regulation and activation, as two macro
curve tables over one engine, with the mode a non-structural setting, so
switching it is a ramped transition on the running field rather than a reset.
Activation carries its own presets, a polychrome palette that puts contrasting
hue families in different regions, brisker event envelopes, and a harder shear
on the trail; the flash-safety ceilings are one table serving both. Every
activation endpoint is measured or certified (§14.8), and none of that says
whether the mode is *pleasant*, which is the same caveat the paragraph below
makes about the defaults.

**Not implemented:**

- **The morphology work in §4.7 is now built through step 6, plus the sensing
  saturation that step 6's postmortem turned out to need.** Feature size is
  polydisperse and migrating (the third climate pair), its global mean is
  closed-loop (step 5's ℓ controller), the trail hubs that turned out to *be*
  the reported dots have a deposit capacity working against them and a shading
  knee stopping what remains from clipping, and the trail rides the velocity
  field (step 6). What the agents *sense* now saturates (`agents.sense_cap`),
  which is the change that makes the layer grow an anastomosing network from
  scratch instead of the field of stationary knots it had always actually
  produced — see "The network that was never there" in §4.7. Step 4 is in:
  agents repel from junctions where the climate asks them to, respawns land
  in founding cohorts on bare ground, and a `rift` event takes a region's
  network apart and lets it heal. As with step 4, what the tests assert for
  the trail advection is the mechanism and the invariants; its aggregate
  effect did not resolve above run-to-run variance at test resolution — see
  §4.7. Flux pruning (step 3) is still switched off *by default*; the
  `stability` macro (§9) now runs it up to gain 3 at the sturdy end of its
  travel, so it is a character the panel can ask for rather than a default
  anyone is given.
- **Device-loss recovery** is scaffolded in `device.py` but the rebuild path is
  untested, since a software adapter offers no way to provoke a device loss.

**Not assessable from here:** how it actually looks, and whether the defaults
sit in the right place perceptually. The software adapter renders correct pixels
far too slowly to watch, so every perceptual question has to leave this
environment to be answered. That round trip is the project's normal way of
working and has been made many times — the four findings named at the top of
this section are all its results — and what it leaves open at any moment is
whatever has changed since the last viewing. The numbers say the simulation is
alive, structured, and stable; whether it is *pleasant* is a judgement that
needs the real GPU and a pair of eyes.

That caveat now has two specific things attached to it, both starting from
§4.7 step 5.

The shading balance was changed on the strength of two measures of the rendered
image — how many separate bright blobs it is made of, and how varied their size
is — and those two pull in opposite directions: gating the reaction harder wins
the first by throwing away the second, which is the polydispersity the rest of
§4.7 exists to produce. The shipped point was chosen where the first has most
of its improvement and the second is nearly intact, on a 160-cell field on the
software adapter. It is the single default most likely to want moving at the
next viewing, and `pigment.v_needs_trail` is the knob.

And it costs the slab more than the stack, which puts it on the list of
slab-specific numbers this section already keeps. A filament network fills far
less of a volume than of a plane, so gating the reaction on the network removes
more of the slab's density than of the stack's, and the exposure governor makes
it up: measured through the march, its multiplier went from 2.8 to 8.3 against
a hard bound of 20. Step 6 then cost it again, and by more, which nothing
measured at the time — dimming the trail hubs is dimming the brightest thing in
the volume, and the multiplier went to 17.6, with the sensing saturation later
handing part of that back, to 15.5.

Those three numbers are all from a 24-voxel slab: half the shipped thickness,
and the volumetric exposure test's own economy until it was corrected to
measure the shipped 48. Optical depth accumulates per filament a ray crosses,
so it scales with the depth in voxels — through the slab the application
actually grows, the same field asks for 9.8. The whole brightness macro still
works there: measured at the top of that knob the governor settles its target
with the multiplier at 15.5, inside the clamp of 20. So the headroom above the
knob is about 1.3× now, where it was 4× before any of this. The settled image
lightness is unchanged throughout and nothing is saturating; what is thinning
is the room above the knob, and it has thinned twice now without either change
noticing. The lever is the march's `extinction` calibration (§5.1), which was
set against a density scale two changes ago; it is shared with the compositor,
so moving it is not free, and it should wait for the next viewing, along with
everything else here that is waiting for one. Until it moves, the test holds
the line where it belongs: its ceiling is derived from the governor's clamp and
the top of the brightness macro rather than chosen, so the next change to the
slab's density is measured against the setting it would cost rather than
against a number someone raised.

That caveat is heavier for the volumetric backend than for the layered one, and
worth being explicit about. Its invariants are checked and hold -- the flow is
divergence-free to the storage precision, the slab wraps on all three axes with
no accumulation at the faces, the depth axis carries structure of its own, the
homeostat keeps mean V in band over a long run, and the flash-safety bound holds
through the ray march exactly as it does through the compositor. One statement
about the *image* survives too, and it is the only one: over 700 frames the
exposure governor settles the mean image lightness on its target under the slab
as it does under the stack (0.163 against a target of 0.156), with the exposure
multiplier inside its bounds by the margin above -- so the march is handing the
output stage
something it can work with, rather than a field too sparse or too dense for the
knobs the two backends share to mean the same thing. But §5.1's own warning that
a volume "makes every parameter harder to reason about" is unaddressed by any of
that. The numbers a slab needs are not the numbers a sheet
needs, and the ones most likely to want moving at the next viewing are
the agent density (a filament network occupies a much smaller fraction of a
volume than of a plane), the depth anisotropy, the light's ambient floor, and
now the thickness -- which has a defensible range and a cost curve but no
measured answer for where inside that range the image stops improving, since
that is exactly the judgement a software adapter cannot make.
The layered path stays the default until that judgement has been made.
