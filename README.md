# anastomosis

A long-running generative visual field, built for stimming and self-regulation.

Filaments grow, seek each other, and fuse — the way fungal hyphae do, which is
where the name comes from — inside a slowly breathing medium whose colour drifts
with what the simulation is actually doing. It is designed to run for days on a
secondary display while you get on with something else on your main one.

Four things it will not do, by construction:

- **It will not flash.** The output stage enforces a hard per-pixel rate limit on
  lightness, in Oklab, measured against the motion-compensated previous frame.
  The bound holds regardless of what the simulation does — see [Safety](#safety).
- **It will not loop.** Nothing anywhere is a function of the clock. All slow
  variation comes from integrating random increments into stored fields, so there
  is no periodic component to find.
- **It will not settle.** The governing parameters are themselves a slowly
  drifting spatial field, regulated by a loose, slow homeostat. The system is
  never solving the same equation twice.
- **It will not become a field of identical dots.** That is a separate promise
  from the one above, and a harder one: a field can be alive by every measure
  the homeostat takes and still be the same picture it was an hour ago, because
  none of those measures can see how the material is *arranged*. Left alone it
  arranges itself into a dense array of similar-sized round features — which
  some people find actively unpleasant to look at — and it does so twice over:
  the reaction settles into a lattice of same-sized spots, and the agents pile
  into round stationary knots. Both are counteracted at the source. Feature
  size is measured and steered — it differs from region to region, and the
  field's overall scale is held to a setpoint that keeps moving — and what the
  agents *sense* saturates, so a knot can never out-attract a filament: the
  agents spread into an anastomosing network instead of orbiting their own
  strongest deposit, and the whole network rides the flow instead of sitting
  still. See `DESIGN.md` §4.7.

`DESIGN.md` indexes the design documentation: the architecture and the
reasoning live in `docs/`, one file per section, and `DESIGN.md` maps section
numbers (§n, as cited in code comments and tests) to the files. This file is how
to run it.

## Install

Python 3.11 or newer, and a GPU with Vulkan, Metal, or DX12.

```bash
pip install -e ".[ui]"     # [ui] adds PySide6 for the control panel
```

Without `[ui]` everything still works; you just adjust parameters by editing the
config file, which is hot-reloaded.

## Run

```bash
anastomosis                          # windowed, 1280x720
anastomosis --fullscreen             # borderless fullscreen
anastomosis --preset quiet           # start from a named preset
anastomosis --backend volumetric     # the raymarched slab instead of layers
anastomosis --volume-detail fine     # a sharper slab, if the card has room
anastomosis --width 2560 --height 1440
```

Drag the window to your secondary display and press **F11** to go fullscreen
there; F11 again brings the window back exactly as it was. `--fullscreen` does
the same thing at launch. It uses borderless windowed fullscreen rather than
exclusive, deliberately: exclusive fullscreen can stall the compositor on your
*other* monitor and steal focus, which would defeat the point.

F11 works from the control panel too, so you do not have to click on the
visual — and therefore give it focus — just to toggle it.

Resizing the window — dragging an edge, maximising, going fullscreen — does not
interrupt the simulation. The field keeps running with everything it has grown,
and only the presentation follows the new size. It also keeps the resolution it
started with, so if you want the extra detail at a much larger window, pass
`--width`/`--height` at launch (or set `render.base_scale` higher) rather than
resizing into it.

The control panel opens as a separate, ordinary window on your main display. It
is not always-on-top and minimises freely — tuck it away and bring it back from
the taskbar when you want it.

Under **Events** there is a button for each kind of slow perturbation the
simulation produces on its own — a bloom, a dieback, a shift in current, a tint,
a rift. Pressing one asks for that event *now* instead of waiting for it to come
up by chance. It is the same event either way: it arrives in one region, builds
over a minute or two, holds, and fades, and it counts against the same limit on
how many can run at once — so the buttons go quiet while that limit is reached,
and come back as events fade. Nothing here cuts. (In the activation mode, a
high **Tempo** also makes events build in tens of seconds rather than minutes,
and two more may run at once — same shape, same size limits, quicker.)

Above the buttons, **How often** sets how frequently those events arrive on
their own, from about one every two hours at the left to about one every three
minutes at the right; the reading beside it tells you where you are in plain
minutes rather than a number. It lives here rather than with the other knobs
because it belongs with the buttons that ask for the same thing by hand.

This is a timing control and nothing more. An event is the same size, the same
strength and the same length however often they come, so the fast end is a field
that spends more of its time inside something happening — not one that gets hit
harder. Arrivals stay random, too: what the control sets is the *average* gap, so
any particular wait can be a good deal shorter or longer, and the next one is
never predictable from the last.

Closing the render window quits: the field is saved, the control panel closes
with it, and the process ends and gives you your terminal back. `Ctrl-C` in that
terminal, and `kill` — which is what logging out sends — end the same way, with
the field on disk.

```bash
anastomosis --no-ui                  # no control panel
anastomosis --list-presets
anastomosis --write-config           # write a config file and exit
```

## Picking up where you left off

Closing the window doesn't throw the field away. The simulation state is saved
every five minutes and again on exit, and **the next launch continues from it**
— you come back to the network you left rather than to fresh noise. A field
that has been running for hours looks materially different from one that
started a minute ago, and growing that back takes a while.

To start over, press **Reset simulation** in the control panel (it asks first —
the old field is gone once the saved state is), or:

```bash
anastomosis --reset                  # ignore the saved state this launch
anastomosis --no-checkpoint          # don't save or resume at all
anastomosis --checkpoint-interval 60 # save more often than every 5 minutes
```

The state lives in `~/.local/state/anastomosis/checkpoint.npz` — about 150 MB at
1440p, rewritten in place (the volumetric backend keeps its own,
`checkpoint-volumetric.npz`), with the fields the engine recomputes every tick left
out. It records the simulation geometry it was taken at, and the next launch
builds itself in that shape before loading it, so reopening at a different window
size — or on another monitor, or after editing the layer count in the config —
resumes the field rather than discarding it. The window is presentation: the
image is simply shown at the new size, exactly as it is when you resize a running
session. A resumed field keeps the resolution it was grown at, so the structural
config values (layer count, base scale, agent density) take effect on the next
new field — press **Reset simulation** when you want them now.

A file that is missing, damaged, or from a version this build cannot read costs
you the field, never the session: that launch quietly starts from seeds.

The save itself is a GPU readback on the render thread followed by a disk write
on a worker, so a frame is held rather than dropped, and nothing about the image
changes while it happens.

A new field, whether from a reset or a first run, comes up through the same
slew limiter as everything else, so it grows in rather than cutting.

## Three tunings

At the top of the control panel, **Mode** chooses which tuning the eight
knobs move through: **Regulation** — calm and slow, the original, built for
settling — **Activation**, the same instrument tuned for sensory seeking:
more motion, more colour, more happening — or **Resonance**, activation's
tuning driven by whatever your computer is playing (below). Switching is
nothing like the Depth choice below it: no reset, no dialog — the field on
screen keeps everything it has grown and changes character over a few
seconds, and switching back is just as smooth. The flash-safety bound is
identical in all modes; none can flash — see [Safety](#safety).

What activation changes is where the knobs *reach*, never what they mean:
the same slider goes further — faster flow and agents, quicker weather, a
hue rotation up to a turn every seven minutes, more of the colour circle in
play at once, more saturated throughout, events up to one every ninety
seconds — while the bottom of every travel stays where regulation has it,
so the two tunings overlap rather than abut. Set it in the config with
`mode = "activation"`.

Activation also has a colour mechanism of its own: turn **Intensity** up
and different regions of the field settle into *contrasting* colour
families, three of them spaced around the hue circle, chosen region by
region by the same slow climate that decides everything else — so the
families drift and hand over the way regimes already do, rather than the
whole image shifting at once. **Palette** still says where on the circle
the whole arrangement sits.

### Resonance: a music visualizer that cannot strobe

**Resonance** wires the field to whatever audio the machine is playing.
The current surges with the bass and the network is sheared by it, colour
saturates with the highs, the palette wheels faster when the music is busy,
and a strong onset — a downbeat, a drop, a phrase starting — blooms one of
the same slow events the other modes get by chance, through the same
envelopes and the same caps. The filaments themselves are never driven by
the music: the organism keeps its own behaviour and rides audio-driven
weather, which is what keeps this anastomosis rather than a spectrum
display. When the music stops, the field is simply itself again — silence
is a first-class state, not an error.

One thing this mode will never be: a strobe. The flash-safety bound of
[Safety](#safety) is identical here, and the music is deliberately not
allowed to touch brightness at all — its energy goes to motion, colour and
incident, the channels the safety argument leaves free. The field dances
with the music; it cannot flash with it.

Capture needs the optional extra:

```bash
pip install -e ".[audio]"
```

It prefers a device that is the machine's own output — PulseAudio/PipeWire
monitors on Linux, "Stereo Mix" where Windows drivers offer it, BlackHole
on macOS if you have routed through it — and falls back to the microphone,
which still works with speakers playing. The panel shows what it is
actually listening to; `audio_device = "name"` in the config picks one by
hand. Without the extra, a device, or any sound, the mode runs as a plain
activation-tuned field and the status line says why.

Under Resonance the Mode box also grows a checkbox: **Draw the filament
network**. Untick it and the network fades out of the image while the
organism keeps running underneath — the medium alone carries the picture,
still structured by the invisible network the reaction grows on. It is a
fade, never a cut, and the setting is saved (`filaments = false`).

The preset list follows the mode — and the world. `prism` (colour first),
`cascade` (motion first, events every few minutes) and `spark` (fine,
dense, brisk) are activation's; `pulse` (motion-forward, headroom left for
the drive) is resonance's; `loam` (the balanced default), `meadow`
(fine, fibrous, eager) and `taproot` (sparse, deep, austere) are the
rhizotron's. Choosing one — including with `--preset` — brings its tuning
and its world with it: `--preset meadow` opens the rhizotron.

The rhizotron itself has **one** tuning: under it the knobs always move
through the regulation table, the Mode selector greys out and says so,
and switching back to a fungal view finds the mode where you left it.

The filament network rides the flow in both modes; at higher **Tempo**
under activation it rides it *harder* than the colour it carries, and that
difference is what stretches and pinches the filaments as they go.

## Two ways of drawing depth

There are two backends, and the **Depth** selector at the top of the control
panel switches between them.

**Layered** (the default) simulates three independent 2D fields at different
scales and tempos and composites them back to front, with parallax, focus
falloff and atmosphere between them. It has by far the finest filament detail —
the front sheet runs at your full display resolution — and it is the cheaper of
the two.

**Volumetric** simulates one continuous slab of about seven million voxels and
raymarches it. Material genuinely passes in front of and behind other material
rather than living in three discrete sheets, dense structure actually attenuates
what is behind it, and a single soft light casts shade *into* the network. It is
laterally coarser — 512 voxels across where the layered front sheet has 2560 —
and it asks more of the card and a good deal more of its memory.

Everything after the image is formed is identical between them: the same colour
mapping, the same exposure governor, the same flash-safety limiter, the same
dither. Every knob below means the same thing under either.

The choice is structural, so it applies to a *new* field rather than the one on
screen. Switching is not destructive, though: each backend keeps its own saved
field, so you can try the other one and come back to find yours where you left
it. Set it permanently with `backend = "volumetric"` in the config file, or for
one session with `--backend`.

### How finely the volume is simulated

The lateral coarseness above is the volumetric backend's one real weakness, and
the **Detail** selector under **Depth** is the knob for it. Three sizes, here at
the default thickness:

| Detail | Slab at 16:9 | Filament width at 1440p | Memory | Simulation cost |
|---|---|---|---|---|
| **Standard** | 512 × 288 × 48 | ~5 px | ~650 MB | 1× |
| **Fine** | 768 × 432 × 48 | ~3.3 px | ~1.5 GB | ~2.3× |
| **Finest** | 1024 × 576 × 48 | ~2.5 px | ~2.7 GB | ~4× |

Wider is sharper rather than merely bigger: the thickness is a separate knob and
this one leaves it alone, while the height follows your window, so the voxels
stay cubic and a filament — which is a fixed handful of voxels across — lands on
proportionally fewer display pixels. If 1080p looks soft and 1440p softer, this
is the setting that answers it.

What it costs is the simulation, and only the simulation. Drawing a frame costs
the same at all three sizes: the raymarch is one ray per output pixel and its
step count follows the slab's *thickness*, which this setting does not touch. So
the extra work is the per-tick passes over the volume, which scale with the
voxel count — the square of the width ratio — along with memory. On the RTX 3080
of DESIGN.md §8.1 the standard size sits around 3% of the card's bandwidth, so
even the finest stays inside roughly a tenth of it.

If **Finest** runs long on your card, the cheapest thing to give back is the
tick rate — `sim_hz = 14` instead of 20. The interpolator hides it completely,
which is why the budget governor throttles that and nothing else.

Set it with `volume_detail = "fine"` in the config file, or `--volume-detail`
for one session. Like the backend it is structural, but unlike the backend it
cannot keep what it is leaving: there is one volumetric field, and a slab of a
different width is a different field. So changing it grows a new one from
seeds — the panel asks first, and the **Depth** status line tells you when a
saved field is still running at a size you have moved away from.

### How thick the slab is

Under **Volumetric**, the **Thickness** slider below the selector sets how many
voxels deep the slab is — from 8 up to however many the shorter of its other two
axes has, which is 288 on a 16:9 display at **Standard** detail. The default 48
is a slab a ray passes one or two filaments through, so the depth cues are all
present but quiet; more depth means more material between you and the far face,
and so more occlusion, more shading and more atmosphere.

Two things to know before moving it. It is what this view's memory is spent on,
linearly — about 650 MB at the default and about 3.9 GB at the ceiling — and the
line under the slider prices whatever it is pointing at. And the returns flatten
before the ceiling does: past some thickness the near structure is opaque enough
that the far face is no longer contributing anything you can see, and where that
happens depends on **Intensity**, since that is what decides how much material
there is. Somewhere in the low hundreds is where it is worth looking first.

Changing it grows a new field — a slab of a different depth is a differently
shaped field, and unlike a backend switch there is nothing to come back to — so
the button beside the slider asks first, and the image settles down and grows
back over a few minutes. The setting is saved, so the next launch opens at it.

The two settings meet in one place: the thickness ceiling is the shorter lateral
axis, so a wider slab can also be a deeper one — at **Finest** the ceiling rises
from 288 to 576. Memory is the product of both, so if you raise both, read the
line under the thickness slider before committing.

## A second world, growing in

There is a third entry in the **Depth** selector, and unlike the other two it
is not another way of drawing the fungal field: **Rhizotron** is a different
metaphor — a pane of glass pressed against living soil. Stratified earth in
real soil-chart colours, stones the water pools above, rain events that soak
down through the column over minutes, the whole window sinking at hour-hand
speed through soil that is generated below, retired above, and can never
repeat — and a plant growing through all of it: pale root tips steered by
gravity, water and stone, plunging axes throwing oblique laterals and fine
fuzz, ivory at the living front and browning with age. Where the fungal
field converges — filaments seeking and fusing into a network — the roots
diverge, one crown ramifying and never rejoining, which is why the two
worlds cannot be mistaken for each other at a glance. And it is a
succession, not a specimen: fine roots fade in minutes while the woody
skeleton persists, plants spend their lives in a quarter-hour and rest as
seeds, rain wakes the seed bank, and the descent leans after whatever is
growing — so the window never runs out of plants, and never holds the same
community twice. `DESIGN.md` §15 is the design and the build record.

It is a backend like the others: structural, so it applies to a new field,
and it keeps its own saved column — switching away and back finds it where
you left it, deeper. The flash-safety bound of [Safety](#safety) is the same
output stage and holds identically; the descent is honest motion the
limiter's reprojection is told about, so it costs the luminance budget
nothing. Start it with `--backend rhizotron`, or from the panel.

## Adjusting it

Eight knobs, all 0–1:

| Knob | What it changes |
|---|---|
| **Intensity** | How much is happening — network density, contrast, colour activity |
| **Scale** | Feature size, from fine filaments to broad forms |
| **Tempo** | Speed of flow, drift, and colour rotation |
| **Palette** | Where the colour range sits on the hue circle |
| **Brightness** | Overall level and the background |
| **Filament glow** | How luminous the filaments are against the ground |
| **Depth** | Focus falloff, atmosphere, and how far the back fades |
| **Parallax** | How far the viewpoint drifts, and how briskly |
| **How often** | How frequently events arrive on their own — under **Events**, not with the rest |

Presets: `default`, `quiet`, `dense`, `deep`, `ember`, `luminous`, `current`.
All of them keep a dark ground.

**Parallax** is the one to reach for if either view looks flat. Everything
**Depth** moves is a shading trick applied to a *normalised* depth — how much
the far material is fogged, dimmed, desaturated and blurred — so it says the
same thing about the back of the scene however far back that actually is.
Parallax is the only cue that comes from the scene *moving*, and it is what
lets you see past the near material rather than being told it is nearer.

The readout is what you get: the share of the screen's width that the near
material slides against the far material. At the top of the knob that is a
quarter of the width, spent at about 8 px/s on a 1440p display. It cannot
flicker at any setting — the drift is a random walk behind a lag, so
consecutive frames move in the same direction, and what you see is a slow pan
rather than a shake.

**Under the volumetric view, the slab's thickness caps it, and that is the
thing worth understanding.** Parallax is thickness times the tangent of the
viewing angle. The default slab is 48 voxels deep against 512 wide — a sheet of
paper — and there is only so much depth to be had by walking around a sheet of
paper. Swinging further does not find more; it finds the same sheet seen
edge-on. So the two knobs compound, and neither does much alone:

| Thickness | Parallax at max | Near/far travel at 1440p |
|---|---|---|
| 48 (default) | held to 8% | ~170 px |
| 96 | held to 15% | ~350 px |
| 144 | 22% | ~520 px |
| 288 | 25% | ~580 px |

If the parallax readout stops rising as you drag it, the Thickness slider is
what is holding it. **Turn both up together.**

Every change — slider, preset switch, or file edit — is **ramped, never stepped**,
so adjusting something can't itself produce a visual jolt. Switching presets is a
slow transition rather than a cut.

### The config file

`~/.config/anastomosis/config.toml`, hot-reloaded on save.

```toml
preset_name = "default"
backend = "layered"          # or "volumetric"
volume_detail = "standard"   # "fine" or "finest"; volumetric only

[macros]
intensity = 0.5
filament_glow = 0.45
event_rate = 0.5

[overrides]
# Pin individual primitives by dotted path; these beat the macros.
"render.filament_luma" = 0.42
"reaction.feed" = 0.019
"volume.depth" = 96          # what the Thickness slider writes
"render.parallax" = 0.30     # viewpoint drift, as a fraction of screen width
"render.parallax_tau" = 60   # seconds; how long it takes to change its mind
```

There are around 70 primitive parameters underneath the macros; the field names
in `anastomosis/config.py` are the dotted paths. Safety-relevant values are
clamped to hard ceilings on load — an out-of-range value is corrected with a
warning rather than rejected, because a typo shouldn't end a session that has
been running for two days.

## Safety

The output stage bounds how fast any pixel's lightness can change, in Oklab,
against the motion-compensated previous frame. At the default of 1% per frame at
30 FPS, a 10% luminance excursion takes at least 333 ms and an opposing pair at
least 667 ms — **1.5 flashes/second against the WCAG 2.3.1 limit of 3**. The
user-settable ceiling is 0.012 (1.8/s).

This holds even if the simulation blows up, a parameter is set absurdly, or a
shader has a bug — the limiter is downstream of all of it. The test suite asserts
it two ways: with flow disabled, so reprojection is the identity and the
per-pixel bound is exact; and under normal operation against the WCAG area
criterion. Both are checked while parameters are slammed between extremes.

The two ceilings hold as a pair, not just individually: the per-frame limit
and the frame-rate cap multiply out to a per-second budget, and raising the
frame rate in the config automatically shrinks the per-frame allowance so the
worst case stays 1.8 flashes/second at any frame rate the settings allow.

If you are photosensitive, note that this is a well-tested engineering bound, not
a medical assurance.

## Performance

Sized for an RTX 3080 at 1440p, where it uses **under 10% of the GPU** — normal
desktop work won't notice it. It is not sized to run alongside a game.

Simulation and presentation are decoupled: the sim runs at ~20 Hz, the display at
30, with motion-compensated interpolation between sim states. That makes 20 Hz
look *smoother* than simulating at 30, and lets the budget governor throttle the
tick rate invisibly if frames run long. Resolution is never changed at runtime,
because that would be plainly visible.

## If it freezes

A frozen window used to leave nothing behind: the process stayed up, the last
log line was ordinary telemetry from before it went wrong, and there was
nothing to go on. There is now a watchdog on a thread of its own that notices
the frame loop has stopped and writes a report while it is still stuck.

Reports land in `~/.local/state/anastomosis/diagnostics/`:

```
stall-20260820-141133.txt   one freeze, sampled every 30s while it lasts
dump-20260820-141133.txt    a fault noticed by something other than the watchdog
anastomosis.log             the log, kept here because a freeze needs the
                            lines before it and a desktop has no console
crash.log                   hard crashes, and manual dumps (below)
```

A stall report says which phase of the frame the loop went into and never came
out of — `tick`, `acquire`, `render`, `telemetry`, `checkpoint`, or `idle` for
a loop that simply stopped being asked to paint — carries a stack for every
thread in the process, and repeats both every 30 seconds while the freeze
lasts. Two samples with the same stacks and an unmoving CPU counter mean
genuinely wedged; a moving one means merely slow. It says what the window was
doing at the time too — on screen or not, and the size the canvas had against
the size the window had — which is what separates a loop nobody is asking to
draw from one that is stuck. It logs an ERROR line at the
same time, and a WARNING if the loop later recovers.

Being asked to paint is not guaranteed — a minimised or occluded window
legitimately stops — so a loop between frames is given 45 seconds before it
counts as stalled, against 10 seconds inside a frame. Adjust with
`--stall-timeout SECONDS`, or pass `0` to switch the watchdog off. While the
window is off screen the gap between frames stops counting entirely, because no
timeout can tell a window left minimised over lunch from a freeze.

Most of the freezes that get this far are not the loop being stuck at all: the
window is up, every thread is idle, and the render scheduler has just stopped
asking for frames — usually after a window event went missing, which a
fullscreen toggle can cause. A poll of its own now watches for that, re-asserts
what the window is actually doing a couple of times a second, and forces a
frame if none has been asked for in three seconds. In the log:

```
WARNING no frame was asked for in 4.2s with the window up at 1920x1080; forced one (1 in a row)
```

One of those is a hiccup that was recovered from. A run of them means the
scheduler is gone and the session is being kept alive a frame at a time — it
writes a `dump-*.txt` report saying so, and is a good moment to save and
restart.

If it ever freezes so completely that even the watchdog cannot run, which is
what happens if a driver call wedges while holding Python's interpreter lock,
there is still a way in from outside:

```bash
kill -USR1 $(pgrep -f anastomosis)   # appends every thread's stack to crash.log
```

That is a C-level handler, so it works when nothing else does, and the program
carries on running afterwards.

## Development

```bash
pip install -e ".[dev]"
pytest -m "not slow"          # ~1 min; skips the Gray-Scott sweeps and GPU soaks
pytest -n 4 --dist worksteal  # everything, ~4 min on a 4-core software adapter
pytest                        # the same on one core, ~7 min
```

The slow tests are almost entirely llvmpipe compute, so they parallelise
cleanly across workers; `--dist worksteal` keeps the two long volumetric tests
from serialising behind each other the way per-file distribution would.

The suite runs headless on a software adapter (Mesa's lavapipe), so it works in
CI without a GPU:

```bash
apt-get install mesa-vulkan-drivers
```

Worth knowing about the tests, because they are load-bearing rather than
decorative — between them they caught a leak in the flash-safety bound, an
arithmetic error in the documented safety ceiling, two ways for regions of the
simulation to die permanently, and an absorbing state that made the whole field
unrecoverable:

- `test_flash_safety.py` — the invariant above. The most important file here.
- `test_regime.py` — locks in the Gray-Scott parameter choices, which were
  measured by sweeping the map rather than guessed. Marked slow.
- `test_soak.py` — non-repetition, NaN recovery, and no per-frame allocation.
- `test_parity.py` — the WGSL reaction against a numpy reference, so shader bugs
  surface as failures rather than as "it looks a bit wrong".
- `test_agents.py` — the agent layer's junction behaviour and respawn, by
  dispatching the compute shader against a world with one filament in it and a
  known right answer.
- `test_checkpoint.py` — that a resumed engine evolves *bit-identically* to the
  one it was captured from, which is the only way to catch a piece of state
  quietly left out of the snapshot.
- `test_shutdown.py` — that closing the window saves the field and really ends
  the process, the last part in a subprocess with a live Qt loop, because a
  session left running behind a closed window leaves no other trace.
- `test_audio.py` — the resonance mode's front end and drive (§16): that the
  feature extractor is deterministic in the sample stream whatever the
  chunking, bounded under hostile input, and treats silence as zeros; that
  the modulation layer is the identity at silence, touches exactly its
  whitelist — which is asserted to exclude every brightness and glow path —
  and stays inside the motion envelope §14's sweep certified; and that every
  way capture can be unavailable is a status line rather than an exception.
  The flash suite additionally slams the drive's whole reach, un-ramped,
  against the per-pixel bound.
- `test_rhizotron.py` — the plant-root backend (§15, steps 1–3): that the
  descent really is an exact integer translation, bit for bit, with fresh
  soil generated below; that the soil is deterministic in its seed and
  different ten thousand rows down; that the scroll reaches the safety
  stage's reprojection with the right sign and magnitude; that the flash
  bound holds exactly with the descent stopped while the wetting machinery is
  slammed, and by the WCAG area criterion with it running in the rain; that
  rain soaks *downward*, isolated against a control run; that gravitropism
  turns a sideways tip down, the branch tree is consistent slot by slot,
  structure builds downward and ages upward, stones cost the plant travel,
  and the shipped crispness sits inside `crisp_sweep.py`'s measured
  certificate; and that a resumed column — descent counters, structure and
  every tip included — evolves bit-identically.
- `test_volume.py` — the volumetric backend: that its flow really is
  divergence-free (checked numerically, because the failure it prevents is
  pigment slowly pooling over hours), that the slab wraps on all three axes and
  carries structure through depth, that the flash-safety bound holds under it
  too, that switching backends keeps both fields, and that changing the slab's
  thickness grows a new field without moving anything the march is calibrated
  with. Also the three lateral sizes: that each grows the shape it names, that
  a wider one is sharper rather than merely bigger, and that changing size
  regrows the field at the new width rather than the old.

## Layout

```
anastomosis/
  app.py          window, frame pacing, hot reload, budget governor
  audio.py        the resonance mode's ears: capture, features, modulation (§16)
  backend.py      what every backend shares: output chain, safety, plumbing
  engine.py       the layered 2.5D backend
  volume.py       the volumetric slab backend
  rhizotron.py    the plant-root world: soil, moisture, the descent (§15)
  config.py       parameters, macros, safety ceilings, ramping
  gpu_params.py   GPU struct layout (generates the WGSL, drives the packing)
  events.py       Poisson-arrival slow events
  checkpoint.py   periodic save and restore of the simulation state
  diagnostics.py  stall watchdog and crash handler
  bluenoise.py    void-and-cluster dither mask
  shaders/        32 top-level WGSL modules, plus three shared includes
  ui/             Qt control panel
```

## Licence

MIT. See `LICENSE`.
