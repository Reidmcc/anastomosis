# anastomosis

A long-running generative visual field, built for stimming and self-regulation.

Filaments grow, seek each other, and fuse — the way fungal hyphae do, which is
where the name comes from — inside a slowly breathing medium whose colour drifts
with what the simulation is actually doing. It is designed to run for days on a
secondary display while you get on with something else on your main one.

Three things it will not do, by construction:

- **It will not flash.** The output stage enforces a hard per-pixel rate limit on
  lightness, in Oklab, measured against the motion-compensated previous frame.
  The bound holds regardless of what the simulation does — see [Safety](#safety).
- **It will not loop.** Nothing anywhere is a function of the clock. All slow
  variation comes from integrating random increments into stored fields, so there
  is no periodic component to find.
- **It will not settle.** The governing parameters are themselves a slowly
  drifting spatial field, regulated by a loose, slow homeostat. The system is
  never solving the same equation twice.

`DESIGN.md` explains the architecture and the reasoning. This file is how to run
it.

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
and come back as events fade. Nothing here cuts.

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

## Two tunings

At the top of the control panel, **Mode** chooses which tuning the eight
knobs move through: **Regulation** — calm and slow, the original, built for
settling — or **Activation**, the same instrument tuned for sensory seeking:
more motion, more colour, more happening. Switching is nothing like the
Depth choice below it: no reset, no dialog — the field on screen keeps
everything it has grown and changes character over a few seconds, and
switching back is just as smooth. The flash-safety bound is identical in
both modes; neither can flash — see [Safety](#safety).

The activation tuning itself is still being built (`DESIGN.md` §14): its
curve table is currently an exact copy of regulation's, so switching does
not change what you see yet, and it has no presets of its own. The
selector, the config key (`mode = "activation"`) and the plumbing are in
place so the tuning can land as measurements finish.

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

## Development

```bash
pip install -e ".[dev]"
pytest                      # ~8 min on a software adapter
pytest -m "not slow"        # ~40s; skips the Gray-Scott sweeps and GPU soaks
```

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
  backend.py      what the two depth backends share: output chain, safety, plumbing
  engine.py       the layered 2.5D backend
  volume.py       the volumetric slab backend
  config.py       parameters, macros, safety ceilings, ramping
  gpu_params.py   GPU struct layout (generates the WGSL, drives the packing)
  events.py       Poisson-arrival slow events
  checkpoint.py   periodic save and restore of the simulation state
  bluenoise.py    void-and-cluster dither mask
  shaders/        30 top-level WGSL modules, plus two shared includes
  ui/             Qt control panel
```

## Licence

MIT. See `LICENSE`.
