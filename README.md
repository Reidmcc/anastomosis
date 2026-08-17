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
anastomosis --width 2560 --height 1440
```

Drag the window to your secondary display and go fullscreen there. It uses
borderless windowed fullscreen rather than exclusive, deliberately: exclusive
fullscreen can stall the compositor on your *other* monitor and steal focus,
which would defeat the point.

Resizing the window — dragging an edge, maximising, going fullscreen — does not
interrupt the simulation. The field keeps running with everything it has grown,
and only the presentation follows the new size. It also keeps the resolution it
started with, so if you want the extra detail at a much larger window, pass
`--width`/`--height` at launch (or set `render.base_scale` higher) rather than
resizing into it.

The control panel opens as a separate, ordinary window on your main display. It
is not always-on-top and minimises freely — tuck it away and bring it back from
the taskbar when you want it.

```bash
anastomosis --no-ui                  # no control panel
anastomosis --list-presets
anastomosis --write-config           # write a config file and exit
```

## Adjusting it

Seven knobs, all 0–1:

| Knob | What it changes |
|---|---|
| **Intensity** | How much is happening — network density, event rate |
| **Scale** | Feature size, from fine filaments to broad forms |
| **Tempo** | Speed of flow, drift, and colour rotation |
| **Palette** | Where the colour range sits on the hue circle |
| **Brightness** | Overall level and the background |
| **Filament glow** | How luminous the filaments are against the ground |
| **Depth** | Parallax, focus falloff, and atmosphere |

Presets: `default`, `quiet`, `dense`, `deep`, `ember`, `luminous`, `current`.
All of them keep a dark ground.

Every change — slider, preset switch, or file edit — is **ramped, never stepped**,
so adjusting something can't itself produce a visual jolt. Switching presets is a
slow transition rather than a cut.

### The config file

`~/.config/anastomosis/config.toml`, hot-reloaded on save.

```toml
preset_name = "default"

[macros]
intensity = 0.5
filament_glow = 0.45

[overrides]
# Pin individual primitives by dotted path; these beat the macros.
"render.filament_luma" = 0.42
"reaction.feed" = 0.019
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
pytest                      # ~60s
pytest -m "not slow"        # skip the Gray-Scott sweeps and GPU soaks
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

## Layout

```
anastomosis/
  app.py          window, frame pacing, hot reload, budget governor
  engine.py       GPU resources, pipelines, tick and render
  config.py       parameters, macros, safety ceilings, ramping
  gpu_params.py   GPU struct layout (generates the WGSL, drives the packing)
  events.py       Poisson-arrival slow events
  bluenoise.py    void-and-cluster dither mask
  shaders/        17 WGSL modules
  ui/             Qt control panel
```

## Licence

MIT. See `LICENSE`.
