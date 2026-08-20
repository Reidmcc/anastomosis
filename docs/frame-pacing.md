> Part of the [Anastomosis design](../DESIGN.md). Section numbers (§n) cited here, in other docs, and in code comments resolve via the index in `DESIGN.md`.

## 8. Frame pacing and GPU budget

**Simulation is decoupled from presentation.** Sim runs at ~15 Hz (tunable 8–30);
render at 30 Hz. Between sim ticks, the renderer **motion-compensated interpolates**:
rather than a naive lerp of two states (which is mushy and reintroduces crawl), both
states are advected by the fractional velocity toward the intermediate time and
blended. Motion looks *smoother* than running the sim at 30 Hz, at half the cost.

This decoupling is also what makes the frame budget adaptive without visible
artefacts: if a frame exceeds `gpu_budget_ms`, the governor lowers the **sim tick
rate**, which the interpolator hides completely. It never changes resolution at
runtime — that would be a visible discontinuity.

### 8.1 Target profile — RTX 3080, 2560×1440

Reference numbers: ~760 GB/s memory bandwidth, ~465 Gtexel/s bilinear fill, 5 MB L2.
These passes are bandwidth-bound stencil and gather work, so bandwidth is the
binding constraint, not FLOPs.

| Stage | Work | Cost |
|---|---|---|
| Sim | 4.9 M cells (1440p + 1/2 + 1/4) × 6 passes × 20 Hz = 588 M cell-updates/s, ~24 B effective each | **~14 GB/s** (1.9% of bandwidth) |
| Agents | 1.55 M agents × 20 Hz = 31 M steps/s → ~93 M sensor samples, ~124 M atomic adds/s | negligible |
| Render | 3.7 M px × ~30 taps × 30 Hz = 3.3 G taps/s bilinear `rgba16f` | ~0.7% of texture fill |

Total steady state lands **under 10% of the card**, including present and driver
overhead. That is comfortably inside "leave the machine usable" — normal desktop
work, video, and an IDE will not notice it. It is *not* sized to coexist with a
game, which matches the stated requirement.

Because the headroom is large, it is spent on quality rather than banked:

- front layer simulated at **native 1440p** (no upscale) for fine filament detail;
- sim tick at **20 Hz** rather than 15, so the interpolator extrapolates less;
- ~1.5 M agents total, enough for dense network structure without individual
  deposits ever being visible;
- wider, higher-quality separable diffusion kernels rather than minimal 5-tap
  stencils — smoother fields, which directly serves the no-punctuation goal;
- multiple RD substeps per tick for finer temporal resolution in the reaction term.

The budget governor is still worth building: the user may be running other GPU work,
and a 3080 driving a second display through the compositor has variable overhead. It
throttles the **sim tick rate** only, which the interpolator hides completely.

**Headroom check:** even the volumetric slab backend (§5.1) fits inside ~10% on this
card, so the depth-backend decision can be made on aesthetics rather than cost.

**Secondary-display specifics:**

- Borderless windowed fullscreen, never exclusive fullscreen — exclusive mode can
  stall the compositor on the other display and steal focus. `rendercanvas` has no
  fullscreen API of its own, so this reaches past it to the native window
  (`anastomosis/window.py`): Qt's `showFullScreen`, which is a window state and never
  a mode change, and for glfw an undecorated window sized to the monitor it is
  already on — deliberately *not* `set_window_monitor`, which is the exclusive path.
  Toggled with **F11**, from the render window or the control panel, and exactly
  reversible: leaving restores the frame the window had, maximised included.
- Never take input focus or capture the cursor.
- `PresentMode::Fifo` (vsync). At 60 Hz, 30 FPS is exactly every other vsync. At 144
  Hz, 30 is not an integer divisor (144/30 = 4.8) — `rendercanvas`'s `max_fps` will
  pace to the nearest vsync, giving slight cadence jitter. Because the renderer
  interpolates to the *actual* elapsed time rather than assuming a fixed step, this
  jitter is not visible.
- Survive the display sleeping, waking, or being unplugged: reconfigure the surface
  rather than treating it as fatal.
- **A window resize rebuilds the presentation chain only.** Simulation resolution is
  fixed when the session starts, for the same reason the governor never touches it:
  re-resolving a running field is a visible discontinuity, and rebuilding the layers
  would additionally discard every field, every agent and the tick count — a hard
  restart in the middle of a session meant to run for days. Only the HDR target, the
  final ping-pong and the exposure partials follow the window; the compositor samples
  layers in normalised coordinates, so they need not match it. Two details make the
  seam invisible: the frame on screen is resampled into the new history buffer (the
  slew limiter emits `history + bounded step`, so an empty history would fade up from
  black over about a second), and a shape change is absorbed by sampling *more* of the
  toroidal field along the axis that grew, rather than by stretching it. Sizes are
  applied once they have held for ~150 ms, so dragging an edge reallocates once.
- Optionally drop to a lower sim rate when the window is not visible.

### 8.2 When the loop stops — `diagnostics.py`

Everything above is about a loop that runs slowly. A loop that stops is a
different failure with a different problem attached to it: it produces no
evidence. There is no exception, so no traceback; the window keeps showing the
last frame it drew, so nothing changes on screen except that it stopped
changing; the process stays up, so the shell says nothing; and the log's last
line is an ordinary telemetry line from a minute before it went wrong. A user
who reports one has nothing to hand over, and the state that would explain it
dies with the process they eventually have to kill.

So the evidence is taken while the freeze is still happening, by a thread that
is not the one that is stuck. The frame loop marks each phase as it enters it —
`tick`, `acquire`, `render`, `telemetry`, `checkpoint`, and `idle` for the gap
between frames — which is one tuple assignment and one clock read, cheap enough
for a path that runs thirty times a second. A watchdog thread notices those
marks stop moving and writes a report: the phase, the simulation's state going
into it, and a stack for every thread in the process.

Three constraints shape it, and each rules something out:

- **It cannot need the stuck thread.** No lock the frame loop could hold, no
  GPU call — the readback it would queue is quite possibly the thing that is
  stuck — and no wgpu or Qt. The phase mark is a whole-tuple rebind rather than
  two fields precisely so that reading it needs no lock.
- **A single stack cannot tell a deadlock from slowness.** So the report is
  sampled again every 30 seconds while the freeze lasts, with the process CPU
  counters beside each sample. Identical stacks and unmoving counters mean
  wedged; either one moving means grinding.
- **Some freezes stop Python itself.** A driver call that wedges while holding
  the GIL stops every Python thread, the watchdog's included. That case cannot
  be covered from inside, so it is covered from outside: `faulthandler`'s
  C-level handlers, one armed for a hard crash and one on `SIGUSR1`, both
  writing to a file opened at startup because a handler that has to allocate is
  a handler that cannot run.

The design pressure that is easy to miss is **false alarms**. A report written
every time the window is minimised is a report nobody reads, and one nobody
reads is worth nothing on the day it matters — so a loop between frames is
given 45 seconds against 10 inside a frame. Not being asked to paint is
ordinary; a frame that never returns is not. That leash is not enough by
itself, though, because no timeout is long enough for a window left minimised
over lunch and short enough to be a watchdog: so the application, which *can*
see the window, tells the watchdog when the loop is parked on purpose, and only
the patient phases are excused by it. A frame that never returns is still a
fault whether or not anybody is looking.

## Not being asked to draw

The loop does not drive itself. `rendercanvas` owns a scheduler that asks for
each frame, and every phase above happens because it asked. When it stops
asking, the loop is neither wedged nor slow — it is simply never called again,
and because the Qt backend re-blits the last bitmap on every expose, the window
goes on looking like an ordinary window that has stopped.

Two things pause that scheduler, and both are the same shape: it stops when the
backend reports the window minimised, and it cancels every frame when the
canvas' cached size is zero. Neither fact is re-derived. Both are written from a
single window event — a state change, a resize — so an event that never
arrives, or one that arrives while the native window is being rebuilt (which is
what a fullscreen transition can do), leaves the canvas holding a belief about
the window that nothing will ever correct. A freeze of exactly this shape is
what `app.py`'s window poll exists for, and the stall report that led to it had
every thread in the process idle.

So the application asks the window instead of waiting to be told. Every two
seconds, off a timer that shares nothing with the scheduler, it reads what the
window says about itself — on screen or not, and how big — and pushes that back
into the canvas *whether or not it has changed*. That is the whole trick: an
edge-triggered fact becomes a level-triggered one, and a missed event costs one
poll rather than the session. A size the canvas disagrees with the window about
is corrected the way the backend's own resize handler would have written it,
and zero is never written back — a window that really is zero-sized really has
nothing to draw.

If frames have stopped anyway, and the window is up and has a size, one is
forced. A forced frame does not go through the scheduler at all: the canvas
draws and presents on the spot, and telling the scheduler that a frame is done
is part of that — which is exactly what a scheduler waiting on a frame it was
never told about is waiting for. One forced frame and it is running again.

When it is not, the session is carried by the poll at a frame every few
seconds. That is a bad way to run and a much better way to stop: the field
keeps its state, the panel keeps working, and the user can save and quit rather
than killing a wedged process. Three forced frames in a row say the scheduler
is not coming back, and write a report — the loop is alive, so the watchdog
never would, and the stacks are the only thing that says why it stopped.

