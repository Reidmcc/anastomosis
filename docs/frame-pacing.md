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

When it is not, three forced frames in a row say the scheduler is not coming
back, and write a report — the loop is alive, so the watchdog never would, and
the stacks are the only thing that says why it stopped. From that point the
poll stops being a reconciliation and becomes the frame clock: it runs at the
frame interval and forces every frame itself, so a session the scheduler has
abandoned keeps the rate it was asked for instead of creeping at a frame every
few seconds. The simulation is paced off elapsed real time and cannot tell the
difference. The moment a frame arrives that the poll did not force, it hands
the pacing back.

That last part is not a refinement, it is the second freeze. A frame every few
seconds was described here as a bad way to run and a much better way to stop —
the field keeps its state, the panel keeps working, the user can save and quit
rather than kill a wedged process. What a second report made plain is that from
the outside it is not a way to *run* at all: a window that moves once every four
seconds is a frozen window, indistinguishable from the fault it is recovering
from, and a session nobody can tell is being carried is a session nobody knows
to restart.

## What "cannot say" means

The poll asks the window three questions and can get three answers to the
first: on screen, off screen, and no answer at all — a canvas with no window
behind it, or a window that raised on being asked. The kick has always treated
the third as "up": a window that will not say whether it is on screen is not a
window that has said it is off, and refusing to draw into it is how a session
stops for no reason.

The pause did not. It was pushed only on a definite answer, which left one
route by which the original fault could survive an unlimited number of polls: a
canvas paused by a minimise event that never had its counterpart, plus a window
that has stopped answering, is a session that is never drawn again. The forced
frame cannot save it — going around the scheduler is the whole point of forcing
one, and it leaves the pause exactly where it found it — so the session is
kicked for as long as it is left running and never comes back. That is the
second freeze, exactly: the report said the window was on screen, the scheduler
was paused, and every frame in the last four minutes was one the poll had
forced.

So both answers now follow the same rule. "Cannot say" is pushed as "up", and
its worst case is one poll's drawing into a window that really was off screen,
which the next poll that gets an answer undoes. What is not assumed is the rest
of it: the watchdog is only told the loop is parked on purpose when the window
actually said so, and the report prints "cannot say" rather than the last
definite answer — which is the line that would have explained the second freeze
on sight instead of claiming a window that had stopped speaking was on screen.

Two more things a report of this shape has to carry, both learned the same way.
What the *scheduler* was doing, since "it stopped asking for frames" has three
quite different causes — it was paused, it is waiting for a frame it asked for
and was never told about, or it is asking and the canvas is throwing every
frame away — and which one it is decides whether the session was recoverable
and by what. And whether the forced frames were actually drawn: a canvas
cancels every frame it is given when it believes it has no size or has been
closed, forced frames included, and a kick that counted itself rather than the
frame described a session drawing nothing whatever as one being carried.

### 8.3 The other machine — laptops and integrated GPUs

Everything above §8.2 is sized against one desktop card. §8.1's budget spends
its headroom deliberately — native 1440p on the front layer, 20 Hz rather than
15, wide diffusion kernels — and every one of those is a decision made *because*
there was headroom. On an integrated GPU there is not, and the interesting thing
is which parts of the design that changes and which it does not.

**It was never a portability question.** The pipeline is core WebGPU throughout:
no optional features, no raised limits, `rgba16float` chosen precisely because it
is both storage-capable and filterable in core (see `engine.py`'s resource notes).
Counted out of the WGSL, the worst pass uses 3 storage textures against a
guaranteed 4, 5 storage buffers against 8, 6 sampled textures against 16, 12,288
bytes of workgroup memory against 16,384, and 256 invocations per workgroup
against 256. Every binding sits inside the specification's guaranteed minima,
which is what an integrated adapter actually reports, so the whole application
*builds* on one. `test_integrated.py` asserts this against the source rather than
against a device, because no machine anybody develops on would ever show it
breaking.

So it is a question of cost, and cost on this hardware has a different shape.

**Bandwidth is the machine's, not the card's.** §8.1's passes are bandwidth-bound
stencil and gather work, and it prices them against 760 GB/s of dedicated VRAM.
An integrated GPU has no VRAM: it reads system memory, at something between
68 GB/s (dual-channel LPDDR4x) and ~136 GB/s (LPDDR5x), *shared with the CPU and
the compositor*. The simulation at §8.1's own arithmetic is ~14 GB/s at 1440p —
2% of the target card and 20% of a laptop's entire memory bus, before the render
side or anything else the machine is doing. The requirement was never "runs"; it
was "leave the machine usable" (§1), and that is the number that has to stay
small.

Four changes follow, and they are the same change four times: something decided
once, for a machine with headroom, decided instead from what is there.

**Which GPU.** `power_preference` was pinned to `"high-performance"`, which on a
laptop with switchable graphics is a request for the discrete card — for a
program explicitly sized to leave the machine usable and expected to run for days
on a battery. It now passes no preference by default, which leaves the choice to
the platform; `--gpu integrated` and `--gpu discrete` say otherwise. On a machine
with one GPU all three find it. An adapter that comes back empty is now a
diagnosable message rather than an `AttributeError` from inside `device.py`.

**How much is simulated.** `render.cell_budget` caps the whole stack's cell count
and `Geometry.derive` shrinks the layers to fit it. Zero — no ceiling — remains
the default, so the target card keeps its native 1440p front layer; an integrated
adapter gets 3 M cells unless `--scale`, `render.base_scale` or
`render.cell_budget` has already answered.

The lever this uses was already there and already free. §8 makes simulation
resolution independent of the window on purpose — the compositor samples layers
in normalised coordinates, a resize rebuilds the presentation chain only — so
simulating smaller is not an approximation of anything, it is the same field with
fewer cells in it. It costs less sharpness than it sounds like, too: §4.7's whole
feature-size apparatus is calibrated in *cells*, so a smaller grid gives the same
morphology at slightly larger on-screen features rather than a blurrier version of
the same picture.

The rhizotron (§15) is under the same ceiling. Its soil pane is one layer rather
than three, but it is a *full-window* layer, so at 1600p it is 4.1 M cells —
more than the stack's front sheet — and its passes read the same shared memory.
Sizing two backends of three would have been an odd place to stop.

Three million is one 1080p stack (1920×1080 + 960×540 + 480×270 = 2.72 M), so the
commonest laptop panel is untouched. Where it bites is the panels that arrived
after §8.1 was written:

| Window | Uncapped | Capped at 3 M | Field memory |
|---|---|---|---|
| 1920×1080 | 2.72 M cells | unchanged | 271 MB |
| 2560×1440 | 4.84 M | 2.87 M | 482 → 285 MB |
| 2560×1600 | 5.38 M | 2.83 M | 535 → 282 MB |
| 2880×1800 | 6.81 M | 2.83 M | 678 → 282 MB |

At 3 M cells the simulation moves about 8.6 GB/s, the per-frame interpolation and
depth of field about 2.9 GB/s at 30 Hz, and the window-sized output stage a few
more: call it 16–17 GB/s at a 1600p window, about a quarter of a dual-channel
laptop's bus. Uncapped at 1800p it would be over half.

**The governor's second lever.** §8's governor throttles the sim tick rate and
nothing else, on the reasoning that the interpolator hides it completely. That is
right, and it is only *sufficient* while the simulation is what costs. On an
integrated GPU the per-frame render work is at least as likely to be what is over
budget — and against a render-bound frame, lowering the tick rate degrades motion
quality and recovers exactly nothing: the governor walks the tick rate to its
0.35 floor, the interpolator starts extrapolating over intervals three times
longer, and the frame is still late.

So there is a second lever, reached only when the first is spent: the presented
frame rate. Recovery runs in the opposite order — the visible degradation is
given back before the invisible one — and is judged against the slot at the
*full* rate rather than the reduced one, or the two would chase each other, since
dropping to 20 fps makes the slot half again as long and that would itself read
as headroom.

Resolution is still never touched at runtime, for the reason §8 gives: it would
be a plainly visible discontinuity. The ceiling above decides it once, when a
field is grown.

Lowering the presented rate is not a flash-safety question, and it is worth
saying why it cannot become one. `config.validate` sizes the per-frame lightness
allowance as `MAX_LUMA_PER_SECOND / max_fps` — against the *cap*, not against the
rate achieved — so presenting fewer frames only slows the worst case further.
Presenting more would be the dangerous direction, and `_apply_frame_rate` cannot:
it clamps to `max_fps` on the way out, which `test_integrated.py` asserts.

**Battery.** Nothing here had ever had to care where the electricity came from. A
laptop makes it matter: a field left running holds the GPU out of its idle states
for the whole of a battery. `power.py` reads the machine's power source on each
platform's own mechanism — `/sys/class/power_supply` on Linux,
`GetSystemPowerStatus` on Windows, `pmset` on macOS — off a thread, once a
minute, because the macOS read is a subprocess and the frame loop may not block
on one.

The answer has three states, not two, and the third is the important one: on
mains, on battery, and *cannot say*. "Cannot say" reads as mains, which is the
same rule the window poll follows for a window that will not say whether it is on
screen (§8.2) — a desktop reports no power source at all, and throttling one for
a battery it does not have is the failure here nobody would ever diagnose. On
battery the backoff uses the governor's own two levers, in the same order: tick
rate to 0.6, frame cap to 20. Resolution, again, untouched.

**Device loss stops being scaffolding.** §13 recorded the rebuild path as present
but untested, which was defensible while the target was a desktop card: losing a
device there means a driver reset somebody is already looking at. On a laptop the
same event is a lid closing, a dock being pulled, or a hybrid-graphics switch —
routine, nightly, to a session meant to run for days. So it is built. Everything
on the far side of the device goes and everything is asked for again in the order
the launch asks for it, retried on a human timescale rather than every frame,
with the work done on the frame loop rather than in the driver's callback.

The field comes back from the checkpoint on disk, and that cost is not
recoverable: reading the live field back would have needed the device that went
away. What returns is the last save — up to fifteen minutes old at the default
interval — and the log says so, rather than letting a silently younger field look
like a clean recovery. This is the one place the interval below is felt.

**Checkpoint interval: five minutes to fifteen.** What a save costs scales with
the simulation, not with the interval: a readback of every field plus an
uncompressed write, about 165 MB for a 1600p stack. At five minutes that is 2 GB
an hour and 48 GB a day, on a drive whose endurance is a consumable, for a
program whose whole proposition is being left running for days. What the shorter
interval bought back was bounded — the field is rolled back, never lost, into a
simulation built to keep growing, and fifteen minutes of a multi-day field is not
a loss anybody can see.

**What is not covered.** The volumetric slab (§5.1) is 666 MB at `standard` and
2.7 GB at `finest`, in *shared* memory, and its raymarch is up to 48 steps of
4-tap depth of field plus a 6-step shadow march per contributing sample. The
early-outs help and it is still not integrated-GPU work at a laptop's native
resolution. Layered is the backend this section is about; nothing stops the slab
being selected on such a machine, and nothing here makes it a good idea.

### 8.4 The other users of the GPU — queue occupancy and `gpu_nice`

Everything above is about this program hitting its own deadlines. This section
is about everyone else's, because the failure that motivated it (issue #40) was
not ours: with an overnight soak running, the *desktop compositor* stopped
compositing — browser panes froze, screenshots timed out — while the GPU sat at
single-digit average utilisation. A machine that looks idle and acts starved.

**The mechanism is occupancy, not utilisation.** Windows preempts GPU work
between submissions, not inside them, and `tick()` never waits for anything.
The interactive app is innocent — its frame loop paces ticks to vsync, so its
queue drains every frame — but a loop that calls `tick()` as fast as Python
can (a soak test, a headless capture) piles submissions into the hardware
queue without bound. Measured on the 3080: the volumetric tick costs ~6.6 ms
of GPU time and ~1.2 ms to encode, so a free-running loop is ~70 ticks deep
after its first hundred and minutes deep after an hour. Every small job the
rest of the desktop submits then waits behind that queue. A one-workgroup
probe dispatch (`tools/gpu_probe.py` — a stand-in for the compositor) that
completes in 1 ms on an idle desktop takes 8 ms at the median and 20 ms at
p99 under a free-running soak; a 60 Hz compositor has 16.7 ms for everything.

**The remedy is two waits at the submission seam.** Every backend's tick ends
at `Backend._submit_tick`, which applies the `anastomosis.nice.GpuNice`
policy: *drain* — wait for this tick's own GPU work before returning, so this
process never holds more than one tick in the queue — and *yield* — sleep
3 ms before the next tick, a window in which this process provably has
nothing queued at all. No dispatch needed splitting: the largest single
dispatch is ~1.7 ms (the volumetric reaction pass), well inside any frame
budget; the entire problem was the unbounded pileup of submissions.

Measured on the 3080, volumetric soak load, probe percentiles:

| condition            | p50    | p99     | max     | soak throughput |
|----------------------|--------|---------|---------|-----------------|
| idle desktop         | 1.0 ms | 6.2 ms  | 6.2 ms  | —               |
| free-run (before)    | 8.4 ms | 19.7 ms | 28.1 ms | unbounded queue |
| drain only           | 2.4 ms | 7.9 ms  | 35.5 ms | 126 ticks/s     |
| drain + 3 ms yield   | 0.5 ms | 6.8 ms  | 6.9 ms  | 89 ticks/s      |

With the full policy the compositor stand-in cannot tell the soak is running.
The throughput cost is real and accepted: a soak's job is to accumulate
simulated hours on a machine somebody is also using.

**Who gets it.** Auto by default: on for hardware adapters, off for software
ones (CI's lavapipe has no compositor to protect, and its tests should not
sleep). Tests and headless scripts construct backends directly and inherit
that. The interactive app overrides it explicitly from config
(`gpu_nice`, CLI `--gpu-nice`, default off) — its pacing already keeps the
queue shallow, and it is the one entry point where throughput is latency.
The sleep goes through the policy's injectable seam, never a raw
`time.sleep` in a timed path, so the pacing tests stay deterministic.
