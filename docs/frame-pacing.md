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
